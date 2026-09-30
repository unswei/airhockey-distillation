#!/usr/bin/env python3
"""Analyse every prespecified noise condition using paired cluster resampling."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyse_principal_subgroups import (
    bootstrap_rates, bootstrap_weights, index_episodes, interval, sampling_design,
)
from scripts.analyse_principal_sweep import SAVE_OUTCOMES

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_record(record, opening, opening_hash, family, seed):
    cfg = opening["config"]
    name = "teacher" if seed is None else f"{family}-{seed}"
    if (record["status"] != "completed" or record["classification"] != cfg["classification"] or
            record["opening_sha256"] != opening_hash or record["family_id"] != family or
            record["training_seed"] != seed or
            record["checkpoint_sha256"] != opening["checkpoints"][name]["sha256"]):
        raise ValueError(f"provenance mismatch: {name}")
    rows = record["episodes"]
    if len(rows) != opening["episodes_per_policy"]:
        raise ValueError(f"incomplete episode grid: {name}")
    if set(r["noise_std_mm"] for r in rows) != set(cfg["noise_std_mm"]):
        raise ValueError("noise levels differ from protocol")
    reference = None
    indexed = {}
    for sigma in cfg["noise_std_mm"]:
        by_condition, metadata = index_episodes(
            [r for r in rows if r["noise_std_mm"] == sigma], tuple(cfg["blackout_steps"]))
        if reference is None:
            reference = metadata
        elif reference != metadata:
            raise ValueError("shot metadata changed across noise levels")
        for (shot, blackout), row in by_condition.items():
            if not 1 <= row["steps"] <= cfg["timeout_steps"]:
                raise ValueError("invalid episode length")
            if row["outcome"] not in {
                "returned", "arrested", "safe_deflection", "goal_conceded",
                "missed_goal_without_contact", "upstream_terminal_after_contact",
                "upstream_terminal_without_contact", "timeout_after_contact", "timeout_without_contact",
            }:
                raise ValueError("unknown or incomplete outcome")
            indexed[(shot, blackout, sigma)] = row
    return indexed, reference


def analyse_records(opening, opening_hash, records):
    cfg = opening["config"]
    seeds = cfg["training_seeds"]
    families = ["teacher", *cfg["families"]]
    reference = None
    indices = {}
    for family in families:
        indices[family] = []
        for seed in ([None] if family == "teacher" else seeds):
            name = "teacher" if seed is None else f"{family}-{seed}"
            indexed, meta = validate_record(records[name], opening, opening_hash, family, seed)
            if reference is None:
                reference = meta
            elif meta != reference:
                raise ValueError("shots are not paired across policies")
            indices[family].append(indexed)
    if len(reference) != 225:
        raise ValueError("expected 225 fresh shots")
    frozen_meta = {s["shot"]["shot_id"]: (s.get("alias_family_id"), s["launch_region"], s["target_region"])
                   for s in opening["shots"]}
    if reference != frozen_meta:
        raise ValueError("episode shots do not match the frozen opening")
    units, strata = sampling_design(reference)
    expected = {"alias": 90, "support_near_post_left": 20, "support_near_post_right": 20, "support_goal_centre": 5}
    if {k: len(v) for k, v in strata.items()} != expected:
        raise ValueError("unexpected shot composition")
    unit_index = {u: i for i, u in enumerate(units)}
    replicates = cfg["analysis"]["bootstrap_replicates"]
    sw, uw = bootstrap_weights(len(seeds), strata, replicates, cfg["analysis"]["bootstrap_seed"])
    groups = {}
    for group in cfg["analysis"]["groups"]:
        selected = [s for s, m in reference.items()
                    if group == "all" or (bool(m[0]) if group == "alias" else not m[0])]
        counts = np.zeros(len(units), dtype=int)
        for shot in selected:
            counts[unit_index[str(reference[shot][0] or shot)]] += 1
        conditions, samples = {}, {}
        for blackout in cfg["blackout_steps"]:
            conditions[str(blackout * 20)] = {}
            for sigma in cfg["noise_std_mm"]:
                policies = {}
                for family in families:
                    sums = np.zeros((len(indices[family]), len(units)))
                    outcomes = {}
                    for j, record in enumerate(indices[family]):
                        for shot in selected:
                            outcome = record[(shot, blackout, sigma)]["outcome"]
                            outcomes[outcome] = outcomes.get(outcome, 0) + 1
                            sums[j, unit_index[str(reference[shot][0] or shot)]] += outcome in SAVE_OUTCOMES
                    weights = np.ones((replicates, 1), dtype=int) if family == "teacher" else sw
                    samples[(family, blackout, sigma)] = 100 * bootstrap_rates(sums, counts, weights, uw)
                    policies[family] = {
                        "save_rate_percent": float(sums.sum() / (len(indices[family]) * len(selected)) * 100),
                        "pointwise_95_interval_percent": interval(samples[(family, blackout, sigma)]),
                        "saves": int(sums.sum()), "episodes": int(len(indices[family]) * len(selected)),
                        "outcomes": outcomes,
                        "seed_save_rates_percent": {str(s): float(v / len(selected) * 100)
                            for s, v in zip(["frozen"] if family == "teacher" else seeds, sums.sum(axis=1), strict=True)},
                    }
                conditions[str(blackout * 20)][str(sigma)] = {"policies": policies}
        for blackout in cfg["blackout_steps"]:
            for sigma in cfg["noise_std_mm"]:
                entry = conditions[str(blackout * 20)][str(sigma)]
                clean = conditions[str(blackout * 20)]["0"]["policies"]
                entry["noise_minus_clean"] = {family: {
                    "estimate_points": entry["policies"][family]["save_rate_percent"] - clean[family]["save_rate_percent"],
                    "pointwise_95_interval_points": interval(samples[(family, blackout, sigma)] - samples[(family, blackout, 0)]),
                } for family in families}
                entry["paired_policy_contrasts"] = {f"structured_k0_minus_{other}": {
                    "estimate_points": entry["policies"]["structured_k0"]["save_rate_percent"] - entry["policies"][other]["save_rate_percent"],
                    "pointwise_95_interval_points": interval(samples[("structured_k0", blackout, sigma)] - samples[(other, blackout, sigma)]),
                } for other in ("gru_n64", "structured_k4", "teacher")}
        groups[group] = {"shots_per_policy_condition": len(selected), "by_blackout_ms": conditions}
    return {"experiment_id": cfg["experiment_id"], "classification": cfg["classification"],
            "analysis": cfg["analysis"], "episode_rows": sum(len(r["episodes"]) for r in records.values()),
            "groups": groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--archive", type=Path, help="Reanalyse the compact public episode archive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-archive", type=Path)
    args = parser.parse_args()
    if bool(args.raw_root) == bool(args.archive):
        parser.error("choose exactly one of raw-root or archive")
    if args.raw_root:
        opening_text = (args.raw_root / "opening.json").read_text()
        opening = json.loads(opening_text)
        names = ["teacher", *(f"{f}-{s}" for f in opening["config"]["families"] for s in opening["config"]["training_seeds"])]
        texts = {name: (args.raw_root / f"{name}.json").read_text() for name in names}
        bundle = {"opening_json": opening_text, "result_json": texts}
    else:
        bundle = json.loads(gzip.decompress(args.archive.read_bytes()))
        opening_text, texts = bundle["opening_json"], bundle["result_json"]
        opening = json.loads(opening_text)
    opening_hash = hashlib.sha256(opening_text.encode()).hexdigest()
    result = analyse_records(opening, opening_hash, {n: json.loads(t) for n, t in texts.items()})
    result["provenance"] = {
        "opening_sha256": opening_hash, "analysis_script_sha256": digest(Path(__file__)),
        "bootstrap_helper_sha256": digest(ROOT / "scripts/analyse_principal_subgroups.py"),
        "raw_results_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in texts.items()},
        "config_sha256": opening["config_sha256"],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if args.write_archive:
        with args.write_archive.open("xb") as stream:
            stream.write(gzip.compress(json.dumps(bundle, sort_keys=True).encode(), mtime=0))
    print(json.dumps({"episodes": result["episode_rows"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
