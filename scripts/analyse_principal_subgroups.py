#!/usr/bin/env python3
"""Post-hoc subgroup analysis of existing principal-test episode records.

Run from the repository with PYTHONPATH=src python -m
scripts.analyse_principal_subgroups --test-root PATH --output NEW_FILE.json.
No policies are run and no principal artefacts are modified. Intervals are
exploratory, pointwise percentile intervals, not multiplicity-adjusted tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyse_principal_sweep import SAVE_OUTCOMES


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/principal_sweep_v1_statistics.json"
FROZEN_SHA256 = "55ad1f50399e54487728ade10e6605a993064369a80ec48c0674704f2f378902"
BLACKOUTS = (0, 5, 10, 15, 20, 25)
GROUPS = (
    "alias", "support", "target_left", "target_right", "target_centre",
    "alias_left", "alias_right",
)
CONTRASTS = (
    ("structured_k0", "feed_forward"),
    ("structured_k0", "finite_stack_10"),
    ("structured_k0", "gru_n64"),
    ("structured_k0", "teacher"),
    ("structured_k1", "structured_k0"),
    ("structured_k2", "structured_k0"),
    ("structured_k4", "structured_k0"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_episodes(
    episodes: list[dict[str, Any]], blackouts: tuple[int, ...] = BLACKOUTS,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, tuple[Any, ...]]]:
    """Require one unchanged shot grid across horizons, including complete pairs."""
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    metadata: dict[str, tuple[Any, ...]] = {}
    for episode in episodes:
        shot = str(episode["shot_id"])
        blackout = int(episode["blackout_steps"])
        key = (shot, blackout)
        if key in indexed:
            raise ValueError(f"duplicate shot/horizon: {key}")
        if blackout not in blackouts:
            raise ValueError(f"unexpected blackout: {blackout}")
        observed = (
            episode.get("alias_family_id"), episode["launch_region"],
            episode["target_region"],
        )
        if observed[2] not in {"near_post_left", "near_post_right", "goal_centre"}:
            raise ValueError(f"unknown target region: {observed[2]}")
        if shot in metadata and metadata[shot] != observed:
            raise ValueError(f"shot metadata changes across horizons: {shot}")
        metadata[shot] = observed
        indexed[key] = episode
    if not metadata or set(indexed) != {(s, b) for s in metadata for b in blackouts}:
        raise ValueError("incomplete shot/horizon grid")
    pairs: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for alias, launch, target in metadata.values():
        if alias:
            pairs[str(alias)].append((launch, target))
    for alias, members in pairs.items():
        if sorted(members) != [("centre", "near_post_left"), ("centre", "near_post_right")]:
            raise ValueError(f"incomplete or invalid alias pair: {alias}")
    return indexed, metadata


def membership(meta: tuple[Any, ...], group: str) -> bool:
    alias, _, target = meta
    return {
        "alias": bool(alias),
        "support": not alias,
        "target_left": target == "near_post_left",
        "target_right": target == "near_post_right",
        "target_centre": target == "goal_centre",
        "alias_left": bool(alias) and target == "near_post_left",
        "alias_right": bool(alias) and target == "near_post_right",
    }[group]


def sampling_design(metadata: dict[str, tuple[Any, ...]]) -> tuple[list[str], dict[str, list[int]]]:
    """Alias families are units; support shots are separate target-stratified units."""
    strata_by_unit: dict[str, str] = {}
    for shot, (alias, _, target) in metadata.items():
        unit = str(alias or shot)
        stratum = "alias" if alias else f"support_{target}"
        if unit in strata_by_unit and strata_by_unit[unit] != stratum:
            raise ValueError("unit spans incompatible strata")
        strata_by_unit[unit] = stratum
    units = sorted(strata_by_unit)
    strata: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        strata[strata_by_unit[unit]].append(index)
    return units, dict(strata)


def bootstrap_weights(
    n_seeds: int, strata: dict[str, list[int]], replicates: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One set of shared draws for all models, subgroups and blackout durations."""
    if n_seeds < 1 or replicates < 1 or not strata:
        raise ValueError("bootstrap dimensions must be positive")
    columns = [i for indices in strata.values() for i in indices]
    if sorted(columns) != list(range(len(columns))):
        raise ValueError("strata must partition unit columns")
    rng = np.random.default_rng(seed)
    seed_weights = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=replicates)
    unit_weights = np.zeros((replicates, len(columns)), dtype=np.int64)
    for stratum in sorted(strata):
        indices = strata[stratum]
        count = len(indices)
        unit_weights[:, indices] = rng.multinomial(count, np.full(count, 1 / count), size=replicates)
    return seed_weights, unit_weights


def bootstrap_rates(
    sums: np.ndarray, counts: np.ndarray,
    seed_weights: np.ndarray, unit_weights: np.ndarray,
) -> np.ndarray:
    """Ratios of shot-weighted cluster sums, with common seed and unit draws."""
    denominators = seed_weights.sum(axis=1) * (unit_weights @ counts)
    if np.any(denominators == 0):
        raise ValueError("empty subgroup in a bootstrap replicate")
    return ((seed_weights @ sums) * unit_weights).sum(axis=1) / denominators


def interval(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def analyse(test_root: Path, replicates: int = 10_000, seed: int = 9302026) -> dict[str, Any]:
    if digest(FROZEN) != FROZEN_SHA256:
        raise ValueError("frozen principal statistics hash mismatch")
    frozen = json.loads(FROZEN.read_text())
    families = list(frozen["student_save_rates"])
    seeds = sorted(int(s) for s in frozen["student_seed_save_rates"][families[0]]["0"])
    indices: dict[str, list[dict[tuple[str, int], dict[str, Any]]]] = {}
    source_hashes: dict[str, str] = {}
    reference = None
    rows = 0
    for family in ["teacher", *families]:
        indices[family] = []
        for training_seed in ([None] if family == "teacher" else seeds):
            relative = "teacher.json" if family == "teacher" else f"students/{family}-{training_seed}.json"
            path = test_root / relative
            raw = path.read_bytes()
            record = json.loads(raw)
            source_hashes[relative] = hashlib.sha256(raw).hexdigest()
            if record.get("status") != "completed":
                raise ValueError(f"incomplete result: {relative}")
            if record.get("evaluation_schedule_sha256") != frozen["principal_test_schedule_sha256"]:
                raise ValueError(f"schedule mismatch: {relative}")
            if record.get("protocol_sha256") != frozen["protocol_sha256"]:
                raise ValueError(f"protocol mismatch: {relative}")
            if training_seed is not None and (
                record.get("family_id") != family or record.get("training_seed") != training_seed
            ):
                raise ValueError(f"family/seed mismatch: {relative}")
            indexed, metadata = index_episodes(record["episodes"])
            if reference is None:
                reference = metadata
            elif metadata != reference:
                raise ValueError(f"paired shot metadata mismatch: {relative}")
            rows += len(indexed)
            indices[family].append(indexed)
    assert reference is not None
    units, strata = sampling_design(reference)
    if len(reference) != 225 or len(seeds) != 5 or rows != 48_600:
        raise ValueError("unexpected principal grid dimensions")
    expected_strata = {"alias": 90, "support_near_post_left": 20,
                       "support_near_post_right": 20, "support_goal_centre": 5}
    if {s: len(i) for s, i in strata.items()} != expected_strata:
        raise ValueError("unexpected principal subgroup composition")
    unit_index = {unit: i for i, unit in enumerate(units)}
    seed_weights, unit_weights = bootstrap_weights(len(seeds), strata, replicates, seed)
    groups = {
        group: {
            "shots_per_policy": sum(membership(m, group) for m in reference.values()),
            "cluster_units": len({str(m[0] or s) for s, m in reference.items() if membership(m, group)}),
            "by_blackout_ms": {},
        }
        for group in GROUPS
    }
    asymmetry: dict[str, Any] = {}
    reconciled = 0
    for blackout in BLACKOUTS:
        samples_by_group: dict[str, dict[str, np.ndarray]] = {g: {} for g in GROUPS}
        for group in GROUPS:
            selected = [s for s, m in reference.items() if membership(m, group)]
            counts = np.zeros(len(units), dtype=np.int64)
            for shot in selected:
                counts[unit_index[str(reference[shot][0] or shot)]] += 1
            policies = {}
            for family, records in indices.items():
                sums = np.zeros((len(records), len(units)), dtype=np.float64)
                for seed_index, record in enumerate(records):
                    for shot in selected:
                        sums[seed_index, unit_index[str(reference[shot][0] or shot)]] += (
                            record[(shot, blackout)]["outcome"] in SAVE_OUTCOMES
                        )
                weights = np.ones((replicates, 1), dtype=np.int64) if family == "teacher" else seed_weights
                samples = bootstrap_rates(sums, counts, weights, unit_weights) * 100
                samples_by_group[group][family] = samples
                policies[family] = {
                    "save_rate_percent": float(sums.sum() / (len(records) * len(selected)) * 100),
                    "pointwise_95_interval_percent": interval(samples),
                    "saves": int(sums.sum()),
                    "episode_rows": len(records) * len(selected),
                    "seed_save_rates_percent": {
                        str(s): float(n / len(selected) * 100)
                        for s, n in zip(["frozen"] if family == "teacher" else seeds, sums.sum(axis=1), strict=True)
                    },
                }
            contrasts = {}
            for left, right in CONTRASTS:
                contrasts[f"{left}_minus_{right}"] = {
                    "estimate_points": policies[left]["save_rate_percent"] - policies[right]["save_rate_percent"],
                    "pointwise_95_interval_points": interval(samples_by_group[group][left] - samples_by_group[group][right]),
                }
            groups[group]["by_blackout_ms"][str(blackout * 20)] = {"policies": policies, "paired_contrasts": contrasts}
        for family, records in indices.items():
            rates = [np.mean([r[(s, blackout)]["outcome"] in SAVE_OUTCOMES for s in reference]) for r in records]
            expected_rates = ([frozen["teacher_save_rates"][str(blackout)]] if family == "teacher" else
                              [frozen["student_seed_save_rates"][family][str(blackout)][str(s)] for s in seeds])
            np.testing.assert_allclose(rates, expected_rates, rtol=0, atol=1e-12)
            weighted = sum(groups[g]["shots_per_policy"] * groups[g]["by_blackout_ms"][str(blackout * 20)]["policies"][family]["save_rate_percent"] for g in ("alias", "support")) / 225
            np.testing.assert_allclose(weighted, 100 * np.mean(rates), rtol=0, atol=1e-12)
            reconciled += len(rates)
        asymmetry[str(blackout * 20)] = {}
        for name, left, right in (("all_targets_left_minus_right", "target_left", "target_right"),
                                  ("alias_left_minus_right", "alias_left", "alias_right")):
            asymmetry[str(blackout * 20)][name] = {
                family: {
                    "estimate_points": groups[left]["by_blackout_ms"][str(blackout * 20)]["policies"][family]["save_rate_percent"] - groups[right]["by_blackout_ms"][str(blackout * 20)]["policies"][family]["save_rate_percent"],
                    "pointwise_95_interval_points": interval(samples_by_group[left][family] - samples_by_group[right][family]),
                }
                for family in indices
            }
    return {
        "schema_version": 1,
        "status": "completed_post_hoc_descriptive_analysis",
        "analysis_scope": "Existing principal-test episodes only; no new evaluation, training or checkpoint selection.",
        "provenance": {
            "principal_statistics_sha256": FROZEN_SHA256,
            "principal_test_schedule_sha256": frozen["principal_test_schedule_sha256"],
            "protocol_sha256": frozen["protocol_sha256"],
            "analysis_script_sha256": digest(Path(__file__)),
            "test_files_sha256": source_hashes,
            "episode_rows": rows,
            "seed_horizon_rates_reconciled": reconciled,
        },
        "bootstrap": {
            "replicates": replicates, "seed": seed,
            "method": "Paired hierarchical seed/cluster bootstrap, stratified by alias family and support target region.",
            "stratum_unit_counts": expected_strata,
            "shared_draws": "Common training-seed draws and unit draws across all models, groups and horizons; both alias members retained.",
            "teacher": "One frozen policy; shot-cluster resampling only, no training-seed uncertainty.",
            "interval_scope": "Pointwise 95% percentile intervals, conditional on observed subgroup composition; not adjusted for multiple comparisons.",
            "not_principal_intervals": "Additional post-hoc stratification differs from the original principal bootstrap; no original intervals are replaced.",
        },
        "limitations": [
            "Post-hoc subgroups of one task are not independent tasks or fresh generalisation evidence.",
            "Only five training seeds; failure-free subgroups can have degenerate empirical bootstrap intervals, not proof of perfect performance.",
            "Goal-centre support contains only five unique shots and cannot establish robustness.",
            "Alias left/right comparisons share 90 paired families; overall left/right additionally includes distinct support shots.",
            "All six existing horizons and all seven student families are retained; no favourable subset selection.",
        ],
        "groups": groups,
        "left_right_asymmetry": asymmetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=9302026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyse(args.test_root, args.replicates, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "episode_rows": result["provenance"]["episode_rows"],
                      "reconciled_seed_horizon_rates": result["provenance"]["seed_horizon_rates_reconciled"]}))


if __name__ == "__main__":
    main()
