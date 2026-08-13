#!/usr/bin/env python3
"""Generate the predeclared principal statistics, figure and efficiency table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.principal_sweep import load_principal_protocol, sha256_file
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS

SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--accounting-root", type=Path, required=True)
    parser.add_argument("--latency-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    test_root = args.test_root.resolve()
    blackouts = [
        *protocol["evaluation"]["test"]["core_blackout_steps"],
        *protocol["evaluation"]["test"]["extrapolation_blackout_steps"],
    ]
    seeds = [int(value) for value in protocol["matched_seeds"]["training"]]
    results: dict[str, dict[int, dict[str, Any]]] = {}
    schedule_hash: str | None = None
    for family_id in PRINCIPAL_FAMILY_IDS:
        results[family_id] = {}
        for seed in seeds:
            path = test_root / "students" / f"{family_id}-{seed}.json"
            value = json.loads(path.read_text())
            if value.get("status") != "completed" or value.get("family_id") != family_id:
                raise ValueError(f"invalid test result: {path}")
            if int(value.get("training_seed", -1)) != seed:
                raise ValueError(f"test result seed mismatch: {path}")
            observed_hash = str(value["evaluation_schedule_sha256"])
            schedule_hash = schedule_hash or observed_hash
            if observed_hash != schedule_hash:
                raise ValueError("test results do not share one schedule")
            results[family_id][seed] = value
    teacher = json.loads((test_root / "teacher.json").read_text())
    if teacher.get("evaluation_schedule_sha256") != schedule_hash:
        raise ValueError("teacher and students do not share one test schedule")

    replicates = int(protocol["measurements"]["uncertainty"]["replicates"])
    bootstrap_seed = int(protocol["measurements"]["uncertainty"]["seed"])
    point_rates: dict[str, dict[str, float]] = {}
    intervals: dict[str, dict[str, list[float]]] = {}
    outcome_counts: dict[str, dict[str, int]] = {}
    outcome_counts_by_blackout: dict[str, dict[str, dict[str, int]]] = {}
    seed_save_rates: dict[str, dict[str, dict[str, float]]] = {}
    draws: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for blackout in blackouts:
        reference = results[PRINCIPAL_FAMILY_IDS[0]][seeds[0]]["episodes"]
        selected = [e for e in reference if int(e["blackout_steps"]) == blackout]
        units = sorted({_unit_id(e) for e in selected})
        rng = np.random.default_rng(bootstrap_seed + int(blackout))
        draws[blackout] = (
            rng.integers(0, len(seeds), size=(replicates, len(seeds))),
            rng.integers(0, len(units), size=(replicates, len(units))),
            units,
        )
    for family_id in PRINCIPAL_FAMILY_IDS:
        point_rates[family_id] = {}
        intervals[family_id] = {}
        counts: Counter[str] = Counter()
        for seed in seeds:
            counts.update(str(e["outcome"]) for e in results[family_id][seed]["episodes"])
        outcome_counts[family_id] = dict(sorted(counts.items()))
        outcome_counts_by_blackout[family_id] = {}
        seed_save_rates[family_id] = {}
        for blackout in blackouts:
            matrices = _cluster_matrices(results[family_id], seeds, blackout, draws[blackout][2])
            samples = cluster_bootstrap_rates(matrices[0], matrices[1], *draws[blackout][:2])
            raw = [
                _is_save(e)
                for seed in seeds
                for e in results[family_id][seed]["episodes"]
                if int(e["blackout_steps"]) == blackout
            ]
            point_rates[family_id][str(blackout)] = float(np.mean(raw))
            intervals[family_id][str(blackout)] = [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ]
            blackout_counts: Counter[str] = Counter()
            seed_save_rates[family_id][str(blackout)] = {}
            for seed in seeds:
                seed_episodes = [
                    e
                    for e in results[family_id][seed]["episodes"]
                    if int(e["blackout_steps"]) == blackout
                ]
                blackout_counts.update(str(e["outcome"]) for e in seed_episodes)
                seed_save_rates[family_id][str(blackout)][str(seed)] = float(
                    np.mean([_is_save(e) for e in seed_episodes])
                )
            outcome_counts_by_blackout[family_id][str(blackout)] = dict(
                sorted(blackout_counts.items())
            )

    differences = _planned_differences(
        protocol, results, seeds, blackouts, draws
    )
    teacher_rates = {
        str(blackout): float(
            np.mean(
                [
                    _is_save(e)
                    for e in teacher["episodes"]
                    if int(e["blackout_steps"]) == blackout
                ]
            )
        )
        for blackout in blackouts
    }
    efficiency_rows = _efficiency_rows(
        protocol,
        results,
        seeds,
        Path(args.accounting_root).resolve(),
        Path(args.latency_root).resolve(),
    )
    teacher_relative_gap_points = {
        family_id: {
            str(blackout): 100.0
            * (teacher_rates[str(blackout)] - point_rates[family_id][str(blackout)])
            for blackout in blackouts
        }
        for family_id in PRINCIPAL_FAMILY_IDS
    }
    recovery = _gap_recovery(point_rates, protocol)
    statistics = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "principal_test_schedule_sha256": schedule_hash,
        "bootstrap": {
            "method": protocol["measurements"]["uncertainty"]["method"],
            "replicates": replicates,
            "seed": bootstrap_seed,
            "evaluation_unit": protocol["measurements"]["uncertainty"]["evaluation_unit"],
        },
        "student_save_rates": point_rates,
        "student_seed_save_rates": seed_save_rates,
        "student_save_rate_95_intervals": intervals,
        "teacher_save_rates": teacher_rates,
        "teacher_relative_gap_points": teacher_relative_gap_points,
        "planned_differences_at_20_steps": differences,
        "descriptive_gap_recovery_at_20_steps": recovery,
        "student_outcome_counts": outcome_counts,
        "student_outcome_counts_by_blackout": outcome_counts_by_blackout,
        "efficiency_seed_points": _efficiency_seed_points(
            Path(args.latency_root).resolve(), seeds
        ),
        "raw_training_seed_results_reported": True,
    }
    statistics_path = output / "statistics.json"
    statistics_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    _write_table(efficiency_rows, output)
    _write_figure(point_rates, intervals, teacher_rates, blackouts, output / "principal_save_rate.png")
    artefacts = [statistics_path, output / "efficiency_table.csv", output / "efficiency_table.md", output / "principal_save_rate.png"]
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "principal_test_schedule_sha256": schedule_hash,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artefacts
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def cluster_bootstrap_rates(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
    seed_draws: np.ndarray,
    unit_draws: np.ndarray,
) -> np.ndarray:
    """Apply matched hierarchical seed/unit draws to clustered binary outcomes."""

    samples = np.empty(len(seed_draws), dtype=np.float64)
    for index, (seed_draw, unit_draw) in enumerate(zip(seed_draws, unit_draws, strict=True)):
        samples[index] = cluster_sums[np.ix_(seed_draw, unit_draw)].sum() / (
            len(seed_draw) * cluster_counts[unit_draw].sum()
        )
    return samples


def _unit_id(episode: dict[str, Any]) -> str:
    return str(episode.get("alias_family_id") or episode["shot_id"])


def _is_save(episode: dict[str, Any]) -> float:
    return float(str(episode["outcome"]) in SAVE_OUTCOMES)


def _cluster_matrices(
    family: dict[int, dict[str, Any]],
    seeds: list[int],
    blackout: int,
    units: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    index = {unit: value for value, unit in enumerate(units)}
    sums = np.zeros((len(seeds), len(units)), dtype=np.float64)
    counts = np.zeros(len(units), dtype=np.int64)
    for seed_index, seed in enumerate(seeds):
        local_counts = np.zeros(len(units), dtype=np.int64)
        for episode in family[seed]["episodes"]:
            if int(episode["blackout_steps"]) != blackout:
                continue
            unit = index[_unit_id(episode)]
            sums[seed_index, unit] += _is_save(episode)
            local_counts[unit] += 1
        if seed_index == 0:
            counts = local_counts
        elif not np.array_equal(local_counts, counts):
            raise ValueError("paired test results have different cluster membership")
    if np.any(counts == 0):
        raise ValueError("test cluster is empty")
    return sums, counts


def _difference(
    left: str,
    right: str,
    results: dict[str, dict[int, dict[str, Any]]],
    seeds: list[int],
    blackout: int,
    draw: tuple[np.ndarray, np.ndarray, list[str]],
) -> dict[str, Any]:
    left_m = _cluster_matrices(results[left], seeds, blackout, draw[2])
    right_m = _cluster_matrices(results[right], seeds, blackout, draw[2])
    left_samples = cluster_bootstrap_rates(left_m[0], left_m[1], draw[0], draw[1])
    right_samples = cluster_bootstrap_rates(right_m[0], right_m[1], draw[0], draw[1])
    differences = left_samples - right_samples
    raw_left = np.mean([_is_save(e) for seed in seeds for e in results[left][seed]["episodes"] if int(e["blackout_steps"]) == blackout])
    raw_right = np.mean([_is_save(e) for seed in seeds for e in results[right][seed]["episodes"] if int(e["blackout_steps"]) == blackout])
    return {
        "left": left,
        "right": right,
        "estimate_points": float(100.0 * (raw_left - raw_right)),
        "percentile_95_interval_points": [
            float(100.0 * np.quantile(differences, 0.025)),
            float(100.0 * np.quantile(differences, 0.975)),
        ],
    }


def _planned_differences(protocol: dict[str, Any], results: dict[str, dict[int, dict[str, Any]]], seeds: list[int], blackouts: list[int], draws: dict[int, tuple[np.ndarray, np.ndarray, list[str]]]) -> dict[str, Any]:
    blackout = 20
    pairs: dict[str, tuple[str, str]] = {}
    for name in protocol["measurements"]["planned_primary_differences_at_20_steps"]:
        left, right = str(name).split("_minus_", 1)
        pairs[str(name)] = (left, right)
    recurrent = ["structured_k0", "structured_k1", "structured_k2", "structured_k4", "gru_n64"]
    for family in recurrent:
        pairs[f"{family}_minus_feed_forward"] = (family, "feed_forward")
        pairs[f"{family}_minus_finite_stack_10"] = (family, "finite_stack_10")
    for family in ("structured_k0", "structured_k1", "structured_k2", "structured_k4"):
        pairs[f"{family}_minus_gru_n64"] = (family, "gru_n64")
    return {name: _difference(left, right, results, seeds, blackout, draws[blackout]) for name, (left, right) in pairs.items()}


def _gap_recovery(
    rates: dict[str, dict[str, float]], protocol: dict[str, Any]
) -> dict[str, Any]:
    denominator = rates["gru_n64"]["20"] - rates["structured_k0"]["20"]
    threshold = float(
        protocol["measurements"]["descriptive_gap_recovery_at_20_steps"][
            "report_only_when_absolute_denominator_at_least_points"
        ]
    ) / 100.0
    if abs(denominator) < threshold:
        return {
            "reported": False,
            "denominator_points": 100.0 * denominator,
            "reason": "absolute denominator is below the predeclared threshold",
        }
    return {
        "reported": True,
        "denominator_points": 100.0 * denominator,
        "structured_fraction_recovered": {
            family_id: (
                rates[family_id]["20"] - rates["structured_k0"]["20"]
            )
            / denominator
            for family_id in (
                "structured_k0",
                "structured_k1",
                "structured_k2",
                "structured_k4",
            )
        },
    }


def _efficiency_seed_points(
    latency_root: Path, seeds: list[int]
) -> dict[str, list[dict[str, float | int]]]:
    return {
        family_id: [
            {
                "training_seed": seed,
                "median_microseconds": float(
                    json.loads(
                        (latency_root / f"{family_id}-{seed}.json").read_text()
                    )["measurements"]["median_microseconds"]
                ),
                "p95_microseconds": float(
                    json.loads(
                        (latency_root / f"{family_id}-{seed}.json").read_text()
                    )["measurements"]["p95_microseconds"]
                ),
            }
            for seed in seeds
        ]
        for family_id in PRINCIPAL_FAMILY_IDS
    }


def _efficiency_rows(protocol: dict[str, Any], results: dict[str, dict[int, dict[str, Any]]], seeds: list[int], accounting_root: Path, latency_root: Path) -> list[dict[str, Any]]:
    rows = []
    core_blackouts = set(int(x) for x in protocol["evaluation"]["test"]["core_blackout_steps"])
    for family_id in PRINCIPAL_FAMILY_IDS:
        accountings = [json.loads((accounting_root / f"{family_id}-{seed}.json").read_text()) for seed in seeds]
        latencies = [json.loads((latency_root / f"{family_id}-{seed}.json").read_text()) for seed in seeds]
        first = accountings[0]
        for value in accountings[1:]:
            for name in ("total_trainable_parameters", "core_parameters", "recurrent_memory_bytes", "total_policy_carry_bytes", "multiply_adds_per_step"):
                if value[name] != first[name]:
                    raise ValueError(f"accounting varies within {family_id}")
        saves = [_is_save(e) for seed in seeds for e in results[family_id][seed]["episodes"] if int(e["blackout_steps"]) in core_blackouts]
        rows.append({
            "family": family_id,
            "total_trainable_parameters": first["total_trainable_parameters"],
            "core_parameters": first["core_parameters"],
            "recurrent_memory_bytes": first["recurrent_memory_bytes"],
            "total_policy_carry_bytes": first["total_policy_carry_bytes"],
            "multiply_adds_per_step": first["multiply_adds_per_step"],
            "cpu_latency_median_microseconds": float(np.median([x["measurements"]["median_microseconds"] for x in latencies])),
            "cpu_latency_p95_microseconds": float(np.median([x["measurements"]["p95_microseconds"] for x in latencies])),
            "overall_core_test_save_rate": float(np.mean(saves)),
        })
    return rows


def _write_table(rows: list[dict[str, Any]], output: Path) -> None:
    fields = list(rows[0])
    with (output / "efficiency_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[name]) for name in fields) + " |")
    (output / "efficiency_table.md").write_text("\n".join(lines) + "\n")


def _write_figure(rates: dict[str, dict[str, float]], intervals: dict[str, dict[str, list[float]]], teacher: dict[str, float], blackouts: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for family_id in PRINCIPAL_FAMILY_IDS:
        values = np.asarray([rates[family_id][str(x)] for x in blackouts]) * 100.0
        bounds = np.asarray([intervals[family_id][str(x)] for x in blackouts]) * 100.0
        axis.plot(blackouts, values, marker="o", label=family_id)
        axis.fill_between(blackouts, bounds[:, 0], bounds[:, 1], alpha=0.10)
    axis.plot(blackouts, [100.0 * teacher[str(x)] for x in blackouts], color="black", linestyle="--", marker="s", label="teacher reference")
    axis.set(xlabel="Blackout length (steps)", ylabel="Save rate (%)", xticks=blackouts, ylim=(0, 102))
    axis.grid(alpha=0.25); axis.legend(ncol=2, fontsize=8)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
