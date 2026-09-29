#!/usr/bin/env python3
"""Recheck frozen principal statistics and the manuscript comparison table.

Run from the code repository with --raw-root containing read-only copies of
test/, accounting/ and latency/. This reads existing evidence; it does not
evaluate policies, reopen the test experimentally, or change frozen results.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import analyse_principal_sweep as analysis
from airhockey_distill.principal_sweep import load_principal_protocol
from plot_paper_figure3 import load_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    frozen = load_statistics()  # Includes SHA-256 validation of frozen statistics.
    protocol = load_principal_protocol(Path("configs/experiments/principal_sweep_execution_v1.yaml"))
    seeds = protocol["matched_seeds"]["training"]
    families = list(frozen["student_save_rates"])
    read = lambda path: json.loads(path.read_text())
    results = {family: {seed: read(args.raw_root / "test/students" / f"{family}-{seed}.json")
                        for seed in seeds} for family in families}
    teacher = read(args.raw_root / "test/teacher.json")
    checks = 0

    def check(actual, expected, label):
        nonlocal checks
        if isinstance(actual, (int, float, list, np.ndarray)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=label)
        else:
            assert actual == expected, label
        checks += 1

    records = [teacher] + [results[f][s] for f in families for s in seeds]
    for record in records:
        check(record["evaluation_schedule_sha256"], frozen["principal_test_schedule_sha256"], "schedule")
        check(len(record["episodes"]), 1350, "episode count")
    draws = {}
    for blackout in (0, 5, 10, 15, 20, 25):
        key = str(blackout)
        selected = [e for e in teacher["episodes"] if e["blackout_steps"] == blackout]
        check(len(selected), 225, "teacher shots")
        check(np.mean([analysis._is_save(e) for e in selected]), frozen["teacher_save_rates"][key], "teacher rate")
        units = sorted({analysis._unit_id(e) for e in selected})
        rng = np.random.default_rng(frozen["bootstrap"]["seed"] + blackout)
        reps = frozen["bootstrap"]["replicates"]
        draws[blackout] = (rng.integers(0, len(seeds), size=(reps, len(seeds))),
                           rng.integers(0, len(units), size=(reps, len(units))), units)
        for family in families:
            rates = []
            for seed in seeds:
                episodes = [e for e in results[family][seed]["episodes"] if e["blackout_steps"] == blackout]
                check(len(episodes), 225, "student shots")
                rate = float(np.mean([analysis._is_save(e) for e in episodes]))
                check(rate, frozen["student_seed_save_rates"][family][key][str(seed)], "seed rate")
                rates.append(rate)
            check(np.mean(rates), frozen["student_save_rates"][family][key], "family mean")
            sums, counts = analysis._cluster_matrices(results[family], seeds, blackout, units)
            samples = analysis.cluster_bootstrap_rates(sums, counts, *draws[blackout][:2])
            check(np.quantile(samples, [.025, .975]), frozen["student_save_rate_95_intervals"][family][key], "family interval")
    for name, expected in frozen["planned_differences_at_20_steps"].items():
        actual = analysis._difference(expected["left"], expected["right"], results, seeds, 20, draws[20])
        check(actual["estimate_points"], expected["estimate_points"], name)
        check(actual["percentile_95_interval_points"], expected["percentile_95_interval_points"], name)

    rows = analysis._efficiency_rows(protocol, results, seeds, args.raw_root / "accounting", args.raw_root / "latency")
    tracked = {r["family"]: r for r in csv.DictReader(Path("results/principal_sweep_v1_efficiency_table.csv").open())}
    for row in rows:
        for key, value in row.items():
            check(value, tracked[row["family"]][key] if key == "family" else float(tracked[row["family"]][key]), key)
    for record in (read(p) for p in (args.raw_root / "latency").glob("*.json")):
        means = record["measurements"]["repetition_mean_microseconds"]
        check(np.median(means), record["measurements"]["median_microseconds"], "latency median")
        check(np.quantile(means, .95), record["measurements"]["p95_microseconds"], "latency p95")
        check(record["runtime"]["cpu_model"], "Intel(R) Core(TM) Ultra 9 285", "CPU")
        check(record["runtime"]["logical_cpu_affinity"], [0], "CPU affinity")

    labels = {"Feed-forward": "feed_forward", "Ten-step stack": "finite_stack_10",
              "GRU-64": "gru_n64", **{f"Structured $k={k}$": f"structured_k{k}" for k in (0, 1, 2, 4)}}
    table_rows = 0
    for line in args.paper.read_text().splitlines():
        cells = [c.strip() for c in line.split("&")]
        if cells[0] not in labels:
            continue
        family = labels[cells[0]]
        efficiency = tracked[family]
        values = [float(n.replace(",", "")) for n in re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", " ".join(cells[1:]))]
        expected = [round(100 * frozen["student_save_rates"][family]["20"], 1),
                    *[round(100*x, 1) for x in frozen["student_save_rate_95_intervals"][family]["20"]],
                    round(100 * float(efficiency["overall_core_test_save_rate"]), 1),
                    *[float(efficiency[k]) for k in ("total_trainable_parameters", "core_parameters", "total_policy_carry_bytes", "recurrent_memory_bytes", "multiply_adds_per_step")],
                    *[round(float(efficiency[k]), 2) for k in ("cpu_latency_median_microseconds", "cpu_latency_p95_microseconds")]]
        check(values, expected, f"manuscript table: {family}")
        table_rows += 1
    check(table_rows, 7, "all table rows")
    print(json.dumps({"status": "PASS", "checks": checks, "episode_rows": sum(len(r["episodes"]) for r in records),
                      "family_intervals_recomputed": 42, "contrasts_recomputed": len(frozen["planned_differences_at_20_steps"]),
                      "table_numeric_entries": 77, "scope": "principal statistics, timings, accounting and manuscript table; prose/protocol claims audited separately"}, indent=2))


if __name__ == "__main__":
    main()
