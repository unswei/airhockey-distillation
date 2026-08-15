from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EFFICIENCY_TABLE = (
    REPOSITORY_ROOT / "results" / "principal_sweep_v1_efficiency_table.csv"
)
STATISTICS = REPOSITORY_ROOT / "results" / "principal_sweep_v1_statistics.json"
EXPECTED_TABLE_SHA256 = (
    "a6e8edc0ebb541c3922b9b589926527519045f370c46c370c78fc749392e9311"
)
FAMILIES = (
    "feed_forward",
    "finite_stack_10",
    "structured_k0",
    "structured_k1",
    "structured_k2",
    "structured_k4",
    "gru_n64",
)
CORE_BLACKOUTS = ("0", "5", "10", "15", "20")
SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def test_complete_comparison_uses_the_frozen_efficiency_table() -> None:
    assert hashlib.sha256(EFFICIENCY_TABLE.read_bytes()).hexdigest() == (
        EXPECTED_TABLE_SHA256
    )
    with EFFICIENCY_TABLE.open(newline="", encoding="utf-8") as stream:
        rows = {row["family"]: row for row in csv.DictReader(stream)}
    assert tuple(rows) == FAMILIES

    accounting = {
        family: tuple(
            int(rows[family][field])
            for field in (
                "total_trainable_parameters",
                "core_parameters",
                "total_policy_carry_bytes",
                "recurrent_memory_bytes",
                "multiply_adds_per_step",
            )
        )
        for family in FAMILIES
    }
    assert accounting == {
        "feed_forward": (5602, 0, 0, 0, 5440),
        "finite_stack_10": (7330, 0, 120, 0, 7168),
        "structured_k0": (12002, 2304, 264, 256, 11776),
        "structured_k1": (12165, 2467, 264, 256, 11938),
        "structured_k2": (12328, 2630, 264, 256, 12100),
        "structured_k4": (12654, 2956, 264, 256, 12424),
        "gru_n64": (28898, 19200, 264, 256, 28352),
    }


def test_complete_comparison_rates_and_latency_match_principal_statistics() -> None:
    principal = json.loads(STATISTICS.read_text(encoding="utf-8"))
    with EFFICIENCY_TABLE.open(newline="", encoding="utf-8") as stream:
        rows = {row["family"]: row for row in csv.DictReader(stream)}

    for family in FAMILIES:
        latency_points = principal["efficiency_seed_points"][family]
        assert float(rows[family]["cpu_latency_median_microseconds"]) == (
            statistics.median(
                float(point["median_microseconds"]) for point in latency_points
            )
        )
        assert float(rows[family]["cpu_latency_p95_microseconds"]) == (
            statistics.median(
                float(point["p95_microseconds"]) for point in latency_points
            )
        )

        counts = principal["student_outcome_counts_by_blackout"][family]
        saved = sum(
            int(counts[blackout].get(outcome, 0))
            for blackout in CORE_BLACKOUTS
            for outcome in SAVE_OUTCOMES
        )
        episodes = sum(
            sum(int(value) for value in counts[blackout].values())
            for blackout in CORE_BLACKOUTS
        )
        assert float(rows[family]["overall_core_test_save_rate"]) == saved / episodes
