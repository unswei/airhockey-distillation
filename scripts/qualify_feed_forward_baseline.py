#!/usr/bin/env python3
"""Select and qualify a visible-observation baseline without opening test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from airhockey_distill.evaluation.memory_gate import SAVE_OUTCOMES

FAULT_OUTCOMES = frozenset({"simulator_fault", "non_finite_state", "safety_fault"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument(
        "--baseline-result", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    qualification = config["baseline_qualification"]
    teacher_path = args.teacher_result.resolve()
    teacher = json.loads(teacher_path.read_text())
    candidates = [
        (path.resolve(), json.loads(path.resolve().read_text()))
        for path in args.baseline_result
    ]
    _validate_inputs(config, teacher, [result for _, result in candidates])

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item[1]["summary"]["save_rate"]),
            -float(item[1]["summary"]["mean_score"]),
            int(item[1]["training_seed"]),
        ),
    )
    selected_path, selected = ranked[0]
    teacher_index = _index_no_blackout(teacher["episodes"])
    baseline_index = _index_no_blackout(selected["episodes"])
    if set(teacher_index) != set(baseline_index):
        raise ValueError("teacher and baseline qualification shots differ")
    teacher_saves = sum(
        teacher_index[key]["outcome"] in SAVE_OUTCOMES for key in teacher_index
    )
    baseline_saves = sum(
        baseline_index[key]["outcome"] in SAVE_OUTCOMES for key in teacher_index
    )
    episodes = len(teacher_index)
    baseline_save_rate = baseline_saves / episodes
    teacher_advantage = (teacher_saves - baseline_saves) / episodes
    fault_count = sum(
        episode["outcome"] in FAULT_OUTCOMES for episode in selected["episodes"]
    )
    checks = [
        _check(
            "credible_visible_feed_forward_baseline",
            baseline_save_rate
            >= float(qualification["minimum_feed_forward_no_blackout_save_rate"]),
            baseline_save_rate,
            f">= {qualification['minimum_feed_forward_no_blackout_save_rate']}",
        ),
        _check(
            "comparable_no_blackout_performance",
            teacher_advantage
            <= float(qualification["maximum_no_blackout_teacher_advantage"]),
            teacher_advantage,
            f"<= {qualification['maximum_no_blackout_teacher_advantage']}",
        ),
        _check(
            "simulator_faults",
            fault_count <= int(qualification["maximum_simulator_faults"]),
            fault_count,
            f"<= {qualification['maximum_simulator_faults']}",
        ),
    ]
    blocking = [check["check_id"] for check in checks if not check["passed"]]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "decision": "GO" if not blocking else "NO_GO",
        "interpretation": (
            "visible-observation baseline is qualified for held-out confirmation"
            if not blocking
            else "visible-observation baseline is not yet qualified"
        ),
        "code_commit": args.code_commit,
        "selection_metric": qualification["selection_metric"],
        "tie_breaks": qualification["tie_breaks"],
        "selected_training_seed": int(selected["training_seed"]),
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_result": str(selected_path),
        "selected_result_sha256": _sha256(selected_path),
        "candidate_summaries": [
            {
                "training_seed": int(result["training_seed"]),
                "save_rate": float(result["summary"]["save_rate"]),
                "mean_score": float(result["summary"]["mean_score"]),
                "result": str(path),
                "result_sha256": _sha256(path),
            }
            for path, result in ranked
        ],
        "observed": {
            "episodes": episodes,
            "teacher_save_rate": teacher_saves / episodes,
            "feed_forward_save_rate": baseline_save_rate,
            "paired_teacher_advantage": teacher_advantage,
            "simulator_faults": fault_count,
        },
        "checks": checks,
        "blocking_checks": blocking,
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "teacher_result": str(teacher_path),
            "teacher_result_sha256": _sha256(teacher_path),
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _validate_inputs(
    config: dict[str, Any], teacher: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    expected_seed_set = {int(value) for value in config["full"]["seeds"]}
    candidate_seed_set = {int(result["training_seed"]) for result in candidates}
    if candidate_seed_set != expected_seed_set:
        raise ValueError("qualification candidates do not match predeclared seeds")
    expected_episodes = int(config["qualification_evaluation"]["shot_count"])
    if len(_index_no_blackout(teacher["episodes"])) != expected_episodes:
        raise ValueError("teacher qualification evaluation has the wrong size")
    for result in candidates:
        if (
            result.get("status") != "completed"
            or result.get("policy") != "feed_forward_ppo"
            or result.get("evaluation") != "qualification_evaluation"
            or not result.get("memoryless")
        ):
            raise ValueError("invalid feed-forward qualification result")
        if len(_index_no_blackout(result["episodes"])) != expected_episodes:
            raise ValueError("baseline qualification evaluation has the wrong size")


def _index_no_blackout(
    episodes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected = [episode for episode in episodes if int(episode["blackout_steps"]) == 0]
    indexed = {str(episode["shot_id"]): episode for episode in selected}
    if len(indexed) != len(selected) or not indexed:
        raise ValueError("no-blackout episodes must have unique shot identifiers")
    return indexed


def _check(
    check_id: str, passed: bool, observed: float | int, required: str
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
