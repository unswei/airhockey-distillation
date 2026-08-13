#!/usr/bin/env python3
"""Apply the predeclared no-blackout gate to the frozen GRU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def apply_gate(
    config: dict[str, Any],
    training_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    *,
    evaluation_code_commit: str,
) -> dict[str, Any]:
    gate = config["no_blackout_gate"]
    provenance = config["provenance"]
    required_blackouts = [int(value) for value in gate["required_blackout_steps"]]
    observed_blackouts = sorted(
        int(value) for value in evaluation_result["summary"]["by_blackout_steps"]
    )
    if training_result["code_commit"] != provenance["training_code_commit"]:
        raise ValueError("training result code commit does not match provenance")
    if evaluation_result["code_commit"] != evaluation_code_commit:
        raise ValueError("evaluation result code commit does not match")
    if training_result["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise ValueError("training checkpoint hash does not match provenance")
    if evaluation_result["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise ValueError("evaluation checkpoint hash does not match provenance")
    if observed_blackouts != required_blackouts:
        raise ValueError("evaluation opened an unapproved blackout condition")

    summary = evaluation_result["summary"]
    outcome_counts = summary["outcome_counts"]
    faults = sum(
        int(outcome_counts.get(name, 0)) for name in ("simulator_fault", "safety_fault")
    )
    checks = {
        "episode_count": {
            "observed": int(summary["episodes"]),
            "required": int(gate["required_episodes"]),
            "passed": int(summary["episodes"]) == int(gate["required_episodes"]),
        },
        "no_blackout_only": {
            "observed": observed_blackouts,
            "required": required_blackouts,
            "passed": observed_blackouts == required_blackouts,
        },
        "save_rate": {
            "observed": float(summary["save_rate"]),
            "minimum": float(gate["minimum_save_rate"]),
            "passed": float(summary["save_rate"]) >= float(gate["minimum_save_rate"]),
        },
        "simulator_or_safety_faults": {
            "observed": faults,
            "maximum": int(gate["maximum_simulator_or_safety_faults"]),
            "passed": faults <= int(gate["maximum_simulator_or_safety_faults"]),
        },
    }
    decision = "GO" if all(check["passed"] for check in checks.values()) else "NO_GO"
    return {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": evaluation_code_commit,
        "training_code_commit": training_result["code_commit"],
        "pilot_scope": config["pilot_scope"],
        "decision": decision,
        "next_action": (
            gate["decision_if_go"] if decision == "GO" else gate["decision_if_no_go"]
        ),
        "checks": checks,
        "student": {
            "training_seed": int(training_result["training_seed"]),
            "selected_epoch": int(training_result["selected_epoch"]),
            "checkpoint_sha256": training_result["checkpoint_sha256"],
            "validation_action_mse": float(
                training_result["selected_metrics"]["validation"]["action_mse"]
            ),
        },
        "evaluation": {
            "save_count": int(summary["save_count"]),
            "save_rate": float(summary["save_rate"]),
            "outcome_counts": outcome_counts,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config_path = args.config.resolve()
    training_path = args.training_result.resolve()
    evaluation_path = args.evaluation_result.resolve()
    config = yaml.safe_load(config_path.read_text())
    if _sha256(training_path) != config["provenance"]["training_result_sha256"]:
        raise ValueError("training result hash mismatch")
    result = apply_gate(
        config,
        json.loads(training_path.read_text()),
        json.loads(evaluation_path.read_text()),
        evaluation_code_commit=args.code_commit,
    )
    result["config_sha256"] = _sha256(config_path)
    result["training_result_sha256"] = _sha256(training_path)
    result["evaluation_result_sha256"] = _sha256(evaluation_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
