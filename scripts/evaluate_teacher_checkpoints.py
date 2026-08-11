#!/usr/bin/env python3
"""Evaluate and select all retained Dreamer teacher checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airhockey_distill.teachers import (
    list_complete_checkpoints,
    select_best_validation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher/dreamerv3.yaml"),
    )
    parser.add_argument("--profile", default="full")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = list_complete_checkpoints(args.checkpoints.resolve())
    if not checkpoints:
        raise ValueError("no complete checkpoints to evaluate")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    evaluator = Path(__file__).with_name("evaluate_teacher_checkpoint.py")
    result_paths: list[Path] = []
    untrained = output / "untrained.json"
    _evaluate_if_missing(evaluator, args, untrained)
    result_paths.append(untrained)
    for step, checkpoint in checkpoints:
        result_path = output / f"checkpoint-{step:09d}.json"
        _evaluate_if_missing(evaluator, args, result_path, checkpoint)
        result_paths.append(result_path)

    results = [json.loads(path.read_text()) for path in result_paths]
    _validate_results(results, args, checkpoints)
    selected = select_best_validation_result(results)
    selection = {
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "profile": args.profile,
        "selection_metric": "validation_save_rate_across_blackouts",
        "tie_breaks": ["mean_score", "earlier_checkpoint_step"],
        "evaluated_checkpoint_steps": [step for step, _ in checkpoints],
        "untrained_summary": results[0]["summary"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_step": selected["checkpoint_step"],
        "selected_summary": selected["summary"],
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    return selection


def _evaluate_if_missing(
    evaluator: Path,
    args: argparse.Namespace,
    output: Path,
    checkpoint: Path | None = None,
) -> None:
    if output.exists():
        return
    command = [
        sys.executable,
        str(evaluator),
        "--config",
        str(args.config),
        "--profile",
        args.profile,
        "--output",
        str(output),
        "--code-commit",
        args.code_commit,
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    subprocess.run(command, check=True)


def _validate_results(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    checkpoints: tuple[tuple[int, Path], ...],
) -> None:
    expected_steps = [step for step, _ in checkpoints]
    actual_steps = sorted(
        int(result["checkpoint_step"])
        for result in results
        if result["policy"] == "checkpoint"
    )
    if actual_steps != expected_steps:
        raise ValueError("checkpoint evaluation set does not match retained checkpoints")
    for result in results:
        if result.get("status") != "completed":
            raise ValueError("incomplete validation result")
        if result.get("code_commit") != args.code_commit:
            raise ValueError("validation result code commit mismatch")
        if result.get("profile") != args.profile:
            raise ValueError("validation result profile mismatch")


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
