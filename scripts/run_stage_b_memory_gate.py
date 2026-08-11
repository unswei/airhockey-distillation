#!/usr/bin/env python3
"""Apply the predeclared paired memory gate to teacher and student results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from airhockey_distill.evaluation import evaluate_memory_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument("--feed-forward-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text())
    teacher = json.loads(args.teacher_result.read_text())
    feed_forward = json.loads(args.feed_forward_result.read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_step = int(config["provenance"]["teacher_selection_step"])
    if int(teacher["checkpoint_step"]) != expected_step:
        raise ValueError("teacher result is not for the selected checkpoint")
    report = evaluate_memory_gate(
        teacher["episodes"], feed_forward["episodes"], config["memory_gate"]
    )
    result = {
        **report,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config.resolve()),
            "teacher_result": str(args.teacher_result.resolve()),
            "teacher_result_sha256": _sha256(args.teacher_result.resolve()),
            "feed_forward_result": str(args.feed_forward_result.resolve()),
            "feed_forward_result_sha256": _sha256(args.feed_forward_result.resolve()),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


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
