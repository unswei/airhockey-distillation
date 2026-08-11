#!/usr/bin/env python3
"""Apply predeclared readiness thresholds to a teacher evaluation result."""

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
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text())
    result = json.loads(args.teacher_result.read_text())
    thresholds = config["teacher_readiness"]
    evaluation = config["evaluation"]
    expected_episodes = int(evaluation["shot_count"]) * len(
        evaluation["blackout_steps"]
    )
    overall = result["summary"]
    by_blackout = overall["by_blackout_steps"]
    long_key = str(max(int(value) for value in evaluation["blackout_steps"]))
    checks = [
        _check(
            "complete_evaluation",
            int(overall["episodes"]) == expected_episodes,
            int(overall["episodes"]),
            expected_episodes,
        ),
        _check(
            "no_blackout_save_rate",
            float(by_blackout["0"]["save_rate"])
            >= float(thresholds["minimum_no_blackout_save_rate"]),
            float(by_blackout["0"]["save_rate"]),
            thresholds["minimum_no_blackout_save_rate"],
        ),
        _check(
            "overall_save_rate",
            float(overall["save_rate"])
            >= float(thresholds["minimum_overall_save_rate"]),
            float(overall["save_rate"]),
            thresholds["minimum_overall_save_rate"],
        ),
        _check(
            "long_blackout_save_rate",
            float(by_blackout[long_key]["save_rate"])
            >= float(thresholds["minimum_long_blackout_save_rate"]),
            float(by_blackout[long_key]["save_rate"]),
            thresholds["minimum_long_blackout_save_rate"],
        ),
    ]
    blocking = [check["check_id"] for check in checks if not check["passed"]]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "decision": "GO" if not blocking else "NO_GO",
        "code_commit": args.code_commit,
        "checks": checks,
        "blocking_checks": blocking,
        "teacher_result_sha256": _sha256(args.teacher_result.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _check(
    check_id: str, passed: bool, observed: float, required: float
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "observed": observed,
        "required_minimum_or_exact": required,
    }


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
