#!/usr/bin/env python3
"""Apply the paired gate for resetting teacher state at blackout onset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from airhockey_distill.evaluation import evaluate_causal_memory_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recurrent-result", type=Path, required=True)
    parser.add_argument("--reset-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text())
    recurrent = json.loads(args.recurrent_result.read_text())
    reset = json.loads(args.reset_result.read_text())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate_causal_memory_ablation(
        recurrent["episodes"], reset["episodes"], config["causal_ablation_gate"]
    )
    result = {
        **report,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "inputs": {
            "config_sha256": _sha256(args.config.resolve()),
            "recurrent_result_sha256": _sha256(args.recurrent_result.resolve()),
            "reset_result_sha256": _sha256(args.reset_result.resolve()),
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
