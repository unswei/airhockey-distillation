#!/usr/bin/env python3
"""Freeze exact parameter and runtime-state accounting for one final policy."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from airhockey_distill.principal_efficiency import (
    load_final_principal_policy,
    policy_accounting,
)
from airhockey_distill.principal_sweep import load_principal_protocol, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    checkpoint = args.checkpoint.resolve()
    policy = load_final_principal_policy(
        protocol, protocol_path, args.family, args.seed, checkpoint
    )
    accounting = policy_accounting(protocol, args.family, policy)
    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "family_id": args.family,
        "training_seed": args.seed,
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        **accounting,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
