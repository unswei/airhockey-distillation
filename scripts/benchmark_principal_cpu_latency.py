#!/usr/bin/env python3
"""Run the predeclared exported-NumPy latency benchmark on Marvin."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from airhockey_distill.principal_efficiency import (
    benchmark_policy_calls,
    load_final_principal_policy,
    validate_latency_runtime,
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
    parser.add_argument("--container-digest", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    specification = protocol["measurements"]["cpu_latency"]
    if args.container_digest != protocol["provenance"]["container_digest"]:
        raise ValueError("container digest does not match the predeclaration")
    runtime = validate_latency_runtime(specification)
    if runtime["decision"] != "GO":
        raise RuntimeError("; ".join(runtime["failures"]))
    checkpoint = args.checkpoint.resolve()
    policy = load_final_principal_policy(
        protocol, protocol_path, args.family, args.seed, checkpoint
    )
    measurements = benchmark_policy_calls(
        policy,
        warmup_calls=int(specification["warmup_calls_per_checkpoint"]),
        timed_calls_per_repetition=int(
            specification["timed_calls_per_repetition"]
        ),
        repetitions=int(specification["repetitions_per_checkpoint"]),
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "decision": "GO",
        "created_at": datetime.now(UTC).isoformat(),
        "family_id": args.family,
        "training_seed": args.seed,
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "runtime_contract": {
            "runtime": specification["runtime"],
            "processes": specification["processes"],
            "threads": specification["threads"],
            "include": specification["include"],
            "exclude": specification["exclude"],
            "container_digest": args.container_digest,
        },
        "runtime": runtime,
        "measurements": measurements,
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


if __name__ == "__main__":
    main()
