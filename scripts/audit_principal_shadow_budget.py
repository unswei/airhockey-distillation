#!/usr/bin/env python3
"""Audit one family's complete, fixed principal shadow budget."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airhockey_distill.principal_sweep import (
    DETERMINISTIC_ACTION_SEMANTICS,
    complete_shadow_schedule_sha256,
    load_principal_protocol,
    principal_family_spec,
    sha256_file,
    shadow_schedule_sha256,
    validate_shadow_partition_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--shadow-partition", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    principal_family_spec(protocol, args.family)
    expected_seeds = [int(value) for value in protocol["matched_seeds"]["training"]]
    expected_partition_episodes = int(
        protocol["shadow_labelling"]["episodes_per_collector"]
    )
    expected_family_episodes = int(protocol["shadow_labelling"]["episodes_per_family"])
    failures: list[str] = []
    partitions: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    teacher_hashes: set[str] = set()
    total_episodes = 0
    total_queries = 0

    if len(args.shadow_partition) != len(expected_seeds):
        failures.append("one partition is required for each matched collector seed")
    for raw_directory in args.shadow_partition:
        directory = raw_directory.resolve()
        manifest_path = directory / "manifest.json"
        try:
            manifest_hash = sha256_file(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            seed = int(manifest["collector_training_seed"])
            if seed in seen_seeds:
                raise ValueError(f"duplicate collector seed {seed}")
            seen_seeds.add(seed)
            if seed not in expected_seeds:
                raise ValueError(f"unexpected collector seed {seed}")
            if manifest.get("status") != "completed":
                raise ValueError("partition is not complete")
            if manifest.get("family_id") != args.family:
                raise ValueError("partition belongs to a different family")
            if manifest.get("protocol_sha256") != sha256_file(protocol_path):
                raise ValueError("partition protocol hash mismatch")
            if manifest.get("teacher_action_semantics") != DETERMINISTIC_ACTION_SEMANTICS:
                raise ValueError("partition does not contain deterministic teacher means")
            if not bool(manifest.get("deterministic_inference")):
                raise ValueError("partition did not use deterministic teacher inference")
            if manifest.get("teacher_previous_action") != (
                "previous_behaviour_policy_requested_command"
            ):
                raise ValueError("partition has the wrong previous-action semantics")
            if manifest.get("shadow_schedule_sha256") != shadow_schedule_sha256(
                protocol, seed
            ):
                raise ValueError("partition schedule hash mismatch")
            if int(manifest.get("episode_count", -1)) != expected_partition_episodes:
                raise ValueError("partition has the wrong episode budget")
            counts = validate_shadow_partition_shards(
                protocol, seed, directory, manifest
            )
            total_episodes += counts["episode_count"]
            total_queries += counts["teacher_query_count"]
            teacher_hashes.add(str(manifest["teacher_checkpoint_sha256"]))
            partitions.append(
                {
                    "collector_training_seed": seed,
                    "directory": str(directory),
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_hash,
                    "student_checkpoint_sha256": str(
                        manifest["student_checkpoint_sha256"]
                    ),
                    "teacher_checkpoint_sha256": str(
                        manifest["teacher_checkpoint_sha256"]
                    ),
                    **counts,
                }
            )
        except Exception as error:
            failures.append(f"{directory}: {error}")

    if seen_seeds != set(expected_seeds):
        failures.append("partitions do not cover the five matched collector seeds")
    if total_episodes != expected_family_episodes:
        failures.append(
            f"realised family episode budget is {total_episodes}, expected {expected_family_episodes}"
        )
    if teacher_hashes and teacher_hashes != {str(protocol["teacher"]["actor_sha256"])}:
        failures.append("partitions do not all use the frozen teacher checkpoint")

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "decision": "GO" if not failures else "NO_GO",
        "family_id": args.family,
        "code_commit": args.code_commit,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "immutable_protocol_sha256": protocol["immutable_protocol_sha256"],
        "shadow_schedule_sha256": complete_shadow_schedule_sha256(protocol),
        "budget_unit": "complete_paired_rollout_episode",
        "expected_episode_budget": expected_family_episodes,
        "realised_episode_budget": total_episodes,
        "realised_teacher_query_count": total_queries,
        "adaptive_top_up_used": False,
        "partitions": sorted(
            partitions, key=lambda value: value["collector_training_seed"]
        ),
        "failures": failures,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = audit(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
