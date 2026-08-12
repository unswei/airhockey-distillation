#!/usr/bin/env python3
"""Audit deterministic targets and previous-action semantics in a dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.envs.policy_interface import END_EFFECTOR_XY_SLICE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-lock-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def audit_dataset(
    dataset_directory: Path,
    *,
    action_lock_steps: int,
    code_commit: str,
) -> dict[str, Any]:
    if action_lock_steps < 0:
        raise ValueError("action lock steps must be non-negative")
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    requested_previous_matches = True
    post_lock_previous_matches = True
    locked_squared_error = 0.0
    locked_action_values = 0
    locked_maximum_absolute_error = 0.0
    episode_count = 0
    first_by_shot_blackout: dict[tuple[str, int], tuple[bytes, bytes]] = {}
    group_summaries: dict[tuple[str, int], list[int | bool]] = {}
    target_by_input: dict[bytes, bytes] = {}
    conflicting_inputs: set[bytes] = set()

    for shard_entry in manifest["shards"]:
        shard_path = dataset_directory / shard_entry["file"]
        if _sha256(shard_path) != shard_entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            offsets = shard["episode_offsets"]
            for local_index in range(len(offsets) - 1):
                start = int(offsets[local_index])
                stop = int(offsets[local_index + 1])
                observations = shard["observations"][start:stop]
                previous_actions = shard["previous_actions"][start:stop]
                targets = shard["teacher_actions"][start:stop]
                requested_previous_matches &= bool(
                    np.array_equal(previous_actions[1:], targets[:-1])
                )
                post_lock_start = action_lock_steps + 1
                if len(targets) > post_lock_start:
                    post_lock_previous_matches &= bool(
                        np.array_equal(
                            previous_actions[post_lock_start:],
                            targets[action_lock_steps:-1],
                        )
                    )
                compared_lock_steps = min(action_lock_steps, len(targets) - 1)
                if compared_lock_steps:
                    difference = (
                        previous_actions[1 : compared_lock_steps + 1]
                        - observations[:compared_lock_steps, END_EFFECTOR_XY_SLICE]
                    )
                    locked_squared_error += float(np.sum(difference**2))
                    locked_action_values += int(difference.size)
                    locked_maximum_absolute_error = max(
                        locked_maximum_absolute_error,
                        float(np.max(np.abs(difference))),
                    )

                input_hash = _array_digest(observations, previous_actions)
                target_hash = _array_digest(targets)
                key = (
                    str(shard["episode_shot_ids"][local_index]),
                    int(shard["episode_blackout_steps"][local_index]),
                )
                if key not in first_by_shot_blackout:
                    first_by_shot_blackout[key] = (input_hash, target_hash)
                    group_summaries[key] = [1, True, True]
                else:
                    first_input, first_target = first_by_shot_blackout[key]
                    group = group_summaries[key]
                    group[0] = int(group[0]) + 1
                    group[1] = bool(group[1]) and input_hash == first_input
                    group[2] = bool(group[2]) and target_hash == first_target
                if (
                    input_hash in target_by_input
                    and target_hash != target_by_input[input_hash]
                ):
                    conflicting_inputs.add(input_hash)
                else:
                    target_by_input.setdefault(input_hash, target_hash)
                episode_count += 1

    repeated = [values for values in group_summaries.values() if int(values[0]) > 1]
    return {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(manifest_path),
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "episode_count": episode_count,
        "action_lock_steps": action_lock_steps,
        "previous_action_semantics": {
            "stored_previous_equals_prior_teacher_command_all": (
                requested_previous_matches
            ),
            "post_lock_previous_equals_prior_teacher_command_all": (
                post_lock_previous_matches
            ),
            "locked_prior_command_vs_applied_hold_mse": (
                locked_squared_error / locked_action_values
                if locked_action_values
                else None
            ),
            "locked_prior_command_vs_applied_hold_maximum_absolute_error": (
                locked_maximum_absolute_error
            ),
            "locked_action_values_compared": locked_action_values,
        },
        "deterministic_target_check": {
            "shot_blackout_groups": len(group_summaries),
            "repeated_shot_blackout_groups": len(repeated),
            "episodes_in_repeated_groups": sum(int(v[0]) for v in repeated),
            "repeated_groups_with_identical_full_inputs": sum(
                bool(v[1]) for v in repeated
            ),
            "repeated_groups_with_identical_targets": sum(bool(v[2]) for v in repeated),
            "unique_full_input_trajectories": len(target_by_input),
            "full_input_trajectories_with_conflicting_targets": len(conflicting_inputs),
        },
    }


def _array_digest(*arrays: np.ndarray[Any, Any]) -> bytes:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.tobytes())
    return digest.digest()


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
    result = audit_dataset(
        args.dataset.resolve(),
        action_lock_steps=args.action_lock_steps,
        code_commit=args.code_commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
