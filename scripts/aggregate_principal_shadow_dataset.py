#!/usr/bin/env python3
"""Aggregate five matched shadow partitions with the common teacher data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.principal_sweep import (
    DETERMINISTIC_ACTION_SEMANTICS,
    complete_shadow_schedule_sha256,
    load_principal_protocol,
    principal_family_spec,
    sha256_file,
    shadow_schedule_sha256,
    shadow_schedule_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument(
        "--shadow-partition", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    principal_family_spec(protocol, args.family)
    base_directory = args.base_dataset.resolve()
    base_manifest_path = base_directory / "manifest.json"
    if sha256_file(base_manifest_path) != (
        protocol["base_teacher_dataset"]["manifest_sha256"]
    ):
        raise ValueError("base dataset manifest hash mismatch")
    base = json.loads(base_manifest_path.read_text())
    if base["dataset_id"] != protocol["base_teacher_dataset"]["id"]:
        raise ValueError("base dataset id mismatch")
    seeds = [int(value) for value in protocol["matched_seeds"]["training"]]
    if len(args.shadow_partition) != len(seeds):
        raise ValueError("one shadow partition is required for every matched seed")
    shadow_sources = []
    seen_seeds: set[int] = set()
    for raw_directory in args.shadow_partition:
        directory = raw_directory.resolve()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        seed = int(manifest["collector_training_seed"])
        if seed in seen_seeds or seed not in seeds:
            raise ValueError("shadow partitions have duplicate or unexpected seeds")
        seen_seeds.add(seed)
        if manifest["family_id"] != args.family:
            raise ValueError("shadow partition belongs to a different family")
        if manifest.get("status") != "completed":
            raise ValueError("shadow partition is incomplete")
        if manifest.get("teacher_action_semantics") != (
            DETERMINISTIC_ACTION_SEMANTICS
        ) or not bool(manifest.get("deterministic_inference")):
            raise ValueError("shadow partition does not contain deterministic means")
        if manifest.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("shadow partition protocol hash mismatch")
        if manifest["shadow_schedule_sha256"] != shadow_schedule_sha256(
            protocol, seed
        ):
            raise ValueError("shadow partition does not use the matched schedule")
        if int(manifest["episode_count"]) != int(
            protocol["shadow_labelling"]["episodes_per_collector"]
        ):
            raise ValueError("shadow partition episode count mismatch")
        _validate_actual_partition_schedule(protocol, seed, directory, manifest)
        shadow_sources.append((seed, directory, manifest_path, manifest))
    if seen_seeds != set(seeds):
        raise ValueError("shadow partitions do not cover every matched seed")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    all_indices: set[int] = set()
    outcomes: Counter[str] = Counter()
    transitions = 0
    source_manifests: dict[str, str] = {
        "teacher_controlled": sha256_file(base_manifest_path)
    }
    sources = [("teacher_controlled", base_directory, base)]
    for seed, directory, manifest_path, manifest in sorted(shadow_sources):
        name = f"shadow_seed_{seed}"
        sources.append((name, directory, manifest))
        source_manifests[name] = sha256_file(manifest_path)
    for source_name, directory, manifest in sources:
        for entry in manifest["shards"]:
            path = directory / entry["file"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"source shard hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as shard:
                indices = {int(value) for value in shard["episode_indices"]}
                if all_indices & indices:
                    raise ValueError("aggregate contains duplicate episode indices")
                all_indices.update(indices)
                outcomes.update(str(value) for value in shard["episode_outcomes"])
                shard_transitions = int(shard["observations"].shape[0])
            transitions += shard_transitions
            shards.append(
                {
                    "file": str(path.resolve()),
                    "source": source_name,
                    "first_episode": min(indices),
                    "episodes": len(indices),
                    "transitions": shard_transitions,
                    "bytes": path.stat().st_size,
                    "sha256": entry["sha256"],
                }
            )
    expected = int(
        protocol["training"]["final_principal_stage"]["data_per_family"][
            "total_episodes"
        ]
    )
    if all_indices != set(range(expected)):
        raise ValueError("aggregate episode indices are not the canonical range")
    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": f"principal_{args.family}_teacher_plus_shadow_v1",
        "dataset_schema_version": 2,
        "family_id": args.family,
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "episode_count": expected,
        "transition_count": transitions,
        "shadow_schedule_sha256": complete_shadow_schedule_sha256(protocol),
        "outcome_counts": dict(sorted(outcomes.items())),
        "source_manifests": source_manifests,
        "shards": shards,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _validate_actual_partition_schedule(
    protocol: dict[str, Any],
    seed: int,
    directory: Path,
    manifest: dict[str, Any],
) -> None:
    expected = shadow_schedule_records(protocol, seed)
    observed: list[dict[str, Any]] = []
    for entry in manifest["shards"]:
        path = directory / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"shadow shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            if int(shard["dataset_schema_version"]) != 2:
                raise ValueError("unsupported shadow shard schema")
            if str(shard["teacher_action_semantics"]) != (
                DETERMINISTIC_ACTION_SEMANTICS
            ):
                raise ValueError("shadow shard does not contain teacher means")
            for episode_index, offset, reset_seed, shot_id, blackout_steps in zip(
                shard["episode_indices"],
                shard["episode_collection_offsets"],
                shard["episode_reset_seeds"],
                shard["episode_shot_ids"],
                shard["episode_blackout_steps"],
                strict=True,
            ):
                observed.append(
                    {
                        "episode_index": int(episode_index),
                        "collection_offset": int(offset),
                        "reset_seed": int(reset_seed),
                        "shot_id": str(shot_id),
                        "blackout_steps": int(blackout_steps),
                    }
                )
    observed.sort(key=lambda value: value["collection_offset"])
    if observed != expected:
        raise ValueError("shadow shard contents do not match the paired schedule")


def main() -> None:
    result = aggregate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
