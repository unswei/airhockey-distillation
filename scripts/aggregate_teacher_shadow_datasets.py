#!/usr/bin/env python3
"""Build a hash-verified manifest over teacher and shadow shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DETERMINISTIC_ACTION_SEMANTICS = "deterministic_actor_mean_after_public_adapter_clip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--shadow-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def aggregate(
    config: dict[str, Any],
    original_directory: Path,
    shadow_directory: Path,
    output_directory: Path,
    *,
    code_commit: str,
) -> dict[str, Any]:
    specification = config["aggregate"]
    original_manifest_path = original_directory / "manifest.json"
    shadow_manifest_path = shadow_directory / "manifest.json"
    original = json.loads(original_manifest_path.read_text())
    shadow = json.loads(shadow_manifest_path.read_text())
    if _sha256(original_manifest_path) != specification["original_manifest_sha256"]:
        raise ValueError("original dataset manifest hash mismatch")
    _validate_source_manifest(
        original,
        dataset_id=specification["original_dataset_id"],
        episode_count=int(specification["original_episodes"]),
    )
    _validate_source_manifest(
        shadow,
        dataset_id=config["shadow_collection"]["id"],
        episode_count=int(specification["shadow_episodes"]),
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True)
    )

    shards = []
    all_indices: set[int] = set()
    outcomes: Counter[str] = Counter()
    transitions = 0
    for source_name, directory, manifest in (
        ("teacher_controlled", original_directory, original),
        ("student_controlled_shadow", shadow_directory, shadow),
    ):
        for entry in manifest["shards"]:
            source_path = directory / entry["file"]
            if _sha256(source_path) != entry["sha256"]:
                raise ValueError(f"source shard hash mismatch: {source_path}")
            with np.load(source_path, allow_pickle=False) as shard:
                indices = {int(value) for value in shard["episode_indices"]}
                if all_indices & indices:
                    raise ValueError("aggregate contains duplicate episode indices")
                all_indices.update(indices)
                shard_outcomes = Counter(
                    str(value) for value in shard["episode_outcomes"]
                )
                outcomes.update(shard_outcomes)
                shard_transitions = int(shard["observations"].shape[0])
            transitions += shard_transitions
            shards.append(
                {
                    "file": str(source_path.resolve()),
                    "source": source_name,
                    "first_episode": min(indices),
                    "episodes": len(indices),
                    "transitions": shard_transitions,
                    "bytes": source_path.stat().st_size,
                    "sha256": entry["sha256"],
                }
            )
    expected_episodes = int(specification["total_episodes"])
    if len(all_indices) != expected_episodes:
        raise ValueError("aggregate episode count mismatch")
    if all_indices != set(range(expected_episodes)):
        raise ValueError("aggregate episode indices must be contiguous")
    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": specification["id"],
        "dataset_schema_version": 2,
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "code_commit": code_commit,
        "config_sha256": _sha256(output_directory / "config.yaml"),
        "episode_count": expected_episodes,
        "transition_count": transitions,
        "outcome_counts": dict(sorted(outcomes.items())),
        "source_manifests": {
            "teacher_controlled": _sha256(original_manifest_path),
            "student_controlled_shadow": _sha256(shadow_manifest_path),
        },
        "shards": shards,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _validate_source_manifest(
    manifest: dict[str, Any], *, dataset_id: str, episode_count: int
) -> None:
    if manifest["status"] != "completed":
        raise ValueError("source dataset is incomplete")
    if manifest["dataset_id"] != dataset_id:
        raise ValueError("source dataset id mismatch")
    if manifest["teacher_action_semantics"] != DETERMINISTIC_ACTION_SEMANTICS:
        raise ValueError("source dataset does not contain deterministic means")
    if not bool(manifest["deterministic_inference"]):
        raise ValueError("source dataset inference is not deterministic")
    if int(manifest["episode_count"]) != episode_count:
        raise ValueError("source dataset episode count mismatch")


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    result = aggregate(
        _load_mapping(args.config.resolve()),
        args.original_dataset.resolve(),
        args.shadow_dataset.resolve(),
        args.output.resolve(),
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "episode_count", "transition_count")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
