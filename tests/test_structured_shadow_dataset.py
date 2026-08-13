import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_teacher_shadow_datasets import aggregate
from scripts.collect_student_shadow_dataset import (
    DETERMINISTIC_ACTION_SEMANTICS,
    _validate_config,
)


def test_shadow_config_binds_student_history_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "student.npz"
    checkpoint.write_bytes(b"student")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "agent.pkl").write_bytes(b"teacher")
    config = {
        "shadow_collection": {
            "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
            "teacher_previous_action": "previous_student_requested_command",
        },
        "behaviour_policy": {
            "checkpoint_sha256": hashlib.sha256(b"student").hexdigest()
        },
        "provenance": {"teacher_actor_sha256": hashlib.sha256(b"teacher").hexdigest()},
    }

    _validate_config(config, checkpoint, teacher)

    config["shadow_collection"]["teacher_previous_action"] = "teacher_action"
    with pytest.raises(ValueError, match="student command history"):
        _validate_config(config, checkpoint, teacher)


def test_aggregate_rejects_overlapping_episode_indices(tmp_path):
    original = _dataset(tmp_path / "original", "original", [0, 1])
    shadow = _dataset(tmp_path / "shadow", "shadow", [1, 2])
    config = _aggregate_config(original / "manifest.json")

    with pytest.raises(ValueError, match="duplicate episode indices"):
        aggregate(
            config,
            original,
            shadow,
            tmp_path / "aggregate",
            code_commit="abc",
        )


def test_aggregate_preserves_disjoint_shards_without_copying(tmp_path):
    original = _dataset(tmp_path / "original", "original", [0, 1])
    shadow = _dataset(tmp_path / "shadow", "shadow", [2, 3])
    config = _aggregate_config(original / "manifest.json")

    result = aggregate(
        config,
        original,
        shadow,
        tmp_path / "aggregate",
        code_commit="abc",
    )

    assert result["episode_count"] == 4
    assert result["transition_count"] == 8
    assert {entry["source"] for entry in result["shards"]} == {
        "teacher_controlled",
        "student_controlled_shadow",
    }
    assert all(Path(entry["file"]).is_absolute() for entry in result["shards"])


def _dataset(directory, dataset_id, indices):
    shards = directory / "shards"
    shards.mkdir(parents=True)
    observations = np.zeros((4, 19), dtype=np.float32)
    shard_path = shards / "shard-0000.npz"
    np.savez_compressed(
        shard_path,
        observations=observations,
        episode_indices=np.asarray(indices, dtype=np.int64),
        episode_outcomes=np.asarray(["returned", "goal_conceded"]),
    )
    shard_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest = {
        "status": "completed",
        "dataset_id": dataset_id,
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "episode_count": 2,
        "shards": [{"file": "shards/shard-0000.npz", "sha256": shard_hash}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def _aggregate_config(original_manifest):
    return {
        "aggregate": {
            "id": "aggregate",
            "original_dataset_id": "original",
            "original_manifest_sha256": hashlib.sha256(
                original_manifest.read_bytes()
            ).hexdigest(),
            "original_episodes": 2,
            "shadow_episodes": 2,
            "total_episodes": 4,
        },
        "shadow_collection": {"id": "shadow"},
    }
