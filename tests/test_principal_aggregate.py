import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airhockey_distill.principal_sweep import (
    DETERMINISTIC_ACTION_SEMANTICS,
    load_principal_protocol,
    shadow_schedule_records,
)
from scripts.aggregate_principal_shadow_dataset import (
    _validate_actual_partition_schedule,
)


def test_aggregate_rechecks_actual_shadow_shard_schedule(tmp_path):
    protocol = load_principal_protocol("configs/experiments/principal_sweep_execution_v1.yaml")
    records = shadow_schedule_records(protocol, 14303)[:3]
    shard_directory = tmp_path / "shards"
    shard_directory.mkdir()
    shard = shard_directory / "shard-0000.npz"
    np.savez(
        shard,
        dataset_schema_version=np.asarray(2),
        teacher_action_semantics=np.asarray(DETERMINISTIC_ACTION_SEMANTICS),
        episode_indices=np.asarray([value["episode_index"] for value in records]),
        episode_collection_offsets=np.asarray(
            [value["collection_offset"] for value in records]
        ),
        episode_reset_seeds=np.asarray([value["reset_seed"] for value in records]),
        episode_shot_ids=np.asarray([value["shot_id"] for value in records]),
        episode_blackout_steps=np.asarray(
            [value["blackout_steps"] for value in records]
        ),
    )
    manifest = {
        "shards": [
            {
                "file": "shards/shard-0000.npz",
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ]
    }
    # Restrict the protocol to the three records represented by this fixture.
    protocol["shadow_labelling"]["collection_offsets_by_seed"]["14303"] = [0, 2]
    protocol["shadow_labelling"]["episodes_per_collector"] = 3

    _validate_actual_partition_schedule(protocol, 14303, tmp_path, manifest)

    with np.load(shard, allow_pickle=False) as loaded:
        payload = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload["episode_reset_seeds"][1] += 1
    np.savez(shard, **payload)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="paired schedule"):
        _validate_actual_partition_schedule(protocol, 14303, tmp_path, manifest)
