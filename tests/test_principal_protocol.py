import json
from pathlib import Path

import numpy as np
import pytest

from airhockey_distill.principal_sweep import (
    DETERMINISTIC_ACTION_SEMANTICS,
    complete_shadow_schedule_sha256,
    evaluation_schedule,
    evaluation_schedule_sha256,
    episode_index_sha256,
    load_principal_episode_splits,
    load_principal_protocol,
    shadow_partition,
    shadow_schedule_records,
    shadow_schedule_sha256,
    validate_principal_dataset_manifest,
)
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS

PROTOCOL = Path("configs/experiments/principal_sweep_execution_v1.yaml")


def test_shadow_partitions_cover_one_common_schedule_without_overlap():
    protocol = load_principal_protocol(PROTOCOL)
    records = []
    ranges = []
    for seed in protocol["matched_seeds"]["training"]:
        first, last = shadow_partition(protocol, seed)
        ranges.append(set(range(first, last + 1)))
        partition = shadow_schedule_records(protocol, seed)
        assert len(partition) == 4000
        assert shadow_schedule_sha256(protocol, seed)
        records.extend(partition)

    assert all(not left & right for index, left in enumerate(ranges) for right in ranges[index + 1 :])
    assert [value["collection_offset"] for value in records] == list(range(20000))
    assert [value["episode_index"] for value in records] == list(range(20000, 40000))
    assert [value["reset_seed"] for value in records] == list(range(52303, 72303))
    assert all(value["shot_id"] for value in records)
    assert all(0 <= value["blackout_steps"] <= 20 for value in records)
    assert complete_shadow_schedule_sha256(protocol) == (
        "8a199955f58cdbe631dfee0373b21c1de348f47e8960465e6634cf718ed95daa"
    )


def test_all_seven_architecture_configs_are_hash_bound():
    protocol = load_principal_protocol(PROTOCOL)

    architectures = protocol["execution"]["architecture_configs"]
    assert set(architectures) == set(PRINCIPAL_FAMILY_IDS)
    for architecture in architectures.values():
        assert Path(architecture["path"]).is_file()
        assert len(architecture["sha256"]) == 64


def test_shared_loader_gives_every_family_identical_episode_splits(tmp_path):
    protocol = load_principal_protocol(PROTOCOL)
    protocol["base_teacher_dataset"]["episodes"] = 10
    lengths = np.asarray([3] * 10, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    transitions = int(offsets[-1])
    shard = tmp_path / "shard.npz"
    np.savez(
        shard,
        dataset_schema_version=np.asarray(2),
        teacher_action_semantics=np.asarray(DETERMINISTIC_ACTION_SEMANTICS),
        observations=np.zeros((transitions, 19), dtype=np.float32),
        previous_actions=np.zeros((transitions, 2), dtype=np.float32),
        teacher_actions=np.zeros((transitions, 2), dtype=np.float32),
        puck_visible=np.ones(transitions, dtype=bool),
        episode_offsets=offsets,
        episode_indices=np.arange(10, dtype=np.int64),
    )
    manifest = {
        "episode_count": 10,
        "shards": [
            {
                "file": shard.name,
                "sha256": __import__("hashlib").sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }

    splits = load_principal_episode_splits(tmp_path, manifest, protocol)
    hashes = {
        family_id: {
            name: episode_index_sha256(split) for name, split in splits.items()
        }
        for family_id in PRINCIPAL_FAMILY_IDS
    }

    assert len({json.dumps(value, sort_keys=True) for value in hashes.values()}) == 1
    assert splits["train"].episode_indices.tolist() == list(range(8))
    assert splits["validation"].episode_indices.tolist() == [8]
    assert splits["internal_test"].episode_indices.tolist() == [9]


def test_every_family_resolves_the_identical_shadow_and_validation_schedules():
    protocol = load_principal_protocol(PROTOCOL)
    shadow_hash = complete_shadow_schedule_sha256(protocol)
    validation_hash = evaluation_schedule_sha256(protocol)

    observed = {
        family_id: (shadow_hash, validation_hash)
        for family_id in PRINCIPAL_FAMILY_IDS
    }

    assert len(set(observed.values())) == 1
    assert validation_hash == (
        "f148c56eb6ce26701abc39419c9efe6c80bab16c493a78ddf8bd95e8a4dc2993"
    )
    schedule = evaluation_schedule(protocol)
    assert len(schedule) == 1125
    assert [value for _, value in schedule[:5]] == [0, 5, 10, 15, 20]


def test_principal_test_cannot_be_opened_through_the_validation_pipeline():
    protocol = load_principal_protocol(PROTOCOL)

    with pytest.raises(ValueError, match="remains closed"):
        evaluation_schedule(protocol, "test")


def test_collector_manifest_must_be_the_common_hash_bound_base_data():
    protocol = load_principal_protocol(PROTOCOL)
    base = protocol["base_teacher_dataset"]
    manifest = {
        "status": "completed",
        "dataset_id": base["id"],
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "episode_count": base["episodes"],
    }

    for family_id in PRINCIPAL_FAMILY_IDS:
        validate_principal_dataset_manifest(
            protocol,
            manifest,
            base["manifest_sha256"],
            family_id=family_id,
            stage="collector",
        )

    with pytest.raises(ValueError, match="frozen common base"):
        validate_principal_dataset_manifest(
            protocol,
            manifest,
            "0" * 64,
            family_id="feed_forward",
            stage="collector",
        )


def test_final_manifest_is_family_specific_but_base_and_schedule_matched():
    protocol = load_principal_protocol(PROTOCOL)
    manifest = {
        "status": "completed",
        "dataset_id": "aggregate",
        "family_id": "structured_k1",
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "episode_count": 40000,
        "protocol_sha256": protocol["resolved_sha256"],
        "shadow_schedule_sha256": complete_shadow_schedule_sha256(protocol),
        "source_manifests": {
            "teacher_controlled": protocol["base_teacher_dataset"][
                "manifest_sha256"
            ],
            **{
                f"shadow_seed_{seed}": f"manifest-{seed}"
                for seed in protocol["matched_seeds"]["training"]
            },
        },
    }

    validate_principal_dataset_manifest(
        protocol,
        manifest,
        "unused-for-family-aggregate",
        family_id="structured_k1",
        stage="final",
    )
    manifest["family_id"] = "structured_k2"
    with pytest.raises(ValueError, match="different family"):
        validate_principal_dataset_manifest(
            protocol,
            manifest,
            "unused",
            family_id="structured_k1",
            stage="final",
        )
