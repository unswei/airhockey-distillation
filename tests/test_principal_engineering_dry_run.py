import hashlib
import json

import numpy as np

from airhockey_distill.principal_sweep import load_principal_protocol
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS
from scripts.run_principal_engineering_dry_run import (
    NON_PRINCIPAL_CLASSIFICATION,
    load_tiny_split,
    paired_shadow_schedule_check,
    short_validation_schedule,
    tiny_overfit_passed,
)


def test_dry_run_uses_identical_small_shadow_schedule_for_all_families():
    protocol = load_principal_protocol(
        "configs/experiments/principal_sweep_execution_v1.yaml"
    )

    result = paired_shadow_schedule_check(protocol)

    assert result["passed"]
    assert result["records_per_family"] == 10
    assert set(result["family_schedule_sha256"]) == set(PRINCIPAL_FAMILY_IDS)
    assert len(set(result["family_schedule_sha256"].values())) == 1


def test_dry_run_validation_is_a_short_prefix_of_the_open_validation_only():
    protocol = load_principal_protocol(
        "configs/experiments/principal_sweep_execution_v1.yaml"
    )

    schedule = short_validation_schedule(protocol, shots=2, blackouts=2)

    assert len(schedule) == 4
    assert [blackout for _, blackout in schedule] == [0, 5, 0, 5]
    assert all(generated.split == "validation" for generated, _ in schedule)


def test_tiny_loader_selects_one_deterministic_mean_episode(tmp_path):
    shard = tmp_path / "shard.npz"
    observations = np.zeros((5, 19), np.float32)
    previous = np.zeros((5, 2), np.float32)
    targets = np.zeros((5, 2), np.float32)
    np.savez(
        shard,
        dataset_schema_version=np.asarray(2),
        observations=observations,
        previous_actions=previous,
        teacher_actions=targets,
        puck_visible=np.asarray([True, True, False, False, True]),
        episode_offsets=np.asarray([0, 3, 5]),
        episode_indices=np.asarray([11, 12]),
        episode_shot_ids=np.asarray(["shot-a", "shot-b"]),
        episode_blackout_steps=np.asarray([2, 0]),
    )
    manifest = {
        "status": "completed",
        "dataset_id": "tiny",
        "teacher_action_semantics": (
            "deterministic_actor_mean_after_public_adapter_clip"
        ),
        "episode_count": 2,
        "shards": [
            {
                "file": shard.name,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }

    split, source = load_tiny_split(tmp_path, manifest)

    assert split.episode_count == 1
    assert split.episode_lengths.tolist() == [2]
    assert split.observations.shape == (1, 128, 19)
    assert source == {
        "dataset_id": "tiny",
        "episode_index": 12,
        "shot_id": "shot-b",
        "blackout_steps": 0,
        "episode_steps": 2,
        "teacher_action_semantics": (
            "deterministic_actor_mean_after_public_adapter_clip"
        ),
    }


def test_dry_run_classification_cannot_be_mistaken_for_principal_evidence():
    assert NON_PRINCIPAL_CLASSIFICATION == "non_principal_engineering_dry_run"
    assert "principal" in NON_PRINCIPAL_CLASSIFICATION
    assert NON_PRINCIPAL_CLASSIFICATION.startswith("non_")


def test_dry_run_overfit_requires_absolute_and_fractional_improvement():
    assert tiny_overfit_passed(
        initial_mse=0.546,
        selected_mse=0.00818,
        maximum_mse=0.01,
        minimum_fractional_reduction=0.98,
    )
    assert not tiny_overfit_passed(
        initial_mse=0.02,
        selected_mse=0.009,
        maximum_mse=0.01,
        minimum_fractional_reduction=0.98,
    )
    assert not tiny_overfit_passed(
        initial_mse=0.5,
        selected_mse=0.02,
        maximum_mse=0.01,
        minimum_fractional_reduction=0.98,
    )
