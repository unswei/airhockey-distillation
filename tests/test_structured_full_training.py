import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from airhockey_distill.students import (
    StructuredRecurrentPolicy,
    initialise_structured_parameters,
    save_structured_checkpoint,
)
from scripts.train_structured_student import (
    DETERMINISTIC_ACTION_SEMANTICS,
    chunk_slices,
    evaluate_split,
    load_episode_splits,
    pad_episode_split,
    split_residue_mapping,
    train_epoch,
    validate_deterministic_manifest,
    validate_full_training_config,
)


def test_full_config_predeclares_project_sequence_and_episode_splits():
    path = Path("configs/student/structured_n64_k2_full_seed_14303.yaml")
    config = yaml.safe_load(path.read_text())

    validate_full_training_config(config)

    assert config["dataset"]["episodes"] == 20000
    assert len(config["dataset"]["train_residues"]) == 8
    assert len(config["dataset"]["validation_residues"]) == 1
    assert len(config["dataset"]["internal_test_residues"]) == 1
    assert config["training"]["sequence_length"] == 64
    assert config["training"]["burn_in_steps"] == 16
    assert config["training"]["loss_steps"] == 48
    assert config["training"]["batch_size"] == 128


def test_all_steps_correction_changes_only_the_predeclared_loss_window():
    path = Path("configs/student/structured_n64_k2_full_seed_14303_all_steps_v1.yaml")
    config = yaml.safe_load(path.read_text())
    parent = yaml.safe_load(
        Path("configs/student/structured_n64_k2_full_seed_14303_v2.yaml").read_text()
    )

    validate_full_training_config(config)

    assert config["policy"] == parent["policy"]
    assert config["dataset"] == parent["dataset"]
    assert config["export"] == parent["export"]
    assert config["provenance"] == {
        **parent["provenance"],
        "dataset_manifest_sha256": (
            "923497e3622209c7060196ff8322f4d333626138048ad94577034c67f01c60cf"
        ),
    }
    unchanged_training = set(parent["training"]) - {"burn_in_steps", "loss_steps"}
    assert all(
        config["training"][key] == parent["training"][key] for key in unchanged_training
    )
    assert config["training"]["burn_in_steps"] == 0
    assert config["training"]["loss_steps"] == 64
    assert config["evaluation"]["blackout_steps"] == [0]
    assert config["no_blackout_gate"]["minimum_save_rate"] == 0.75


def test_zero_burn_in_supervises_every_valid_step():
    split = pad_episode_split(
        [_episode(episode_index=0, length=20)],
        maximum_episode_steps=128,
        burn_in_steps=0,
    )

    assert np.all(split.loss_mask[0, :20])
    assert not np.any(split.loss_mask[0, 20:])


def test_v2_all_steps_correction_changes_only_export_tolerance():
    v1 = yaml.safe_load(
        Path(
            "configs/student/structured_n64_k2_full_seed_14303_all_steps_v1.yaml"
        ).read_text()
    )
    v2 = yaml.safe_load(
        Path(
            "configs/student/structured_n64_k2_full_seed_14303_all_steps_v2.yaml"
        ).read_text()
    )

    assert v2["policy"] == v1["policy"]
    assert v2["dataset"] == v1["dataset"]
    assert v2["training"] == v1["training"]
    assert v2["evaluation"] == v1["evaluation"]
    assert v2["no_blackout_gate"] == v1["no_blackout_gate"]
    assert v2["provenance"] == v1["provenance"]
    assert v2["export"] == {
        **v1["export"],
        "maximum_absolute_error": 0.00002,
    }


def test_full_trainer_rejects_sampled_action_manifest():
    dataset = _dataset_config()
    manifest = _manifest()
    manifest["teacher_action_semantics"] = "executed_after_public_adapter_clip"
    manifest["deterministic_inference"] = False

    with pytest.raises(ValueError, match="deterministic means"):
        validate_deterministic_manifest(manifest, dataset)


def test_episode_split_padding_masks_only_boundary_burn_in(tmp_path):
    dataset = _dataset_config()
    shard_path = tmp_path / "shard-0000.npz"
    lengths = [20] * 9 + [70]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    transition_count = int(offsets[-1])
    observations = np.arange(transition_count * 19, dtype=np.float32).reshape(
        transition_count, 19
    )
    previous_actions = np.full((transition_count, 2), 0.25, dtype=np.float32)
    teacher_actions = np.full((transition_count, 2), -0.5, dtype=np.float32)
    for offset in offsets[:-1]:
        previous_actions[int(offset)] = 0.0
    np.savez_compressed(
        shard_path,
        dataset_schema_version=np.asarray(2, dtype=np.int64),
        teacher_action_semantics=np.asarray(DETERMINISTIC_ACTION_SEMANTICS),
        observations=observations,
        previous_actions=previous_actions,
        teacher_actions=teacher_actions,
        puck_visible=np.ones(transition_count, dtype=bool),
        episode_offsets=offsets,
        episode_indices=np.arange(10, dtype=np.int64),
    )
    manifest = _manifest()
    manifest["shards"] = [
        {
            "file": shard_path.name,
            "sha256": _sha256(shard_path),
        }
    ]

    splits = load_episode_splits(
        tmp_path,
        manifest,
        dataset,
        maximum_episode_steps=128,
        burn_in_steps=16,
    )

    assert {name: value.episode_count for name, value in splits.items()} == {
        "train": 8,
        "validation": 1,
        "internal_test": 1,
    }
    assert splits["train"].loss_mask.sum(axis=1).tolist() == [4] * 8
    internal = splits["internal_test"]
    assert internal.episode_lengths.tolist() == [70]
    assert not np.any(internal.loss_mask[0, :16])
    assert np.all(internal.loss_mask[0, 16:70])
    assert not np.any(internal.loss_mask[0, 70:])
    assert chunk_slices(128, 64) == (slice(0, 64), slice(64, 128))


def test_split_residues_must_be_disjoint_and_complete():
    dataset = _dataset_config()
    dataset["validation_residues"] = [7]

    with pytest.raises(ValueError, match="disjoint"):
        split_residue_mapping(dataset)


def test_training_smoke_carries_across_two_chunks():
    import torch

    from airhockey_distill.students.structured_torch import (
        StructuredRecurrentModule,
    )

    episodes = []
    for episode_index in range(4):
        length = 70
        previous_actions = np.full((length, 2), 0.1, dtype=np.float32)
        previous_actions[0] = 0.0
        episodes.append(
            {
                "episode_index": episode_index,
                "observations": np.full(
                    (length, 19), episode_index / 10.0, dtype=np.float32
                ),
                "previous_actions": previous_actions,
                "teacher_actions": np.full((length, 2), -0.2, dtype=np.float32),
                "puck_visible": np.arange(length) < 20,
            }
        )
    split = pad_episode_split(
        episodes,
        maximum_episode_steps=128,
        burn_in_steps=16,
    )
    module = StructuredRecurrentModule(seed=7)
    optimiser = torch.optim.AdamW(module.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(7)

    training_mse = train_epoch(
        module,
        optimiser,
        split,
        batch_size=2,
        sequence_length=64,
        gradient_clip=1.0,
        generator=generator,
    )
    metrics = evaluate_split(module, split, batch_size=2, sequence_length=64)

    assert np.isfinite(training_mse)
    assert np.isfinite(metrics["action_mse"])
    assert metrics["loss_bearing_action_values"] == 4 * (70 - 16) * 2


def test_full_checkpoint_uses_student_id_without_architecture_conflict(tmp_path):
    path = tmp_path / "student.npz"
    metadata = {
        "schema_version": 1,
        "student_id": "structured_n64_k2",
        "training_seed": 14303,
    }

    save_structured_checkpoint(path, initialise_structured_parameters(14303), metadata)
    restored = StructuredRecurrentPolicy.load(path)

    assert restored.metadata["policy"] == "structured_recurrent_n64_k2"
    assert restored.metadata["student_id"] == "structured_n64_k2"


def _dataset_config():
    return {
        "id": "teacher_v3_structured_n64_k2_full_deterministic_v1",
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "episodes": 10,
        "episode_split_modulus": 10,
        "train_residues": list(range(8)),
        "validation_residues": [8],
        "internal_test_residues": [9],
    }


def _manifest():
    return {
        "status": "completed",
        "dataset_id": "teacher_v3_structured_n64_k2_full_deterministic_v1",
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "episode_count": 10,
        "shards": [],
    }


def _episode(*, episode_index, length):
    return {
        "episode_index": episode_index,
        "observations": np.zeros((length, 19), dtype=np.float32),
        "previous_actions": np.zeros((length, 2), dtype=np.float32),
        "teacher_actions": np.zeros((length, 2), dtype=np.float32),
        "puck_visible": np.ones(length, dtype=bool),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
