from types import SimpleNamespace

import numpy as np
import pytest

from airhockey_distill.principal_sweep import (
    DETERMINISTIC_ACTION_SEMANTICS,
    load_principal_protocol,
    shadow_schedule_records,
)
from airhockey_distill.students import (
    PRINCIPAL_FAMILY_IDS,
    principal_policy_from_parameters,
)
from airhockey_distill.students.feed_forward import (
    initialise_feed_forward_parameters,
)
from airhockey_distill.students.finite_stack import (
    initialise_finite_stack_parameters,
)
from airhockey_distill.students.gru import initialise_gru_parameters
from airhockey_distill.students.structured import initialise_structured_parameters
from scripts.collect_principal_shadow_dataset import (
    _validate_checkpoints,
    validate_shadow_shard,
)


def _policy(family_id):
    if family_id == "feed_forward":
        parameters = initialise_feed_forward_parameters(14303)
    elif family_id == "finite_stack_10":
        parameters = initialise_finite_stack_parameters(14303)
    elif family_id == "gru_n64":
        parameters = initialise_gru_parameters(14303)
    else:
        rank = int(family_id.removeprefix("structured_k"))
        parameters = initialise_structured_parameters(14303, innovation_rank=rank)
    return principal_policy_from_parameters(family_id, parameters, {})


@pytest.mark.parametrize("family_id", PRINCIPAL_FAMILY_IDS)
def test_common_policy_loop_resets_each_family_only_at_episode_boundary(family_id):
    policy = _policy(family_id)
    observation = np.linspace(-0.5, 0.5, 19, dtype=np.float32)

    first, carried = policy.act(observation, policy.initial_carry())
    second, _ = policy.act(observation, carried)
    repeated, _ = policy.act(observation, policy.initial_carry())

    np.testing.assert_array_equal(repeated, first)
    if family_id == "feed_forward":
        np.testing.assert_array_equal(second, first)
    else:
        assert not np.array_equal(second, first)


def test_existing_shadow_shard_must_preserve_schedule_and_command_history(tmp_path):
    protocol = load_principal_protocol("configs/experiments/principal_sweep_execution_v1.yaml")
    records = shadow_schedule_records(protocol, 14303)[:2]
    path = tmp_path / "shard.npz"
    behaviour = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], np.float32)
    previous = np.asarray([[0.0, 0.0], [0.1, 0.2], [0.0, 0.0]], np.float32)
    np.savez(
        path,
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
        episode_offsets=np.asarray([0, 2, 3]),
        previous_actions=previous,
        behaviour_actions=behaviour,
    )

    validate_shadow_shard(path, records)

    with np.load(path, allow_pickle=False) as values:
        payload = {name: np.asarray(values[name]) for name in values.files}
    payload["episode_reset_seeds"][0] += 1
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="paired schedule"):
        validate_shadow_shard(path, records)


def test_shadow_checkpoint_must_be_matching_base_only_collector(tmp_path):
    protocol = load_principal_protocol("configs/experiments/principal_sweep_execution_v1.yaml")
    checkpoint = tmp_path / "student.npz"
    checkpoint.write_bytes(b"checkpoint")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "agent.pkl").write_bytes(b"teacher")
    metadata = {
        "student_id": "gru_n64",
        "training_seed": 14303,
        "training_stage": "collector",
        "protocol_sha256": protocol["resolved_sha256"],
        "dataset_manifest_sha256": protocol["base_teacher_dataset"][
            "manifest_sha256"
        ],
    }
    protocol["teacher"]["actor_sha256"] = __import__("hashlib").sha256(
        b"teacher"
    ).hexdigest()

    _validate_checkpoints(
        protocol, metadata, "gru_n64", 14303, checkpoint, teacher
    )

    metadata["training_stage"] = "final"
    with pytest.raises(ValueError, match="collector checkpoint"):
        _validate_checkpoints(
            protocol, metadata, "gru_n64", 14303, checkpoint, teacher
        )
