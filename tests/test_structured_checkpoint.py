import numpy as np
import pytest

from airhockey_distill.students import (
    StructuredRecurrentPolicy,
    initialise_structured_parameters,
    save_structured_checkpoint,
)


def test_structured_checkpoint_round_trip_reproduces_action_sequence(tmp_path):
    rng = np.random.default_rng(31)
    parameters = initialise_structured_parameters(31)
    policy = StructuredRecurrentPolicy(parameters, {})
    observations = rng.normal(size=(17, 19)).astype(np.float32)
    previous_actions = rng.uniform(-1.0, 1.0, size=(17, 2)).astype(np.float32)
    previous_actions[0] = 0.0
    expected_actions, expected_states = policy.teacher_forced_sequence(
        observations, previous_actions
    )
    path = tmp_path / "structured_n64_k2.npz"

    save_structured_checkpoint(path, parameters, {"training_seed": 31})
    restored = StructuredRecurrentPolicy.load(path)
    restored_actions, restored_states = restored.teacher_forced_sequence(
        observations, previous_actions
    )

    np.testing.assert_array_equal(restored_actions, expected_actions)
    np.testing.assert_array_equal(restored_states, expected_states)
    assert restored.metadata["training_seed"] == 31
    assert restored.metadata["state_dimension"] == 64
    assert restored.metadata["innovation_rank"] == 2
    assert restored.metadata["include_previous_action"] is True


def test_structured_checkpoint_rejects_conflicting_architecture_metadata(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        save_structured_checkpoint(
            tmp_path / "wrong.npz",
            initialise_structured_parameters(32),
            {"innovation_rank": 4},
        )
