import numpy as np
import pytest

from airhockey_distill.students import (
    GRURecurrentPolicy,
    initialise_gru_parameters,
    save_gru_checkpoint,
)


def test_gru_checkpoint_round_trip_reproduces_action_and_state_sequence(tmp_path):
    rng = np.random.default_rng(58)
    parameters = initialise_gru_parameters(58)
    policy = GRURecurrentPolicy(parameters, {})
    observations = rng.normal(size=(17, 19)).astype(np.float32)
    previous_actions = rng.uniform(-1.0, 1.0, size=(17, 2)).astype(np.float32)
    previous_actions[0] = 0.0
    expected_actions, expected_states = policy.teacher_forced_sequence(
        observations, previous_actions
    )
    path = tmp_path / "gru_n64.npz"

    save_gru_checkpoint(path, parameters, {"training_seed": 58})
    restored = GRURecurrentPolicy.load(path)
    restored_actions, restored_states = restored.teacher_forced_sequence(
        observations, previous_actions
    )

    np.testing.assert_array_equal(restored_actions, expected_actions)
    np.testing.assert_array_equal(restored_states, expected_states)
    assert restored.metadata["training_seed"] == 58
    assert restored.metadata["state_dimension"] == 64
    assert restored.metadata["include_previous_action"] is True


def test_gru_checkpoint_rejects_conflicting_architecture_metadata(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        save_gru_checkpoint(
            tmp_path / "wrong.npz",
            initialise_gru_parameters(59),
            {"state_dimension": 32},
        )
