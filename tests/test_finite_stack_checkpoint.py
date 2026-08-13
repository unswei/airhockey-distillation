import numpy as np
import pytest

from airhockey_distill.students import (
    FiniteStackPolicy,
    initialise_finite_stack_parameters,
    save_finite_stack_checkpoint,
)


def test_finite_stack_checkpoint_round_trip_reproduces_action_and_history(tmp_path):
    rng = np.random.default_rng(69)
    parameters = initialise_finite_stack_parameters(69)
    policy = FiniteStackPolicy(parameters, {})
    observations = rng.normal(size=(17, 19)).astype(np.float32)
    expected_actions, expected_histories = policy.sequence(observations)
    path = tmp_path / "finite_stack_10.npz"

    save_finite_stack_checkpoint(path, parameters, {"training_seed": 69})
    restored = FiniteStackPolicy.load(path)
    restored_actions, restored_histories = restored.sequence(observations)

    np.testing.assert_array_equal(restored_actions, expected_actions)
    np.testing.assert_array_equal(restored_histories, expected_histories)
    assert restored.metadata["training_seed"] == 69
    assert restored.metadata["policy"] == "finite_stack_10"
    assert restored.metadata["puck_history_steps"] == 10
    assert restored.metadata["network_input_dimension"] == 46
    assert restored.metadata["include_previous_action"] is False


def test_finite_stack_checkpoint_rejects_conflicting_metadata(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        save_finite_stack_checkpoint(
            tmp_path / "wrong.npz",
            initialise_finite_stack_parameters(70),
            {"puck_history_steps": 8},
        )
