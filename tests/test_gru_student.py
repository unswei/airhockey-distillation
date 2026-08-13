import numpy as np
import pytest

from airhockey_distill.students import (
    GRU_PARAMETER_SHAPES,
    GRU_STATE_DIM,
    GRURecurrentPolicy,
    initialise_gru_parameters,
)


def test_gru_student_has_matched_hidden_size_64_architecture():
    policy = GRURecurrentPolicy(initialise_gru_parameters(51), {})

    assert GRU_STATE_DIM == 64
    assert GRU_PARAMETER_SHAPES["encoder_0_weight"] == (64, 19)
    assert GRU_PARAMETER_SHAPES["encoder_1_weight"] == (32, 64)
    assert GRU_PARAMETER_SHAPES["gru_input_weight"] == (192, 34)
    assert GRU_PARAMETER_SHAPES["action_hidden_weight"] == (64, 96)
    assert policy.parameter_count == 28898
    assert policy.recurrent_parameter_count == 19200


def test_gru_closed_loop_carry_resets_only_when_explicitly_reinitialised():
    policy = GRURecurrentPolicy(initialise_gru_parameters(52), {})
    observation = np.linspace(-0.8, 0.8, 19, dtype=np.float32)

    first_action, carried = policy.act(observation, policy.initial_carry())
    continued_action, continued = policy.act(observation, carried)
    repeated_action, repeated = policy.act(observation, policy.initial_carry())

    np.testing.assert_array_equal(repeated_action, first_action)
    np.testing.assert_array_equal(repeated.memory, carried.memory)
    assert not np.array_equal(continued.memory, carried.memory)
    assert not np.array_equal(continued_action, first_action)
    np.testing.assert_array_equal(carried.previous_action, first_action)


def test_previous_action_enters_gru_update():
    parameters = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in GRU_PARAMETER_SHAPES.items()
    }
    # PyTorch gate order is reset, update, new. Drive the first candidate unit.
    parameters["gru_input_weight"][128, 32] = 2.0
    policy = GRURecurrentPolicy(parameters, {})
    observation = np.zeros(19, dtype=np.float32)
    state = np.zeros(64, dtype=np.float32)

    _, zero_action_state = policy.step(observation, np.zeros(2), state)
    _, driven_state = policy.step(
        observation, np.asarray([0.5, 0.0], dtype=np.float32), state
    )

    np.testing.assert_array_equal(zero_action_state, np.zeros(64, dtype=np.float32))
    assert driven_state[0] == pytest.approx(0.5 * np.tanh(1.0))


def test_gru_batched_step_matches_individual_steps():
    rng = np.random.default_rng(53)
    policy = GRURecurrentPolicy(initialise_gru_parameters(53), {})
    observations = rng.normal(size=(4, 19)).astype(np.float32)
    previous_actions = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    previous_states = rng.normal(size=(4, 64)).astype(np.float32)

    batch_actions, batch_states = policy.step(
        observations, previous_actions, previous_states
    )
    for index in range(4):
        action, state = policy.step(
            observations[index], previous_actions[index], previous_states[index]
        )
        np.testing.assert_allclose(action, batch_actions[index], atol=3e-7)
        np.testing.assert_allclose(state, batch_states[index], atol=3e-7)


def test_gru_student_rejects_invalid_shapes_and_values():
    policy = GRURecurrentPolicy(initialise_gru_parameters(54), {})
    carry = policy.initial_carry()

    with pytest.raises(ValueError, match="observation"):
        policy.act(np.zeros(18, dtype=np.float32), carry)
    with pytest.raises(ValueError, match="previous action"):
        policy.step(np.zeros(19), np.zeros(3), np.zeros(64))
    observation = np.zeros(19, dtype=np.float32)
    observation[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.act(observation, carry)
