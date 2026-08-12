import numpy as np
import pytest

from airhockey_distill.students import (
    INNOVATION_RANK,
    STATE_DIM,
    STRUCTURED_PARAMETER_SHAPES,
    StructuredRecurrentPolicy,
    initialise_structured_parameters,
)


def test_structured_student_has_fixed_n64_k2_architecture():
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(42), {})

    assert STATE_DIM == 64
    assert INNOVATION_RANK == 2
    assert STRUCTURED_PARAMETER_SHAPES["innovation_output_weight"] == (64, 2)
    assert STRUCTURED_PARAMETER_SHAPES["innovation_state_weight"] == (2, 64)
    assert policy.parameter_count == 12328
    assert policy.recurrent_parameter_count == 2630


def test_diagonal_initialisation_spans_declared_time_constants():
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(7), {})
    diagonal = policy.diagonal_dynamics.astype(np.float64)
    time_constants_ms = -20.0 / np.log(diagonal)

    assert np.all((diagonal > 0.0) & (diagonal < 1.0))
    assert time_constants_ms[0] == pytest.approx(40.0, rel=1e-6)
    assert time_constants_ms[-1] == pytest.approx(2000.0, rel=1e-5)
    assert np.all(np.diff(time_constants_ms) > 0.0)


def test_closed_loop_carry_resets_only_when_explicitly_reinitialised():
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(8), {})
    observation = np.linspace(-0.8, 0.8, 19, dtype=np.float32)
    initial = policy.initial_carry()

    first_action, carried = policy.act(observation, initial)
    continued_action, continued = policy.act(observation, carried)
    repeated_action, repeated = policy.act(observation, policy.initial_carry())

    np.testing.assert_array_equal(repeated_action, first_action)
    np.testing.assert_array_equal(repeated.memory, carried.memory)
    assert not np.array_equal(continued.memory, carried.memory)
    assert not np.array_equal(continued_action, first_action)
    np.testing.assert_array_equal(carried.previous_action, first_action)


def test_previous_action_enters_the_recurrent_update():
    parameters = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in STRUCTURED_PARAMETER_SHAPES.items()
    }
    parameters["recurrence_action_weight"][0, 0] = 2.0
    parameters["innovation_action_weight"][0, 1] = 1.0
    parameters["innovation_output_weight"][1, 0] = 3.0
    policy = StructuredRecurrentPolicy(parameters, {})
    observation = np.zeros(19, dtype=np.float32)
    state = np.zeros(64, dtype=np.float32)

    _, zero_action_state = policy.step(
        observation, np.zeros(2, dtype=np.float32), state
    )
    _, driven_state = policy.step(
        observation, np.asarray([0.5, 0.25], dtype=np.float32), state
    )

    np.testing.assert_array_equal(zero_action_state, np.zeros(64, dtype=np.float32))
    assert driven_state[0] == pytest.approx(1.0)
    assert driven_state[1] == pytest.approx(3.0 * np.tanh(0.25))


def test_batched_step_matches_individual_steps():
    rng = np.random.default_rng(10)
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(10), {})
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
        np.testing.assert_allclose(action, batch_actions[index], atol=2e-7, rtol=1e-6)
        np.testing.assert_allclose(state, batch_states[index], atol=2e-7, rtol=1e-6)


def test_structured_student_rejects_invalid_shapes_and_values():
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(11), {})
    carry = policy.initial_carry()

    with pytest.raises(ValueError, match="observation"):
        policy.act(np.zeros(18, dtype=np.float32), carry)
    with pytest.raises(ValueError, match="previous action"):
        policy.step(
            np.zeros(19, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
        )
    observation = np.zeros(19, dtype=np.float32)
    observation[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.act(observation, carry)
