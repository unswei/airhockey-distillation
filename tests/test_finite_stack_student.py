from pathlib import Path

import numpy as np
import pytest
import yaml

from airhockey_distill.students import (
    FINITE_STACK_HISTORY_DIM,
    FINITE_STACK_HISTORY_STEPS,
    FINITE_STACK_INPUT_DIM,
    FINITE_STACK_PARAMETER_SHAPES,
    FINITE_STACK_PROPRIOCEPTION_DIM,
    FiniteStackPolicy,
    initialise_finite_stack_parameters,
)


def _observation(step: int, *, visible: bool = True) -> np.ndarray:
    observation = np.zeros(19, dtype=np.float32)
    observation[:16] = step / 100.0
    if visible:
        observation[16:18] = (step, -step)
        observation[18] = 1.0
    return observation


def test_finite_stack_matches_predeclared_architecture_and_counts():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(61), {})
    config = yaml.safe_load(
        Path("configs/student/finite_stack_10.yaml").read_text()
    )

    assert FINITE_STACK_HISTORY_STEPS == 10
    assert FINITE_STACK_PROPRIOCEPTION_DIM == 16
    assert FINITE_STACK_HISTORY_DIM == 30
    assert FINITE_STACK_INPUT_DIM == 46
    assert FINITE_STACK_PARAMETER_SHAPES["encoder_0_weight"] == (64, 46)
    assert policy.parameter_count == 7330
    assert config["policy"]["puck_history"]["steps"] == 10
    assert config["policy"]["network_input_dimension"] == 46
    assert config["expected_counts"]["total_trainable_parameters"] == 7330
    assert config["expected_counts"]["persistent_state_float32_values"] == 30


def test_history_is_oldest_to_newest_and_retains_exactly_ten_triples():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(62), {})
    history = policy.initial_carry().puck_history

    for step in range(1, 12):
        policy_input, updated, single = policy.prepare_input(
            _observation(step),
            history,
        )
        history = updated[0]
        assert single
        np.testing.assert_array_equal(policy_input[0, :16], step / 100.0)
        np.testing.assert_array_equal(policy_input[0, 16:], history)

    expected = np.asarray(
        [(step, -step, 1.0) for step in range(2, 12)],
        dtype=np.float32,
    ).reshape(-1)
    np.testing.assert_array_equal(history, expected)


def test_episode_start_padding_and_blackout_triples_are_zero():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(63), {})
    initial = policy.initial_carry()

    _, visible = policy.act(_observation(1), initial)
    _, hidden = policy.act(_observation(2, visible=False), visible)

    np.testing.assert_array_equal(initial.puck_history, np.zeros(30, np.float32))
    np.testing.assert_array_equal(
        visible.puck_history[-3:],
        np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(hidden.puck_history[-3:], np.zeros(3, np.float32))
    np.testing.assert_array_equal(
        hidden.puck_history[-6:-3],
        visible.puck_history[-3:],
    )


def test_visibility_change_does_not_reset_older_history():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(64), {})
    history = policy.initial_carry().puck_history
    observations = np.stack(
        [_observation(1), _observation(2), _observation(3, visible=False)]
    )

    _, histories = policy.sequence(observations, history)

    np.testing.assert_array_equal(
        histories[-1, -9:],
        np.asarray(
            [1.0, -1.0, 1.0, 2.0, -2.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_only_current_proprioception_is_outside_the_puck_history():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(65), {})
    observation = np.arange(19, dtype=np.float32)
    history = np.arange(30, dtype=np.float32) + 100.0

    policy_input, updated, _ = policy.prepare_input(observation, history)

    np.testing.assert_array_equal(policy_input[0, :16], observation[:16])
    np.testing.assert_array_equal(updated[0, -3:], observation[16:19])
    np.testing.assert_array_equal(policy_input[0, 16:], updated[0])
    assert policy_input.shape == (1, 46)


def test_closed_loop_history_resets_only_when_explicitly_reinitialised():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(66), {})
    observation = _observation(4)

    first_action, carried = policy.act(observation, policy.initial_carry())
    continued_action, continued = policy.act(observation, carried)
    repeated_action, repeated = policy.act(observation, policy.initial_carry())

    np.testing.assert_array_equal(repeated_action, first_action)
    np.testing.assert_array_equal(repeated.puck_history, carried.puck_history)
    assert not np.array_equal(continued.puck_history, carried.puck_history)
    assert not np.array_equal(continued_action, first_action)


def test_batched_step_matches_individual_steps():
    rng = np.random.default_rng(67)
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(67), {})
    observations = rng.normal(size=(4, 19)).astype(np.float32)
    histories = rng.normal(size=(4, 30)).astype(np.float32)

    batch_actions, batch_histories = policy.step(observations, histories)
    for index in range(4):
        action, history = policy.step(observations[index], histories[index])
        np.testing.assert_allclose(action, batch_actions[index], atol=3e-7)
        np.testing.assert_array_equal(history, batch_histories[index])


def test_finite_stack_rejects_invalid_shapes_and_values():
    policy = FiniteStackPolicy(initialise_finite_stack_parameters(68), {})

    with pytest.raises(ValueError, match="observation"):
        policy.act(np.zeros(18, dtype=np.float32), policy.initial_carry())
    with pytest.raises(ValueError, match="previous history"):
        policy.step(np.zeros(19, dtype=np.float32), np.zeros(29, dtype=np.float32))
    observation = np.zeros(19, dtype=np.float32)
    observation[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.act(observation, policy.initial_carry())
