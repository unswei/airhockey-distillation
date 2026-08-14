import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from airhockey_distill.students import (
    INNOVATION_RANK,
    STATE_DIM,
    SUPPORTED_INNOVATION_RANKS,
    STRUCTURED_PARAMETER_SHAPES,
    StructuredRecurrentPolicy,
    initialise_structured_parameters,
    structured_parameter_shapes,
)
from airhockey_distill.students.structured import (
    NUMPY_PAIRWISE_FLOAT32_IMPLEMENTATION,
)


def test_structured_student_has_fixed_n64_k2_architecture():
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(42), {})

    assert STATE_DIM == 64
    assert INNOVATION_RANK == 2
    assert STRUCTURED_PARAMETER_SHAPES["innovation_output_weight"] == (64, 2)
    assert STRUCTURED_PARAMETER_SHAPES["innovation_state_weight"] == (2, 64)
    assert policy.parameter_count == 12328
    assert policy.recurrent_parameter_count == 2630
    assert (
        policy.batch_one_inference_implementation
        == NUMPY_PAIRWISE_FLOAT32_IMPLEMENTATION
    )


@pytest.mark.parametrize(
    ("rank", "parameters", "recurrent_parameters"),
    ((0, 12002, 2304), (1, 12165, 2467), (2, 12328, 2630), (4, 12654, 2956)),
)
def test_structured_student_supports_predeclared_ranks(
    rank, parameters, recurrent_parameters
):
    policy = StructuredRecurrentPolicy(
        initialise_structured_parameters(420 + rank, innovation_rank=rank),
        {},
    )

    assert SUPPORTED_INNOVATION_RANKS == (0, 1, 2, 4)
    assert policy.innovation_rank == rank
    assert policy.parameter_count == parameters
    assert policy.recurrent_parameter_count == recurrent_parameters
    if rank == 0:
        assert not any(name.startswith("innovation_") for name in policy.parameters)
    else:
        assert policy.parameters["innovation_output_weight"].shape == (64, rank)
        assert policy.parameters["innovation_state_weight"].shape == (rank, 64)


@pytest.mark.parametrize(
    ("rank", "parameters", "recurrent_parameters"),
    ((0, 12002, 2304), (1, 12165, 2467), (4, 12654, 2956)),
)
def test_new_rank_configs_match_runtime_counts(rank, parameters, recurrent_parameters):
    config = yaml.safe_load(
        Path(f"configs/student/structured_n64_k{rank}.yaml").read_text()
    )
    policy = StructuredRecurrentPolicy(
        initialise_structured_parameters(427, innovation_rank=rank),
        {},
    )

    assert config["policy"]["recurrence"]["innovation_rank"] == rank
    assert config["expected_counts"]["total_trainable_parameters"] == parameters
    assert (
        config["expected_counts"]["recurrent_core_parameters"]
        == recurrent_parameters
    )
    assert policy.parameter_count == parameters
    assert policy.recurrent_parameter_count == recurrent_parameters


def test_default_k2_initialisation_is_exactly_backwards_compatible():
    default = initialise_structured_parameters(423)
    explicit = initialise_structured_parameters(423, innovation_rank=2)

    assert STRUCTURED_PARAMETER_SHAPES == structured_parameter_shapes(2)
    assert default.keys() == explicit.keys()
    for name in default:
        np.testing.assert_array_equal(default[name], explicit[name])


def test_default_k2_initialisation_matches_legacy_parameter_digest():
    parameters = initialise_structured_parameters(42)
    digest = hashlib.sha256()
    for name, value in parameters.items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())

    assert digest.hexdigest() == (
        "93930bc5bf38543ef6accb75209409f6e577f957c02aff46582037d26c611b42"
    )


def test_matched_seed_shares_every_noninnovation_parameter_across_ranks():
    by_rank = {
        rank: initialise_structured_parameters(426, innovation_rank=rank)
        for rank in SUPPORTED_INNOVATION_RANKS
    }
    common_names = {
        name
        for name in by_rank[2]
        if not name.startswith("innovation_")
    }

    for name in common_names:
        for rank in SUPPORTED_INNOVATION_RANKS:
            np.testing.assert_array_equal(by_rank[rank][name], by_rank[2][name])


def test_k0_is_exactly_linear_in_the_previous_state():
    rng = np.random.default_rng(424)
    policy = StructuredRecurrentPolicy(
        initialise_structured_parameters(424, innovation_rank=0),
        {},
    )
    observation = rng.normal(size=19).astype(np.float32)
    previous_action = rng.normal(size=2).astype(np.float32)
    previous_state = rng.normal(size=64).astype(np.float32)

    jacobian = policy.recurrent_jacobian(
        observation,
        previous_action,
        previous_state,
    )

    np.testing.assert_array_equal(jacobian, np.diag(policy.diagonal_dynamics))


@pytest.mark.parametrize("rank", (-1, 3, 5, True))
def test_structured_student_rejects_unapproved_ranks(rank):
    with pytest.raises(ValueError, match="0, 1, 2 or 4"):
        initialise_structured_parameters(425, innovation_rank=rank)


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


@pytest.mark.parametrize("rank", (0, 1, 2, 4))
def test_canonical_batch_one_act_is_bit_exact_with_general_step(rank):
    rng = np.random.default_rng(801 + rank)
    policy = StructuredRecurrentPolicy(
        initialise_structured_parameters(91 + rank, innovation_rank=rank),
        {"student_id": f"structured_k{rank}"},
        innovation_rank=rank,
    )
    carry = policy.initial_carry()
    reference_state = carry.memory
    reference_previous_action = carry.previous_action

    for _ in range(200):
        observation = rng.normal(size=19).astype(np.float32)
        expected_action, expected_state = policy.step(
            observation,
            reference_previous_action,
            reference_state,
        )
        action, carry = policy.act(observation, carry)

        np.testing.assert_array_equal(action, expected_action)
        np.testing.assert_array_equal(carry.memory, expected_state)
        np.testing.assert_array_equal(carry.previous_action, expected_action)
        reference_state = expected_state
        reference_previous_action = expected_action


def test_canonical_batched_act_keeps_the_general_batch_contract():
    rng = np.random.default_rng(811)
    policy = StructuredRecurrentPolicy(
        initialise_structured_parameters(92, innovation_rank=2),
        {"student_id": "structured_k2"},
        innovation_rank=2,
    )
    observations = rng.normal(size=(3, 19)).astype(np.float32)
    carry = policy.initial_carry(batch_size=3)

    expected_action, expected_state = policy.step(
        observations,
        carry.previous_action,
        carry.memory,
    )
    action, next_carry = policy.act(observations, carry)

    np.testing.assert_array_equal(action, expected_action)
    np.testing.assert_array_equal(next_carry.memory, expected_state)


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
