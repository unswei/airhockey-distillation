"""Structured recurrent student with a rank-two nonlinear innovation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from airhockey_distill.envs.policy_interface import (
    PUBLIC_ACTION_DIM,
    PUBLIC_OBSERVATION_DIM,
)

ENCODED_DIM = 32
STATE_DIM = 64
INNOVATION_RANK = 2
CONTROL_PERIOD_MS = 20.0

STRUCTURED_PARAMETER_SHAPES = {
    "encoder_0_weight": (64, PUBLIC_OBSERVATION_DIM),
    "encoder_0_bias": (64,),
    "encoder_1_weight": (ENCODED_DIM, 64),
    "encoder_1_bias": (ENCODED_DIM,),
    "recurrence_alpha": (STATE_DIM,),
    "recurrence_input_weight": (STATE_DIM, ENCODED_DIM),
    "recurrence_action_weight": (STATE_DIM, PUBLIC_ACTION_DIM),
    "recurrence_bias": (STATE_DIM,),
    "innovation_output_weight": (STATE_DIM, INNOVATION_RANK),
    "innovation_state_weight": (INNOVATION_RANK, STATE_DIM),
    "innovation_input_weight": (INNOVATION_RANK, ENCODED_DIM),
    "innovation_action_weight": (INNOVATION_RANK, PUBLIC_ACTION_DIM),
    "innovation_bias": (INNOVATION_RANK,),
    "action_hidden_weight": (64, STATE_DIM + ENCODED_DIM),
    "action_hidden_bias": (64,),
    "action_output_weight": (PUBLIC_ACTION_DIM, 64),
    "action_output_bias": (PUBLIC_ACTION_DIM,),
}

RECURRENT_PARAMETER_NAMES = frozenset(
    {
        "recurrence_alpha",
        "recurrence_input_weight",
        "recurrence_action_weight",
        "recurrence_bias",
        "innovation_output_weight",
        "innovation_state_weight",
        "innovation_input_weight",
        "innovation_action_weight",
        "innovation_bias",
    }
)

_CHECKPOINT_ARCHITECTURE = {
    "schema_version": 1,
    "policy": "structured_recurrent_n64_k2",
    "observation_dimension": PUBLIC_OBSERVATION_DIM,
    "encoded_dimension": ENCODED_DIM,
    "state_dimension": STATE_DIM,
    "innovation_rank": INNOVATION_RANK,
    "action_dimension": PUBLIC_ACTION_DIM,
    "include_previous_action": True,
    "action_output": "tanh_mean",
}


@dataclass(frozen=True)
class StructuredPolicyCarry:
    """Memory and previous executed action carried between control steps."""

    memory: NDArray[np.float32]
    previous_action: NDArray[np.float32]


@dataclass(frozen=True)
class StructuredRecurrentPolicy:
    """NumPy inference for the principal ``n=64, k=2`` student."""

    parameters: Mapping[str, NDArray[np.float32]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, shape in STRUCTURED_PARAMETER_SHAPES.items():
            if name not in self.parameters:
                raise ValueError(f"missing structured parameter {name}")
            value = np.asarray(self.parameters[name])
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite float32 values")
        for key, expected in _CHECKPOINT_ARCHITECTURE.items():
            if key in self.metadata and self.metadata[key] != expected:
                raise ValueError(
                    f"checkpoint metadata {key} must be {expected!r}, "
                    f"got {self.metadata[key]!r}"
                )

    @classmethod
    def load(cls, path: str | Path) -> StructuredRecurrentPolicy:
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata_json"].item()))
            parameters = {
                name: np.asarray(checkpoint[name], dtype=np.float32)
                for name in STRUCTURED_PARAMETER_SHAPES
            }
        for key, expected in _CHECKPOINT_ARCHITECTURE.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"checkpoint metadata {key} must be {expected!r}, "
                    f"got {metadata.get(key)!r}"
                )
        return cls(parameters=parameters, metadata=metadata)

    @property
    def parameter_count(self) -> int:
        return sum(
            int(np.prod(shape)) for shape in STRUCTURED_PARAMETER_SHAPES.values()
        )

    @property
    def recurrent_parameter_count(self) -> int:
        return sum(
            int(np.prod(STRUCTURED_PARAMETER_SHAPES[name]))
            for name in RECURRENT_PARAMETER_NAMES
        )

    @property
    def diagonal_dynamics(self) -> NDArray[np.float32]:
        """Return the stable diagonal entries ``tanh(alpha)``."""

        return np.tanh(self.parameters["recurrence_alpha"]).astype(np.float32)

    def initial_carry(self, batch_size: int | None = None) -> StructuredPolicyCarry:
        """Create the only carry reset used at an episode boundary."""

        if batch_size is None:
            memory_shape = (STATE_DIM,)
            action_shape = (PUBLIC_ACTION_DIM,)
        else:
            if batch_size <= 0:
                raise ValueError("batch size must be positive")
            memory_shape = (batch_size, STATE_DIM)
            action_shape = (batch_size, PUBLIC_ACTION_DIM)
        return StructuredPolicyCarry(
            memory=np.zeros(memory_shape, dtype=np.float32),
            previous_action=np.zeros(action_shape, dtype=np.float32),
        )

    def act(
        self,
        observation: ArrayLike,
        carry: StructuredPolicyCarry,
    ) -> tuple[NDArray[np.float32], StructuredPolicyCarry]:
        """Advance a closed-loop carry and return the deterministic action mean."""

        action, memory = self.step(
            observation,
            carry.previous_action,
            carry.memory,
        )
        return action, StructuredPolicyCarry(
            memory=np.asarray(memory, dtype=np.float32),
            previous_action=np.asarray(action, dtype=np.float32),
        )

    def step(
        self,
        observation: ArrayLike,
        previous_action: ArrayLike,
        previous_state: ArrayLike,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Apply one recurrent update with an explicit previous action and state."""

        observations, single = _feature_batch(
            "observation", observation, PUBLIC_OBSERVATION_DIM
        )
        batch_size = observations.shape[0]
        actions = _matching_batch(
            "previous action",
            previous_action,
            PUBLIC_ACTION_DIM,
            batch_size,
            single,
        )
        states = _matching_batch(
            "previous state", previous_state, STATE_DIM, batch_size, single
        )
        encoded = self._encode_batch(observations)
        next_state = self._update_batch(encoded, actions, states)
        action_input = np.concatenate((next_state, encoded), axis=1)
        action_hidden = _silu(
            action_input @ self.parameters["action_hidden_weight"].T
            + self.parameters["action_hidden_bias"]
        )
        action = np.tanh(
            action_hidden @ self.parameters["action_output_weight"].T
            + self.parameters["action_output_bias"]
        ).astype(np.float32)
        if single:
            return action[0], next_state[0]
        return action, next_state

    def teacher_forced_sequence(
        self,
        observations: ArrayLike,
        previous_actions: ArrayLike,
        initial_state: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Evaluate one contiguous episode sequence without implicit resets."""

        observation_array = np.asarray(observations, dtype=np.float32)
        action_array = np.asarray(previous_actions, dtype=np.float32)
        if (
            observation_array.ndim != 2
            or observation_array.shape[1] != PUBLIC_OBSERVATION_DIM
            or not len(observation_array)
        ):
            raise ValueError("observations must have non-empty shape (time, 19)")
        if action_array.shape != (len(observation_array), PUBLIC_ACTION_DIM):
            raise ValueError("previous actions must have shape (time, 2)")
        if not np.all(np.isfinite(observation_array)) or not np.all(
            np.isfinite(action_array)
        ):
            raise ValueError("sequence inputs must contain only finite values")
        state = (
            np.zeros(STATE_DIM, dtype=np.float32)
            if initial_state is None
            else _single_feature("initial state", initial_state, STATE_DIM)
        )
        predicted_actions = []
        states = []
        for observation, previous_action in zip(
            observation_array, action_array, strict=True
        ):
            action, state = self.step(observation, previous_action, state)
            predicted_actions.append(action)
            states.append(state)
        return (
            np.stack(predicted_actions).astype(np.float32),
            np.stack(states).astype(np.float32),
        )

    def recurrent_jacobian(
        self,
        observation: ArrayLike,
        previous_action: ArrayLike,
        previous_state: ArrayLike,
    ) -> NDArray[np.float32]:
        """Return ``dz_t/dz_(t-1)`` for one transition."""

        observations, single = _feature_batch(
            "observation", observation, PUBLIC_OBSERVATION_DIM
        )
        if not single:
            raise ValueError("recurrent Jacobian requires one observation")
        action = _single_feature(
            "previous action", previous_action, PUBLIC_ACTION_DIM
        )
        state = _single_feature("previous state", previous_state, STATE_DIM)
        encoded = self._encode_batch(observations)[0]
        preactivation = (
            self.parameters["innovation_state_weight"] @ state
            + self.parameters["innovation_input_weight"] @ encoded
            + self.parameters["innovation_action_weight"] @ action
            + self.parameters["innovation_bias"]
        )
        derivative = 1.0 - np.tanh(preactivation) ** 2
        innovation_jacobian = (
            self.parameters["innovation_output_weight"]
            @ np.diag(derivative)
            @ self.parameters["innovation_state_weight"]
        )
        return (
            np.diag(self.diagonal_dynamics) + innovation_jacobian
        ).astype(np.float32)

    def _encode_batch(
        self, observations: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        hidden = _silu(
            observations @ self.parameters["encoder_0_weight"].T
            + self.parameters["encoder_0_bias"]
        )
        return _silu(
            hidden @ self.parameters["encoder_1_weight"].T
            + self.parameters["encoder_1_bias"]
        ).astype(np.float32)

    def _update_batch(
        self,
        encoded: NDArray[np.float32],
        previous_actions: NDArray[np.float32],
        previous_states: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        linear = (
            previous_states * self.diagonal_dynamics
            + encoded @ self.parameters["recurrence_input_weight"].T
            + previous_actions @ self.parameters["recurrence_action_weight"].T
            + self.parameters["recurrence_bias"]
        )
        innovation = np.tanh(
            previous_states @ self.parameters["innovation_state_weight"].T
            + encoded @ self.parameters["innovation_input_weight"].T
            + previous_actions @ self.parameters["innovation_action_weight"].T
            + self.parameters["innovation_bias"]
        )
        return (
            linear
            + innovation @ self.parameters["innovation_output_weight"].T
        ).astype(np.float32)


def initialise_structured_parameters(
    seed: int,
    *,
    minimum_time_constant_ms: float = 40.0,
    maximum_time_constant_ms: float = 2000.0,
    control_period_ms: float = CONTROL_PERIOD_MS,
) -> dict[str, NDArray[np.float32]]:
    """Initialise the fixed ``n=64, k=2`` architecture reproducibly."""

    if minimum_time_constant_ms <= 0:
        raise ValueError("minimum time constant must be positive")
    if maximum_time_constant_ms < minimum_time_constant_ms:
        raise ValueError("maximum time constant must not be smaller than minimum")
    if control_period_ms <= 0:
        raise ValueError("control period must be positive")
    rng = np.random.default_rng(seed)

    def weight(output_width: int, input_width: int) -> NDArray[np.float32]:
        standard_deviation = np.sqrt(2.0 / (input_width + output_width))
        return rng.normal(
            0.0, standard_deviation, (output_width, input_width)
        ).astype(np.float32)

    time_constants = np.geomspace(
        minimum_time_constant_ms, maximum_time_constant_ms, STATE_DIM
    )
    diagonal = np.exp(-control_period_ms / time_constants)
    alpha = np.arctanh(diagonal).astype(np.float32)
    return {
        "encoder_0_weight": weight(64, PUBLIC_OBSERVATION_DIM),
        "encoder_0_bias": np.zeros(64, dtype=np.float32),
        "encoder_1_weight": weight(ENCODED_DIM, 64),
        "encoder_1_bias": np.zeros(ENCODED_DIM, dtype=np.float32),
        "recurrence_alpha": alpha,
        "recurrence_input_weight": weight(STATE_DIM, ENCODED_DIM),
        "recurrence_action_weight": weight(STATE_DIM, PUBLIC_ACTION_DIM),
        "recurrence_bias": np.zeros(STATE_DIM, dtype=np.float32),
        "innovation_output_weight": weight(STATE_DIM, INNOVATION_RANK),
        "innovation_state_weight": weight(INNOVATION_RANK, STATE_DIM),
        "innovation_input_weight": weight(INNOVATION_RANK, ENCODED_DIM),
        "innovation_action_weight": weight(INNOVATION_RANK, PUBLIC_ACTION_DIM),
        "innovation_bias": np.zeros(INNOVATION_RANK, dtype=np.float32),
        "action_hidden_weight": weight(64, STATE_DIM + ENCODED_DIM),
        "action_hidden_bias": np.zeros(64, dtype=np.float32),
        "action_output_weight": weight(PUBLIC_ACTION_DIM, 64),
        "action_output_bias": np.zeros(PUBLIC_ACTION_DIM, dtype=np.float32),
    }


def save_structured_checkpoint(
    path: str | Path,
    parameters: Mapping[str, ArrayLike],
    metadata: Mapping[str, Any],
) -> None:
    """Save a self-describing, framework-neutral structured checkpoint."""

    checkpoint_metadata = dict(metadata)
    for key, expected in _CHECKPOINT_ARCHITECTURE.items():
        if key in checkpoint_metadata and checkpoint_metadata[key] != expected:
            raise ValueError(f"metadata {key} conflicts with the architecture")
        checkpoint_metadata[key] = expected
    checked = StructuredRecurrentPolicy(
        parameters={
            name: np.asarray(value, dtype=np.float32)
            for name, value in parameters.items()
        },
        metadata=checkpoint_metadata,
    )
    payload: dict[str, Any] = {
        name: checked.parameters[name] for name in STRUCTURED_PARAMETER_SHAPES
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(checkpoint_metadata, sort_keys=True)
    )
    np.savez(Path(path), **payload)


def _feature_batch(
    name: str, value: ArrayLike, width: int
) -> tuple[NDArray[np.float32], bool]:
    array = np.asarray(value, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (batch, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array, single


def _matching_batch(
    name: str,
    value: ArrayLike,
    width: int,
    batch_size: int,
    expect_single: bool,
) -> NDArray[np.float32]:
    array, single = _feature_batch(name, value, width)
    if single != expect_single or array.shape[0] != batch_size:
        raise ValueError(f"{name} batch shape must match the observation")
    return array


def _single_feature(name: str, value: ArrayLike, width: int) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _silu(value: NDArray[np.float32]) -> NDArray[np.float32]:
    exponent = np.clip(-value, -60.0, 60.0)
    return value / (1.0 + np.exp(exponent))
