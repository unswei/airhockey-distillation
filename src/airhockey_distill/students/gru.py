"""Framework-neutral NumPy inference for the matched GRU-64 student."""

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

GRU_ENCODED_DIM = 32
GRU_STATE_DIM = 64
GRU_INPUT_DIM = GRU_ENCODED_DIM + PUBLIC_ACTION_DIM

GRU_PARAMETER_SHAPES = {
    "encoder_0_weight": (64, PUBLIC_OBSERVATION_DIM),
    "encoder_0_bias": (64,),
    "encoder_1_weight": (GRU_ENCODED_DIM, 64),
    "encoder_1_bias": (GRU_ENCODED_DIM,),
    "gru_input_weight": (3 * GRU_STATE_DIM, GRU_INPUT_DIM),
    "gru_hidden_weight": (3 * GRU_STATE_DIM, GRU_STATE_DIM),
    "gru_input_bias": (3 * GRU_STATE_DIM,),
    "gru_hidden_bias": (3 * GRU_STATE_DIM,),
    "action_hidden_weight": (64, GRU_STATE_DIM + GRU_ENCODED_DIM),
    "action_hidden_bias": (64,),
    "action_output_weight": (PUBLIC_ACTION_DIM, 64),
    "action_output_bias": (PUBLIC_ACTION_DIM,),
}

GRU_RECURRENT_PARAMETER_NAMES = frozenset(
    {
        "gru_input_weight",
        "gru_hidden_weight",
        "gru_input_bias",
        "gru_hidden_bias",
    }
)

_CHECKPOINT_ARCHITECTURE = {
    "schema_version": 1,
    "policy": "gru_recurrent_n64",
    "observation_dimension": PUBLIC_OBSERVATION_DIM,
    "encoded_dimension": GRU_ENCODED_DIM,
    "state_dimension": GRU_STATE_DIM,
    "action_dimension": PUBLIC_ACTION_DIM,
    "include_previous_action": True,
    "action_output": "tanh_mean",
}


@dataclass(frozen=True)
class GRUPolicyCarry:
    """GRU memory and previous requested command carried between steps."""

    memory: NDArray[np.float32]
    previous_action: NDArray[np.float32]


@dataclass(frozen=True)
class GRURecurrentPolicy:
    """NumPy inference for the matched hidden-size-64 GRU student."""

    parameters: Mapping[str, NDArray[np.float32]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, shape in GRU_PARAMETER_SHAPES.items():
            if name not in self.parameters:
                raise ValueError(f"missing GRU parameter {name}")
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
    def load(cls, path: str | Path) -> GRURecurrentPolicy:
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata_json"].item()))
            parameters = {
                name: np.asarray(checkpoint[name], dtype=np.float32)
                for name in GRU_PARAMETER_SHAPES
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
        return sum(int(np.prod(shape)) for shape in GRU_PARAMETER_SHAPES.values())

    @property
    def recurrent_parameter_count(self) -> int:
        return sum(
            int(np.prod(GRU_PARAMETER_SHAPES[name]))
            for name in GRU_RECURRENT_PARAMETER_NAMES
        )

    def initial_carry(self, batch_size: int | None = None) -> GRUPolicyCarry:
        """Create the only carry reset used at an episode boundary."""

        if batch_size is None:
            memory_shape = (GRU_STATE_DIM,)
            action_shape = (PUBLIC_ACTION_DIM,)
        else:
            if batch_size <= 0:
                raise ValueError("batch size must be positive")
            memory_shape = (batch_size, GRU_STATE_DIM)
            action_shape = (batch_size, PUBLIC_ACTION_DIM)
        return GRUPolicyCarry(
            memory=np.zeros(memory_shape, dtype=np.float32),
            previous_action=np.zeros(action_shape, dtype=np.float32),
        )

    def act(
        self,
        observation: ArrayLike,
        carry: GRUPolicyCarry,
    ) -> tuple[NDArray[np.float32], GRUPolicyCarry]:
        """Advance the closed-loop carry and return the deterministic mean."""

        action, memory = self.step(
            observation,
            carry.previous_action,
            carry.memory,
        )
        return action, GRUPolicyCarry(
            memory=np.asarray(memory, dtype=np.float32),
            previous_action=np.asarray(action, dtype=np.float32),
        )

    def step(
        self,
        observation: ArrayLike,
        previous_action: ArrayLike,
        previous_state: ArrayLike,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Apply one standard PyTorch-compatible GRUCell update."""

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
            "previous state", previous_state, GRU_STATE_DIM, batch_size, single
        )
        encoded = self._encode_batch(observations)
        next_state = self._update_batch(encoded, actions, states)
        action_hidden = _silu(
            np.concatenate((next_state, encoded), axis=1)
            @ self.parameters["action_hidden_weight"].T
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
        """Evaluate one contiguous sequence without implicit carry resets."""

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
            np.zeros(GRU_STATE_DIM, dtype=np.float32)
            if initial_state is None
            else _single_feature("initial state", initial_state, GRU_STATE_DIM)
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
        recurrent_input = np.concatenate((encoded, previous_actions), axis=1)
        input_gates = (
            recurrent_input @ self.parameters["gru_input_weight"].T
            + self.parameters["gru_input_bias"]
        )
        hidden_gates = (
            previous_states @ self.parameters["gru_hidden_weight"].T
            + self.parameters["gru_hidden_bias"]
        )
        input_reset, input_update, input_new = np.split(input_gates, 3, axis=1)
        hidden_reset, hidden_update, hidden_new = np.split(hidden_gates, 3, axis=1)
        reset = _sigmoid(input_reset + hidden_reset)
        update = _sigmoid(input_update + hidden_update)
        candidate = np.tanh(input_new + reset * hidden_new)
        return (candidate * (1.0 - update) + previous_states * update).astype(
            np.float32
        )


def initialise_gru_parameters(seed: int) -> dict[str, NDArray[np.float32]]:
    """Initialise the matched GRU-64 architecture reproducibly."""

    rng = np.random.default_rng(seed)

    def weight(output_width: int, input_width: int) -> NDArray[np.float32]:
        standard_deviation = np.sqrt(2.0 / (input_width + output_width))
        return rng.normal(
            0.0, standard_deviation, (output_width, input_width)
        ).astype(np.float32)

    return {
        "encoder_0_weight": weight(64, PUBLIC_OBSERVATION_DIM),
        "encoder_0_bias": np.zeros(64, dtype=np.float32),
        "encoder_1_weight": weight(GRU_ENCODED_DIM, 64),
        "encoder_1_bias": np.zeros(GRU_ENCODED_DIM, dtype=np.float32),
        "gru_input_weight": weight(3 * GRU_STATE_DIM, GRU_INPUT_DIM),
        "gru_hidden_weight": weight(3 * GRU_STATE_DIM, GRU_STATE_DIM),
        "gru_input_bias": np.zeros(3 * GRU_STATE_DIM, dtype=np.float32),
        "gru_hidden_bias": np.zeros(3 * GRU_STATE_DIM, dtype=np.float32),
        "action_hidden_weight": weight(64, GRU_STATE_DIM + GRU_ENCODED_DIM),
        "action_hidden_bias": np.zeros(64, dtype=np.float32),
        "action_output_weight": weight(PUBLIC_ACTION_DIM, 64),
        "action_output_bias": np.zeros(PUBLIC_ACTION_DIM, dtype=np.float32),
    }


def save_gru_checkpoint(
    path: str | Path,
    parameters: Mapping[str, ArrayLike],
    metadata: Mapping[str, Any],
) -> None:
    """Save a self-describing, framework-neutral GRU checkpoint."""

    checkpoint_metadata = dict(metadata)
    for key, expected in _CHECKPOINT_ARCHITECTURE.items():
        if key in checkpoint_metadata and checkpoint_metadata[key] != expected:
            raise ValueError(f"metadata {key} conflicts with the architecture")
        checkpoint_metadata[key] = expected
    checked = GRURecurrentPolicy(
        parameters={
            name: np.asarray(value, dtype=np.float32)
            for name, value in parameters.items()
        },
        metadata=checkpoint_metadata,
    )
    payload: dict[str, Any] = {
        name: checked.parameters[name] for name in GRU_PARAMETER_SHAPES
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


def _sigmoid(value: NDArray[np.float32]) -> NDArray[np.float32]:
    exponent = np.clip(-value, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(exponent))).astype(np.float32)


def _silu(value: NDArray[np.float32]) -> NDArray[np.float32]:
    return (value * _sigmoid(value)).astype(np.float32)
