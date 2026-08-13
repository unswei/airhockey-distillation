"""Framework-neutral NumPy inference for the ten-step finite-stack student."""

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
    PUCK_POSITION_XY_SLICE,
)

FINITE_STACK_HISTORY_STEPS = 10
FINITE_STACK_PUCK_FEATURE_DIM = 3
FINITE_STACK_PROPRIOCEPTION_DIM = int(PUCK_POSITION_XY_SLICE.start)
FINITE_STACK_HISTORY_DIM = (
    FINITE_STACK_HISTORY_STEPS * FINITE_STACK_PUCK_FEATURE_DIM
)
FINITE_STACK_INPUT_DIM = (
    FINITE_STACK_PROPRIOCEPTION_DIM + FINITE_STACK_HISTORY_DIM
)

FINITE_STACK_PARAMETER_SHAPES = {
    "encoder_0_weight": (64, FINITE_STACK_INPUT_DIM),
    "encoder_0_bias": (64,),
    "encoder_1_weight": (32, 64),
    "encoder_1_bias": (32,),
    "action_hidden_weight": (64, 32),
    "action_hidden_bias": (64,),
    "action_output_weight": (PUBLIC_ACTION_DIM, 64),
    "action_output_bias": (PUBLIC_ACTION_DIM,),
}

_CHECKPOINT_ARCHITECTURE = {
    "schema_version": 1,
    "policy": "finite_stack_10",
    "observation_dimension": PUBLIC_OBSERVATION_DIM,
    "current_proprioception_dimension": FINITE_STACK_PROPRIOCEPTION_DIM,
    "puck_history_steps": FINITE_STACK_HISTORY_STEPS,
    "puck_feature_dimension": FINITE_STACK_PUCK_FEATURE_DIM,
    "history_dimension": FINITE_STACK_HISTORY_DIM,
    "network_input_dimension": FINITE_STACK_INPUT_DIM,
    "history_order": "oldest_to_newest",
    "episode_start_padding": "zero_position_and_zero_visibility",
    "include_previous_action": False,
    "action_dimension": PUBLIC_ACTION_DIM,
    "action_output": "tanh_mean",
}


@dataclass(frozen=True)
class FiniteStackCarry:
    """Ten masked puck triples carried between control steps."""

    puck_history: NDArray[np.float32]


@dataclass(frozen=True)
class FiniteStackPolicy:
    """NumPy inference for the predeclared ten-step finite-stack student."""

    parameters: Mapping[str, NDArray[np.float32]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, shape in FINITE_STACK_PARAMETER_SHAPES.items():
            if name not in self.parameters:
                raise ValueError(f"missing finite-stack parameter {name}")
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
    def load(cls, path: str | Path) -> FiniteStackPolicy:
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata_json"].item()))
            parameters = {
                name: np.asarray(checkpoint[name], dtype=np.float32)
                for name in FINITE_STACK_PARAMETER_SHAPES
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
            int(np.prod(shape))
            for shape in FINITE_STACK_PARAMETER_SHAPES.values()
        )

    def initial_carry(self, batch_size: int | None = None) -> FiniteStackCarry:
        """Return zero-position, zero-visibility episode-start padding."""

        if batch_size is None:
            shape = (FINITE_STACK_HISTORY_DIM,)
        else:
            if batch_size <= 0:
                raise ValueError("batch size must be positive")
            shape = (batch_size, FINITE_STACK_HISTORY_DIM)
        return FiniteStackCarry(puck_history=np.zeros(shape, dtype=np.float32))

    def act(
        self,
        observation: ArrayLike,
        carry: FiniteStackCarry,
    ) -> tuple[NDArray[np.float32], FiniteStackCarry]:
        """Append the current puck triple and emit the deterministic mean."""

        action, history = self.step(observation, carry.puck_history)
        return action, FiniteStackCarry(
            puck_history=np.asarray(history, dtype=np.float32)
        )

    def step(
        self,
        observation: ArrayLike,
        previous_history: ArrayLike,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Advance one stack step with explicit previous history."""

        policy_input, history, single = self.prepare_input(
            observation,
            previous_history,
        )
        hidden = _silu(
            policy_input @ self.parameters["encoder_0_weight"].T
            + self.parameters["encoder_0_bias"]
        )
        encoded = _silu(
            hidden @ self.parameters["encoder_1_weight"].T
            + self.parameters["encoder_1_bias"]
        )
        action_hidden = _silu(
            encoded @ self.parameters["action_hidden_weight"].T
            + self.parameters["action_hidden_bias"]
        )
        action = np.tanh(
            action_hidden @ self.parameters["action_output_weight"].T
            + self.parameters["action_output_bias"]
        ).astype(np.float32)
        if single:
            return action[0], history[0]
        return action, history

    def prepare_input(
        self,
        observation: ArrayLike,
        previous_history: ArrayLike,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], bool]:
        """Build the 46-value input and expose the updated history for auditing."""

        observations, single = _feature_batch(
            "observation",
            observation,
            PUBLIC_OBSERVATION_DIM,
        )
        histories = _matching_batch(
            "previous history",
            previous_history,
            FINITE_STACK_HISTORY_DIM,
            observations.shape[0],
            single,
        )
        current_puck = observations[:, FINITE_STACK_PROPRIOCEPTION_DIM:]
        history = np.concatenate(
            (histories[:, FINITE_STACK_PUCK_FEATURE_DIM:], current_puck),
            axis=1,
        ).astype(np.float32)
        policy_input = np.concatenate(
            (observations[:, :FINITE_STACK_PROPRIOCEPTION_DIM], history),
            axis=1,
        ).astype(np.float32)
        return policy_input, history, single

    def sequence(
        self,
        observations: ArrayLike,
        initial_history: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Evaluate one contiguous sequence without implicit history resets."""

        observation_array = np.asarray(observations, dtype=np.float32)
        if (
            observation_array.ndim != 2
            or observation_array.shape[1] != PUBLIC_OBSERVATION_DIM
            or not len(observation_array)
        ):
            raise ValueError("observations must have non-empty shape (time, 19)")
        if not np.all(np.isfinite(observation_array)):
            raise ValueError("observations must contain only finite values")
        history = (
            np.zeros(FINITE_STACK_HISTORY_DIM, dtype=np.float32)
            if initial_history is None
            else _single_feature(
                "initial history",
                initial_history,
                FINITE_STACK_HISTORY_DIM,
            )
        )
        actions = []
        histories = []
        for observation in observation_array:
            action, history = self.step(observation, history)
            actions.append(action)
            histories.append(history)
        return (
            np.stack(actions).astype(np.float32),
            np.stack(histories).astype(np.float32),
        )


def initialise_finite_stack_parameters(
    seed: int,
) -> dict[str, NDArray[np.float32]]:
    """Initialise the fixed ten-step stack network reproducibly."""

    rng = np.random.default_rng(seed)

    def weight(output_width: int, input_width: int) -> NDArray[np.float32]:
        standard_deviation = np.sqrt(2.0 / (input_width + output_width))
        return rng.normal(
            0.0,
            standard_deviation,
            (output_width, input_width),
        ).astype(np.float32)

    return {
        "encoder_0_weight": weight(64, FINITE_STACK_INPUT_DIM),
        "encoder_0_bias": np.zeros(64, dtype=np.float32),
        "encoder_1_weight": weight(32, 64),
        "encoder_1_bias": np.zeros(32, dtype=np.float32),
        "action_hidden_weight": weight(64, 32),
        "action_hidden_bias": np.zeros(64, dtype=np.float32),
        "action_output_weight": weight(PUBLIC_ACTION_DIM, 64),
        "action_output_bias": np.zeros(PUBLIC_ACTION_DIM, dtype=np.float32),
    }


def save_finite_stack_checkpoint(
    path: str | Path,
    parameters: Mapping[str, ArrayLike],
    metadata: Mapping[str, Any],
) -> None:
    """Save a self-describing, framework-neutral finite-stack checkpoint."""

    checkpoint_metadata = dict(metadata)
    for key, expected in _CHECKPOINT_ARCHITECTURE.items():
        if key in checkpoint_metadata and checkpoint_metadata[key] != expected:
            raise ValueError(f"metadata {key} conflicts with the architecture")
        checkpoint_metadata[key] = expected
    checked = FiniteStackPolicy(
        parameters={
            name: np.asarray(value, dtype=np.float32)
            for name, value in parameters.items()
        },
        metadata=checkpoint_metadata,
    )
    payload: dict[str, Any] = {
        name: checked.parameters[name] for name in FINITE_STACK_PARAMETER_SHAPES
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(checkpoint_metadata, sort_keys=True)
    )
    np.savez(Path(path), **payload)


def _feature_batch(
    name: str,
    value: ArrayLike,
    width: int,
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


def _single_feature(
    name: str,
    value: ArrayLike,
    width: int,
) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _silu(value: NDArray[np.float32]) -> NDArray[np.float32]:
    exponent = np.clip(-value, -60.0, 60.0)
    return (value / (1.0 + np.exp(exponent))).astype(np.float32)
