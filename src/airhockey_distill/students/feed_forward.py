"""Small observation-only student used for the Stage B memory check."""

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

PARAMETER_SHAPES = {
    "encoder_0_weight": (64, PUBLIC_OBSERVATION_DIM),
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
    "policy": "feed_forward",
    "observation_dimension": PUBLIC_OBSERVATION_DIM,
    "action_dimension": PUBLIC_ACTION_DIM,
    "include_previous_action": False,
    "action_output": "tanh_mean",
}


@dataclass(frozen=True)
class FeedForwardPolicy:
    """NumPy inference for the frozen two-layer encoder and action head."""

    parameters: Mapping[str, NDArray[np.float32]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, shape in PARAMETER_SHAPES.items():
            if name not in self.parameters:
                raise ValueError(f"missing feed-forward parameter {name}")
            value = np.asarray(self.parameters[name])
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite float32 values")
        if self.metadata.get("policy") == "feed_forward":
            for key, expected in _CHECKPOINT_ARCHITECTURE.items():
                if self.metadata.get(key) != expected:
                    raise ValueError(
                        f"checkpoint metadata {key} must be {expected!r}, "
                        f"got {self.metadata.get(key)!r}"
                    )

    @classmethod
    def load(cls, path: str | Path) -> FeedForwardPolicy:
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata_json"].item()))
            parameters = {
                name: np.asarray(checkpoint[name], dtype=np.float32)
                for name in PARAMETER_SHAPES
            }
        return cls(parameters=parameters, metadata=metadata)

    @property
    def parameter_count(self) -> int:
        return sum(int(np.prod(shape)) for shape in PARAMETER_SHAPES.values())

    def action(self, observation: ArrayLike) -> NDArray[np.float32]:
        value = np.asarray(observation, dtype=np.float32)
        single = value.ndim == 1
        if single:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != PUBLIC_OBSERVATION_DIM:
            raise ValueError("observation must have shape (19,) or (batch, 19)")
        if not np.all(np.isfinite(value)):
            raise ValueError("observation must contain only finite values")

        hidden = _silu(
            value @ self.parameters["encoder_0_weight"].T
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
        return action[0] if single else action


def save_feed_forward_checkpoint(
    path: str | Path,
    parameters: Mapping[str, ArrayLike],
    metadata: Mapping[str, Any],
) -> None:
    checked = FeedForwardPolicy(
        parameters={
            name: np.asarray(value, dtype=np.float32)
            for name, value in parameters.items()
        },
        metadata=dict(metadata),
    )
    payload: dict[str, Any] = {
        name: checked.parameters[name] for name in PARAMETER_SHAPES
    }
    payload["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    np.savez(Path(path), **payload)


def initialise_feed_forward_parameters(
    seed: int,
) -> dict[str, NDArray[np.float32]]:
    """Initialise the principal observation-only network reproducibly."""

    rng = np.random.default_rng(seed)

    def weight(output_width: int, input_width: int) -> NDArray[np.float32]:
        standard_deviation = np.sqrt(2.0 / (input_width + output_width))
        return rng.normal(
            0.0,
            standard_deviation,
            (output_width, input_width),
        ).astype(np.float32)

    return {
        "encoder_0_weight": weight(64, PUBLIC_OBSERVATION_DIM),
        "encoder_0_bias": np.zeros(64, dtype=np.float32),
        "encoder_1_weight": weight(32, 64),
        "encoder_1_bias": np.zeros(32, dtype=np.float32),
        "action_hidden_weight": weight(64, 32),
        "action_hidden_bias": np.zeros(64, dtype=np.float32),
        "action_output_weight": weight(PUBLIC_ACTION_DIM, 64),
        "action_output_bias": np.zeros(PUBLIC_ACTION_DIM, dtype=np.float32),
    }


def save_principal_feed_forward_checkpoint(
    path: str | Path,
    parameters: Mapping[str, ArrayLike],
    metadata: Mapping[str, Any],
) -> None:
    """Save a self-describing principal feed-forward checkpoint."""

    checkpoint_metadata = dict(metadata)
    for key, expected in _CHECKPOINT_ARCHITECTURE.items():
        if key in checkpoint_metadata and checkpoint_metadata[key] != expected:
            raise ValueError(f"metadata {key} conflicts with the architecture")
        checkpoint_metadata[key] = expected
    save_feed_forward_checkpoint(path, parameters, checkpoint_metadata)


def _silu(value: NDArray[np.float32]) -> NDArray[np.float32]:
    exponent = np.clip(-value, -60.0, 60.0)
    return value / (1.0 + np.exp(exponent))
