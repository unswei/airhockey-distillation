"""Trainable PyTorch form of the principal feed-forward student."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.nn import functional

from airhockey_distill.envs.policy_interface import PUBLIC_OBSERVATION_DIM
from airhockey_distill.students.feed_forward import (
    PARAMETER_SHAPES,
    initialise_feed_forward_parameters,
)


class FeedForwardModule(nn.Module):
    """Differentiable 19--64--32--64--2 observation-only policy."""

    def __init__(
        self,
        *,
        seed: int = 0,
        parameters: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__()
        initial = (
            initialise_feed_forward_parameters(seed)
            if parameters is None
            else {
                name: np.asarray(value, dtype=np.float32)
                for name, value in parameters.items()
            }
        )
        for name, shape in PARAMETER_SHAPES.items():
            if name not in initial:
                raise ValueError(f"missing feed-forward parameter {name}")
            value = initial[name]
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid feed-forward parameter {name}")
            self.register_parameter(
                name,
                nn.Parameter(torch.from_numpy(np.array(value, copy=True))),
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, observation: Tensor) -> Tensor:
        if observation.shape[-1] != PUBLIC_OBSERVATION_DIM:
            raise ValueError("observation must end in dimension 19")
        hidden = functional.silu(
            functional.linear(
                observation,
                self.encoder_0_weight,
                self.encoder_0_bias,
            )
        )
        encoded = functional.silu(
            functional.linear(hidden, self.encoder_1_weight, self.encoder_1_bias)
        )
        action_hidden = functional.silu(
            functional.linear(
                encoded,
                self.action_hidden_weight,
                self.action_hidden_bias,
            )
        )
        return torch.tanh(
            functional.linear(
                action_hidden,
                self.action_output_weight,
                self.action_output_bias,
            )
        )

    def export_numpy_parameters(self) -> dict[str, NDArray[np.float32]]:
        return {
            name: parameter.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, parameter in self.named_parameters()
        }
