"""Trainable PyTorch form of the ten-step finite-stack student."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.nn import functional

from airhockey_distill.envs.policy_interface import (
    PUBLIC_ACTION_DIM,
    PUBLIC_OBSERVATION_DIM,
)
from airhockey_distill.students.finite_stack import (
    FINITE_STACK_HISTORY_DIM,
    FINITE_STACK_PARAMETER_SHAPES,
    FINITE_STACK_PROPRIOCEPTION_DIM,
    FINITE_STACK_PUCK_FEATURE_DIM,
    initialise_finite_stack_parameters,
)


class FiniteStackModule(nn.Module):
    """Differentiable stack policy with explicit puck-history carry."""

    def __init__(
        self,
        *,
        seed: int = 0,
        parameters: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__()
        initial = (
            initialise_finite_stack_parameters(seed)
            if parameters is None
            else {
                name: np.asarray(value, dtype=np.float32)
                for name, value in parameters.items()
            }
        )
        for name, shape in FINITE_STACK_PARAMETER_SHAPES.items():
            if name not in initial:
                raise ValueError(f"missing finite-stack parameter {name}")
            value = initial[name]
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid finite-stack parameter {name}")
            self.register_parameter(
                name,
                nn.Parameter(torch.from_numpy(np.array(value, copy=True))),
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_history(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        reference = self.encoder_0_weight
        return torch.zeros(
            (batch_size, FINITE_STACK_HISTORY_DIM),
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def forward_step(
        self,
        observation: Tensor,
        previous_history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if observation.shape[-1] != PUBLIC_OBSERVATION_DIM:
            raise ValueError("observation must end in dimension 19")
        if previous_history.shape[:-1] != observation.shape[:-1] or (
            previous_history.shape[-1] != FINITE_STACK_HISTORY_DIM
        ):
            raise ValueError("previous history shape must match the observation")

        current_puck = observation[..., FINITE_STACK_PROPRIOCEPTION_DIM:]
        history = torch.cat(
            (
                previous_history[..., FINITE_STACK_PUCK_FEATURE_DIM:],
                current_puck,
            ),
            dim=-1,
        )
        policy_input = torch.cat(
            (observation[..., :FINITE_STACK_PROPRIOCEPTION_DIM], history),
            dim=-1,
        )
        hidden = functional.silu(
            functional.linear(
                policy_input,
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
        action = torch.tanh(
            functional.linear(
                action_hidden,
                self.action_output_weight,
                self.action_output_bias,
            )
        )
        return action, history

    def forward_sequence(
        self,
        observations: Tensor,
        initial_history: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Unroll complete histories for supervised sequence training."""

        if observations.ndim != 3 or observations.shape[-1] != PUBLIC_OBSERVATION_DIM:
            raise ValueError("observations must have shape (batch, time, 19)")
        history = (
            self.initial_history(
                observations.shape[0],
                device=observations.device,
                dtype=observations.dtype,
            )
            if initial_history is None
            else initial_history
        )
        if history.shape != (observations.shape[0], FINITE_STACK_HISTORY_DIM):
            raise ValueError("initial history must have shape (batch, 30)")
        actions = []
        histories = []
        for time_index in range(observations.shape[1]):
            action, history = self.forward_step(
                observations[:, time_index],
                history,
            )
            actions.append(action)
            histories.append(history)
        return torch.stack(actions, dim=1), torch.stack(histories, dim=1)

    def export_numpy_parameters(self) -> dict[str, NDArray[np.float32]]:
        """Export weights for ``FiniteStackPolicy`` evaluation."""

        return {
            name: parameter.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, parameter in self.named_parameters()
        }
