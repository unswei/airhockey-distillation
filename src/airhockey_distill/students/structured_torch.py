"""Trainable PyTorch form of the fixed structured recurrent student."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn
from torch.nn import functional as functional

from airhockey_distill.envs.policy_interface import (
    PUBLIC_ACTION_DIM,
    PUBLIC_OBSERVATION_DIM,
)
from airhockey_distill.students.structured import (
    ENCODED_DIM,
    STATE_DIM,
    STRUCTURED_PARAMETER_SHAPES,
    initialise_structured_parameters,
)


class StructuredRecurrentModule(nn.Module):
    """Differentiable ``n=64, k=2`` policy with explicit sequence state."""

    def __init__(
        self,
        *,
        seed: int = 0,
        parameters: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__()
        initial = (
            initialise_structured_parameters(seed)
            if parameters is None
            else {
                name: np.asarray(value, dtype=np.float32)
                for name, value in parameters.items()
            }
        )
        for name, shape in STRUCTURED_PARAMETER_SHAPES.items():
            if name not in initial:
                raise ValueError(f"missing structured parameter {name}")
            value = initial[name]
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid structured parameter {name}")
            self.register_parameter(
                name,
                nn.Parameter(torch.from_numpy(np.array(value, copy=True))),
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        reference = self.recurrence_alpha
        return torch.zeros(
            (batch_size, STATE_DIM),
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def forward_step(
        self,
        observation: Tensor,
        previous_action: Tensor,
        previous_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if observation.shape[-1] != PUBLIC_OBSERVATION_DIM:
            raise ValueError("observation must end in dimension 19")
        if previous_action.shape[:-1] != observation.shape[:-1] or (
            previous_action.shape[-1] != PUBLIC_ACTION_DIM
        ):
            raise ValueError("previous action shape must match the observation")
        if previous_state.shape[:-1] != observation.shape[:-1] or (
            previous_state.shape[-1] != STATE_DIM
        ):
            raise ValueError("previous state shape must match the observation")

        encoded = functional.silu(
            functional.linear(
                observation, self.encoder_0_weight, self.encoder_0_bias
            )
        )
        encoded = functional.silu(
            functional.linear(encoded, self.encoder_1_weight, self.encoder_1_bias)
        )
        state = (
            torch.tanh(self.recurrence_alpha) * previous_state
            + functional.linear(
                encoded, self.recurrence_input_weight, self.recurrence_bias
            )
            + functional.linear(
                previous_action, self.recurrence_action_weight, bias=None
            )
        )
        innovation = torch.tanh(
            functional.linear(
                previous_state,
                self.innovation_state_weight,
                self.innovation_bias,
            )
            + functional.linear(
                encoded, self.innovation_input_weight, bias=None
            )
            + functional.linear(
                previous_action, self.innovation_action_weight, bias=None
            )
        )
        state = state + functional.linear(
            innovation, self.innovation_output_weight, bias=None
        )
        action_hidden = functional.silu(
            functional.linear(
                torch.cat((state, encoded), dim=-1),
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
        return action, state

    def forward_sequence(
        self,
        observations: Tensor,
        previous_actions: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Unroll a contiguous batch of episodes for supervised training."""

        if observations.ndim != 3 or observations.shape[-1] != PUBLIC_OBSERVATION_DIM:
            raise ValueError("observations must have shape (batch, time, 19)")
        if previous_actions.shape != (
            observations.shape[0],
            observations.shape[1],
            PUBLIC_ACTION_DIM,
        ):
            raise ValueError("previous actions must have shape (batch, time, 2)")
        state = (
            self.initial_state(
                observations.shape[0],
                device=observations.device,
                dtype=observations.dtype,
            )
            if initial_state is None
            else initial_state
        )
        if state.shape != (observations.shape[0], STATE_DIM):
            raise ValueError("initial state must have shape (batch, 64)")
        actions = []
        states = []
        for time_index in range(observations.shape[1]):
            action, state = self.forward_step(
                observations[:, time_index],
                previous_actions[:, time_index],
                state,
            )
            actions.append(action)
            states.append(state)
        return torch.stack(actions, dim=1), torch.stack(states, dim=1)

    def export_numpy_parameters(self) -> dict[str, NDArray[np.float32]]:
        """Export weights for ``StructuredRecurrentPolicy`` evaluation."""

        return {
            name: parameter.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, parameter in self.named_parameters()
        }
