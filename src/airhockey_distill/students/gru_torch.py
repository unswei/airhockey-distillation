"""Trainable PyTorch form of the matched GRU-64 recurrent student."""

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
from airhockey_distill.students.gru import (
    GRU_ENCODED_DIM,
    GRU_PARAMETER_SHAPES,
    GRU_STATE_DIM,
    initialise_gru_parameters,
)


class GRURecurrentModule(nn.Module):
    """Differentiable hidden-size-64 GRU with explicit sequence state."""

    def __init__(
        self,
        *,
        seed: int = 0,
        parameters: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__()
        initial = (
            initialise_gru_parameters(seed)
            if parameters is None
            else {
                name: np.asarray(value, dtype=np.float32)
                for name, value in parameters.items()
            }
        )
        for name, shape in GRU_PARAMETER_SHAPES.items():
            if name not in initial:
                raise ValueError(f"missing GRU parameter {name}")
            value = initial[name]
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid GRU parameter {name}")
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
        reference = self.gru_input_weight
        return torch.zeros(
            (batch_size, GRU_STATE_DIM),
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
            previous_state.shape[-1] != GRU_STATE_DIM
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
        recurrent_input = torch.cat((encoded, previous_action), dim=-1)
        input_gates = functional.linear(
            recurrent_input, self.gru_input_weight, self.gru_input_bias
        )
        hidden_gates = functional.linear(
            previous_state, self.gru_hidden_weight, self.gru_hidden_bias
        )
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        state = candidate * (1.0 - update) + previous_state * update
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
        if state.shape != (observations.shape[0], GRU_STATE_DIM):
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
        """Export weights for ``GRURecurrentPolicy`` evaluation."""

        return {
            name: parameter.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, parameter in self.named_parameters()
        }
