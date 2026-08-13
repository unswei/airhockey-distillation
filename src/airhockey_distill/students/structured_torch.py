"""Trainable PyTorch form of the rank-configurable structured student."""

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
    CANONICAL_FLOAT32_ARITHMETIC,
    ENCODED_DIM,
    INNOVATION_RANK,
    LEGACY_FLOAT32_ARITHMETIC,
    STATE_DIM,
    SUPPORTED_INFERENCE_ARITHMETICS,
    initialise_structured_parameters,
    infer_structured_innovation_rank,
    structured_parameter_shapes,
)


class StructuredRecurrentModule(nn.Module):
    """Differentiable ``n=64`` policy with explicit sequence state."""

    def __init__(
        self,
        *,
        seed: int = 0,
        innovation_rank: int | None = None,
        inference_arithmetic: str = LEGACY_FLOAT32_ARITHMETIC,
        parameters: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        super().__init__()
        initial = (
            initialise_structured_parameters(
                seed,
                innovation_rank=(
                    INNOVATION_RANK
                    if innovation_rank is None
                    else innovation_rank
                ),
            )
            if parameters is None
            else {
                name: np.asarray(value, dtype=np.float32)
                for name, value in parameters.items()
            }
        )
        rank = infer_structured_innovation_rank(
            initial,
            requested_rank=innovation_rank,
        )
        if inference_arithmetic not in SUPPORTED_INFERENCE_ARITHMETICS:
            raise ValueError("unsupported structured inference arithmetic")
        self.inference_arithmetic = inference_arithmetic
        self.innovation_rank = rank
        self.parameter_shapes = structured_parameter_shapes(rank)
        for name, shape in self.parameter_shapes.items():
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

        parameters = self._forward_parameters()
        return self._forward_step(
            observation,
            previous_action,
            previous_state,
            parameters,
        )

    def _forward_parameters(self) -> dict[str, Tensor]:
        return dict(self.named_parameters())

    def _forward_step(
        self,
        observation: Tensor,
        previous_action: Tensor,
        previous_state: Tensor,
        parameters: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        canonical = (
            self.inference_arithmetic == CANONICAL_FLOAT32_ARITHMETIC
        )
        linear = _pairwise_linear_float32 if canonical else functional.linear

        activation_silu = (
            _canonical_silu_float32 if canonical else functional.silu
        )
        activation_tanh = _canonical_tanh_float32 if canonical else torch.tanh

        encoded = activation_silu(
            linear(
                observation,
                parameters["encoder_0_weight"],
                parameters["encoder_0_bias"],
            )
        )
        encoded = activation_silu(
            linear(
                encoded,
                parameters["encoder_1_weight"],
                parameters["encoder_1_bias"],
            )
        )
        state = (
            activation_tanh(parameters["recurrence_alpha"]) * previous_state
            + linear(
                encoded,
                parameters["recurrence_input_weight"],
                parameters["recurrence_bias"],
            )
            + linear(
                previous_action,
                parameters["recurrence_action_weight"],
                bias=None,
            )
        )
        if self.innovation_rank:
            innovation = activation_tanh(
                linear(
                    previous_state,
                    parameters["innovation_state_weight"],
                    parameters["innovation_bias"],
                )
                + linear(
                    encoded,
                    parameters["innovation_input_weight"],
                    bias=None,
                )
                + linear(
                    previous_action,
                    parameters["innovation_action_weight"],
                    bias=None,
                )
            )
            state = state + linear(
                innovation,
                parameters["innovation_output_weight"],
                bias=None,
            )
        action_hidden = activation_silu(
            linear(
                torch.cat((state, encoded), dim=-1),
                parameters["action_hidden_weight"],
                parameters["action_hidden_bias"],
            )
        )
        action = activation_tanh(
            linear(
                action_hidden,
                parameters["action_output_weight"],
                parameters["action_output_bias"],
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
        parameters = self._forward_parameters()
        for time_index in range(observations.shape[1]):
            action, state = self._forward_step(
                observations[:, time_index],
                previous_actions[:, time_index],
                state,
                parameters,
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


def _pairwise_linear_float32(
    value: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Mirror the exported NumPy policy's balanced float32 reduction tree."""

    terms = value.unsqueeze(-2) * weight.unsqueeze(0)
    while terms.shape[-1] > 1:
        pair_count = terms.shape[-1] // 2
        reduced = (
            terms[..., : 2 * pair_count : 2]
            + terms[..., 1 : 2 * pair_count : 2]
        )
        if terms.shape[-1] % 2:
            reduced = torch.cat((reduced, terms[..., -1:]), dim=-1)
        terms = reduced
    result = terms[..., 0]
    return result if bias is None else result + bias


def _canonical_silu_float32(value: Tensor) -> Tensor:
    exponent = torch.clip(-value, -60.0, 60.0)
    return value * (1.0 / (1.0 + torch.exp(exponent)))


def _canonical_tanh_float32(value: Tensor) -> Tensor:
    bounded = torch.clip(value, -20.0, 20.0)
    exponent = torch.exp(-2.0 * bounded)
    return (1.0 - exponent) / (1.0 + exponent)
