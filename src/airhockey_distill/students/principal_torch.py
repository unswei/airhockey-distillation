"""Family-neutral trainable interface for the seven principal students."""

from __future__ import annotations

import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from airhockey_distill.students.feed_forward_torch import FeedForwardModule
from airhockey_distill.students.finite_stack_torch import FiniteStackModule
from airhockey_distill.students.gru_torch import GRURecurrentModule
from airhockey_distill.students.principal import (
    PRINCIPAL_FAMILY_IDS,
    principal_structured_rank,
)
from airhockey_distill.students.structured_torch import StructuredRecurrentModule


class PrincipalStudentModule(nn.Module):
    """Dispatch family-specific state while keeping one training interface."""

    def __init__(self, family_id: str, *, seed: int) -> None:
        super().__init__()
        if family_id not in PRINCIPAL_FAMILY_IDS:
            raise ValueError(f"unsupported principal family {family_id!r}")
        self.family_id = family_id
        if family_id == "feed_forward":
            self.student = FeedForwardModule(seed=seed)
        elif family_id == "finite_stack_10":
            self.student = FiniteStackModule(seed=seed)
        elif family_id == "gru_n64":
            self.student = GRURecurrentModule(seed=seed)
        else:
            self.student = StructuredRecurrentModule(
                seed=seed,
                innovation_rank=principal_structured_rank(family_id),
            )

    @property
    def parameter_count(self) -> int:
        return int(self.student.parameter_count)

    def forward_sequence(
        self,
        observations: Tensor,
        previous_actions: Tensor,
        initial_carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != 19:
            raise ValueError("observations must have shape (batch, time, 19)")
        if previous_actions.shape != (*observations.shape[:2], 2):
            raise ValueError("previous actions must have shape (batch, time, 2)")
        if self.family_id == "feed_forward":
            if initial_carry is not None:
                raise ValueError("feed-forward carry must be None")
            actions = self.student(observations)
            carries = observations.new_zeros((*observations.shape[:2], 0))
            return actions, carries
        if self.family_id == "finite_stack_10":
            return self.student.forward_sequence(observations, initial_carry)
        return self.student.forward_sequence(
            observations,
            previous_actions,
            initial_carry,
        )

    def export_numpy_parameters(self) -> dict[str, NDArray]:
        return self.student.export_numpy_parameters()
