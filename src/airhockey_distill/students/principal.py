"""Family-neutral NumPy interface for the seven principal students."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from airhockey_distill.students.feed_forward import (
    FeedForwardPolicy,
    save_principal_feed_forward_checkpoint,
)
from airhockey_distill.students.finite_stack import (
    FiniteStackPolicy,
    save_finite_stack_checkpoint,
)
from airhockey_distill.students.gru import GRURecurrentPolicy, save_gru_checkpoint
from airhockey_distill.students.structured import (
    StructuredRecurrentPolicy,
    save_structured_checkpoint,
)

PRINCIPAL_FAMILY_IDS = (
    "feed_forward",
    "finite_stack_10",
    "structured_k0",
    "structured_k1",
    "structured_k2",
    "structured_k4",
    "gru_n64",
)


def principal_structured_rank(family_id: str) -> int | None:
    if family_id.startswith("structured_k"):
        rank = int(family_id.removeprefix("structured_k"))
        if rank in (0, 1, 2, 4):
            return rank
    return None


@dataclass(frozen=True)
class PrincipalPolicy:
    """One common closed-loop and sequence interface across all families."""

    family_id: str
    policy: Any

    def __post_init__(self) -> None:
        if self.family_id not in PRINCIPAL_FAMILY_IDS:
            raise ValueError(f"unsupported principal family {self.family_id!r}")
        declared = self.policy.metadata.get("student_id")
        if declared is not None and declared != self.family_id:
            raise ValueError("checkpoint student_id does not match the family")

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.policy.metadata)

    @property
    def parameter_count(self) -> int:
        return int(self.policy.parameter_count)

    @property
    def core_parameter_count(self) -> int:
        return int(getattr(self.policy, "recurrent_parameter_count", 0))

    @property
    def carry_float32_values(self) -> int:
        if self.family_id == "feed_forward":
            return 0
        if self.family_id == "finite_stack_10":
            return 30
        return 66

    def initial_carry(self, batch_size: int | None = None) -> Any:
        if self.family_id == "feed_forward":
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch size must be positive")
            return None
        return self.policy.initial_carry(batch_size)

    def act(
        self,
        observation: ArrayLike,
        carry: Any,
    ) -> tuple[NDArray[np.float32], Any]:
        if self.family_id == "feed_forward":
            if carry is not None:
                raise ValueError("feed-forward carry must be None")
            return self.policy.action(observation), None
        return self.policy.act(observation, carry)

    def sequence(
        self,
        observations: ArrayLike,
        previous_actions: ArrayLike,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        observation_array = np.asarray(observations, dtype=np.float32)
        previous_action_array = np.asarray(previous_actions, dtype=np.float32)
        if self.family_id == "feed_forward":
            actions = self.policy.action(observation_array)
            carries = np.zeros((len(observation_array), 0), dtype=np.float32)
            return actions, carries
        if self.family_id == "finite_stack_10":
            return self.policy.sequence(observation_array)
        return self.policy.teacher_forced_sequence(
            observation_array,
            previous_action_array,
        )


def load_principal_policy(
    family_id: str,
    checkpoint: str | Path,
) -> PrincipalPolicy:
    """Load one family from its framework-neutral checkpoint."""

    _validate_family_id(family_id)
    path = Path(checkpoint)
    if family_id == "feed_forward":
        policy = FeedForwardPolicy.load(path)
    elif family_id == "finite_stack_10":
        policy = FiniteStackPolicy.load(path)
    elif family_id == "gru_n64":
        policy = GRURecurrentPolicy.load(path)
    else:
        policy = StructuredRecurrentPolicy.load(path)
        if policy.innovation_rank != principal_structured_rank(family_id):
            raise ValueError("structured checkpoint rank does not match the family")
    return PrincipalPolicy(family_id=family_id, policy=policy)


def principal_policy_from_parameters(
    family_id: str,
    parameters: dict[str, NDArray[np.float32]],
    metadata: dict[str, Any] | None = None,
) -> PrincipalPolicy:
    """Construct a common NumPy policy directly from exported parameters."""

    _validate_family_id(family_id)
    checked_metadata = {} if metadata is None else dict(metadata)
    if family_id == "feed_forward":
        policy = FeedForwardPolicy(parameters, checked_metadata)
    elif family_id == "finite_stack_10":
        policy = FiniteStackPolicy(parameters, checked_metadata)
    elif family_id == "gru_n64":
        policy = GRURecurrentPolicy(parameters, checked_metadata)
    else:
        policy = StructuredRecurrentPolicy(
            parameters,
            checked_metadata,
            innovation_rank=principal_structured_rank(family_id),
        )
    return PrincipalPolicy(family_id=family_id, policy=policy)


def save_principal_checkpoint(
    family_id: str,
    path: str | Path,
    parameters: dict[str, NDArray[np.float32]],
    metadata: dict[str, Any],
) -> None:
    """Save one principal family with a shared provenance contract."""

    _validate_family_id(family_id)
    checkpoint_metadata = dict(metadata)
    if checkpoint_metadata.get("student_id", family_id) != family_id:
        raise ValueError("checkpoint student_id conflicts with the family")
    checkpoint_metadata["student_id"] = family_id
    if family_id == "feed_forward":
        save_principal_feed_forward_checkpoint(path, parameters, checkpoint_metadata)
    elif family_id == "finite_stack_10":
        save_finite_stack_checkpoint(path, parameters, checkpoint_metadata)
    elif family_id == "gru_n64":
        save_gru_checkpoint(path, parameters, checkpoint_metadata)
    else:
        save_structured_checkpoint(
            path,
            parameters,
            checkpoint_metadata,
            innovation_rank=principal_structured_rank(family_id),
        )


def _validate_family_id(family_id: str) -> None:
    if family_id not in PRINCIPAL_FAMILY_IDS:
        raise ValueError(f"unsupported principal family {family_id!r}")
