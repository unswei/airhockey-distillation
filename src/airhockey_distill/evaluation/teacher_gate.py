"""Fail-closed evidence gate for beginning teacher training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from airhockey_distill.envs.policy_interface import (
    PUBLIC_OBSERVATION_COMPONENTS,
    PUBLIC_OBSERVATION_DIM,
    PUCK_POSITION_XY_SLICE,
    UPSTREAM_OBSERVATION_COMPONENTS,
    UPSTREAM_OBSERVATION_DIM,
    UPSTREAM_PUCK_VELOCITY_XY_SLICE,
    PublicObservationAdapter,
)
from airhockey_distill.evaluation.rollout import EpisodeTrace

PRIVILEGED_SAVE_OUTCOMES = frozenset(
    {"arrested", "cleared", "returned", "safe_deflection"}
)


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class TeacherGateThresholds:
    """Conservative thresholds used before spending compute on a teacher."""

    minimum_distinct_shots: int = 200
    minimum_inactive_concession_rate: float = 0.8
    minimum_fixed_concession_rate: float = 0.8
    minimum_privileged_save_rate: float = 0.8
    minimum_reliability_episodes: int = 20
    maximum_simulator_faults: int = 0

    def __post_init__(self) -> None:
        if self.minimum_distinct_shots <= 0:
            raise ValueError("minimum_distinct_shots must be positive")
        if self.minimum_reliability_episodes <= 0:
            raise ValueError("minimum_reliability_episodes must be positive")
        if self.maximum_simulator_faults < 0:
            raise ValueError("maximum_simulator_faults must be non-negative")
        rates = (
            self.minimum_inactive_concession_rate,
            self.minimum_fixed_concession_rate,
            self.minimum_privileged_save_rate,
        )
        if any(rate < 0.0 or rate > 1.0 for rate in rates):
            raise ValueError("rate thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class TeacherGateEvidence:
    inactive_traces: tuple[EpisodeTrace, ...]
    fixed_traces: tuple[EpisodeTrace, ...]
    privileged_traces: tuple[EpisodeTrace, ...]
    replay_deterministic: bool
    observation_contract_clean: bool
    reliability_episodes_attempted: int
    reliability_episodes_completed: int
    simulator_faults: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateCheck:
    check_id: str
    status: GateStatus
    observed: dict[str, Any]
    required: dict[str, Any]
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "observed": self.observed,
            "required": self.required,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class TeacherGateReport:
    thresholds: TeacherGateThresholds
    checks: tuple[GateCheck, ...]

    @property
    def decision(self) -> str:
        return (
            "GO"
            if all(check.status is GateStatus.PASS for check in self.checks)
            else "NO_GO"
        )

    @property
    def blocking_checks(self) -> tuple[str, ...]:
        return tuple(
            check.check_id
            for check in self.checks
            if check.status is not GateStatus.PASS
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "decision": self.decision,
            "thresholds": asdict(self.thresholds),
            "blocking_checks": list(self.blocking_checks),
            "checks": [check.as_dict() for check in self.checks],
        }


def audit_public_observation_contract() -> bool:
    """Check exact composition and invariance to hidden puck velocity."""

    if (
        sum(size for _, size in UPSTREAM_OBSERVATION_COMPONENTS)
        != UPSTREAM_OBSERVATION_DIM
    ):
        return False
    if sum(size for _, size in PUBLIC_OBSERVATION_COMPONENTS) != PUBLIC_OBSERVATION_DIM:
        return False

    public_names = {name for name, _ in PUBLIC_OBSERVATION_COMPONENTS}
    if "puck_velocity" in public_names:
        return False
    if any(name.startswith("opponent_") for name in public_names):
        return False

    adapter = PublicObservationAdapter()
    first = np.linspace(-0.9, 0.9, UPSTREAM_OBSERVATION_DIM, dtype=np.float32)
    second = first.copy()
    second[UPSTREAM_PUCK_VELOCITY_XY_SLICE] = (91.0, -37.0)
    visible_first = adapter.adapt(first, puck_visible=True)
    visible_second = adapter.adapt(second, puck_visible=True)
    hidden = adapter.adapt(first, puck_visible=False)

    return bool(
        np.array_equal(visible_first, visible_second)
        and np.array_equal(hidden[PUCK_POSITION_XY_SLICE], np.zeros(2))
    )


def evaluate_teacher_training_gate(
    evidence: TeacherGateEvidence,
    thresholds: TeacherGateThresholds | None = None,
) -> TeacherGateReport:
    """Evaluate all checks; missing evidence can never produce ``GO``."""

    criteria = thresholds or TeacherGateThresholds()
    inactive = _index_unique_shots(evidence.inactive_traces)
    fixed = _index_unique_shots(evidence.fixed_traces)
    privileged = _index_unique_shots(evidence.privileged_traces)
    shot_ids = set(inactive)
    paired = shot_ids == set(fixed) == set(privileged)

    checks = [
        GateCheck(
            check_id="paired_shot_coverage",
            status=GateStatus.PASS if paired and shot_ids else GateStatus.FAIL,
            observed={
                "inactive_distinct_shots": len(inactive),
                "fixed_distinct_shots": len(fixed),
                "privileged_distinct_shots": len(privileged),
                "same_shot_ids": paired,
            },
            required={"same_non_empty_shot_ids": True},
            explanation="All rate controls must be evaluated on the same generated shots.",
        )
    ]

    checks.extend(
        (
            _rate_check(
                check_id="inactive_concession_rate",
                traces=tuple(inactive.values()),
                successes=lambda trace: trace.outcome == "goal_conceded",
                threshold=criteria.minimum_inactive_concession_rate,
                minimum_shots=criteria.minimum_distinct_shots,
                coverage_valid=paired,
                explanation=(
                    "The inactive defender must meet the configured concession-rate "
                    "threshold on paired shots."
                ),
            ),
            _rate_check(
                check_id="fixed_concession_rate",
                traces=tuple(fixed.values()),
                successes=lambda trace: trace.outcome == "goal_conceded",
                threshold=criteria.minimum_fixed_concession_rate,
                minimum_shots=criteria.minimum_distinct_shots,
                coverage_valid=paired,
                explanation=(
                    "The fixed-centre defender must meet the configured concession-rate "
                    "threshold on paired shots."
                ),
            ),
            _rate_check(
                check_id="privileged_save_rate",
                traces=tuple(privileged.values()),
                successes=lambda trace: trace.outcome in PRIVILEGED_SAVE_OUTCOMES,
                threshold=criteria.minimum_privileged_save_rate,
                minimum_shots=criteria.minimum_distinct_shots,
                coverage_valid=paired,
                explanation=(
                    "The privileged controller should explicitly return, clear or arrest "
                    "the configured fraction of paired shots. Bare timeouts are not "
                    "counted as saves."
                ),
            ),
        )
    )

    checks.extend(
        (
            GateCheck(
                check_id="deterministic_shot_replay",
                status=(
                    GateStatus.PASS
                    if evidence.replay_deterministic
                    else GateStatus.FAIL
                ),
                observed={
                    "exact_public_trajectory_match": evidence.replay_deterministic
                },
                required={"exact_public_trajectory_match": True},
                explanation="Repeated shots must reproduce the exact public trajectory.",
            ),
            GateCheck(
                check_id="public_observation_isolation",
                status=(
                    GateStatus.PASS
                    if evidence.observation_contract_clean
                    else GateStatus.FAIL
                ),
                observed={
                    "puck_velocity_absent": evidence.observation_contract_clean,
                    "opponent_information_absent": evidence.observation_contract_clean,
                },
                required={
                    "puck_velocity_absent": True,
                    "opponent_information_absent": True,
                },
                explanation=(
                    "Defender joint velocity is intentional; puck velocity and opponent "
                    "features must be absent."
                ),
            ),
            _reliability_check(evidence, criteria),
        )
    )
    return TeacherGateReport(thresholds=criteria, checks=tuple(checks))


def _index_unique_shots(traces: tuple[EpisodeTrace, ...]) -> dict[str, EpisodeTrace]:
    indexed: dict[str, EpisodeTrace] = {}
    for trace in traces:
        if trace.shot_id in indexed:
            raise ValueError(f"duplicate trace for shot_id {trace.shot_id!r}")
        indexed[trace.shot_id] = trace
    return indexed


def _rate_check(
    *,
    check_id: str,
    traces: tuple[EpisodeTrace, ...],
    successes: Callable[[EpisodeTrace], bool],
    threshold: float,
    minimum_shots: int,
    coverage_valid: bool,
    explanation: str,
) -> GateCheck:
    success_count = sum(bool(successes(trace)) for trace in traces)
    rate = success_count / len(traces) if traces else None
    if not coverage_valid or len(traces) < minimum_shots:
        status = GateStatus.INSUFFICIENT_EVIDENCE
    elif rate is not None and rate >= threshold:
        status = GateStatus.PASS
    else:
        status = GateStatus.FAIL
    return GateCheck(
        check_id=check_id,
        status=status,
        observed={
            "rate": rate,
            "successes": success_count,
            "distinct_shots": len(traces),
        },
        required={"minimum_rate": threshold, "minimum_distinct_shots": minimum_shots},
        explanation=explanation,
    )


def _reliability_check(
    evidence: TeacherGateEvidence,
    thresholds: TeacherGateThresholds,
) -> GateCheck:
    fault_count = len(evidence.simulator_faults)
    if fault_count > thresholds.maximum_simulator_faults:
        status = GateStatus.FAIL
    elif (
        evidence.reliability_episodes_completed
        < thresholds.minimum_reliability_episodes
    ):
        status = GateStatus.INSUFFICIENT_EVIDENCE
    else:
        status = GateStatus.PASS
    return GateCheck(
        check_id="marvin_simulator_reliability",
        status=status,
        observed={
            "episodes_attempted": evidence.reliability_episodes_attempted,
            "episodes_completed": evidence.reliability_episodes_completed,
            "fault_count": fault_count,
            "faults": list(evidence.simulator_faults),
        },
        required={
            "minimum_completed_episodes": thresholds.minimum_reliability_episodes,
            "maximum_faults": thresholds.maximum_simulator_faults,
        },
        explanation="Marvin must complete the smoke episodes without simulator faults.",
    )
