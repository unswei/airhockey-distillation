"""Paired rollout, metrics, latency, probing and visualisation tools."""

from .baselines import (
    FixedCentreController,
    InactiveController,
    PrivilegedInterceptController,
)
from .memory_gate import evaluate_causal_memory_ablation, evaluate_memory_gate
from .rollout import (
    EpisodeTrace,
    assert_equivalent_replay,
    rollout_privileged_controller,
    rollout_public_controller,
)
from .teacher_gate import (
    GateCheck,
    GateStatus,
    TeacherGateEvidence,
    TeacherGateReport,
    TeacherGateThresholds,
    audit_public_observation_contract,
    evaluate_teacher_training_gate,
)

__all__ = [
    "EpisodeTrace",
    "FixedCentreController",
    "GateCheck",
    "GateStatus",
    "InactiveController",
    "PrivilegedInterceptController",
    "TeacherGateEvidence",
    "TeacherGateReport",
    "TeacherGateThresholds",
    "assert_equivalent_replay",
    "audit_public_observation_contract",
    "evaluate_causal_memory_ablation",
    "evaluate_memory_gate",
    "evaluate_teacher_training_gate",
    "rollout_privileged_controller",
    "rollout_public_controller",
]
