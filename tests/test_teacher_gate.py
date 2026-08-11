import numpy as np

from airhockey_distill.envs.policy_interface import PUBLIC_OBSERVATION_COMPONENTS
from airhockey_distill.evaluation import (
    GateStatus,
    TeacherGateEvidence,
    audit_public_observation_contract,
    evaluate_teacher_training_gate,
)
from airhockey_distill.evaluation.rollout import EpisodeTrace


def trace(controller: str, shot_number: int, outcome: str) -> EpisodeTrace:
    return EpisodeTrace(
        controller_name=controller,
        shot_id=f"shot-{shot_number:03d}",
        observations=(np.zeros(19, dtype=np.float32),),
        actions=(),
        visibility=(True,),
        total_reward=0.0,
        terminated=True,
        truncated=False,
        outcome=outcome,
    )


def evidence(
    *,
    shots: int,
    inactive_outcome: str = "goal_conceded",
    fixed_outcome: str = "goal_conceded",
    privileged_outcome: str = "returned",
    replay: bool = True,
    observation_clean: bool = True,
    completed: int = 20,
    faults: tuple[str, ...] = (),
) -> TeacherGateEvidence:
    return TeacherGateEvidence(
        inactive_traces=tuple(
            trace("inactive", index, inactive_outcome) for index in range(shots)
        ),
        fixed_traces=tuple(
            trace("fixed_centre", index, fixed_outcome) for index in range(shots)
        ),
        privileged_traces=tuple(
            trace("privileged_intercept", index, privileged_outcome)
            for index in range(shots)
        ),
        replay_deterministic=replay,
        observation_contract_clean=observation_clean,
        reliability_episodes_attempted=20,
        reliability_episodes_completed=completed,
        simulator_faults=faults,
    )


def checks_by_id(report):  # type: ignore[no-untyped-def]
    return {check.check_id: check for check in report.checks}


def test_gate_passes_only_with_sufficient_passing_evidence() -> None:
    report = evaluate_teacher_training_gate(evidence(shots=200))

    assert report.decision == "GO"
    assert report.blocking_checks == ()


def test_single_shot_rates_are_insufficient_even_when_outcomes_look_good() -> None:
    report = evaluate_teacher_training_gate(evidence(shots=1))
    checks = checks_by_id(report)

    assert report.decision == "NO_GO"
    assert checks["inactive_concession_rate"].status is GateStatus.INSUFFICIENT_EVIDENCE
    assert checks["fixed_concession_rate"].status is GateStatus.INSUFFICIENT_EVIDENCE
    assert checks["privileged_save_rate"].status is GateStatus.INSUFFICIENT_EVIDENCE
    assert checks["deterministic_shot_replay"].status is GateStatus.PASS


def test_gate_fails_bad_rates_and_simulator_faults() -> None:
    mixed_fixed = tuple(
        trace("fixed_centre", index, "goal_conceded" if index < 159 else "timeout")
        for index in range(200)
    )
    gate_evidence = evidence(shots=200, faults=("MuJoCo fault",), completed=19)
    gate_evidence = TeacherGateEvidence(
        inactive_traces=gate_evidence.inactive_traces,
        fixed_traces=mixed_fixed,
        privileged_traces=gate_evidence.privileged_traces,
        replay_deterministic=gate_evidence.replay_deterministic,
        observation_contract_clean=gate_evidence.observation_contract_clean,
        reliability_episodes_attempted=20,
        reliability_episodes_completed=19,
        simulator_faults=gate_evidence.simulator_faults,
    )
    report = evaluate_teacher_training_gate(gate_evidence)
    checks = checks_by_id(report)

    assert report.decision == "NO_GO"
    assert checks["fixed_concession_rate"].status is GateStatus.FAIL
    assert checks["marvin_simulator_reliability"].status is GateStatus.FAIL


def test_public_observation_contract_excludes_puck_velocity_and_opponent() -> None:
    names = {name for name, _ in PUBLIC_OBSERVATION_COMPONENTS}

    assert audit_public_observation_contract()
    assert "puck_velocity" not in names
    assert not any(name.startswith("opponent_") for name in names)
