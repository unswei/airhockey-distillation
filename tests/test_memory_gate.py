import pytest

from airhockey_distill.evaluation import (
    evaluate_causal_memory_ablation,
    evaluate_memory_gate,
)


def episode(shot: int, blackout: int, saved: bool) -> dict[str, object]:
    return {
        "shot_id": f"shot-{shot:03d}",
        "blackout_steps": blackout,
        "outcome": "returned" if saved else "goal_conceded",
    }


def gate_config() -> dict[str, object]:
    return {
        "long_blackout_steps": 20,
        "minimum_feed_forward_no_blackout_save_rate": 0.75,
        "maximum_no_blackout_teacher_advantage": 0.10,
        "minimum_long_blackout_teacher_advantage": 0.15,
        "minimum_advantage_growth": 0.10,
        "require_positive_long_advantage_ci_lower": True,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 10201,
    }


def test_memory_gate_uses_paired_outcomes_and_passes_clear_memory_effect() -> None:
    teacher = []
    feed_forward = []
    for shot in range(100):
        teacher.extend([episode(shot, 0, shot < 85), episode(shot, 20, shot < 80)])
        feed_forward.extend([episode(shot, 0, shot < 80), episode(shot, 20, shot < 45)])

    report = evaluate_memory_gate(teacher, list(reversed(feed_forward)), gate_config())

    assert report["decision"] == "GO"
    assert report["paired_episode_count"] == 200
    assert report["by_blackout_steps"]["20"]["paired_teacher_advantage"] == 0.35


def test_memory_gate_rejects_weak_feed_forward_visible_baseline() -> None:
    teacher = [
        episode(shot, blackout, True) for shot in range(20) for blackout in (0, 20)
    ]
    feed_forward = [
        episode(shot, blackout, shot < 5) for shot in range(20) for blackout in (0, 20)
    ]

    report = evaluate_memory_gate(teacher, feed_forward, gate_config())

    assert report["decision"] == "NO_GO"
    assert "credible_visible_feed_forward_baseline" in report["blocking_checks"]


def test_memory_gate_rejects_unpaired_results() -> None:
    with pytest.raises(ValueError, match="episode keys differ"):
        evaluate_memory_gate(
            [episode(0, 0, True)],
            [episode(1, 0, True)],
            gate_config(),
        )


def test_causal_ablation_gate_requires_paired_long_blackout_drop() -> None:
    recurrent = []
    reset = []
    for shot in range(100):
        recurrent.extend(
            [
                {**episode(shot, 0, shot < 85), "steps": 20, "score": 1.0},
                {**episode(shot, 20, shot < 80), "steps": 20, "score": 1.0},
            ]
        )
        reset.extend(
            [
                {**episode(shot, 0, shot < 85), "steps": 20, "score": 1.0},
                {**episode(shot, 20, shot < 55), "steps": 20, "score": 1.0},
            ]
        )
    config = {
        "long_blackout_steps": 20,
        "minimum_long_blackout_save_rate_drop": 0.10,
        "require_positive_long_drop_ci_lower": True,
        "require_identical_no_blackout_outcomes": True,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 11303,
    }

    report = evaluate_causal_memory_ablation(recurrent, reset, config)

    assert report["decision"] == "GO"
    assert report["identical_no_blackout_control"] is True
    assert report["by_blackout_steps"]["20"]["paired_save_rate_drop"] == 0.25
