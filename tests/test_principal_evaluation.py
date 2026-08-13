from airhockey_distill.principal_sweep import load_principal_protocol
from scripts.evaluate_principal_student import summarise


def test_common_evaluation_summary_uses_protocol_save_outcomes():
    protocol = load_principal_protocol("configs/experiments/principal_sweep_execution_v1.yaml")
    outcomes = [
        "returned",
        "arrested",
        "safe_deflection",
        "goal_conceded",
        "timeout",
    ]
    episodes = [
        {
            "outcome": outcome,
            "blackout_steps": blackout,
            "score": float(index),
        }
        for blackout in (0, 5, 10, 15, 20)
        for index, outcome in enumerate(outcomes)
    ]

    summary = summarise(episodes, [0.001] * len(episodes), protocol)

    assert summary["episodes"] == 25
    assert summary["save_count"] == 15
    assert summary["save_rate"] == 0.6
    for blackout in (0, 5, 10, 15, 20):
        assert summary["by_blackout_steps"][str(blackout)]["save_count"] == 3
