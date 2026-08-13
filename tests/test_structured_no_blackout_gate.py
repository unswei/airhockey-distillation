import copy

import pytest

from scripts.gate_structured_no_blackout import apply_gate


def test_gate_passes_credible_no_blackout_student():
    result = apply_gate(
        _config(), _training(), _evaluation(save_count=180), code_commit="abc"
    )

    assert result["decision"] == "GO"
    assert result["next_action"] == "full"


def test_gate_routes_weak_student_to_shadow_labelling():
    result = apply_gate(
        _config(), _training(), _evaluation(save_count=168), code_commit="abc"
    )

    assert result["decision"] == "NO_GO"
    assert result["next_action"] == "shadow"


def test_gate_rejects_opened_blackout_conditions():
    evaluation = _evaluation(save_count=180)
    evaluation["summary"]["by_blackout_steps"]["20"] = copy.deepcopy(
        evaluation["summary"]["by_blackout_steps"]["0"]
    )

    with pytest.raises(ValueError, match="unapproved blackout"):
        apply_gate(_config(), _training(), evaluation, code_commit="abc")


def _config():
    return {
        "no_blackout_gate": {
            "required_episodes": 225,
            "required_blackout_steps": [0],
            "minimum_save_rate": 0.75,
            "maximum_simulator_or_safety_faults": 0,
            "decision_if_go": "full",
            "decision_if_no_go": "shadow",
        }
    }


def _training():
    return {
        "code_commit": "abc",
        "checkpoint_sha256": "checkpoint",
        "training_seed": 14303,
        "selected_epoch": 10,
        "selected_metrics": {"validation": {"action_mse": 0.01}},
    }


def _evaluation(save_count):
    return {
        "code_commit": "abc",
        "checkpoint_sha256": "checkpoint",
        "summary": {
            "episodes": 225,
            "save_count": save_count,
            "save_rate": save_count / 225,
            "outcome_counts": {
                "goal_conceded": 225 - save_count,
                "returned": save_count,
            },
            "by_blackout_steps": {"0": {"episodes": 225}},
        },
    }
