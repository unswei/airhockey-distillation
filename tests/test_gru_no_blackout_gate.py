import copy
from pathlib import Path

import pytest
import yaml

from scripts.gate_gru_no_blackout import apply_gate


def test_gru_gate_passes_credible_no_blackout_pilot():
    result = apply_gate(
        _config(),
        _training(),
        _evaluation(save_count=180),
        evaluation_code_commit="eval",
    )

    assert result["decision"] == "GO"
    assert result["next_action"] == "paired"
    assert result["pilot_scope"]["classification"].startswith("engineering")


def test_gru_gate_stops_weak_pilot():
    result = apply_gate(
        _config(),
        _training(),
        _evaluation(save_count=168),
        evaluation_code_commit="eval",
    )

    assert result["decision"] == "NO_GO"
    assert result["next_action"] == "stop"


def test_gru_gate_rejects_opened_blackout_and_checkpoint_mismatch():
    evaluation = _evaluation(save_count=180)
    evaluation["summary"]["by_blackout_steps"]["20"] = copy.deepcopy(
        evaluation["summary"]["by_blackout_steps"]["0"]
    )
    with pytest.raises(ValueError, match="unapproved blackout"):
        apply_gate(
            _config(), _training(), evaluation, evaluation_code_commit="eval"
        )

    evaluation = _evaluation(save_count=180)
    evaluation["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint hash"):
        apply_gate(
            _config(), _training(), evaluation, evaluation_code_commit="eval"
        )


def test_predeclared_gru_gate_is_no_blackout_only():
    config = yaml.safe_load(
        Path(
            "configs/student/gru_n64_full_pilot_no_blackout_gate_v1.yaml"
        ).read_text()
    )

    assert config["evaluation"]["shot_count"] == 225
    assert config["evaluation"]["blackout_steps"] == [0]
    assert config["no_blackout_gate"]["minimum_save_rate"] == 0.75
    assert config["no_blackout_gate"]["maximum_simulator_or_safety_faults"] == 0


def test_paired_gru_pilot_is_bound_to_successful_visible_gate():
    config = yaml.safe_load(
        Path("configs/student/gru_n64_full_pilot_paired_v1.yaml").read_text()
    )

    assert config["evaluation"]["shot_count"] == 225
    assert config["evaluation"]["blackout_steps"] == [0, 5, 10, 15, 20]
    assert config["provenance"]["qualification_decision"] == "GO"
    assert config["provenance"]["qualification_gate_sha256"] == (
        "284d0051f52f71ca73a3346fb245f5674036175e23279a5bc48bd47a954cdd31"
    )
    assert config["provenance"]["checkpoint_sha256"] == (
        "13cd9a53f3c1470a24fb777c45ea74ca4f48a8906e1ae7c1353ff0ccd5d24d4a"
    )


def _config():
    return {
        "pilot_scope": {
            "classification": "engineering_pilot_not_principal_family_comparison"
        },
        "provenance": {
            "training_code_commit": "train",
            "checkpoint_sha256": "checkpoint",
        },
        "no_blackout_gate": {
            "required_episodes": 225,
            "required_blackout_steps": [0],
            "minimum_save_rate": 0.75,
            "maximum_simulator_or_safety_faults": 0,
            "decision_if_go": "paired",
            "decision_if_no_go": "stop",
        },
    }


def _training():
    return {
        "code_commit": "train",
        "checkpoint_sha256": "checkpoint",
        "training_seed": 14303,
        "selected_epoch": 97,
        "selected_metrics": {"validation": {"action_mse": 0.045}},
    }


def _evaluation(save_count):
    return {
        "code_commit": "eval",
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
