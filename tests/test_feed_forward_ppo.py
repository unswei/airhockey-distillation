import argparse
import json
from pathlib import Path

import pytest
import yaml

from scripts.evaluate_feed_forward_ppo import _validate_training_result
from scripts.qualify_feed_forward_baseline import run as run_qualification
from scripts.train_feed_forward_ppo import _latest_checkpoint, _validate_seed


def test_latest_ppo_checkpoint_uses_largest_timestep(tmp_path):
    (tmp_path / "ppo_100_steps.zip").touch()
    expected = tmp_path / "ppo_900_steps.zip"
    expected.touch()
    (tmp_path / "unrelated.zip").touch()

    assert _latest_checkpoint(tmp_path) == expected


def test_ppo_seed_must_be_predeclared():
    _validate_seed({"seeds": [3, 5]}, 5)
    with pytest.raises(ValueError, match="not predeclared"):
        _validate_seed({"seeds": [3, 5]}, 4)


def test_ppo_training_result_binds_checkpoint_and_config(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"model")
    from scripts.evaluate_feed_forward_ppo import _sha256

    result = {
        "status": "completed",
        "policy": "feed_forward_ppo",
        "profile": "full",
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": "config-hash",
    }
    _validate_training_result(result, checkpoint, "config-hash")

    with pytest.raises(ValueError, match="different config"):
        _validate_training_result(result, checkpoint, "another-hash")


def test_visible_baseline_qualification_selects_best_predeclared_seed(tmp_path):
    config = {
        "full": {"seeds": [11, 12, 13]},
        "qualification_evaluation": {"shot_count": 10},
        "baseline_qualification": {
            "selection_metric": "no_blackout_save_rate",
            "tie_breaks": ["mean_score", "lower_training_seed"],
            "minimum_feed_forward_no_blackout_save_rate": 0.75,
            "maximum_no_blackout_teacher_advantage": 0.10,
            "maximum_simulator_faults": 0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    teacher_path = tmp_path / "teacher.json"
    teacher_path.write_text(json.dumps({"episodes": _episodes(10)}))
    baseline_paths = []
    for seed, saves in ((11, 8), (12, 9), (13, 9)):
        path = tmp_path / f"baseline-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "policy": "feed_forward_ppo",
                    "evaluation": "qualification_evaluation",
                    "memoryless": True,
                    "training_seed": seed,
                    "checkpoint": f"checkpoint-{seed}.zip",
                    "checkpoint_sha256": f"hash-{seed}",
                    "summary": {
                        "save_rate": saves / 10,
                        "mean_score": saves / 10,
                    },
                    "episodes": _episodes(saves),
                }
            )
        )
        baseline_paths.append(path)

    report = run_qualification(
        argparse.Namespace(
            config=config_path,
            teacher_result=teacher_path,
            baseline_result=baseline_paths,
            output=tmp_path / "qualification.json",
            code_commit="test-commit",
        )
    )

    assert report["decision"] == "GO"
    assert report["selected_training_seed"] == 12
    assert report["observed"]["feed_forward_save_rate"] == 0.9
    assert report["blocking_checks"] == []


def _episodes(save_count: int) -> list[dict[str, object]]:
    return [
        {
            "shot_id": f"shot-{index}",
            "blackout_steps": 0,
            "outcome": "returned" if index < save_count else "goal_conceded",
            "score": 1.0 if index < save_count else -1.0,
            "steps": 10,
        }
        for index in range(10)
    ]
