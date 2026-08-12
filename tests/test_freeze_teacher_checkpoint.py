import argparse
import json
import pickle

from scripts.freeze_teacher_checkpoint import run


def test_frozen_checkpoint_records_explicit_teacher_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "agent.pkl").write_bytes(b"agent")
    (checkpoint / "step.pkl").write_bytes(pickle.dumps(700_000))
    (checkpoint / "done").touch()
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "code_commit": "training-commit",
                "profile": "full",
                "selected_checkpoint_step": 700_000,
                "selection_metric": "validation_save_rate_across_blackouts",
                "selected_summary": {"save_rate": 0.98},
            }
        )
    )
    config = tmp_path / "teacher.yaml"
    config.write_text("schema_version: 1\n")
    output = tmp_path / "frozen"

    manifest = run(
        argparse.Namespace(
            checkpoint=checkpoint,
            selection=selection,
            config=config,
            output=output,
            code_commit="freezing-commit",
            teacher_id="dreamerv3_teacher_v3",
        )
    )

    assert manifest["teacher_id"] == "dreamerv3_teacher_v3"
    assert manifest["checkpoint_step"] == 700_000
    assert manifest["files"]["agent.pkl"]["sha256"]
    assert json.loads((output / "manifest.json").read_text()) == manifest
