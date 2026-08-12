#!/usr/bin/env python3
"""Freeze a validation-selected Dreamer teacher checkpoint with hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airhockey_distill.teachers import read_checkpoint_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--teacher-id", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    selection_path = args.selection.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for name in ("agent.pkl", "step.pkl", "done"):
        if not (checkpoint / name).exists():
            raise FileNotFoundError(checkpoint / name)

    selection = json.loads(selection_path.read_text())
    step = read_checkpoint_step(checkpoint)
    if step != int(selection["selected_checkpoint_step"]):
        raise ValueError("checkpoint step does not match the selection report")

    output.mkdir(parents=True)
    for name in ("agent.pkl", "step.pkl", "done"):
        shutil.copy2(checkpoint / name, output / name)

    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "teacher_id": args.teacher_id,
        "frozen_at": datetime.now(UTC).isoformat(),
        "freezing_code_commit": args.code_commit,
        "training_code_commit": selection["code_commit"],
        "training_profile": selection["profile"],
        "checkpoint_step": step,
        "selection_metric": selection["selection_metric"],
        "selection_summary": selection["selected_summary"],
        "source": {
            "checkpoint": str(checkpoint),
            "selection": str(selection_path),
            "config": str(config_path),
        },
        "source_hashes": {
            "selection.json": _sha256(selection_path),
            config_path.name: _sha256(config_path),
        },
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in ("agent.pkl", "step.pkl", "done")
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for path in output.iterdir():
        os.chmod(path, 0o444)
    os.chmod(output, 0o555)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
