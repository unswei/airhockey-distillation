#!/usr/bin/env python3
"""Run a bounded DreamerV3 training compatibility test on Marvin."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import gymnasium
import jax
import mujoco
import yaml

from airhockey_distill.envs import DirectLaunchTrainingEnv

ENVIRONMENT_ID = "AirHockeyDefendShotTrackingLoss-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher/dreamerv3.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    smoke = config["smoke"]
    training = config["training"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "code_commit": args.code_commit,
        "config": config,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "mujoco": mujoco.__version__,
        },
    }
    metadata_path = output / "smoke_result.json"
    _write_json(metadata_path, metadata)

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=training["distribution_config"],
            split=training["distribution_split"],
            sampling_seed=int(training["sampling_seed"]),
            blackout_start_observation_step=int(
                training["blackout_start_observation_step"]
            ),
            minimum_blackout_steps=int(training["minimum_blackout_steps"]),
            maximum_blackout_steps=int(training["maximum_blackout_steps"]),
            **kwargs,
        )

    gymnasium.register(id=ENVIRONMENT_ID, entry_point=make_environment)

    dreamer_logdir = output / "dreamer"
    dreamer_args = [
        "--configs",
        str(smoke["model_preset"]),
        "--task",
        f"gymnasium_{ENVIRONMENT_ID}",
        "--logdir",
        str(dreamer_logdir),
        "--seed",
        str(smoke["seed"]),
        "--batch_size",
        str(smoke["batch_size"]),
        "--batch_length",
        str(smoke["batch_length"]),
        "--report_length",
        str(smoke["report_length"]),
        "--replay.size",
        str(smoke["replay_size"]),
        "--run.steps",
        str(smoke["steps"]),
        "--run.train_ratio",
        str(smoke["train_ratio"]),
        "--run.envs",
        "1",
        "--run.log_every",
        "1",
        "--run.report_every",
        "3600",
        "--run.save_every",
        "1",
        "--run.debug",
        "true",
        "--jax.platform",
        "cuda",
        "--jax.prealloc",
        "false",
        "--logger.outputs",
        "jsonl",
        "--errfile",
        "true",
    ]
    metadata["dreamer_arguments"] = dreamer_args
    _write_json(metadata_path, metadata)

    before = monotonic()
    try:
        from dreamerv3 import main as dreamer_main

        dreamer_main.main(dreamer_args)
    except BaseException as error:
        metadata.update(
            status="failed",
            duration_seconds=monotonic() - before,
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(metadata_path, metadata)
        raise

    metrics_path = dreamer_logdir / "metrics.jsonl"
    checkpoints = tuple((dreamer_logdir / "ckpt").glob("**/*"))
    metadata.update(
        status="completed",
        completed_at=datetime.now(UTC).isoformat(),
        duration_seconds=monotonic() - before,
        metrics_lines=_line_count(metrics_path),
        checkpoint_file_count=sum(path.is_file() for path in checkpoints),
    )
    _write_json(metadata_path, metadata)
    return metadata


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("teacher config must be a mapping")
    for key in ("teacher", "training", "smoke", "provenance"):
        if not isinstance(raw.get(key), dict):
            raise TypeError(f"teacher config section {key!r} must be a mapping")
    return raw


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(1 for _ in stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
