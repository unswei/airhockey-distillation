#!/usr/bin/env python3
"""Run a versioned DreamerV3 teacher training profile on Marvin."""

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
from airhockey_distill.teachers import (
    RetainingCheckpointFactory,
    StepCheckpointClockFactory,
    build_training_arguments,
    list_complete_checkpoints,
)

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
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    if not isinstance(config.get(args.profile), dict):
        raise ValueError(f"unknown teacher training profile {args.profile!r}")
    run_config = config[args.profile]
    training = config["training"]
    output = args.output.resolve()
    metadata_path = output / f"{args.profile}_result.json"
    previous: dict[str, Any] | None = None
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            _validate_resume(previous, args, config)
            if previous.get("status") == "completed":
                return previous
    else:
        output.mkdir(parents=True)

    started_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": (
            previous.get("started_at", started_at.isoformat())
            if previous
            else started_at.isoformat()
        ),
        "code_commit": args.code_commit,
        "profile": args.profile,
        "config": config,
        "attempts": [
            *(previous.get("attempts", []) if previous else []),
            {"started_at": started_at.isoformat()},
        ],
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "mujoco": mujoco.__version__,
        },
    }
    _write_json(metadata_path, metadata)

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=training["distribution_config"],
            reward_config=training["reward_config"],
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
    dreamer_args = build_training_arguments(
        run_config,
        environment_id=ENVIRONMENT_ID,
        logdir=dreamer_logdir,
    )
    metadata["dreamer_arguments"] = dreamer_args
    _write_json(metadata_path, metadata)

    before = monotonic()
    try:
        import elements
        import embodied
        from embodied.envs import from_gymnasium
        from dreamerv3 import main as dreamer_main

        # The pinned fork references elements.Space without importing elements.
        from_gymnasium.elements = elements
        original_checkpoint = elements.Checkpoint
        original_local_clock = embodied.LocalClock
        checkpoint_factory = RetainingCheckpointFactory(
            original_checkpoint,
            keep=int(run_config.get("checkpoint_keep", 1)),
        )
        elements.Checkpoint = checkpoint_factory
        if "checkpoint_every_steps" in run_config:
            embodied.LocalClock = StepCheckpointClockFactory(
                original_local_clock,
                save_every_seconds=float(run_config["save_every"]),
                checkpoint_every_steps=int(run_config["checkpoint_every_steps"]),
            )
        try:
            dreamer_main.main(dreamer_args)
            # The pinned loop otherwise saves only on a wall-clock interval and
            # can leave the retained policy behind the completed run step.
            checkpoint_factory.save_final()
        finally:
            elements.Checkpoint = original_checkpoint
            embodied.LocalClock = original_local_clock
    except BaseException as error:
        metadata["attempts"][-1].update(
            completed_at=datetime.now(UTC).isoformat(),
            status="failed",
        )
        metadata.update(
            status="failed",
            duration_seconds=monotonic() - before,
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_json(metadata_path, metadata)
        raise

    metrics_path = dreamer_logdir / "metrics.jsonl"
    metrics = _read_json_lines(metrics_path)
    training_metric_keys = sorted(
        {
            key
            for record in metrics
            for key in record
            if key.startswith("train/")
        }
    )
    checkpoints = list_complete_checkpoints(dreamer_logdir / "ckpt")
    episode_scores = [
        float(record["episode/score"])
        for record in metrics
        if "episode/score" in record
    ]
    nonzero_episode_scores = [score for score in episode_scores if score != 0.0]
    metadata.update(
        completed_at=datetime.now(UTC).isoformat(),
        duration_seconds=monotonic() - before,
        metrics_lines=len(metrics),
        final_metrics_step=max(
            (int(record.get("step", 0)) for record in metrics),
            default=0,
        ),
        training_metric_keys=training_metric_keys,
        episode_score_count=len(episode_scores),
        nonzero_episode_score_count=len(nonzero_episode_scores),
        episode_score_range=(
            {"minimum": min(episode_scores), "maximum": max(episode_scores)}
            if episode_scores
            else None
        ),
        checkpoint_file_count=sum(
            path.is_file()
            for _, checkpoint in checkpoints
            for path in checkpoint.iterdir()
        ),
        checkpoint_steps=[step for step, _ in checkpoints],
        final_checkpoint_step=(checkpoints[-1][0] if checkpoints else None),
    )
    if not training_metric_keys:
        metadata.update(
            status="failed",
            error={
                "type": "TrainingCriterionError",
                "message": "no train/* metrics were written",
            },
        )
        _write_json(metadata_path, metadata)
        raise RuntimeError("Dreamer run wrote no train/* metrics")
    if not nonzero_episode_scores:
        metadata.update(
            status="failed",
            error={
                "type": "TrainingCriterionError",
                "message": "no non-zero episode score was written",
            },
        )
        _write_json(metadata_path, metadata)
        raise RuntimeError("Dreamer run wrote no non-zero episode score")
    if not checkpoints or checkpoints[-1][0] < int(run_config["steps"]):
        metadata.update(
            status="failed",
            error={
                "type": "TrainingCriterionError",
                "message": "no final checkpoint at the requested training step",
            },
        )
        _write_json(metadata_path, metadata)
        raise RuntimeError("Dreamer run wrote no final checkpoint")

    metadata["status"] = "completed"
    metadata["attempts"][-1].update(
        completed_at=metadata["completed_at"],
        status="completed",
    )
    _write_json(metadata_path, metadata)
    return metadata


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("teacher config must be a mapping")
    for key in ("teacher", "training", "smoke", "evaluation", "provenance"):
        if not isinstance(raw.get(key), dict):
            raise TypeError(f"teacher config section {key!r} must be a mapping")
    return raw


def _validate_resume(
    previous: dict[str, Any],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    expected = {
        "code_commit": args.code_commit,
        "profile": args.profile,
        "config": config,
    }
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f"cannot resume with different {key}")


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
