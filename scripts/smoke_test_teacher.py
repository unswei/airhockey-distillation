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
    parser.add_argument("--profile", choices=("smoke", "diagnostic"), default="smoke")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    run_config = config[args.profile]
    training = config["training"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "code_commit": args.code_commit,
        "profile": args.profile,
        "config": config,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "mujoco": mujoco.__version__,
        },
    }
    metadata_path = output / f"{args.profile}_result.json"
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
    dreamer_args = [
        "--configs",
        str(run_config["model_preset"]),
        "--task",
        f"gymnasium_{ENVIRONMENT_ID}",
        "--logdir",
        str(dreamer_logdir),
        "--seed",
        str(run_config["seed"]),
        "--batch_size",
        str(run_config["batch_size"]),
        "--batch_length",
        str(run_config["batch_length"]),
        "--report_length",
        str(run_config["report_length"]),
        "--replay.size",
        str(run_config["replay_size"]),
        "--run.steps",
        str(run_config["steps"]),
        "--run.train_ratio",
        str(run_config["train_ratio"]),
        "--run.envs",
        str(run_config.get("log_every", 1)),
        "--run.log_every",
        "1",
        "--run.report_every",
        str(run_config.get("report_every", 3600)),
        "--run.save_every",
        str(run_config.get("save_every", 1)),
        "--run.debug",
        "True",
        "--jax.platform",
        "cuda",
        "--jax.prealloc",
        "False",
        "--logger.outputs",
        "jsonl",
        "--errfile",
        "True",
    ]
    metadata["dreamer_arguments"] = dreamer_args
    _write_json(metadata_path, metadata)

    before = monotonic()
    try:
        import elements
        from embodied.envs import from_gymnasium
        from dreamerv3 import main as dreamer_main

        # The pinned fork references elements.Space without importing elements.
        from_gymnasium.elements = elements
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
    metrics = _read_json_lines(metrics_path)
    training_metric_keys = sorted(
        {
            key
            for record in metrics
            for key in record
            if key.startswith("train/")
        }
    )
    checkpoints = tuple((dreamer_logdir / "ckpt").glob("**/*"))
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
        checkpoint_file_count=sum(path.is_file() for path in checkpoints),
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

    metadata["status"] = "completed"
    _write_json(metadata_path, metadata)
    return metadata


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("teacher config must be a mapping")
    for key in ("teacher", "training", "smoke", "diagnostic", "evaluation", "provenance"):
        if not isinstance(raw.get(key), dict):
            raise TypeError(f"teacher config section {key!r} must be a mapping")
    return raw


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
