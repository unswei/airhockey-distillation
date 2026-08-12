#!/usr/bin/env python3
"""Evaluate a memoryless PPO policy on fixed direct-launch episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from airhockey_distill.envs import (
    BlackoutSchedule,
    DefenceRewardTracker,
    DefendShotTrackingLoss,
    load_defence_reward,
    load_direct_launch_distribution,
)

SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evaluation",
        choices=("qualification_evaluation", "confirmation_evaluation"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco
    import stable_baselines3
    from stable_baselines3 import PPO

    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    evaluation = config[args.evaluation]
    task = config["task"]
    checkpoint = args.checkpoint.resolve()
    training_result_path = args.training_result.resolve()
    training_result = json.loads(training_result_path.read_text())
    _validate_training_result(training_result, checkpoint, _sha256(config_path))

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(checkpoint, device="cpu")
    if model.observation_space.shape != (19,) or model.action_space.shape != (2,):
        raise ValueError("PPO checkpoint does not use the public 19-by-2 interface")

    distribution = load_direct_launch_distribution(
        evaluation["distribution_config"]
    )
    generated = distribution.generate(evaluation["distribution_split"])
    selected = _select_evenly(generated, int(evaluation["shot_count"]))
    blackouts = tuple(int(value) for value in evaluation["blackout_steps"])
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(
            load_defence_reward(task["reward_config"])
        ),
        action_lock_steps=int(task["action_lock_steps"]),
    )

    episodes: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    try:
        for generated_shot in selected:
            for blackout_steps in blackouts:
                environment.blackout = BlackoutSchedule(
                    start_observation_step=int(
                        task["blackout_start_observation_step"]
                    ),
                    length_steps=blackout_steps,
                )
                observation, _ = environment.reset(shot=generated_shot.shot)
                score = 0.0
                steps = 0
                outcome = "rollout_limit"
                while steps < environment.timeout_steps:
                    before = perf_counter()
                    action, state = model.predict(observation, deterministic=True)
                    inference_seconds.append(perf_counter() - before)
                    if state is not None:
                        raise RuntimeError("feed-forward PPO unexpectedly returned state")
                    observation, reward, terminated, truncated, info = (
                        environment.step(np.asarray(action, dtype=np.float32))
                    )
                    score += float(reward)
                    steps += 1
                    if terminated or truncated:
                        outcome = str(info["outcome"])
                        break
                episodes.append(
                    {
                        "shot_id": generated_shot.shot.shot_id,
                        "launch_region": generated_shot.launch_region,
                        "target_region": generated_shot.target_region,
                        "blackout_steps": blackout_steps,
                        "outcome": outcome,
                        "score": score,
                        "steps": steps,
                    }
                )
    finally:
        environment.close()

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "feed_forward_ppo",
        "memoryless": True,
        "deterministic_evaluation": True,
        "training_seed": int(training_result["training_seed"]),
        "evaluation": args.evaluation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_result": str(training_result_path),
        "training_result_sha256": _sha256(training_result_path),
        "code_commit": args.code_commit,
        "config_sha256": _sha256(config_path),
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "mujoco": mujoco.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "summary": _summarise(episodes, inference_seconds),
        "episodes": episodes,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _validate_training_result(
    result: dict[str, Any], checkpoint: Path, config_sha256: str
) -> None:
    if result.get("status") != "completed" or result.get("policy") != (
        "feed_forward_ppo"
    ):
        raise ValueError("PPO training result is incomplete or has the wrong policy")
    if result.get("profile") != "full":
        raise ValueError("only a full-profile PPO checkpoint may be evaluated")
    if _sha256(checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError("PPO checkpoint hash does not match its training result")
    if result.get("config_sha256") != config_sha256:
        raise ValueError("PPO checkpoint was trained with a different config")


def _select_evenly(values: tuple[Any, ...], count: int) -> tuple[Any, ...]:
    if count <= 0 or count > len(values):
        raise ValueError("evaluation shot count must lie inside the split size")
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    if len({int(index) for index in indices}) != count:
        raise ValueError("evaluation shot selection produced duplicate indices")
    return tuple(values[int(index)] for index in indices)


def _summarise(
    episodes: list[dict[str, Any]], inference_seconds: list[float]
) -> dict[str, Any]:
    overall = _episode_summary(episodes)
    by_blackout = {
        str(blackout): _episode_summary(
            [episode for episode in episodes if episode["blackout_steps"] == blackout]
        )
        for blackout in sorted({episode["blackout_steps"] for episode in episodes})
    }
    milliseconds = np.asarray(inference_seconds, dtype=np.float64) * 1000.0
    return {
        **overall,
        "by_blackout_steps": by_blackout,
        "inference_milliseconds": {
            "median": float(np.median(milliseconds)),
            "p95": float(np.quantile(milliseconds, 0.95)),
            "maximum": float(np.max(milliseconds)),
        },
    }


def _episode_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(str(episode["outcome"]) for episode in episodes)
    saves = sum(outcomes[outcome] for outcome in SAVE_OUTCOMES)
    concessions = outcomes["goal_conceded"]
    return {
        "episodes": len(episodes),
        "save_count": saves,
        "save_rate": saves / len(episodes),
        "concession_count": concessions,
        "concession_rate": concessions / len(episodes),
        "mean_score": float(np.mean([episode["score"] for episode in episodes])),
        "outcome_counts": dict(sorted(outcomes.items())),
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
