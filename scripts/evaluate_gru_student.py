#!/usr/bin/env python3
"""Evaluate a frozen GRU-64 student on paired validation episodes."""

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

import mujoco
import numpy as np
import yaml

from airhockey_distill.envs import (
    BlackoutSchedule,
    DefenceRewardTracker,
    DefendShotTrackingLoss,
    load_defence_reward,
    load_direct_launch_distribution,
)
from airhockey_distill.students import GRURecurrentPolicy

SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_mapping(args.config.resolve())
    evaluation = config["evaluation"]
    dataset = config["dataset"]
    checkpoint = args.checkpoint.resolve()
    if _sha256(checkpoint) != config["provenance"]["checkpoint_sha256"]:
        raise ValueError("frozen GRU checkpoint hash mismatch")
    policy = GRURecurrentPolicy.load(checkpoint)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    distribution = load_direct_launch_distribution(evaluation["distribution_config"])
    generated = distribution.generate(evaluation["distribution_split"])
    selected = _select_evenly(generated, int(evaluation["shot_count"]))
    blackouts = tuple(int(value) for value in evaluation["blackout_steps"])
    reward_specification = load_defence_reward(dataset["reward_config"])
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(reward_specification),
        action_lock_steps=int(dataset.get("action_lock_steps", 0)),
    )

    episodes: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    try:
        for generated_shot in selected:
            for blackout_steps in blackouts:
                environment.blackout = BlackoutSchedule(
                    start_observation_step=int(
                        dataset["blackout_start_observation_step"]
                    ),
                    length_steps=blackout_steps,
                )
                observation, _ = environment.reset(shot=generated_shot.shot)
                carry = policy.initial_carry()
                score = 0.0
                steps = 0
                outcome = "rollout_limit"
                while steps < environment.timeout_steps:
                    before = perf_counter()
                    action, carry = policy.act(observation, carry)
                    inference_seconds.append(perf_counter() - before)
                    observation, reward, terminated, truncated, info = environment.step(
                        action
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
        "policy": "gru_n64",
        "pilot_scope": config["pilot_scope"],
        "policy_metadata": policy.metadata,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "code_commit": args.code_commit,
        "config_sha256": _sha256(args.config.resolve()),
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "mujoco": mujoco.__version__,
        },
        "summary": summarise(episodes, inference_seconds),
        "episodes": episodes,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _select_evenly(values: tuple[Any, ...], count: int) -> tuple[Any, ...]:
    if count <= 0 or count > len(values):
        raise ValueError("evaluation shot count must lie inside the split size")
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    if len({int(index) for index in indices}) != count:
        raise ValueError("evaluation shot selection produced duplicate indices")
    return tuple(values[int(index)] for index in indices)


def summarise(
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
