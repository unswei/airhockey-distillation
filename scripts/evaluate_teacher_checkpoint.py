#!/usr/bin/env python3
"""Evaluate an untrained or checkpointed Dreamer policy on paired validation cases."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import gymnasium
import jax
import mujoco
import numpy as np
import yaml
from ruamel import yaml as ruamel_yaml

from airhockey_distill.envs import (
    BlackoutSchedule,
    DefenceRewardTracker,
    DefendShotTrackingLoss,
    DirectLaunchTrainingEnv,
    load_defence_reward,
    load_direct_launch_distribution,
)
from airhockey_distill.teachers import read_checkpoint_step

ENVIRONMENT_ID = "AirHockeyDefendShotEvaluation-v0"
SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher/dreamerv3.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--profile", default="diagnostic")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    if not isinstance(config.get(args.profile), dict):
        raise ValueError(f"unknown teacher profile {args.profile!r}")
    profile = config[args.profile]
    evaluation_name = str(profile.get("evaluation", "evaluation"))
    evaluation = config[evaluation_name]
    training = config["training"]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is not None and not (checkpoint / "done").exists():
        raise FileNotFoundError(f"incomplete checkpoint {checkpoint}")

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=evaluation["distribution_config"],
            reward_config=training["reward_config"],
            split=evaluation["distribution_split"],
            sampling_seed=int(training["sampling_seed"]),
            minimum_blackout_steps=0,
            maximum_blackout_steps=0,
            **kwargs,
        )

    gymnasium.register(id=ENVIRONMENT_ID, entry_point=make_environment)

    import elements
    import portal
    from dreamerv3 import agent as dreamer_agent
    from dreamerv3 import main as dreamer_main
    from embodied.envs import from_gymnasium

    from_gymnasium.elements = elements
    upstream = ruamel_yaml.YAML(typ="safe").load(
        (Path(dreamer_agent.__file__).parent / "configs.yaml").read_text()
    )
    dreamer_config = elements.Config(upstream["defaults"])
    dreamer_config = dreamer_config.update(
        upstream[str(profile["model_preset"])]
    )
    dreamer_config = dreamer_config.update(
        {
            "task": f"gymnasium_{ENVIRONMENT_ID}",
            "logdir": str(output.parent / f"{output.stem}-runtime"),
            "seed": int(profile["seed"]),
            "batch_size": int(profile["batch_size"]),
            "batch_length": int(profile["batch_length"]),
            "report_length": int(profile["report_length"]),
            "jax.platform": "cuda",
            "jax.prealloc": False,
            "run.envs": 1,
            "run.debug": True,
        }
    )

    portal.setup(
        errfile=False,
        clientkw={"logging_color": "cyan"},
        serverkw={"logging_color": "cyan"},
        ipv6=False,
    )
    agent = dreamer_main.make_agent(dreamer_config)
    if checkpoint is not None:
        loader = elements.Checkpoint()
        loader.agent = agent
        loader.load(checkpoint, keys=["agent"])

    distribution = load_direct_launch_distribution(
        evaluation["distribution_config"]
    )
    generated = distribution.generate(evaluation["distribution_split"])
    selected = _select_evenly(generated, int(evaluation["shot_count"]))
    blackouts = tuple(int(value) for value in evaluation["blackout_steps"])
    reward_specification = load_defence_reward(training["reward_config"])
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(reward_specification)
    )

    episodes: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    try:
        for generated_shot in selected:
            for blackout_steps in blackouts:
                environment.blackout = BlackoutSchedule(
                    start_observation_step=int(
                        training["blackout_start_observation_step"]
                    ),
                    length_steps=blackout_steps,
                )
                observation, _ = environment.reset(shot=generated_shot.shot)
                carry = agent.init_policy(1)
                reward = 0.0
                score = 0.0
                steps = 0
                outcome = "rollout_limit"
                is_first = True
                while steps < environment.timeout_steps:
                    policy_observation = {
                        "image": np.asarray([observation], dtype=np.float32),
                        "reward": np.asarray([reward], dtype=np.float32),
                        "is_first": np.asarray([is_first], dtype=bool),
                        "is_last": np.asarray([False], dtype=bool),
                        "is_terminal": np.asarray([False], dtype=bool),
                    }
                    before = perf_counter()
                    carry, actions, _ = agent.policy(
                        carry,
                        policy_observation,
                        mode="eval",
                    )
                    inference_seconds.append(perf_counter() - before)
                    action = np.asarray(actions["action"])[0]
                    observation, reward, terminated, truncated, info = (
                        environment.step(action)
                    )
                    score += float(reward)
                    steps += 1
                    is_first = False
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
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "checkpoint" if checkpoint is not None else "untrained",
        "profile": args.profile,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_step": (
            read_checkpoint_step(checkpoint) if checkpoint is not None else None
        ),
        "code_commit": args.code_commit,
        "config": config,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "mujoco": mujoco.__version__,
        },
        "summary": _summarise(episodes, inference_seconds),
        "episodes": episodes,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _select_evenly(values: tuple[Any, ...], count: int) -> tuple[Any, ...]:
    if count <= 0 or count > len(values):
        raise ValueError("evaluation shot count must lie inside the split size")
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    if len(set(int(index) for index in indices)) != count:
        raise ValueError("evaluation shot selection produced duplicate indices")
    return tuple(values[int(index)] for index in indices)


def _summarise(
    episodes: list[dict[str, Any]],
    inference_seconds: list[float],
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


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("teacher config must be a mapping")
    for key in ("teacher", "training", "evaluation", "provenance"):
        if not isinstance(raw.get(key), dict):
            raise TypeError(f"teacher config section {key!r} must be a mapping")
    return raw


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
