#!/usr/bin/env python3
"""Evaluate any principal family on the canonical paired validation schedule."""

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

import numpy as np

from airhockey_distill.envs import (
    BlackoutSchedule,
    DefenceRewardTracker,
    DefendShotTrackingLoss,
    load_defence_reward,
)
from airhockey_distill.principal_sweep import (
    evaluation_schedule,
    evaluation_schedule_sha256,
    load_principal_protocol,
    principal_family_spec,
    sha256_file,
    validate_training_seed,
)
from airhockey_distill.students import load_principal_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    principal_family_spec(protocol, args.family)
    validate_training_seed(protocol, args.seed)
    checkpoint = args.checkpoint.resolve()
    policy = load_principal_policy(args.family, checkpoint)
    if policy.metadata.get("student_id") != args.family:
        raise ValueError("checkpoint family mismatch")
    if int(policy.metadata.get("training_seed", -1)) != args.seed:
        raise ValueError("checkpoint seed mismatch")
    if policy.metadata.get("training_stage") != "final":
        raise ValueError("principal evaluation requires a final-stage checkpoint")
    if policy.metadata.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("checkpoint protocol hash mismatch")

    evaluation = protocol["evaluation"]["validation"]
    reward_specification = load_defence_reward(
        protocol["shadow_labelling"]["reward_config"]
    )
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(reward_specification),
        action_lock_steps=int(protocol["shadow_labelling"]["action_lock_steps"]),
    )
    episodes: list[dict[str, Any]] = []
    inference_seconds = []
    try:
        for schedule_index, (generated, blackout_steps) in enumerate(
            evaluation_schedule(protocol)
        ):
            environment.blackout = BlackoutSchedule(
                start_observation_step=int(
                    protocol["shadow_labelling"][
                        "blackout_start_observation_step"
                    ]
                ),
                length_steps=blackout_steps,
            )
            observation, _ = environment.reset(shot=generated.shot)
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
                    "schedule_index": schedule_index,
                    "shot_id": generated.shot.shot_id,
                    "alias_family_id": generated.alias_family_id,
                    "launch_region": generated.launch_region,
                    "target_region": generated.target_region,
                    "blackout_steps": blackout_steps,
                    "outcome": outcome,
                    "score": score,
                    "steps": steps,
                }
            )
    finally:
        environment.close()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "family_id": args.family,
        "training_seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "evaluation_split": evaluation["split"],
        "evaluation_schedule_sha256": evaluation_schedule_sha256(protocol),
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "mujoco": mujoco.__version__,
        },
        "summary": summarise(episodes, inference_seconds, protocol),
        "episodes": episodes,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarise(
    episodes: list[dict[str, Any]],
    inference_seconds: list[float],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    save_outcomes = frozenset(protocol["evaluation"]["save_outcomes"])

    def episode_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        outcomes = Counter(str(value["outcome"]) for value in selected)
        saves = sum(outcomes[value] for value in save_outcomes)
        return {
            "episodes": len(selected),
            "save_count": saves,
            "save_rate": saves / len(selected),
            "concession_count": outcomes["goal_conceded"],
            "outcome_counts": dict(sorted(outcomes.items())),
            "mean_score": float(np.mean([value["score"] for value in selected])),
        }

    milliseconds = np.asarray(inference_seconds, dtype=np.float64) * 1000.0
    return {
        **episode_summary(episodes),
        "by_blackout_steps": {
            str(blackout): episode_summary(
                [value for value in episodes if value["blackout_steps"] == blackout]
            )
            for blackout in protocol["evaluation"]["validation"]["blackout_steps"]
        },
        "inference_milliseconds": {
            "median": float(np.median(milliseconds)),
            "p95": float(np.quantile(milliseconds, 0.95)),
            "maximum": float(np.max(milliseconds)),
        },
    }


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
