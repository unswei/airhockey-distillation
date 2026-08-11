#!/usr/bin/env python3
"""Evaluate the fail-closed gate for beginning teacher training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.envs import DEFAULT_DIRECT_LAUNCH_SHOT, DefendShotTrackingLoss
from airhockey_distill.evaluation import (
    FixedCentreController,
    InactiveController,
    PrivilegedInterceptController,
    TeacherGateEvidence,
    TeacherGateThresholds,
    audit_public_observation_contract,
    evaluate_teacher_training_gate,
    rollout_privileged_controller,
    rollout_public_controller,
)
from airhockey_distill.evaluation.rollout import EpisodeTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-distinct-shots", type=int, default=200)
    parser.add_argument("--minimum-rate", type=float, default=0.8)
    parser.add_argument("--reliability-episodes", type=int, default=20)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = TeacherGateThresholds(
        minimum_distinct_shots=args.minimum_distinct_shots,
        minimum_inactive_concession_rate=args.minimum_rate,
        minimum_fixed_concession_rate=args.minimum_rate,
        minimum_privileged_save_rate=args.minimum_rate,
        minimum_reliability_episodes=args.reliability_episodes,
    )
    environment = DefendShotTrackingLoss()
    try:
        inactive = InactiveController()
        fixed = FixedCentreController(environment.ee_workspace_xy)
        privileged = PrivilegedInterceptController(environment.ee_workspace_xy)

        inactive_trace = rollout_public_controller(
            environment, inactive, DEFAULT_DIRECT_LAUNCH_SHOT
        )
        fixed_trace = rollout_public_controller(
            environment, fixed, DEFAULT_DIRECT_LAUNCH_SHOT
        )
        privileged_trace = rollout_privileged_controller(
            environment, privileged, DEFAULT_DIRECT_LAUNCH_SHOT
        )

        reliability_traces: list[EpisodeTrace] = []
        faults: list[str] = []
        for episode in range(args.reliability_episodes):
            try:
                trace = rollout_public_controller(
                    environment, fixed, DEFAULT_DIRECT_LAUNCH_SHOT
                )
                _require_finite_trace(trace)
                reliability_traces.append(trace)
            except Exception as error:  # noqa: BLE001 - faults belong in the report
                faults.append(f"episode {episode}: {type(error).__name__}: {error}")

        signatures = {trace.signature() for trace in reliability_traces}
        replay_deterministic = (
            len(reliability_traces) == args.reliability_episodes
            and len(signatures) == 1
            and fixed_trace.signature() in signatures
        )
        evidence = TeacherGateEvidence(
            inactive_traces=(inactive_trace,),
            fixed_traces=(fixed_trace,),
            privileged_traces=(privileged_trace,),
            replay_deterministic=replay_deterministic,
            observation_contract_clean=audit_public_observation_contract(),
            reliability_episodes_attempted=args.reliability_episodes,
            reliability_episodes_completed=len(reliability_traces),
            simulator_faults=tuple(faults),
        )
        report = evaluate_teacher_training_gate(evidence, thresholds).as_dict()
        report["scope"] = {
            "launch_mode": "direct_launch",
            "available_distinct_shots": 1,
            "shot": DEFAULT_DIRECT_LAUNCH_SHOT.as_dict(),
        }
        report["baseline_observations"] = [
            inactive_trace.summary(),
            fixed_trace.summary(),
            privileged_trace.summary(),
        ]
        return report
    finally:
        environment.close()


def _require_finite_trace(trace: EpisodeTrace) -> None:
    arrays = (*trace.observations, *trace.actions)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError(
            "trajectory contains a non-finite observation or action"
        )
    if not np.isfinite(trace.total_reward):
        raise FloatingPointError("trajectory contains a non-finite reward")


def main() -> None:
    args = parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if report["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
