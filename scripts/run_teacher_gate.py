#!/usr/bin/env python3
"""Evaluate the fail-closed gate for beginning teacher training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.envs import (
    DefendShotTrackingLoss,
    load_direct_launch_distribution,
    summarise_distribution,
)
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
    parser.add_argument(
        "--distribution-config",
        type=Path,
        default=Path("configs/env/direct_launch_v2.yaml"),
    )
    parser.add_argument("--distribution-split", default="calibration")
    parser.add_argument("--minimum-distinct-shots", type=int, default=200)
    parser.add_argument("--minimum-rate", type=float, default=0.8)
    parser.add_argument("--reliability-episodes", type=int, default=20)
    parser.add_argument("--action-lock-steps", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    distribution = load_direct_launch_distribution(args.distribution_config)
    generated_shots = distribution.generate(args.distribution_split)
    thresholds = TeacherGateThresholds(
        minimum_distinct_shots=args.minimum_distinct_shots,
        minimum_inactive_concession_rate=args.minimum_rate,
        minimum_fixed_concession_rate=args.minimum_rate,
        minimum_privileged_save_rate=args.minimum_rate,
        minimum_reliability_episodes=args.reliability_episodes,
    )
    environment = DefendShotTrackingLoss(action_lock_steps=args.action_lock_steps)
    try:
        inactive = InactiveController()
        fixed = FixedCentreController(environment.ee_workspace_xy)
        privileged = PrivilegedInterceptController(environment.ee_workspace_xy)

        inactive_traces: list[EpisodeTrace] = []
        fixed_traces: list[EpisodeTrace] = []
        privileged_traces: list[EpisodeTrace] = []
        for generated in generated_shots:
            inactive_traces.append(
                rollout_public_controller(environment, inactive, generated.shot)
            )
            fixed_traces.append(
                rollout_public_controller(environment, fixed, generated.shot)
            )
            privileged_traces.append(
                rollout_privileged_controller(environment, privileged, generated.shot)
            )

        reliability_traces: list[EpisodeTrace] = []
        faults: list[str] = []
        reliability_shot = generated_shots[0].shot
        for episode in range(args.reliability_episodes):
            try:
                trace = rollout_public_controller(environment, fixed, reliability_shot)
                _require_finite_trace(trace)
                reliability_traces.append(trace)
            except Exception as error:  # noqa: BLE001 - faults belong in the report
                faults.append(f"episode {episode}: {type(error).__name__}: {error}")

        signatures = {trace.signature() for trace in reliability_traces}
        replay_deterministic = (
            len(reliability_traces) == args.reliability_episodes
            and len(signatures) == 1
            and fixed_traces[0].signature() in signatures
        )
        evidence = TeacherGateEvidence(
            inactive_traces=tuple(inactive_traces),
            fixed_traces=tuple(fixed_traces),
            privileged_traces=tuple(privileged_traces),
            replay_deterministic=replay_deterministic,
            observation_contract_clean=audit_public_observation_contract(),
            reliability_episodes_attempted=args.reliability_episodes,
            reliability_episodes_completed=len(reliability_traces),
            simulator_faults=tuple(faults),
        )
        report = evaluate_teacher_training_gate(evidence, thresholds).as_dict()
        report["scope"] = {
            "launch_mode": "direct_launch",
            "distribution_id": distribution.distribution_id,
            "distribution_split": args.distribution_split,
            "available_distinct_shots": len(generated_shots),
            "distribution_summary": summarise_distribution(generated_shots),
            "reliability_shot_id": reliability_shot.shot_id,
        }
        report["baseline_outcome_counts"] = {
            "inactive": _outcome_counts(inactive_traces),
            "fixed_centre": _outcome_counts(fixed_traces),
            "privileged_intercept": _outcome_counts(privileged_traces),
        }
        report["baseline_outcome_counts_by_region_pair"] = {
            "inactive": _outcome_counts_by_region_pair(
                generated_shots,
                inactive_traces,
            ),
            "fixed_centre": _outcome_counts_by_region_pair(
                generated_shots,
                fixed_traces,
            ),
            "privileged_intercept": _outcome_counts_by_region_pair(
                generated_shots,
                privileged_traces,
            ),
        }
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


def _outcome_counts(traces: list[EpisodeTrace]) -> dict[str, int]:
    return dict(sorted(Counter(trace.outcome for trace in traces).items()))


def _outcome_counts_by_region_pair(
    generated_shots: tuple[Any, ...],
    traces: list[EpisodeTrace],
) -> dict[str, dict[str, int]]:
    if len(generated_shots) != len(traces):
        raise ValueError("generated shots and traces must have equal lengths")
    counts: dict[str, Counter[str]] = {}
    for generated, trace in zip(generated_shots, traces, strict=True):
        if generated.shot.shot_id != trace.shot_id:
            raise ValueError("generated shot and trace order must match")
        pair = f"{generated.launch_region}->{generated.target_region}"
        counts.setdefault(pair, Counter())[trace.outcome] += 1
    return {
        pair: dict(sorted(outcomes.items()))
        for pair, outcomes in sorted(counts.items())
    }


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
