#!/usr/bin/env python3
"""Run the three controls on the deterministic minimal task slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from airhockey_distill.envs import (
    DEFAULT_DIRECT_LAUNCH_SHOT,
    BlackoutSchedule,
    DefendShotTrackingLoss,
)
from airhockey_distill.evaluation import (
    FixedCentreController,
    InactiveController,
    PrivilegedInterceptController,
    assert_equivalent_replay,
    rollout_privileged_controller,
    rollout_public_controller,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run() -> dict[str, Any]:
    environment = DefendShotTrackingLoss(
        blackout=BlackoutSchedule(start_observation_step=5, length_steps=10)
    )
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
        fixed_replay = rollout_public_controller(
            environment, fixed, DEFAULT_DIRECT_LAUNCH_SHOT
        )
        assert_equivalent_replay(fixed_trace, fixed_replay)

        return {
            "task": "DefendShotTrackingLoss",
            "launch_mode": "direct_launch",
            "public_observation_dimension": 19,
            "public_action_dimension": 2,
            "blackout": {
                "start_observation_step": 5,
                "length_steps": 10,
            },
            "shot": DEFAULT_DIRECT_LAUNCH_SHOT.as_dict(),
            "deterministic_replay": True,
            "baselines": [
                inactive_trace.summary(),
                fixed_trace.summary(),
                privileged_trace.summary(),
            ],
        }
    finally:
        environment.close()


def main() -> None:
    args = parse_args()
    report = run()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
