#!/usr/bin/env python3
"""Generate and optionally simulate a versioned direct-launch manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.envs import (
    DefendShotTrackingLoss,
    GeneratedShot,
    load_direct_launch_distribution,
    summarise_distribution,
)
from airhockey_distill.evaluation import InactiveController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/env/direct_launch_v1.yaml"),
    )
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--simulate", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    distribution = load_direct_launch_distribution(args.config)
    shots = distribution.generate(args.split)
    report: dict[str, Any] = {
        "schema_version": 1,
        "distribution_id": distribution.distribution_id,
        "split": args.split,
        "split_purpose": distribution.split(args.split).purpose,
        "summary": summarise_distribution(shots),
        "shots": [shot.as_dict() for shot in shots],
    }
    if args.simulate:
        report["simulation"] = simulate_distribution(
            shots,
            approach_plane_x=distribution.defender_approach_plane_x,
            realised_time_bounds=(
                distribution.realised_approach_time_target_seconds.minimum,
                distribution.realised_approach_time_target_seconds.maximum,
            ),
        )
    return report


def simulate_distribution(
    shots: tuple[GeneratedShot, ...],
    *,
    approach_plane_x: float,
    realised_time_bounds: tuple[float, float],
) -> dict[str, Any]:
    environment = DefendShotTrackingLoss()
    controller = InactiveController()
    arrivals: list[dict[str, Any]] = []
    faults: list[str] = []
    try:
        for generated in shots:
            try:
                observation, _ = environment.reset(shot=generated.shot)
                arrival_step: int | None = None
                outcome = "not_terminal"
                for step in range(1, environment.timeout_steps + 1):
                    observation, _, terminated, truncated, info = environment.step(
                        controller.act(observation)
                    )
                    state = environment.privileged_state()
                    if not np.isfinite(
                        (*state.puck_position_table_xy, *state.puck_velocity_table_xy)
                    ).all():
                        raise FloatingPointError("non-finite puck state")
                    if state.puck_position_table_xy[0] <= approach_plane_x:
                        arrival_step = step
                        break
                    if terminated or truncated:
                        outcome = str(info["outcome"])
                        break
                arrivals.append(
                    {
                        "shot_id": generated.shot.shot_id,
                        "arrival_step": arrival_step,
                        "arrival_time_seconds": (
                            arrival_step / 50.0 if arrival_step is not None else None
                        ),
                        "terminal_outcome_before_arrival": outcome,
                    }
                )
            except Exception as error:  # noqa: BLE001 - preserve fault evidence
                faults.append(
                    f"{generated.shot.shot_id}: {type(error).__name__}: {error}"
                )
    finally:
        environment.close()

    times = tuple(
        float(arrival["arrival_time_seconds"])
        for arrival in arrivals
        if arrival["arrival_time_seconds"] is not None
    )
    lower, upper = realised_time_bounds
    within_bounds = sum(lower <= time <= upper for time in times)
    return {
        "attempted": len(shots),
        "reached_approach_plane": len(times),
        "fault_count": len(faults),
        "faults": faults,
        "realised_time_target_seconds": {"minimum": lower, "maximum": upper},
        "realised_approach_time_seconds": (
            {"minimum": min(times), "maximum": max(times)} if times else None
        ),
        "within_realised_time_target": within_bounds,
        "within_realised_time_target_fraction": (
            within_bounds / len(shots) if shots else 0.0
        ),
        "arrivals": arrivals,
    }


def main() -> None:
    args = parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
