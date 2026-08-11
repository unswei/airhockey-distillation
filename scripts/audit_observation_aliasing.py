#!/usr/bin/env python3
"""Audit realised public-state aliasing at the locked-prefix blackout onset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.envs import (
    BlackoutSchedule,
    DefendShotTrackingLoss,
    GeneratedShot,
    load_direct_launch_distribution,
)
from airhockey_distill.evaluation import PrivilegedInterceptController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/env/direct_launch_v3.yaml"),
    )
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--maximum-last-visible-separation-metres", type=float, default=0.01
    )
    parser.add_argument("--maximum-hidden-public-difference", type=float, default=1e-6)
    parser.add_argument(
        "--minimum-privileged-action-distance", type=float, default=0.25
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    distribution = load_direct_launch_distribution(args.config)
    aliasing = distribution.observation_aliasing
    if aliasing is None:
        raise ValueError("distribution does not define observation aliasing")
    generated = distribution.generate(args.split)
    alias_shots = [shot for shot in generated if shot.alias_family_id is not None]
    families: dict[str, list[GeneratedShot]] = defaultdict(list)
    for shot in alias_shots:
        if shot.alias_family_id is None:
            raise RuntimeError("alias shot has no family id")
        families[shot.alias_family_id].append(shot)

    environment = DefendShotTrackingLoss(
        blackout=BlackoutSchedule(
            start_observation_step=aliasing.blackout_start_observation_step,
            length_steps=20,
        ),
        action_lock_steps=aliasing.blackout_start_observation_step,
    )
    privileged = PrivilegedInterceptController(environment.ee_workspace_xy)
    records: dict[str, dict[str, Any]] = {}
    try:
        for shot in alias_shots:
            observation, info = environment.reset(shot=shot.shot)
            last_visible = None
            last_visible_state = None
            for _ in range(aliasing.blackout_start_observation_step):
                if int(info["observation_step"]) == (
                    aliasing.blackout_start_observation_step - 1
                ):
                    last_visible = observation.copy()
                    last_visible_state = environment.privileged_state()
                observation, _, terminated, truncated, info = environment.step(
                    (1.0, -1.0)
                )
                if terminated or truncated:
                    raise RuntimeError("alias shot terminated during its locked prefix")
            if last_visible is None or last_visible_state is None:
                raise RuntimeError("last visible state was not captured")
            if bool(info["puck_visible"]) or bool(info["action_locked"]):
                raise RuntimeError("blackout and action release must coincide")
            hidden_state = environment.privileged_state()
            records[shot.shot.shot_id] = {
                "last_visible_observation": last_visible,
                "last_visible_puck_table_xy": np.asarray(
                    last_visible_state.puck_position_table_xy, dtype=np.float64
                ),
                "hidden_observation": observation.copy(),
                "hidden_puck_table_xy": np.asarray(
                    hidden_state.puck_position_table_xy, dtype=np.float64
                ),
                "hidden_puck_velocity_table_xy": np.asarray(
                    hidden_state.puck_velocity_table_xy, dtype=np.float64
                ),
                "privileged_action": privileged.act(observation, hidden_state),
            }
    finally:
        environment.close()

    pair_rows = []
    for family_id, family in sorted(families.items()):
        if len(family) != 2:
            raise ValueError(
                f"alias family must contain exactly two shots: {family_id}"
            )
        first, second = sorted(family, key=lambda shot: shot.target_region)
        first_record = records[first.shot.shot_id]
        second_record = records[second.shot.shot_id]
        pair_rows.append(
            {
                "alias_family_id": family_id,
                "shot_ids": [first.shot.shot_id, second.shot.shot_id],
                "target_regions": [first.target_region, second.target_region],
                "target_separation_metres": abs(
                    first.target_goal_y - second.target_goal_y
                ),
                "last_visible_puck_separation_metres": _distance(
                    first_record["last_visible_puck_table_xy"],
                    second_record["last_visible_puck_table_xy"],
                ),
                "first_hidden_puck_separation_metres": _distance(
                    first_record["hidden_puck_table_xy"],
                    second_record["hidden_puck_table_xy"],
                ),
                "first_hidden_velocity_distance_metres_per_second": _distance(
                    first_record["hidden_puck_velocity_table_xy"],
                    second_record["hidden_puck_velocity_table_xy"],
                ),
                "hidden_public_observation_max_abs_difference": float(
                    np.max(
                        np.abs(
                            first_record["hidden_observation"]
                            - second_record["hidden_observation"]
                        )
                    )
                ),
                "privileged_action_distance": _distance(
                    first_record["privileged_action"],
                    second_record["privileged_action"],
                ),
            }
        )

    aggregate = {
        "families": len(pair_rows),
        "shots": len(alias_shots),
        "minimum_target_separation_metres": min(
            row["target_separation_metres"] for row in pair_rows
        ),
        "maximum_last_visible_puck_separation_metres": max(
            row["last_visible_puck_separation_metres"] for row in pair_rows
        ),
        "maximum_first_hidden_puck_separation_metres": max(
            row["first_hidden_puck_separation_metres"] for row in pair_rows
        ),
        "minimum_first_hidden_velocity_distance_metres_per_second": min(
            row["first_hidden_velocity_distance_metres_per_second"] for row in pair_rows
        ),
        "maximum_hidden_public_observation_abs_difference": max(
            row["hidden_public_observation_max_abs_difference"] for row in pair_rows
        ),
        "minimum_privileged_action_distance": min(
            row["privileged_action_distance"] for row in pair_rows
        ),
    }
    checks = [
        _check("at_least_90_alias_families", aggregate["families"] >= 90),
        _check(
            "last_visible_positions_are_close",
            aggregate["maximum_last_visible_puck_separation_metres"]
            <= args.maximum_last_visible_separation_metres,
        ),
        _check(
            "hidden_public_observations_match",
            aggregate["maximum_hidden_public_observation_abs_difference"]
            <= args.maximum_hidden_public_difference,
        ),
        _check(
            "required_actions_diverge",
            aggregate["minimum_privileged_action_distance"]
            >= args.minimum_privileged_action_distance,
        ),
    ]
    blocking = [check["check_id"] for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "distribution_id": distribution.distribution_id,
        "split": args.split,
        "decision": "GO" if not blocking else "NO_GO",
        "thresholds": {
            "minimum_alias_families": 90,
            "maximum_last_visible_separation_metres": args.maximum_last_visible_separation_metres,
            "maximum_hidden_public_difference": args.maximum_hidden_public_difference,
            "minimum_privileged_action_distance": args.minimum_privileged_action_distance,
        },
        "aggregate": aggregate,
        "checks": checks,
        "blocking_checks": blocking,
        "pairs": pair_rows,
    }


def _distance(first: Any, second: Any) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def _check(check_id: str, passed: bool) -> dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed)}


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
