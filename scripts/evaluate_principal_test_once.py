#!/usr/bin/env python3
"""Open the released principal test once and evaluate every frozen policy."""

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
from airhockey_distill.principal_release import (
    principal_test_schedule,
    principal_test_schedule_sha256,
    validate_go_release_report,
)
from airhockey_distill.principal_sweep import (
    load_principal_protocol,
    sha256_file,
)
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS, load_principal_policy
from scripts.collect_principal_shadow_dataset import _load_mapping, _load_teacher
from scripts.evaluate_principal_student import evaluate_policy_on_schedule, summarise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    report_path = args.release_report.resolve()
    release = validate_go_release_report(protocol_path, report_path)
    final_root = args.final_root.resolve()
    teacher_config_path = args.teacher_config.resolve()
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    if teacher_config_path != Path(
        protocol["shadow_labelling"]["teacher_config"]
    ).resolve() or sha256_file(teacher_config_path) != protocol["shadow_labelling"][
        "teacher_config_sha256"
    ]:
        raise ValueError("teacher config does not match the execution specification")
    if sha256_file(teacher_checkpoint / "agent.pkl") != protocol["teacher"][
        "actor_sha256"
    ]:
        raise ValueError("teacher checkpoint hash mismatch")

    output = args.output.resolve()
    opening_path = output / "opening.json"
    expected_opening = {
        "schema_version": 1,
        "opening_id": "principal_test_v1_single_opening",
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "release_report": str(report_path),
        "release_report_sha256": sha256_file(report_path),
        "release_evidence_manifest_sha256": release["evidence_manifest_sha256"],
        "principal_test_schedule_sha256": release[
            "principal_test_schedule_sha256"
        ],
        "status": "opened_after_release_gate_go",
    }
    if output.exists():
        if not opening_path.is_file() or json.loads(opening_path.read_text()) != (
            expected_opening
        ):
            raise FileExistsError(
                "test output exists without the identical frozen opening record"
            )
    else:
        output.mkdir(parents=True)
        opening_path.write_text(
            json.dumps(expected_opening, indent=2, sort_keys=True) + "\n"
        )
    completed_path = output / "result.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text())
        if completed.get("status") != "completed" or completed.get(
            "opening_id"
        ) != expected_opening["opening_id"]:
            raise ValueError("completed test result does not match its opening")
        return completed

    # This is the only principal-test schedule construction in the sweep. A
    # retry with the identical opening record resumes the same opening.
    schedule = principal_test_schedule(protocol)
    schedule_hash = principal_test_schedule_sha256(protocol)
    if schedule_hash != release["principal_test_schedule_sha256"]:
        raise ValueError("released principal-test schedule hash changed")

    student_directory = output / "students"
    student_directory.mkdir(exist_ok=True)
    checkpoints: list[tuple[str, int, Path]] = []
    for family_id in PRINCIPAL_FAMILY_IDS:
        for raw_seed in protocol["matched_seeds"]["training"]:
            seed = int(raw_seed)
            checkpoint = final_root / family_id / str(seed) / "checkpoint.npz"
            policy = load_principal_policy(family_id, checkpoint)
            metadata = policy.metadata
            if metadata.get("student_id") != family_id:
                raise ValueError("final checkpoint family mismatch")
            if int(metadata.get("training_seed", -1)) != seed:
                raise ValueError("final checkpoint seed mismatch")
            if metadata.get("training_stage") != "final":
                raise ValueError("principal test requires final checkpoints")
            if metadata.get("protocol_sha256") != sha256_file(protocol_path):
                raise ValueError("final checkpoint protocol mismatch")
            checkpoints.append((family_id, seed, checkpoint))

    completed = 0
    for family_id, seed, checkpoint in checkpoints:
        result_path = student_directory / f"{family_id}-{seed}.json"
        if result_path.exists():
            _validate_existing_student_result(
                result_path, family_id, seed, checkpoint, schedule_hash
            )
            completed += 1
            continue
        policy = load_principal_policy(family_id, checkpoint)
        episodes, inference = evaluate_policy_on_schedule(policy, schedule, protocol)
        result = {
            "schema_version": 1,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "opening_id": expected_opening["opening_id"],
            "family_id": family_id,
            "training_seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "code_commit": args.code_commit,
            "protocol_sha256": sha256_file(protocol_path),
            "evaluation_split": protocol["evaluation"]["test"]["split"],
            "evaluation_schedule_sha256": schedule_hash,
            "release_report_sha256": sha256_file(report_path),
            "release_evidence_manifest_sha256": release[
                "evidence_manifest_sha256"
            ],
            "runtime": {"hostname": platform.node(), "python": sys.version},
            "summary": summarise(
                episodes,
                inference,
                protocol,
                blackout_steps=[
                    *protocol["evaluation"]["test"]["core_blackout_steps"],
                    *protocol["evaluation"]["test"][
                        "extrapolation_blackout_steps"
                    ],
                ],
            ),
            "episodes": episodes,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        completed += 1
        print(
            json.dumps(
                {"test_students_completed": completed, "family": family_id, "seed": seed},
                sort_keys=True,
            ),
            flush=True,
        )

    teacher_path = output / "teacher.json"
    if not teacher_path.exists():
        teacher, jax_version, devices = _load_teacher(
            protocol,
            _load_mapping(teacher_config_path),
            teacher_checkpoint,
            output,
        )
        episodes, inference = evaluate_teacher_on_schedule(
            teacher, schedule, protocol
        )
        teacher_result = {
            "schema_version": 1,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "opening_id": expected_opening["opening_id"],
            "policy": "frozen_teacher_reference",
            "checkpoint": str(teacher_checkpoint),
            "checkpoint_sha256": sha256_file(teacher_checkpoint / "agent.pkl"),
            "code_commit": args.code_commit,
            "protocol_sha256": sha256_file(protocol_path),
            "evaluation_split": protocol["evaluation"]["test"]["split"],
            "evaluation_schedule_sha256": schedule_hash,
            "release_report_sha256": sha256_file(report_path),
            "runtime": {
                "hostname": platform.node(),
                "python": sys.version,
                "jax": jax_version,
                "jax_devices": devices,
            },
            "summary": summarise(
                episodes,
                inference,
                protocol,
                blackout_steps=[
                    *protocol["evaluation"]["test"]["core_blackout_steps"],
                    *protocol["evaluation"]["test"][
                        "extrapolation_blackout_steps"
                    ],
                ],
            ),
            "episodes": episodes,
        }
        teacher_path.write_text(
            json.dumps(teacher_result, indent=2, sort_keys=True) + "\n"
        )

    result = {
        "schema_version": 1,
        "status": "completed",
        "opening_id": expected_opening["opening_id"],
        "student_results": len(checkpoints),
        "teacher_results": 1,
        "principal_test_schedule_sha256": schedule_hash,
    }
    completed_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def evaluate_teacher_on_schedule(
    teacher: Any,
    schedule: tuple[tuple[Any, int], ...],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float]]:
    reward = load_defence_reward(protocol["shadow_labelling"]["reward_config"])
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(reward),
        action_lock_steps=int(protocol["shadow_labelling"]["action_lock_steps"]),
    )
    episodes: list[dict[str, Any]] = []
    inference: list[float] = []
    try:
        for schedule_index, (generated, blackout_steps) in enumerate(schedule):
            environment.blackout = BlackoutSchedule(
                start_observation_step=int(
                    protocol["shadow_labelling"]["blackout_start_observation_step"]
                ),
                length_steps=blackout_steps,
            )
            observation, _ = environment.reset(shot=generated.shot)
            carry = teacher.init_policy(1)
            previous_reward = 0.0
            score = 0.0
            steps = 0
            outcome = "rollout_limit"
            while steps < environment.timeout_steps:
                policy_observation = {
                    "image": np.asarray([observation], dtype=np.float32),
                    "reward": np.asarray([previous_reward], dtype=np.float32),
                    "is_first": np.asarray([steps == 0], dtype=bool),
                    "is_last": np.asarray([False], dtype=bool),
                    "is_terminal": np.asarray([False], dtype=bool),
                }
                before = perf_counter()
                carry, actions, _ = teacher.policy(
                    carry, policy_observation, mode="eval"
                )
                inference.append(perf_counter() - before)
                action = np.clip(
                    np.asarray(actions["action"], dtype=np.float32)[0], -1.0, 1.0
                )
                observation, previous_reward, terminated, truncated, info = (
                    environment.step(action)
                )
                score += float(previous_reward)
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
    return episodes, inference


def _validate_existing_student_result(
    path: Path,
    family_id: str,
    seed: int,
    checkpoint: Path,
    schedule_hash: str,
) -> None:
    value = json.loads(path.read_text())
    expected = {
        "status": "completed",
        "opening_id": "principal_test_v1_single_opening",
        "family_id": family_id,
        "training_seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluation_schedule_sha256": schedule_hash,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError(f"existing test result does not match opening: {path}")


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
