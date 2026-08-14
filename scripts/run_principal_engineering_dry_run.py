#!/usr/bin/env python3
"""Run a small, explicitly non-principal check of all seven student families."""

from __future__ import annotations

import argparse
import json
import os
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
from airhockey_distill.principal_efficiency import (
    benchmark_policy_calls,
    validate_latency_runtime,
)
from airhockey_distill.principal_sweep import (
    PrincipalEpisodeSplit,
    evaluation_schedule,
    load_principal_protocol,
    sha256_file,
    shadow_schedule_records,
)
from airhockey_distill.students import (
    PRINCIPAL_FAMILY_IDS,
    load_principal_policy,
    principal_policy_from_parameters,
    save_principal_checkpoint,
)
from airhockey_distill.students.principal_torch import PrincipalStudentModule
from scripts.train_principal_student import (
    evaluate_principal_split,
    train_principal_epoch,
    verify_principal_export,
)

NON_PRINCIPAL_CLASSIFICATION = "non_principal_engineering_dry_run"
TRAINING_SEED = 14303


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tiny-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--overfit-mse", type=float, default=0.01)
    parser.add_argument("--minimum-loss-reduction", type=float, default=0.98)
    parser.add_argument("--validation-shots", type=int, default=2)
    parser.add_argument("--validation-blackouts", type=int, default=2)
    parser.add_argument("--latency-warmup", type=int, default=100)
    parser.add_argument("--latency-calls", type=int, default=200)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument("--container-digest", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco
    import torch

    if args.epochs <= 0 or args.overfit_mse <= 0.0 or not (
        0.0 < args.minimum_loss_reduction < 1.0
    ):
        raise ValueError("dry-run training limits must be positive")
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    if args.container_digest != protocol["provenance"]["container_digest"]:
        raise ValueError("dry-run container digest does not match the protocol")
    latency_runtime = validate_latency_runtime(
        protocol["measurements"]["cpu_latency"]
    )
    if latency_runtime["decision"] != "GO":
        raise RuntimeError("; ".join(latency_runtime["failures"]))
    dataset_directory = args.tiny_dataset.resolve()
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split, source = load_tiny_split(dataset_directory, manifest)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()

    shadow = paired_shadow_schedule_check(protocol)
    validation_schedule = short_validation_schedule(
        protocol,
        shots=args.validation_shots,
        blackouts=args.validation_blackouts,
    )
    family_results = []
    for family_index, family_id in enumerate(PRINCIPAL_FAMILY_IDS):
        family_results.append(
            run_family(
                protocol,
                protocol_path,
                family_id,
                split,
                validation_schedule,
                output,
                code_commit=args.code_commit,
                maximum_epochs=args.epochs,
                overfit_mse=args.overfit_mse,
                minimum_loss_reduction=args.minimum_loss_reduction,
                latency_warmup=args.latency_warmup,
                latency_calls=args.latency_calls,
                latency_repetitions=args.latency_repetitions,
                torch=torch,
            )
        )
        print(
            json.dumps(
                {
                    "family": family_id,
                    "completed": family_index + 1,
                    "decision": family_results[-1]["decision"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    checks = {
        "all_seven_families_completed": len(family_results) == 7,
        "all_tiny_overfits_pass": all(
            value["checks"]["tiny_dataset_overfit"] for value in family_results
        ),
        "all_checkpoint_reloads_exact": all(
            value["checks"]["exact_checkpoint_reload"] for value in family_results
        ),
        "all_numpy_pytorch_agree": all(
            value["checks"]["numpy_pytorch_agreement"] for value in family_results
        ),
        "paired_shadow_schedules_identical": bool(shadow["passed"]),
        "all_short_validation_rollouts_complete": all(
            value["checks"]["short_validation_rollout"] for value in family_results
        ),
        "all_latency_measurements_complete": all(
            value["checks"]["latency_measurement"] for value in family_results
        ),
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "classification": NON_PRINCIPAL_CLASSIFICATION,
        "principal_evidence_eligible": False,
        "release_manifest_eligible": False,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "container_digest": args.container_digest,
        "protocol_sha256": sha256_file(protocol_path),
        "dataset": str(dataset_directory),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_source": source,
        "dry_run_limits": {
            "training_episodes": split.episode_count,
            "maximum_epochs": args.epochs,
            "overfit_mse_threshold": args.overfit_mse,
            "minimum_fractional_loss_reduction": args.minimum_loss_reduction,
            "validation_shots": args.validation_shots,
            "validation_blackouts_per_shot": args.validation_blackouts,
            "latency_warmup_calls": args.latency_warmup,
            "latency_timed_calls_per_repetition": args.latency_calls,
            "latency_repetitions": args.latency_repetitions,
        },
        "checks": checks,
        "paired_shadow_schedule": shadow,
        "latency_runtime": latency_runtime,
        "families": family_results,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def run_family(
    protocol: dict[str, Any],
    protocol_path: Path,
    family_id: str,
    split: PrincipalEpisodeSplit,
    validation_schedule: tuple[tuple[Any, int], ...],
    output: Path,
    *,
    code_commit: str,
    maximum_epochs: int,
    overfit_mse: float,
    minimum_loss_reduction: float,
    latency_warmup: int,
    latency_calls: int,
    latency_repetitions: int,
    torch: Any,
) -> dict[str, Any]:
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    module = PrincipalStudentModule(family_id, seed=TRAINING_SEED)
    optimiser = torch.optim.AdamW(module.parameters(), lr=3e-3, weight_decay=0.0)
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    initial = evaluate_principal_split(module, split, 1, 64)
    selected_mse = float(initial["equal_episode_action_mse"])
    initial_mse = selected_mse
    epochs_completed = 0
    started = perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        train_principal_epoch(
            module,
            optimiser,
            split,
            batch_size=1,
            sequence_length=64,
            gradient_clip=1.0,
            generator=generator,
        )
        epochs_completed = epoch
        if epoch % 10 == 0 or epoch == maximum_epochs:
            selected_mse = float(
                evaluate_principal_split(module, split, 1, 64)[
                    "equal_episode_action_mse"
                ]
            )
            if tiny_overfit_passed(
                initial_mse=initial_mse,
                selected_mse=selected_mse,
                maximum_mse=overfit_mse,
                minimum_fractional_reduction=minimum_loss_reduction,
            ):
                break
    checkpoint = output / "checkpoints" / f"{family_id}.npz"
    module.prepare_for_numpy_export()
    metadata = {
        "schema_version": 1,
        "student_id": family_id,
        "training_seed": TRAINING_SEED,
        "training_stage": "non_principal_dry_run",
        "classification": NON_PRINCIPAL_CLASSIFICATION,
        "principal_evidence_eligible": False,
        "code_commit": code_commit,
        "protocol_sha256": sha256_file(protocol_path),
    }
    parameters = module.export_numpy_parameters()
    save_principal_checkpoint(family_id, checkpoint, parameters, metadata)
    exported = principal_policy_from_parameters(family_id, parameters, metadata)
    restored = load_principal_policy(family_id, checkpoint)
    action_error, carry_error, carry_tolerance_fraction, reload_exact = (
        verify_principal_export(
            module, exported, restored, split, episode_count=1
        )
    )
    validation = run_short_validation(
        protocol, restored, validation_schedule
    )
    latency = benchmark_policy_calls(
        restored,
        warmup_calls=latency_warmup,
        timed_calls_per_repetition=latency_calls,
        repetitions=latency_repetitions,
    )
    action_agreement_tolerance = float(
        protocol["training"]["export"]["maximum_absolute_error"]
    )
    fractional_reduction = 1.0 - selected_mse / initial_mse
    checks = {
        "tiny_dataset_overfit": tiny_overfit_passed(
            initial_mse=initial_mse,
            selected_mse=selected_mse,
            maximum_mse=overfit_mse,
            minimum_fractional_reduction=minimum_loss_reduction,
        ),
        "exact_checkpoint_reload": bool(reload_exact),
        "numpy_pytorch_agreement": (
            action_error <= action_agreement_tolerance
            and carry_tolerance_fraction <= 1.0
        ),
        "short_validation_rollout": len(validation["episodes"])
        == len(validation_schedule)
        and all(
            value["outcome"] != "rollout_limit" and int(value["steps"]) > 0
            for value in validation["episodes"]
        ),
        "latency_measurement": (
            len(latency["repetition_mean_microseconds"])
            == latency_repetitions
            and latency["median_microseconds"] >= 0.0
            and latency["p95_microseconds"] >= 0.0
        ),
    }
    return {
        "family_id": family_id,
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "training_seed": TRAINING_SEED,
        "training_episodes": 1,
        "initial_action_mse": initial_mse,
        "selected_action_mse": selected_mse,
        "fractional_loss_reduction": fractional_reduction,
        "overfit_gate": {
            "maximum_action_mse": overfit_mse,
            "minimum_fractional_loss_reduction": minimum_loss_reduction,
        },
        "maximum_epochs": maximum_epochs,
        "epochs_completed": epochs_completed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": restored.parameter_count,
        "numpy_pytorch_action_maximum_absolute_error": action_error,
        "numpy_pytorch_carry_maximum_absolute_error": carry_error,
        "numpy_pytorch_carry_maximum_tolerance_fraction": (
            carry_tolerance_fraction
        ),
        "checkpoint_reload_exact": reload_exact,
        "validation": validation,
        "latency": latency,
        "duration_seconds": perf_counter() - started,
    }


def tiny_overfit_passed(
    *,
    initial_mse: float,
    selected_mse: float,
    maximum_mse: float,
    minimum_fractional_reduction: float,
) -> bool:
    """Use one architecture-neutral engineering overfit criterion."""

    if initial_mse <= 0.0:
        return selected_mse <= maximum_mse
    reduction = 1.0 - selected_mse / initial_mse
    return selected_mse <= maximum_mse and reduction >= minimum_fractional_reduction


def load_tiny_split(
    dataset_directory: Path,
    manifest: dict[str, Any],
) -> tuple[PrincipalEpisodeSplit, dict[str, Any]]:
    if manifest.get("status") != "completed":
        raise ValueError("tiny dataset is incomplete")
    if manifest.get("teacher_action_semantics") != (
        "deterministic_actor_mean_after_public_adapter_clip"
    ):
        raise ValueError("tiny dataset does not contain deterministic teacher means")
    if int(manifest.get("episode_count", -1)) < 1:
        raise ValueError("tiny dataset contains no episodes")
    entry = manifest["shards"][0]
    path = dataset_directory / entry["file"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("tiny dataset shard hash mismatch")
    with np.load(path, allow_pickle=False) as shard:
        if int(shard["dataset_schema_version"]) != 2:
            raise ValueError("tiny dataset has the wrong schema")
        blackout_values = np.asarray(shard["episode_blackout_steps"], dtype=np.int64)
        candidates = np.flatnonzero(blackout_values == 0)
        if len(candidates) == 0:
            raise ValueError(
                "tiny dry-run dataset needs a no-blackout episode so the "
                "memoryless family has a representable target"
            )
        local_episode = int(candidates[0])
        first, last = int(shard["episode_offsets"][local_episode]), int(
            shard["episode_offsets"][local_episode + 1]
        )
        observations = np.asarray(shard["observations"][first:last], np.float32)
        previous_actions = np.asarray(
            shard["previous_actions"][first:last], np.float32
        )
        teacher_actions = np.asarray(
            shard["teacher_actions"][first:last], np.float32
        )
        visible = np.asarray(shard["puck_visible"][first:last], bool)
        source = {
            "dataset_id": manifest["dataset_id"],
            "episode_index": int(shard["episode_indices"][local_episode]),
            "shot_id": str(shard["episode_shot_ids"][local_episode]),
            "blackout_steps": int(shard["episode_blackout_steps"][local_episode]),
            "episode_steps": last - first,
            "teacher_action_semantics": manifest["teacher_action_semantics"],
        }
    length = len(observations)
    maximum_steps = 128
    padded_observations = np.zeros((1, maximum_steps, 19), np.float32)
    padded_previous = np.zeros((1, maximum_steps, 2), np.float32)
    padded_teacher = np.zeros((1, maximum_steps, 2), np.float32)
    padded_visible = np.zeros((1, maximum_steps), bool)
    valid = np.zeros((1, maximum_steps), bool)
    padded_observations[0, :length] = observations
    padded_previous[0, :length] = previous_actions
    padded_teacher[0, :length] = teacher_actions
    padded_visible[0, :length] = visible
    valid[0, :length] = True
    return (
        PrincipalEpisodeSplit(
            observations=padded_observations,
            previous_actions=padded_previous,
            teacher_actions=padded_teacher,
            puck_visible=padded_visible,
            valid_mask=valid,
            episode_indices=np.asarray([source["episode_index"]], np.int64),
            episode_lengths=np.asarray([length], np.int16),
        ),
        source,
    )


def paired_shadow_schedule_check(protocol: dict[str, Any]) -> dict[str, Any]:
    reference = None
    hashes = {}
    for family_id in PRINCIPAL_FAMILY_IDS:
        projected = [
            {
                "collector_training_seed": int(seed),
                **record,
            }
            for seed in protocol["matched_seeds"]["training"]
            for record in shadow_schedule_records(protocol, int(seed))[:2]
        ]
        payload = json.dumps(projected, sort_keys=True, separators=(",", ":")).encode()
        hashes[family_id] = __import__("hashlib").sha256(payload).hexdigest()
        if reference is None:
            reference = projected
        elif projected != reference:
            return {"passed": False, "family_schedule_sha256": hashes}
    return {
        "passed": len(set(hashes.values())) == 1,
        "records_per_family": len(reference or []),
        "collector_seeds": [
            int(value) for value in protocol["matched_seeds"]["training"]
        ],
        "records_per_collector_seed": 2,
        "family_schedule_sha256": hashes,
    }


def short_validation_schedule(
    protocol: dict[str, Any], *, shots: int, blackouts: int
) -> tuple[tuple[Any, int], ...]:
    if shots <= 0 or blackouts <= 0:
        raise ValueError("short validation dimensions must be positive")
    full = evaluation_schedule(protocol)
    blackout_values = len(protocol["evaluation"]["validation"]["blackout_steps"])
    selected = []
    for shot_index in range(shots):
        first = shot_index * blackout_values
        selected.extend(full[first : first + blackouts])
    if len(selected) != shots * blackouts:
        raise ValueError("short validation slice exceeds the canonical schedule")
    return tuple(selected)


def run_short_validation(
    protocol: dict[str, Any],
    policy: Any,
    schedule: tuple[tuple[Any, int], ...],
) -> dict[str, Any]:
    reward = load_defence_reward(protocol["shadow_labelling"]["reward_config"])
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(reward),
        action_lock_steps=int(protocol["shadow_labelling"]["action_lock_steps"]),
    )
    episodes = []
    try:
        for index, (generated, blackout_steps) in enumerate(schedule):
            environment.blackout = BlackoutSchedule(
                start_observation_step=int(
                    protocol["shadow_labelling"]["blackout_start_observation_step"]
                ),
                length_steps=blackout_steps,
            )
            observation, _ = environment.reset(shot=generated.shot)
            carry = policy.initial_carry()
            outcome = "rollout_limit"
            steps = 0
            while steps < environment.timeout_steps:
                action, carry = policy.act(observation, carry)
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                if terminated or truncated:
                    outcome = str(info["outcome"])
                    break
            episodes.append(
                {
                    "schedule_index": index,
                    "shot_id": generated.shot.shot_id,
                    "blackout_steps": blackout_steps,
                    "steps": steps,
                    "outcome": outcome,
                }
            )
    finally:
        environment.close()
    outcomes = Counter(value["outcome"] for value in episodes)
    return {
        "classification": NON_PRINCIPAL_CLASSIFICATION,
        "principal_evidence_eligible": False,
        "episodes": episodes,
        "outcome_counts": dict(sorted(outcomes.items())),
    }


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
