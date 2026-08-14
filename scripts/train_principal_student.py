#!/usr/bin/env python3
"""Train any principal student family under one matched protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from airhockey_distill.principal_sweep import (
    PrincipalEpisodeSplit,
    episode_index_sha256,
    load_principal_episode_splits,
    load_principal_protocol,
    principal_family_spec,
    sha256_file,
    validate_principal_dataset_manifest,
    validate_training_seed,
)
from airhockey_distill.students import (
    load_principal_policy,
    principal_policy_from_parameters,
    save_principal_checkpoint,
)

PRINCIPAL_EXPORT_ACTION_ABSOLUTE_TOLERANCE = 2e-5
PRINCIPAL_EXPORT_CARRY_ABSOLUTE_TOLERANCE = 2e-5
PRINCIPAL_EXPORT_CARRY_RELATIVE_TOLERANCE = 1e-6
PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE = 2e-5


@dataclass(frozen=True)
class PrincipalExportVerification:
    """Complete cross-framework diagnostics for one exported checkpoint."""

    action_maximum_absolute_error: float
    action_worst_case: dict[str, Any] | None
    carry_maximum_absolute_error: float
    carry_absolute_worst_case: dict[str, Any] | None
    carry_maximum_tolerance_fraction: float
    carry_tolerance_worst_case: dict[str, Any] | None
    one_step_action_maximum_absolute_error: float
    one_step_carry_maximum_absolute_error: float
    one_step_worst_case: dict[str, Any] | None
    checkpoint_reload_exact: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("collector", "final"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from airhockey_distill.students.principal_torch import PrincipalStudentModule

    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    family = principal_family_spec(protocol, args.family)
    validate_training_seed(protocol, args.seed)
    training = protocol["training"]
    dataset_directory = args.dataset.resolve()
    manifest_path = dataset_directory / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_principal_dataset_manifest(
        protocol,
        manifest,
        manifest_hash,
        family_id=args.family,
        stage=args.stage,
    )
    splits = load_principal_episode_splits(
        dataset_directory,
        manifest,
        protocol,
    )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    resolved = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "family_id": args.family,
        "training_seed": args.seed,
        "stage": args.stage,
        "dataset_manifest_sha256": manifest_hash,
    }
    (output / "resolved_run.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(
        max(1, min(int(training["torch_threads"]), os.cpu_count() or 1))
    )
    module = PrincipalStudentModule(args.family, seed=args.seed)
    if module.parameter_count != int(family["expected_total_parameters"]):
        raise ValueError("runtime parameter count disagrees with the protocol")
    optimiser = torch.optim.AdamW(
        module.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    batch_size = int(training["batch_size"])
    sequence_length = int(training["sequence_length"])
    maximum_epochs = int(training["maximum_epochs"])
    patience_epochs = int(training["patience_epochs"])
    minimum_improvement = float(training["minimum_validation_improvement"])
    gradient_clip = float(training["gradient_clip_global_norm"])
    if splits["train"].episode_count % batch_size:
        raise ValueError(
            "training episode count must divide into equal complete-episode batches"
        )

    initial_metrics = {
        name: evaluate_principal_split(module, split, batch_size, sequence_length)
        for name, split in splits.items()
        if name != "internal_test"
    }
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    patience = 0
    epochs_completed = 0
    started = perf_counter()
    with (output / "metrics.jsonl").open("w") as metrics_stream:
        _write_metric(
            metrics_stream,
            {
                "epoch": 0,
                "train": initial_metrics["train"],
                "validation": initial_metrics["validation"],
            },
        )
        for epoch in range(1, maximum_epochs + 1):
            online = train_principal_epoch(
                module,
                optimiser,
                splits["train"],
                batch_size=batch_size,
                sequence_length=sequence_length,
                gradient_clip=gradient_clip,
                generator=generator,
            )
            validation = evaluate_principal_split(
                module,
                splits["validation"],
                batch_size,
                sequence_length,
            )
            _write_metric(
                metrics_stream,
                {
                    "epoch": epoch,
                    "online_train_equal_episode_mse": online,
                    "validation": validation,
                },
            )
            epochs_completed = epoch
            validation_mse = float(validation["equal_episode_action_mse"])
            if validation_mse < best_validation - minimum_improvement:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = deepcopy(module.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= patience_epochs:
                    break

    if best_state is None:
        raise RuntimeError("training did not produce a selected checkpoint")
    module.load_state_dict(best_state)
    module.prepare_for_numpy_export()
    module.eval()
    selected_metrics = {
        name: evaluate_principal_split(module, split, batch_size, sequence_length)
        for name, split in splits.items()
    }
    checkpoint_path = output / "checkpoint.npz"
    checkpoint_metadata = {
        "schema_version": 1,
        "student_id": args.family,
        "training_seed": args.seed,
        "training_stage": args.stage,
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": manifest_hash,
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "loss_weighting": "equal_total_weight_per_complete_episode",
        "training_arithmetic": "pytorch_float32",
        "selected_epoch": best_epoch,
        "validation_equal_episode_action_mse": best_validation,
    }
    parameters = module.export_numpy_parameters()
    save_principal_checkpoint(
        args.family,
        checkpoint_path,
        parameters,
        checkpoint_metadata,
    )
    exported = principal_policy_from_parameters(
        args.family,
        parameters,
        checkpoint_metadata,
    )
    restored = load_principal_policy(args.family, checkpoint_path)
    verification = verify_principal_export(
        module,
        exported,
        restored,
        splits["validation"],
        episode_count=int(training["export"]["verification_episodes"]),
    )
    action_tolerance = float(training["export"]["maximum_absolute_error"])
    if action_tolerance != PRINCIPAL_EXPORT_ACTION_ABSOLUTE_TOLERANCE:
        raise ValueError("principal action export tolerance changed unexpectedly")
    gate_failures = principal_export_gate_failures(
        verification,
        action_tolerance=action_tolerance,
        require_exact_checkpoint_reload=bool(
            training["export"]["require_exact_checkpoint_reload"]
        ),
    )
    if gate_failures:
        raise RuntimeError(
            "principal export gate failed: " + ", ".join(gate_failures)
        )

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "family_id": args.family,
        "training_seed": args.seed,
        "training_stage": args.stage,
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_manifest_sha256": manifest_hash,
        "loss_weighting": "equal_total_weight_per_complete_episode",
        "loss_steps": "every_valid_episode_step",
        "split_episode_counts": {
            name: split.episode_count for name, split in splits.items()
        },
        "split_transition_counts": {
            name: split.transition_count for name, split in splits.items()
        },
        "split_episode_index_sha256": {
            name: episode_index_sha256(split) for name, split in splits.items()
        },
        "selected_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "selection_metric": "validation_equal_episode_action_mse",
        "initial_metrics": initial_metrics,
        "selected_metrics": selected_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "export_action_maximum_absolute_error": (
            verification.action_maximum_absolute_error
        ),
        "export_action_absolute_tolerance": action_tolerance,
        "export_action_worst_case": verification.action_worst_case,
        "export_carry_maximum_absolute_error": (
            verification.carry_maximum_absolute_error
        ),
        "export_carry_absolute_tolerance": (
            PRINCIPAL_EXPORT_CARRY_ABSOLUTE_TOLERANCE
        ),
        "export_carry_relative_tolerance": (
            PRINCIPAL_EXPORT_CARRY_RELATIVE_TOLERANCE
        ),
        "export_carry_maximum_absolute_error_worst_case": (
            verification.carry_absolute_worst_case
        ),
        "export_carry_maximum_tolerance_fraction": (
            verification.carry_maximum_tolerance_fraction
        ),
        "export_carry_maximum_tolerance_fraction_worst_case": (
            verification.carry_tolerance_worst_case
        ),
        "export_same_state_one_step_action_maximum_absolute_error": (
            verification.one_step_action_maximum_absolute_error
        ),
        "export_same_state_one_step_carry_maximum_absolute_error": (
            verification.one_step_carry_maximum_absolute_error
        ),
        "export_same_state_one_step_absolute_tolerance": (
            PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE
        ),
        "export_same_state_one_step_worst_case": (
            verification.one_step_worst_case
        ),
        "checkpoint_reload_exact": verification.checkpoint_reload_exact,
        "parameter_count": exported.parameter_count,
        "core_parameter_count": exported.core_parameter_count,
        "carry_float32_values": exported.carry_float32_values,
        "duration_seconds": perf_counter() - started,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def train_principal_epoch(
    module: Any,
    optimiser: Any,
    split: PrincipalEpisodeSplit,
    *,
    batch_size: int,
    sequence_length: int,
    gradient_clip: float,
    generator: Any,
) -> float:
    """Optimise the mean of complete-episode means, independent of length."""

    import torch

    module.train()
    permutation = torch.randperm(split.episode_count, generator=generator).numpy()
    total_episode_mse = 0.0
    episode_count = 0
    for start in range(0, split.episode_count, batch_size):
        indices = permutation[start : start + batch_size]
        episode_squared_error = torch.zeros(len(indices), dtype=torch.float32)
        episode_action_values = torch.from_numpy(
            split.episode_lengths[indices].astype(np.float32) * 2.0
        )
        carry = None
        optimiser.zero_grad(set_to_none=True)
        for chunk in _chunk_slices(split.observations.shape[1], sequence_length):
            observations = torch.from_numpy(split.observations[indices, chunk])
            previous_actions = torch.from_numpy(
                split.previous_actions[indices, chunk]
            )
            targets = torch.from_numpy(split.teacher_actions[indices, chunk])
            mask = torch.from_numpy(split.valid_mask[indices, chunk])
            predictions, carries = module.forward_sequence(
                observations,
                previous_actions,
                carry,
            )
            carry = (
                None
                if carries.shape[-1] == 0
                else carries[:, -1].detach()
            )
            squared_errors = (predictions - targets) ** 2
            expanded = mask.unsqueeze(-1).expand_as(squared_errors)
            chunk_squared_error = (squared_errors * expanded).sum(dim=(1, 2))
            episode_squared_error += chunk_squared_error.detach()
            chunk_loss = (chunk_squared_error / episode_action_values).mean()
            if bool(torch.any(expanded)):
                chunk_loss.backward()
        if not bool(torch.all(episode_action_values > 0)):
            raise RuntimeError("training batch contains an empty episode")
        episode_mse = episode_squared_error / episode_action_values
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(), gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("non-finite gradient norm during principal training")
        optimiser.step()
        total_episode_mse += float(episode_mse.detach().sum())
        episode_count += len(indices)
    return total_episode_mse / episode_count


def evaluate_principal_split(
    module: Any,
    split: PrincipalEpisodeSplit,
    batch_size: int,
    sequence_length: int,
) -> dict[str, float]:
    """Report both equal-episode and descriptive transition-weighted errors."""

    import torch

    equal_episode_sum = 0.0
    transition_squared_error = 0.0
    action_value_count = 0
    module.eval()
    with torch.no_grad():
        for start in range(0, split.episode_count, batch_size):
            indices = np.arange(start, min(start + batch_size, split.episode_count))
            episode_squared_error = torch.zeros(len(indices), dtype=torch.float32)
            episode_action_values = torch.zeros(len(indices), dtype=torch.float32)
            carry = None
            for chunk in _chunk_slices(
                split.observations.shape[1], sequence_length
            ):
                observations = torch.from_numpy(split.observations[indices, chunk])
                previous_actions = torch.from_numpy(
                    split.previous_actions[indices, chunk]
                )
                targets = torch.from_numpy(split.teacher_actions[indices, chunk])
                mask = torch.from_numpy(split.valid_mask[indices, chunk])
                predictions, carries = module.forward_sequence(
                    observations,
                    previous_actions,
                    carry,
                )
                carry = None if carries.shape[-1] == 0 else carries[:, -1]
                errors = (predictions - targets) ** 2
                expanded = mask.unsqueeze(-1).expand_as(errors)
                episode_squared_error += (errors * expanded).sum(dim=(1, 2))
                episode_action_values += expanded.sum(dim=(1, 2))
            episode_mse = episode_squared_error / episode_action_values
            equal_episode_sum += float(episode_mse.sum())
            transition_squared_error += float(episode_squared_error.sum())
            action_value_count += int(episode_action_values.sum())
    return {
        "equal_episode_action_mse": equal_episode_sum / split.episode_count,
        "transition_weighted_action_mse": (
            transition_squared_error / action_value_count
        ),
        "episodes": split.episode_count,
        "valid_action_values": action_value_count,
    }


def verify_principal_export(
    module: Any,
    exported: Any,
    restored: Any,
    split: PrincipalEpisodeSplit,
    *,
    episode_count: int,
) -> PrincipalExportVerification:
    import torch

    action_error = 0.0
    action_worst_case = None
    carry_error = 0.0
    carry_absolute_worst_case = None
    carry_tolerance_fraction = 0.0
    carry_tolerance_worst_case = None
    one_step_action_error = 0.0
    one_step_carry_error = 0.0
    one_step_worst_case = None
    reload_exact = True
    module.eval()
    for index in range(min(episode_count, split.episode_count)):
        length = int(split.episode_lengths[index])
        observations = split.observations[index, :length]
        previous_actions = split.previous_actions[index, :length]
        with torch.no_grad():
            torch_actions, torch_carries = module.forward_sequence(
                torch.from_numpy(observations[None, :]),
                torch.from_numpy(previous_actions[None, :]),
            )
        exported_actions, exported_carries = exported.sequence(
            observations,
            previous_actions,
        )
        restored_actions, restored_carries = restored.sequence(
            observations,
            previous_actions,
        )
        torch_action_values = torch_actions.numpy()[0]
        torch_carry_values = torch_carries.numpy()[0]
        action_errors = np.abs(torch_action_values - exported_actions)
        action_location = tuple(
            int(value)
            for value in np.unravel_index(
                np.argmax(action_errors),
                action_errors.shape,
            )
        )
        episode_action_error = float(action_errors[action_location])
        if action_worst_case is None or episode_action_error > action_error:
            timestep, dimension = action_location
            action_error = episode_action_error
            action_worst_case = {
                "absolute_error": episode_action_error,
                "state_magnitude": _carry_state_magnitude(
                    torch_carry_values,
                    exported_carries,
                    timestep,
                ),
                "verification_episode_offset": index,
                "episode_index": int(split.episode_indices[index]),
                "timestep": timestep,
                "dimension": dimension,
                "torch_value": float(torch_action_values[action_location]),
                "numpy_value": float(exported_actions[action_location]),
            }
        if exported_carries.shape[1]:
            absolute_errors = np.abs(torch_carry_values - exported_carries)
            state_magnitudes = np.maximum(
                np.abs(torch_carry_values),
                np.abs(exported_carries),
            )
            allowed_errors = (
                PRINCIPAL_EXPORT_CARRY_ABSOLUTE_TOLERANCE
                + PRINCIPAL_EXPORT_CARRY_RELATIVE_TOLERANCE
                * state_magnitudes
            )
            tolerance_fractions = absolute_errors / allowed_errors
            absolute_location = tuple(
                int(value)
                for value in np.unravel_index(
                    np.argmax(absolute_errors),
                    absolute_errors.shape,
                )
            )
            episode_carry_error = float(absolute_errors[absolute_location])
            if (
                carry_absolute_worst_case is None
                or episode_carry_error > carry_error
            ):
                timestep, dimension = absolute_location
                carry_absolute_worst_case = {
                    "absolute_error": episode_carry_error,
                    "allowed_error": float(allowed_errors[absolute_location]),
                    "tolerance_fraction": float(
                        tolerance_fractions[absolute_location]
                    ),
                    "state_magnitude": float(
                        state_magnitudes[absolute_location]
                    ),
                    "verification_episode_offset": index,
                    "episode_index": int(split.episode_indices[index]),
                    "timestep": timestep,
                    "dimension": dimension,
                    "torch_value": float(
                        torch_carry_values[absolute_location]
                    ),
                    "numpy_value": float(exported_carries[absolute_location]),
                }
            carry_error = max(carry_error, episode_carry_error)
            tolerance_location = tuple(
                int(value)
                for value in np.unravel_index(
                    np.argmax(tolerance_fractions),
                    tolerance_fractions.shape,
                )
            )
            episode_tolerance_fraction = float(
                tolerance_fractions[tolerance_location]
            )
            if (
                carry_tolerance_worst_case is None
                or episode_tolerance_fraction > carry_tolerance_fraction
            ):
                timestep, dimension = tolerance_location
                carry_tolerance_fraction = episode_tolerance_fraction
                carry_tolerance_worst_case = {
                    "absolute_error": float(
                        absolute_errors[tolerance_location]
                    ),
                    "allowed_error": float(
                        allowed_errors[tolerance_location]
                    ),
                    "tolerance_fraction": float(
                        tolerance_fractions[tolerance_location]
                    ),
                    "state_magnitude": float(
                        state_magnitudes[tolerance_location]
                    ),
                    "verification_episode_offset": index,
                    "episode_index": int(split.episode_indices[index]),
                    "timestep": timestep,
                    "dimension": dimension,
                    "torch_value": float(
                        torch_carry_values[tolerance_location]
                    ),
                    "numpy_value": float(
                        exported_carries[tolerance_location]
                    ),
                }
        (
            episode_one_step_action_error,
            episode_one_step_carry_error,
            episode_one_step_worst_case,
        ) = (
            same_state_one_step_errors(
                module,
                observations,
                previous_actions,
                exported_actions,
                exported_carries,
                verification_episode_offset=index,
                episode_index=int(split.episode_indices[index]),
            )
        )
        one_step_action_error = max(
            one_step_action_error,
            episode_one_step_action_error,
        )
        one_step_carry_error = max(
            one_step_carry_error,
            episode_one_step_carry_error,
        )
        if episode_one_step_worst_case is not None and (
            one_step_worst_case is None
            or episode_one_step_worst_case["absolute_error"]
            > one_step_worst_case["absolute_error"]
        ):
            one_step_worst_case = episode_one_step_worst_case
        reload_exact = reload_exact and np.array_equal(
            restored_actions, exported_actions
        ) and np.array_equal(restored_carries, exported_carries)
    return PrincipalExportVerification(
        action_maximum_absolute_error=action_error,
        action_worst_case=action_worst_case,
        carry_maximum_absolute_error=carry_error,
        carry_absolute_worst_case=carry_absolute_worst_case,
        carry_maximum_tolerance_fraction=carry_tolerance_fraction,
        carry_tolerance_worst_case=carry_tolerance_worst_case,
        one_step_action_maximum_absolute_error=one_step_action_error,
        one_step_carry_maximum_absolute_error=one_step_carry_error,
        one_step_worst_case=one_step_worst_case,
        checkpoint_reload_exact=reload_exact,
    )


def principal_export_gate_failures(
    verification: PrincipalExportVerification,
    *,
    action_tolerance: float = PRINCIPAL_EXPORT_ACTION_ABSOLUTE_TOLERANCE,
    require_exact_checkpoint_reload: bool = True,
) -> tuple[str, ...]:
    """Return every fail-closed export-gate predicate that was violated."""

    failures = []
    action_error = verification.action_maximum_absolute_error
    carry_fraction = verification.carry_maximum_tolerance_fraction
    one_step_action_error = (
        verification.one_step_action_maximum_absolute_error
    )
    one_step_carry_error = verification.one_step_carry_maximum_absolute_error
    if not math.isfinite(action_error) or action_error > action_tolerance:
        failures.append("action_absolute_error")
    if not math.isfinite(carry_fraction) or carry_fraction > 1.0:
        failures.append("carry_scale_aware_error")
    if not math.isfinite(one_step_action_error) or one_step_action_error > (
        PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE
    ):
        failures.append("same_state_one_step_action_absolute_error")
    if not math.isfinite(one_step_carry_error) or one_step_carry_error > (
        PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE
    ):
        failures.append("same_state_one_step_carry_absolute_error")
    if (
        require_exact_checkpoint_reload
        and not verification.checkpoint_reload_exact
    ):
        failures.append("checkpoint_reload_not_exact")
    return tuple(failures)


def same_state_one_step_errors(
    module: Any,
    observations: np.ndarray,
    previous_actions: np.ndarray,
    numpy_actions: np.ndarray,
    numpy_carries: np.ndarray,
    *,
    verification_episode_offset: int = 0,
    episode_index: int = 0,
) -> tuple[float, float, dict[str, Any] | None]:
    """Compare one step at a time from the same NumPy-exported carry."""

    import torch

    action_error = 0.0
    carry_error = 0.0
    worst_case = None
    shared_carry = None
    with torch.no_grad():
        for time_index in range(len(observations)):
            input_state_magnitude = (
                0.0
                if shared_carry is None or not shared_carry.numel()
                else float(torch.max(torch.abs(shared_carry)).item())
            )
            torch_actions, torch_carries = module.forward_sequence(
                torch.from_numpy(observations[None, time_index : time_index + 1]),
                torch.from_numpy(
                    previous_actions[None, time_index : time_index + 1]
                ),
                shared_carry,
            )
            torch_action = torch_actions.numpy()[0, 0]
            action_errors = np.abs(torch_action - numpy_actions[time_index])
            action_dimension = int(np.argmax(action_errors))
            step_action_error = float(action_errors[action_dimension])
            action_error = max(action_error, step_action_error)
            if worst_case is None or step_action_error > worst_case[
                "absolute_error"
            ]:
                worst_case = {
                    "quantity": "action",
                    "absolute_error": step_action_error,
                    "allowed_error": (
                        PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE
                    ),
                    "state_magnitude": input_state_magnitude,
                    "verification_episode_offset": verification_episode_offset,
                    "episode_index": episode_index,
                    "timestep": time_index,
                    "dimension": action_dimension,
                    "torch_value": float(torch_action[action_dimension]),
                    "numpy_value": float(
                        numpy_actions[time_index, action_dimension]
                    ),
                }
            if numpy_carries.shape[1]:
                torch_carry = torch_carries.numpy()[0, 0]
                carry_errors = np.abs(
                    torch_carry - numpy_carries[time_index]
                )
                carry_dimension = int(np.argmax(carry_errors))
                step_carry_error = float(carry_errors[carry_dimension])
                carry_error = max(carry_error, step_carry_error)
                if step_carry_error > worst_case["absolute_error"]:
                    worst_case = {
                        "quantity": "carry",
                        "absolute_error": step_carry_error,
                        "allowed_error": (
                            PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE
                        ),
                        "state_magnitude": input_state_magnitude,
                        "output_magnitude": max(
                            abs(float(torch_carry[carry_dimension])),
                            abs(
                                float(
                                    numpy_carries[
                                        time_index,
                                        carry_dimension,
                                    ]
                                )
                            ),
                        ),
                        "verification_episode_offset": (
                            verification_episode_offset
                        ),
                        "episode_index": episode_index,
                        "timestep": time_index,
                        "dimension": carry_dimension,
                        "torch_value": float(torch_carry[carry_dimension]),
                        "numpy_value": float(
                            numpy_carries[time_index, carry_dimension]
                        ),
                    }
                shared_carry = torch.from_numpy(
                    numpy_carries[None, time_index].copy()
                )
    return action_error, carry_error, worst_case


def _carry_state_magnitude(
    torch_carries: np.ndarray,
    numpy_carries: np.ndarray,
    timestep: int,
) -> float:
    if not numpy_carries.shape[1]:
        return 0.0
    return max(
        float(np.max(np.abs(torch_carries[timestep]))),
        float(np.max(np.abs(numpy_carries[timestep]))),
    )


def scale_aware_carry_error(
    torch_carries: np.ndarray,
    numpy_carries: np.ndarray,
) -> tuple[float, float]:
    """Return absolute error and its fraction of the declared carry tolerance."""

    if torch_carries.shape != numpy_carries.shape:
        raise ValueError("NumPy and PyTorch carry shapes differ")
    absolute_error = np.abs(torch_carries - numpy_carries)
    scale = np.maximum(np.abs(torch_carries), np.abs(numpy_carries))
    allowed_error = (
        PRINCIPAL_EXPORT_CARRY_ABSOLUTE_TOLERANCE
        + PRINCIPAL_EXPORT_CARRY_RELATIVE_TOLERANCE * scale
    )
    return (
        float(np.max(absolute_error, initial=0.0)),
        float(np.max(absolute_error / allowed_error, initial=0.0)),
    )


def _chunk_slices(total_steps: int, sequence_length: int) -> tuple[slice, ...]:
    return tuple(
        slice(start, min(start + sequence_length, total_steps))
        for start in range(0, total_steps, sequence_length)
    )


def _write_metric(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
