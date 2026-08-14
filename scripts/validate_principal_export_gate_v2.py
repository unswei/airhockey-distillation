#!/usr/bin/env python3
"""Revalidate the corrected principal export gate against sealed V2 artefacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from airhockey_distill.principal_sweep import (
    episode_index_sha256,
    load_principal_episode_splits,
    load_principal_protocol,
    sha256_file,
    validate_principal_dataset_manifest,
)
from airhockey_distill.students import (
    load_principal_policy,
    principal_policy_from_parameters,
)
from scripts.train_principal_student import (
    PRINCIPAL_EXPORT_ACTION_ABSOLUTE_TOLERANCE,
    PRINCIPAL_EXPORT_CARRY_ABSOLUTE_TOLERANCE,
    PRINCIPAL_EXPORT_CARRY_RELATIVE_TOLERANCE,
    PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE,
    principal_export_gate_failures,
    verify_principal_export,
)

FAMILY_SEEDS = {
    "feed_forward": (14303, 14304, 14305, 14306, 14307),
    "finite_stack_10": (14303, 14304, 14305, 14306, 14307),
    "structured_k0": (14303, 14304, 14305, 14306),
}
HISTORICAL_FAILED_IDENTITY = ("structured_k0", 14306)
EXPECTED_CORRUPTION_CASES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-attempt-commit", required=True)
    parser.add_argument("--expected-seal-manifest-sha256", required=True)
    parser.add_argument("--expected-failed-checkpoint-sha256", required=True)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def _torch_module_from_checkpoint(
    family_id: str,
    seed: int,
    checkpoint: Path,
) -> tuple[Any, Any]:
    import torch

    from airhockey_distill.students.principal_torch import PrincipalStudentModule

    policy = load_principal_policy(family_id, checkpoint)
    parameters = policy.policy.parameters
    module = PrincipalStudentModule(family_id, seed=seed)
    state = module.state_dict()
    parameter_keys = set(parameters)
    expected_keys = {name.removeprefix("student.") for name in state}
    if parameter_keys != expected_keys:
        raise ValueError(
            f"checkpoint/state keys differ for {family_id}: "
            f"checkpoint-only={sorted(parameter_keys - expected_keys)}, "
            f"torch-only={sorted(expected_keys - parameter_keys)}"
        )
    for name, value in state.items():
        array = np.asarray(parameters[name.removeprefix("student.")])
        if tuple(array.shape) != tuple(value.shape):
            raise ValueError(f"checkpoint shape differs for {name}")
        state[name] = torch.from_numpy(np.array(array, copy=True))
    module.load_state_dict(state, strict=True)
    module.prepare_for_numpy_export()
    module.eval()
    return module, policy


def _verification_record(
    verification: Any,
    *,
    action_tolerance: float,
    require_exact_checkpoint_reload: bool,
) -> dict[str, Any]:
    failures = principal_export_gate_failures(
        verification,
        action_tolerance=action_tolerance,
        require_exact_checkpoint_reload=require_exact_checkpoint_reload,
    )
    record = asdict(verification)
    record["predicates"] = {
        "action_absolute": "action_absolute_error" not in failures,
        "carry_scale_aware": "carry_scale_aware_error" not in failures,
        "same_state_one_step_action_absolute": (
            "same_state_one_step_action_absolute_error" not in failures
        ),
        "same_state_one_step_carry_absolute": (
            "same_state_one_step_carry_absolute_error" not in failures
        ),
        "checkpoint_reload_exact": (
            "checkpoint_reload_not_exact" not in failures
        ),
    }
    record["gate_failures"] = list(failures)
    record["decision"] = "GO" if not failures else "NO_GO"
    return record


def _load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as checkpoint:
        return {
            name: np.array(checkpoint[name], copy=True)
            for name in checkpoint.files
        }


def _write_corrupt_checkpoint(
    source: Path,
    destination: Path,
    mutate: Callable[[dict[str, np.ndarray]], dict[str, Any]],
) -> dict[str, Any]:
    arrays = _load_checkpoint_arrays(source)
    mutation = mutate(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "mutation": mutation,
    }


def _metadata_mutation(
    key: str,
    value: Any,
) -> Callable[[dict[str, np.ndarray]], dict[str, Any]]:
    def mutate(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        metadata = json.loads(str(arrays["metadata_json"].item()))
        old = metadata.get(key)
        metadata[key] = value
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        return {"key": f"metadata_json.{key}", "old": old, "new": value}

    return mutate


def _array_scalar_mutation(
    key: str,
    index: int,
    delta: float,
) -> Callable[[dict[str, np.ndarray]], dict[str, Any]]:
    def mutate(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        array = arrays[key]
        old = float(array.reshape(-1)[index])
        array.reshape(-1)[index] = np.float32(old + delta)
        new = float(array.reshape(-1)[index])
        return {
            "key": key,
            "flat_index": index,
            "requested_delta": delta,
            "old": old,
            "new": new,
            "actual_delta": new - old,
        }

    return mutate


def _loader_rejection_case(
    case_id: str,
    family_id: str,
    source: Path,
    corrupt: dict[str, Any],
) -> dict[str, Any]:
    try:
        load_principal_policy(family_id, Path(corrupt["path"]))
    except Exception as error:  # The loader must reject malformed external input.
        return {
            "case_id": case_id,
            "family_id": family_id,
            "source_checkpoint_sha256": sha256_file(source),
            "corrupt_checkpoint": corrupt,
            "expected": "loader_rejection",
            "observed_exception": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "fail_closed": True,
        }
    return {
        "case_id": case_id,
        "family_id": family_id,
        "source_checkpoint_sha256": sha256_file(source),
        "corrupt_checkpoint": corrupt,
        "expected": "loader_rejection",
        "observed_exception": None,
        "fail_closed": False,
    }


def _semantic_corruption_case(
    case_id: str,
    family_id: str,
    source: Path,
    corrupt: dict[str, Any],
    module: Any,
    split: Any,
    verification_episodes: int,
    action_tolerance: float,
    require_exact_checkpoint_reload: bool,
    *,
    expected_failure: str,
    pristine_export_for_reload: Any | None = None,
) -> dict[str, Any]:
    corrupted = load_principal_policy(family_id, Path(corrupt["path"]))
    if pristine_export_for_reload is None:
        exported = principal_policy_from_parameters(
            family_id,
            corrupted.policy.parameters,
            corrupted.metadata,
        )
    else:
        exported = pristine_export_for_reload
    verification = verify_principal_export(
        module,
        exported,
        corrupted,
        split,
        episode_count=verification_episodes,
    )
    record = _verification_record(
        verification,
        action_tolerance=action_tolerance,
        require_exact_checkpoint_reload=require_exact_checkpoint_reload,
    )
    return {
        "case_id": case_id,
        "family_id": family_id,
        "source_checkpoint_sha256": sha256_file(source),
        "corrupt_checkpoint": corrupt,
        "expected_gate_failure": expected_failure,
        "verification": record,
        "fail_closed": expected_failure in record["gate_failures"],
    }


def _family_corruptions(
    family_id: str,
    attempt: Path,
    corruption_directory: Path,
    split: Any,
    verification_episodes: int,
    action_tolerance: float,
    require_exact_checkpoint_reload: bool,
) -> list[dict[str, Any]]:
    source = attempt / "final" / family_id / "14303" / "checkpoint.npz"
    module, pristine = _torch_module_from_checkpoint(family_id, 14303, source)
    pristine_export = principal_policy_from_parameters(
        family_id,
        pristine.policy.parameters,
        pristine.metadata,
    )
    cases: list[dict[str, Any]] = []

    if family_id == "feed_forward":
        action_corrupt = _write_corrupt_checkpoint(
            source,
            corruption_directory / "action_output_bias_plus_0p05.npz",
            _array_scalar_mutation("action_output_bias", 0, 0.05),
        )
        cases.append(
            _semantic_corruption_case(
                "action_parameter_disagreement",
                family_id,
                source,
                action_corrupt,
                module,
                split,
                verification_episodes,
                action_tolerance,
                require_exact_checkpoint_reload,
                expected_failure="action_absolute_error",
            )
        )
        cases.append(
            _semantic_corruption_case(
                "non_exact_checkpoint_reload",
                family_id,
                source,
                action_corrupt,
                module,
                split,
                verification_episodes,
                action_tolerance,
                require_exact_checkpoint_reload,
                expected_failure="checkpoint_reload_not_exact",
                pristine_export_for_reload=pristine_export,
            )
        )

    if family_id == "finite_stack_10":
        missing = _write_corrupt_checkpoint(
            source,
            corruption_directory / "missing_action_output_weight.npz",
            lambda arrays: {
                "removed_key": "action_output_weight",
                "removed_shape": list(arrays.pop("action_output_weight").shape),
            },
        )
        cases.append(
            _loader_rejection_case(
                "missing_required_array", family_id, source, missing
            )
        )
        truncated_path = corruption_directory / "truncated_checkpoint.npz"
        payload = source.read_bytes()
        truncated_path.write_bytes(payload[: max(1, len(payload) // 2)])
        truncated = {
            "path": str(truncated_path),
            "sha256": sha256_file(truncated_path),
            "mutation": {
                "operation": "truncate",
                "source_bytes": len(payload),
                "corrupt_bytes": truncated_path.stat().st_size,
            },
        }
        cases.append(
            _loader_rejection_case(
                "truncated_npz", family_id, source, truncated
            )
        )

    if family_id == "structured_k0":
        carry_corrupt = _write_corrupt_checkpoint(
            source,
            corruption_directory / "recurrence_alpha_53_plus_5e-6.npz",
            _array_scalar_mutation("recurrence_alpha", 53, 5e-6),
        )
        cases.append(
            _semantic_corruption_case(
                "scale_aware_carry_disagreement",
                family_id,
                source,
                carry_corrupt,
                module,
                split,
                verification_episodes,
                action_tolerance,
                require_exact_checkpoint_reload,
                expected_failure="carry_scale_aware_error",
            )
        )
        one_step_corrupt = _write_corrupt_checkpoint(
            source,
            corruption_directory / "recurrence_alpha_35_plus_5e-6.npz",
            _array_scalar_mutation("recurrence_alpha", 35, 5e-6),
        )
        cases.append(
            _semantic_corruption_case(
                "same_state_one_step_carry_disagreement",
                family_id,
                source,
                one_step_corrupt,
                module,
                split,
                verification_episodes,
                action_tolerance,
                require_exact_checkpoint_reload,
                expected_failure="same_state_one_step_carry_absolute_error",
            )
        )
        mismatch = _write_corrupt_checkpoint(
            source,
            corruption_directory / "student_id_mismatch.npz",
            _metadata_mutation("student_id", "structured_k1"),
        )
        cases.append(
            _loader_rejection_case(
                "student_id_mismatch", family_id, source, mismatch
            )
        )
        arithmetic = _write_corrupt_checkpoint(
            source,
            corruption_directory / "unsupported_inference_arithmetic.npz",
            _metadata_mutation("inference_arithmetic", "corrupt_float32"),
        )
        cases.append(
            _loader_rejection_case(
                "unsupported_inference_arithmetic",
                family_id,
                source,
                arithmetic,
            )
        )

        def wrong_shape(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
            old_shape = list(arrays["action_output_weight"].shape)
            arrays["action_output_weight"] = arrays["action_output_weight"][:1]
            return {
                "key": "action_output_weight",
                "old_shape": old_shape,
                "new_shape": list(arrays["action_output_weight"].shape),
            }

        shape = _write_corrupt_checkpoint(
            source,
            corruption_directory / "wrong_action_output_shape.npz",
            wrong_shape,
        )
        cases.append(
            _loader_rejection_case(
                "wrong_parameter_shape", family_id, source, shape
            )
        )

        def non_finite(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
            old = float(arrays["recurrence_bias"][0])
            arrays["recurrence_bias"][0] = np.float32(np.nan)
            return {
                "key": "recurrence_bias",
                "flat_index": 0,
                "old": old,
                "new": "NaN",
            }

        nan_checkpoint = _write_corrupt_checkpoint(
            source,
            corruption_directory / "nan_recurrence_bias.npz",
            non_finite,
        )
        cases.append(
            _loader_rejection_case(
                "non_finite_parameter", family_id, source, nan_checkpoint
            )
        )
    return cases


def _validate_family(
    family_id: str,
    seeds: tuple[int, ...],
    attempt_string: str,
    protocol_string: str,
    output_string: str,
    verification_episodes: int,
    action_tolerance: float,
    require_exact_checkpoint_reload: bool,
    cpu_ids: tuple[int, ...],
) -> dict[str, Any]:
    import torch

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))
    torch.set_num_threads(len(cpu_ids))
    torch.use_deterministic_algorithms(True)

    attempt = Path(attempt_string)
    protocol_path = Path(protocol_string)
    output = Path(output_string)
    manifest_path = attempt / "datasets" / family_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = load_principal_protocol(protocol_path)
    manifest_sha256 = sha256_file(manifest_path)
    validate_principal_dataset_manifest(
        protocol,
        manifest,
        manifest_sha256,
        family_id=family_id,
        stage="final",
    )
    split = load_principal_episode_splits(
        manifest_path.parent,
        manifest,
        protocol,
    )["validation"]

    checkpoints = []
    for seed in seeds:
        checkpoint = attempt / "final" / family_id / str(seed) / "checkpoint.npz"
        historical_result = checkpoint.with_name("result.json")
        expected_historical_failure = (family_id, seed) == HISTORICAL_FAILED_IDENTITY
        if historical_result.exists() == expected_historical_failure:
            raise ValueError(
                f"unexpected historical result presence for {family_id}/{seed}"
            )
        module, restored = _torch_module_from_checkpoint(
            family_id,
            seed,
            checkpoint,
        )
        exported = principal_policy_from_parameters(
            family_id,
            restored.policy.parameters,
            restored.metadata,
        )
        verification = verify_principal_export(
            module,
            exported,
            restored,
            split,
            episode_count=verification_episodes,
        )
        checkpoints.append(
            {
                "family_id": family_id,
                "training_seed": seed,
                "historical_status": (
                    "failed_seed_14306"
                    if expected_historical_failure
                    else "successful"
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "historical_result_sha256": (
                    None
                    if expected_historical_failure
                    else sha256_file(historical_result)
                ),
                "checkpoint_metadata": restored.metadata,
                "verification": _verification_record(
                    verification,
                    action_tolerance=action_tolerance,
                    require_exact_checkpoint_reload=(
                        require_exact_checkpoint_reload
                    ),
                ),
            }
        )

    corruptions = _family_corruptions(
        family_id,
        attempt,
        output / "corruptions" / family_id,
        split,
        verification_episodes,
        action_tolerance,
        require_exact_checkpoint_reload,
    )
    return {
        "family_id": family_id,
        "worker_pid": os.getpid(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else list(cpu_ids)
        ),
        "torch_threads": torch.get_num_threads(),
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_episode_count": manifest["episode_count"],
        "dataset_transition_count": manifest["transition_count"],
        "validation_episode_count": split.episode_count,
        "validation_episode_index_sha256": episode_index_sha256(split),
        "checkpoints": checkpoints,
        "corruptions": corruptions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    attempt = args.attempt.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    if args.workers != len(FAMILY_SEEDS):
        raise ValueError("this validation requires exactly three family workers")
    if output.exists():
        raise FileExistsError(output)
    if not attempt.is_dir():
        raise FileNotFoundError(attempt)
    if any(attempt.glob("principal_test*")):
        raise RuntimeError("principal_test artefact exists; refusing validation")
    seal_path = attempt / "orchestrator" / "seal_manifest.sha256"
    seal_sha256_before = sha256_file(seal_path)
    if seal_sha256_before != args.expected_seal_manifest_sha256:
        raise ValueError("sealed V2 manifest hash differs from the declaration")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != args.code_commit:
        raise ValueError("checkout does not match --code-commit")

    protocol = load_principal_protocol(protocol_path)
    action_tolerance = float(
        protocol["training"]["export"]["maximum_absolute_error"]
    )
    if action_tolerance != PRINCIPAL_EXPORT_ACTION_ABSOLUTE_TOLERANCE:
        raise ValueError("principal action export tolerance changed unexpectedly")
    verification_episodes = int(
        protocol["training"]["export"]["verification_episodes"]
    )
    require_exact_reload = bool(
        protocol["training"]["export"]["require_exact_checkpoint_reload"]
    )
    failed_checkpoint = (
        attempt / "final" / "structured_k0" / "14306" / "checkpoint.npz"
    )
    if sha256_file(failed_checkpoint) != args.expected_failed_checkpoint_sha256:
        raise ValueError("failed seed-14306 checkpoint hash differs")

    expected_checkpoints = {
        (family_id, str(seed))
        for family_id, seeds in FAMILY_SEEDS.items()
        for seed in seeds
    }
    observed_checkpoints = {
        (checkpoint.parent.parent.name, checkpoint.parent.name)
        for checkpoint in (attempt / "final").glob("*/*/checkpoint.npz")
    }
    if observed_checkpoints != expected_checkpoints:
        raise ValueError(
            "sealed attempt checkpoint inventory differs: "
            f"missing={sorted(expected_checkpoints - observed_checkpoints)}, "
            f"extra={sorted(observed_checkpoints - expected_checkpoints)}"
        )

    for family_id, seeds in FAMILY_SEEDS.items():
        for seed in seeds:
            checkpoint_directory = attempt / "final" / family_id / str(seed)
            metadata = load_principal_policy(
                family_id,
                checkpoint_directory / "checkpoint.npz",
            ).metadata
            if metadata.get("code_commit") != args.expected_attempt_commit:
                raise ValueError(
                    f"attempt commit mismatch in {family_id}/{seed}"
                )
            result_path = checkpoint_directory / "result.json"
            if (family_id, seed) == HISTORICAL_FAILED_IDENTITY:
                if result_path.exists():
                    raise ValueError("failed seed unexpectedly has a result")
            else:
                historical_result = json.loads(result_path.read_text())
                if historical_result.get("status") != "completed":
                    raise ValueError(
                        f"historical result is not completed for {family_id}/{seed}"
                    )
                if historical_result.get("code_commit") != (
                    args.expected_attempt_commit
                ):
                    raise ValueError(
                        f"historical result commit differs for {family_id}/{seed}"
                    )

    output.mkdir(parents=True)
    cpu_count = os.cpu_count() or 1
    if cpu_count < 3:
        raise RuntimeError("three family workers require at least three CPUs")
    available_cpu_ids = sorted(
        os.sched_getaffinity(0)
        if hasattr(os, "sched_getaffinity")
        else range(cpu_count)
    )
    group_size, remainder = divmod(len(available_cpu_ids), 3)
    groups = []
    start = 0
    for index in range(3):
        stop = start + group_size + (1 if index < remainder else 0)
        groups.append(tuple(available_cpu_ids[start:stop]))
        start = stop
    if any(not group for group in groups):
        raise RuntimeError("could not allocate one CPU group per family")
    futures = {}
    with ProcessPoolExecutor(max_workers=3) as executor:
        for index, (family_id, seeds) in enumerate(FAMILY_SEEDS.items()):
            future = executor.submit(
                _validate_family,
                family_id,
                seeds,
                str(attempt),
                str(protocol_path),
                str(output),
                verification_episodes,
                action_tolerance,
                require_exact_reload,
                groups[index],
            )
            futures[future] = family_id
        family_results = {
            futures[future]: future.result()
            for future in as_completed(futures)
        }

    ordered_families = [family_results[name] for name in FAMILY_SEEDS]
    checkpoints = [
        checkpoint
        for family in ordered_families
        for checkpoint in family["checkpoints"]
    ]
    corruptions = [
        corruption
        for family in ordered_families
        for corruption in family["corruptions"]
    ]
    successful = [
        entry for entry in checkpoints if entry["historical_status"] == "successful"
    ]
    historical_failed = [
        entry
        for entry in checkpoints
        if entry["historical_status"] == "failed_seed_14306"
    ]
    summary = {
        "historical_successful_total": len(successful),
        "historical_successful_passed": sum(
            entry["verification"]["decision"] == "GO" for entry in successful
        ),
        "historical_failed_total": len(historical_failed),
        "historical_failed_passed_corrected_gate": sum(
            entry["verification"]["decision"] == "GO"
            for entry in historical_failed
        ),
        "corruption_cases_total": len(corruptions),
        "corruption_cases_failed_closed": sum(
            entry["fail_closed"] for entry in corruptions
        ),
    }
    all_passed = summary == {
        "historical_successful_total": 13,
        "historical_successful_passed": 13,
        "historical_failed_total": 1,
        "historical_failed_passed_corrected_gate": 1,
        "corruption_cases_total": EXPECTED_CORRUPTION_CASES,
        "corruption_cases_failed_closed": EXPECTED_CORRUPTION_CASES,
    }
    seal_sha256_after = sha256_file(seal_path)
    principal_test_absent_after = not any(attempt.glob("principal_test*"))
    all_passed = (
        all_passed
        and seal_sha256_after == seal_sha256_before
        and principal_test_absent_after
    )
    result = {
        "schema_version": 1,
        "status": "completed" if all_passed else "failed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": current_commit,
        "attempt": str(attempt),
        "attempt_code_commit": args.expected_attempt_commit,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "sealed_v2_manifest_sha256_before": seal_sha256_before,
        "sealed_v2_manifest_sha256_after": seal_sha256_after,
        "principal_test_absent_before": True,
        "principal_test_absent_after": principal_test_absent_after,
        "thresholds": {
            "action_absolute": PRINCIPAL_EXPORT_ACTION_ABSOLUTE_TOLERANCE,
            "carry_absolute": PRINCIPAL_EXPORT_CARRY_ABSOLUTE_TOLERANCE,
            "carry_relative": PRINCIPAL_EXPORT_CARRY_RELATIVE_TOLERANCE,
            "same_state_one_step_absolute": (
                PRINCIPAL_EXPORT_ONE_STEP_ABSOLUTE_TOLERANCE
            ),
            "require_exact_checkpoint_reload": require_exact_reload,
            "verification_episodes": verification_episodes,
        },
        "parallel_execution": {
            "workers": 3,
            "strategy": "one worker per family; checkpoints sequential within family",
            "family_workers": {
                family["family_id"]: {
                    "pid": family["worker_pid"],
                    "cpu_affinity": family["cpu_affinity"],
                    "torch_threads": family["torch_threads"],
                }
                for family in ordered_families
            },
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "summary": summary,
        "families": ordered_families,
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    if not all_passed:
        raise RuntimeError(f"principal export gate V2 validation failed: {result_path}")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
