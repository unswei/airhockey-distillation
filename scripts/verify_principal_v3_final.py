#!/usr/bin/env python3
"""Fail-closed verification for one correction-only V3 final checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ACTION_TOLERANCE = 2e-5
CARRY_ABSOLUTE_TOLERANCE = 2e-5
CARRY_RELATIVE_TOLERANCE = 1e-6
ONE_STEP_TOLERANCE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final-directory", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _finite_within(value: Any, tolerance: float) -> bool:
    number = float(value)
    return math.isfinite(number) and number <= tolerance


def _compare_parameter_arrays(
    source: Path,
    candidate: Path,
) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as source_checkpoint, np.load(
        candidate, allow_pickle=False
    ) as candidate_checkpoint:
        source_keys = set(source_checkpoint.files) - {"metadata_json"}
        candidate_keys = set(candidate_checkpoint.files) - {"metadata_json"}
        keys_equal = source_keys == candidate_keys
        mismatches = []
        if keys_equal:
            for key in sorted(source_keys):
                if not np.array_equal(source_checkpoint[key], candidate_checkpoint[key]):
                    mismatches.append(key)
        return {
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "keys_equal": keys_equal,
            "parameter_arrays_exact": keys_equal and not mismatches,
            "mismatched_parameter_arrays": mismatches,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    final_directory = args.final_directory.resolve()
    result_path = final_directory / "result.json"
    checkpoint_path = final_directory / "checkpoint.npz"
    result = _load_json(result_path)
    dataset_manifest_sha256 = sha256_file(args.dataset_manifest.resolve())

    failures = []
    expected_fields = {
        "status": "completed",
        "family_id": args.family,
        "training_seed": args.seed,
        "training_stage": "final",
        "code_commit": args.expected_code_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    for key, expected in expected_fields.items():
        if result.get(key) != expected:
            failures.append(f"result_{key}_mismatch")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        failures.append("checkpoint_sha256_mismatch")
    if not _finite_within(
        result.get("export_action_maximum_absolute_error", math.nan),
        ACTION_TOLERANCE,
    ):
        failures.append("action_export_gate")
    if not _finite_within(
        result.get("export_carry_maximum_tolerance_fraction", math.nan),
        1.0,
    ):
        failures.append("carry_export_gate")
    if not _finite_within(
        result.get(
            "export_same_state_one_step_action_maximum_absolute_error",
            math.nan,
        ),
        ONE_STEP_TOLERANCE,
    ):
        failures.append("same_state_action_export_gate")
    if not _finite_within(
        result.get(
            "export_same_state_one_step_carry_maximum_absolute_error",
            math.nan,
        ),
        ONE_STEP_TOLERANCE,
    ):
        failures.append("same_state_carry_export_gate")
    if result.get("checkpoint_reload_exact") is not True:
        failures.append("checkpoint_reload_not_exact")
    if result.get("export_action_worst_case") is None:
        failures.append("missing_action_worst_case")
    if args.family != "feed_forward":
        if result.get("export_carry_maximum_absolute_error_worst_case") is None:
            failures.append("missing_carry_absolute_worst_case")
        if result.get("export_carry_maximum_tolerance_fraction_worst_case") is None:
            failures.append("missing_carry_tolerance_worst_case")
    if result.get("export_same_state_one_step_worst_case") is None:
        failures.append("missing_same_state_worst_case")

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"].item()))
    expected_metadata = {
        "student_id": args.family,
        "training_seed": args.seed,
        "training_stage": "final",
        "code_commit": args.expected_code_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            failures.append(f"checkpoint_metadata_{key}_mismatch")

    source_equivalence = None
    if args.source_checkpoint is not None:
        source_equivalence = _compare_parameter_arrays(
            args.source_checkpoint.resolve(), checkpoint_path
        )
        if source_equivalence["parameter_arrays_exact"] is not True:
            failures.append("source_parameter_arrays_differ")

    verification = {
        "schema_version": 1,
        "status": "completed" if not failures else "failed",
        "created_at": datetime.now(UTC).isoformat(),
        "family_id": args.family,
        "training_seed": args.seed,
        "final_directory": str(final_directory),
        "training_code_commit": args.expected_code_commit,
        "protocol_sha256": args.expected_protocol_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "gate": {
            "action_absolute_tolerance": ACTION_TOLERANCE,
            "action_maximum_absolute_error": result.get(
                "export_action_maximum_absolute_error"
            ),
            "action_worst_case": result.get("export_action_worst_case"),
            "carry_absolute_tolerance": CARRY_ABSOLUTE_TOLERANCE,
            "carry_relative_tolerance": CARRY_RELATIVE_TOLERANCE,
            "carry_maximum_absolute_error": result.get(
                "export_carry_maximum_absolute_error"
            ),
            "carry_maximum_absolute_error_worst_case": result.get(
                "export_carry_maximum_absolute_error_worst_case"
            ),
            "carry_maximum_tolerance_fraction": result.get(
                "export_carry_maximum_tolerance_fraction"
            ),
            "carry_maximum_tolerance_fraction_worst_case": result.get(
                "export_carry_maximum_tolerance_fraction_worst_case"
            ),
            "same_state_one_step_absolute_tolerance": ONE_STEP_TOLERANCE,
            "same_state_one_step_action_maximum_absolute_error": result.get(
                "export_same_state_one_step_action_maximum_absolute_error"
            ),
            "same_state_one_step_carry_maximum_absolute_error": result.get(
                "export_same_state_one_step_carry_maximum_absolute_error"
            ),
            "same_state_one_step_worst_case": result.get(
                "export_same_state_one_step_worst_case"
            ),
            "checkpoint_reload_exact": result.get("checkpoint_reload_exact"),
        },
        "source_equivalence": source_equivalence,
        "failures": failures,
        "decision": "GO" if not failures else "NO_GO",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise RuntimeError(
            f"V3 final checkpoint verification failed: {', '.join(failures)}"
        )
    return verification


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
