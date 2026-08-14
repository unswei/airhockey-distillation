#!/usr/bin/env python3
"""Freeze the implementation-only structured inference optimisation evidence."""

from __future__ import annotations

import argparse
import json
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airhockey_distill.principal_release import _check_efficiency, _check_validation
from airhockey_distill.principal_sweep import (
    evaluation_schedule_sha256,
    load_principal_protocol,
    sha256_file,
)
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS
from airhockey_distill.students.structured import (
    NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION,
)

STRUCTURED_FAMILIES = (
    "structured_k0",
    "structured_k1",
    "structured_k2",
    "structured_k4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-v3-root", type=Path, required=True)
    parser.add_argument("--optimisation-code-commit", required=True)
    parser.add_argument("--orchestration-code-commit", required=True)
    parser.add_argument("--source-v2-seal-sha256", required=True)
    parser.add_argument("--source-v3-final-manifest-sha256", required=True)
    parser.add_argument("--source-v3-measurement-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be an object: {path}")
    return value


def _require_frozen(path: Path) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError(f"evidence remains writable: {path}")


def _validate_hash(value: str, length: int, label: str) -> None:
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a lowercase hexadecimal hash")


def _check_closed_test_gate(root: Path) -> None:
    if any(root.glob("principal_test*")) or (root / "test").exists():
        raise ValueError(f"principal_test is present under {root}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    run_root = args.run_root.resolve()
    source_v3 = args.source_v3_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for value, label in (
        (args.optimisation_code_commit, "optimisation code commit"),
        (args.orchestration_code_commit, "orchestration code commit"),
    ):
        _validate_hash(value, 40, label)
    for value, label in (
        (args.source_v2_seal_sha256, "source V2 seal"),
        (args.source_v3_final_manifest_sha256, "source V3 final manifest"),
        (
            args.source_v3_measurement_manifest_sha256,
            "source V3 measurement manifest",
        ),
    ):
        _validate_hash(value, 64, label)
    _check_closed_test_gate(run_root)
    _check_closed_test_gate(source_v3)

    final_manifest_path = source_v3 / "orchestrator/final_training_manifest.json"
    measurement_manifest_path = (
        source_v3 / "orchestrator/measurements/manifest.json"
    )
    for path in (final_manifest_path, measurement_manifest_path):
        _require_frozen(path)
    if sha256_file(final_manifest_path) != args.source_v3_final_manifest_sha256:
        raise ValueError("source V3 final-training manifest changed")
    if (
        sha256_file(measurement_manifest_path)
        != args.source_v3_measurement_manifest_sha256
    ):
        raise ValueError("source V3 measurement manifest changed")
    final_manifest = _read_object(final_manifest_path)
    measurement_manifest = _read_object(measurement_manifest_path)
    if (
        final_manifest.get("status") != "completed"
        or int(final_manifest.get("final_models", -1)) != 35
        or final_manifest.get("principal_test_opened") is not False
        or measurement_manifest.get("decision") != "GO"
        or measurement_manifest.get("principal_test_opened") is not False
    ):
        raise ValueError("source V3 is not completed, frozen pre-test evidence")

    seeds = tuple(int(seed) for seed in protocol["matched_seeds"]["training"])
    expected = {
        (family, seed)
        for family in PRINCIPAL_FAMILY_IDS
        for seed in seeds
    }
    structured_expected = {
        identity for identity in expected if identity[0] in STRUCTURED_FAMILIES
    }
    final_records = {
        (str(record["family_id"]), int(record["training_seed"])): record
        for record in final_manifest.get("models", [])
    }
    v3_measurement_records = {
        (str(record["family_id"]), int(record["training_seed"])): record
        for record in measurement_manifest.get("models", [])
    }
    if set(final_records) != expected or set(v3_measurement_records) != expected:
        raise ValueError("source V3 model identities differ from the protocol")

    kernel_manifest_path = run_root / "native/build.json"
    verification_path = run_root / "verification/native-kernel.json"
    for path in (kernel_manifest_path, verification_path):
        _require_frozen(path)
    kernel_manifest = _read_object(kernel_manifest_path)
    native_verification = _read_object(verification_path)
    binaries = tuple((run_root / "native").glob("_airhockey_pairwise_float32*.so"))
    if len(binaries) != 1:
        raise ValueError("V4 must contain exactly one native kernel binary")
    binary = binaries[0]
    _require_frozen(binary)
    if (
        kernel_manifest.get("implementation")
        != NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION
        or kernel_manifest.get("code_commit") != args.optimisation_code_commit
        or kernel_manifest.get("binary_sha256") != sha256_file(binary)
        or native_verification.get("decision") != "GO"
        or native_verification.get("code_commit")
        != args.optimisation_code_commit
        or native_verification.get("loaded_binary_sha256") != sha256_file(binary)
    ):
        raise ValueError("native kernel provenance or verification failed")
    verified_checkpoints = {
        (str(record["family_id"]), int(record["training_seed"])): record
        for record in native_verification.get("checkpoints", [])
    }
    if set(verified_checkpoints) != structured_expected or not all(
        record.get("passed") is True for record in verified_checkpoints.values()
    ):
        raise ValueError("native verification does not cover all structured models")

    failures: list[str] = []
    models = []
    protocol_hash = sha256_file(protocol_path)
    for family, seed in sorted(expected):
        identity = (family, seed)
        checkpoint = source_v3 / "final" / family / str(seed) / "checkpoint.npz"
        _require_frozen(checkpoint)
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != final_records[identity].get("checkpoint_sha256"):
            raise ValueError(f"source V3 checkpoint changed: {family}/{seed}")

        accounting_path = source_v3 / "accounting" / f"{family}-{seed}.json"
        _require_frozen(accounting_path)
        accounting = _read_object(accounting_path)
        _check_efficiency(
            protocol,
            protocol_path,
            "accounting_results",
            identity,
            accounting,
            checkpoint_hash,
            checkpoint,
            failures,
        )

        if family in STRUCTURED_FAMILIES:
            validation_path = run_root / "validation" / f"{family}-{seed}.json"
            reference_validation_path = (
                source_v3 / "validation" / f"{family}-{seed}.json"
            )
            _require_frozen(reference_validation_path)
            reference_validation = _read_object(reference_validation_path)
            validation_source = "V4 native rerun"
        else:
            validation_path = source_v3 / "validation" / f"{family}-{seed}.json"
            reference_validation = None
            validation_source = "V3 unchanged family reuse"
        _require_frozen(validation_path)
        validation = _read_object(validation_path)
        _check_validation(
            protocol,
            protocol_path,
            identity,
            validation,
            checkpoint_hash,
            failures,
        )
        if family in STRUCTURED_FAMILIES:
            if validation.get("code_commit") != args.optimisation_code_commit:
                failures.append(f"validation {family}/{seed}: code commit mismatch")
            if validation.get("runtime", {}).get("inference_implementation") != (
                NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION
            ):
                failures.append(
                    f"validation {family}/{seed}: native implementation absent"
                )
            if validation.get("episodes") != reference_validation.get("episodes"):
                failures.append(
                    f"validation {family}/{seed}: episode rows differ from V3"
                )
            summary = dict(validation.get("summary", {}))
            reference_summary = dict(reference_validation.get("summary", {}))
            summary.pop("inference_milliseconds", None)
            reference_summary.pop("inference_milliseconds", None)
            if summary != reference_summary:
                failures.append(
                    f"validation {family}/{seed}: outcomes differ from V3"
                )

        latency_path = run_root / "latency" / f"{family}-{seed}.json"
        _require_frozen(latency_path)
        latency = _read_object(latency_path)
        _check_efficiency(
            protocol,
            protocol_path,
            "cpu_latency_results",
            identity,
            latency,
            checkpoint_hash,
            checkpoint,
            failures,
        )
        expected_implementation = (
            NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION
            if family in STRUCTURED_FAMILIES
            else "numpy_float32_v1"
        )
        if latency.get("code_commit") != args.optimisation_code_commit:
            failures.append(f"latency {family}/{seed}: code commit mismatch")
        if latency.get("runtime_contract", {}).get(
            "inference_implementation"
        ) != expected_implementation:
            failures.append(f"latency {family}/{seed}: implementation mismatch")

        models.append(
            {
                "family_id": family,
                "training_seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "validation_source": validation_source,
                "validation_sha256": sha256_file(validation_path),
                "accounting_source": "V3 hash reuse",
                "accounting_sha256": sha256_file(accounting_path),
                "cpu_latency_sha256": sha256_file(latency_path),
                "inference_implementation": expected_implementation,
                "validation_save_rate": float(validation["summary"]["save_rate"]),
                "validation_save_rate_at_20_steps": float(
                    validation["summary"]["by_blackout_steps"]["20"]["save_rate"]
                ),
                "cpu_latency_median_microseconds": float(
                    latency["measurements"]["median_microseconds"]
                ),
            }
        )
    if failures:
        raise ValueError("; ".join(failures))

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "decision": "GO",
        "created_at": datetime.now(UTC).isoformat(),
        "optimisation_code_commit": args.optimisation_code_commit,
        "orchestration_code_commit": args.orchestration_code_commit,
        "protocol_sha256": protocol_hash,
        "validation_schedule_sha256": evaluation_schedule_sha256(protocol),
        "source_v2_seal_sha256": args.source_v2_seal_sha256,
        "source_v3_final_manifest_sha256": args.source_v3_final_manifest_sha256,
        "source_v3_measurement_manifest_sha256": (
            args.source_v3_measurement_manifest_sha256
        ),
        "native_build_manifest_sha256": sha256_file(kernel_manifest_path),
        "native_binary_sha256": sha256_file(binary),
        "native_verification_sha256": sha256_file(verification_path),
        "native_linear_bit_exact": native_verification["linear_kernel"][
            "bit_exact"
        ],
        "native_verified_structured_checkpoints": len(verified_checkpoints),
        "structured_validation_reruns": len(structured_expected),
        "structured_validation_episode_rows_identical_to_v3": True,
        "reused_v3_validation_results": len(expected - structured_expected),
        "reused_v3_accounting_results": len(expected),
        "new_isolated_cpu_latency_results": len(expected),
        "principal_test_opened": False,
        "models": models,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    print(json.dumps(verify(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
