#!/usr/bin/env python3
"""Verify and hash the correction-only V3 pre-test measurements."""

from __future__ import annotations

import argparse
import json
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from airhockey_distill.principal_release import (
    _check_efficiency,
    _check_validation,
)
from airhockey_distill.principal_sweep import (
    evaluation_schedule_sha256,
    load_principal_protocol,
    sha256_file,
)
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--final-training-manifest", type=Path, required=True)
    parser.add_argument("--measurement-code-commit", required=True)
    parser.add_argument("--orchestration-code-commit", required=True)
    parser.add_argument("--source-v2-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be an object: {path}")
    return value


def _require_frozen(path: Path) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ValueError(f"measurement remains writable: {path}")


def _validate_commit(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be a 40-character Git hash")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    run_root = args.run_root.resolve()
    final_manifest_path = args.final_training_manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    _validate_commit(args.measurement_code_commit, "measurement code commit")
    _validate_commit(args.orchestration_code_commit, "orchestration code commit")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_v2_seal_sha256):
        raise ValueError("source V2 seal hash must be SHA-256")
    if any(run_root.glob("principal_test*")) or (run_root / "test").exists():
        raise ValueError("principal_test must remain absent during measurement")

    final_manifest = _read_object(final_manifest_path)
    if (
        final_manifest.get("status") != "completed"
        or int(final_manifest.get("final_models", -1)) != 35
        or final_manifest.get("principal_test_opened") is not False
    ):
        raise ValueError("final-training manifest is incomplete")
    if final_manifest.get("training_code_commit") != args.measurement_code_commit:
        raise ValueError("final-training and measurement code commits differ")
    final_records = {
        (str(value["family_id"]), int(value["training_seed"])): value
        for value in final_manifest.get("models", [])
    }
    seeds = tuple(int(value) for value in protocol["matched_seeds"]["training"])
    expected = {
        (family_id, seed)
        for family_id in PRINCIPAL_FAMILY_IDS
        for seed in seeds
    }
    if set(final_records) != expected:
        raise ValueError("final-training manifest identities differ from the protocol")

    records = []
    failures: list[str] = []
    protocol_hash = sha256_file(protocol_path)
    for family_id, seed in sorted(expected):
        identity = (family_id, seed)
        checkpoint = run_root / "final" / family_id / str(seed) / "checkpoint.npz"
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != final_records[identity].get("checkpoint_sha256"):
            raise ValueError(f"final checkpoint changed: {family_id}/{seed}")

        validation_path = run_root / "validation" / f"{family_id}-{seed}.json"
        accounting_path = run_root / "accounting" / f"{family_id}-{seed}.json"
        latency_path = run_root / "latency" / f"{family_id}-{seed}.json"
        for path in (validation_path, accounting_path, latency_path):
            _require_frozen(path)
        validation = _read_object(validation_path)
        accounting = _read_object(accounting_path)
        latency = _read_object(latency_path)

        for label, value in (
            ("validation", validation),
            ("accounting", accounting),
            ("latency", latency),
        ):
            if value.get("code_commit") != args.measurement_code_commit:
                failures.append(
                    f"{label} {family_id} seed {seed}: measurement commit mismatch"
                )
            if value.get("protocol_sha256") != protocol_hash:
                failures.append(
                    f"{label} {family_id} seed {seed}: protocol hash mismatch"
                )

        _check_validation(
            protocol,
            protocol_path,
            identity,
            validation,
            checkpoint_hash,
            failures,
        )
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
        records.append(
            {
                "family_id": family_id,
                "training_seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "validation_sha256": sha256_file(validation_path),
                "accounting_sha256": sha256_file(accounting_path),
                "cpu_latency_sha256": sha256_file(latency_path),
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
        "measurement_code_commit": args.measurement_code_commit,
        "orchestration_code_commit": args.orchestration_code_commit,
        "protocol_sha256": protocol_hash,
        "validation_schedule_sha256": evaluation_schedule_sha256(protocol),
        "source_v2_seal_sha256": args.source_v2_seal_sha256,
        "final_training_manifest_sha256": sha256_file(final_manifest_path),
        "principal_test_opened": False,
        "validation_results": len(records),
        "accounting_results": len(records),
        "cpu_latency_results": len(records),
        "models": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    result = verify(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
