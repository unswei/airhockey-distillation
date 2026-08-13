#!/usr/bin/env python3
"""Build the canonical frozen evidence manifest from completed artefacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from airhockey_distill.principal_sweep import load_principal_protocol, sha256_file
from airhockey_distill.students import load_principal_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--aggregation-code-commit", required=True)
    parser.add_argument("--collector-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--final-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--family-dataset-manifest", type=Path, action="append", default=[])
    parser.add_argument("--shadow-budget-audit", type=Path, action="append", default=[])
    parser.add_argument("--validation-result", type=Path, action="append", default=[])
    parser.add_argument("--accounting-result", type=Path, action="append", default=[])
    parser.add_argument("--cpu-latency-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--declare-principal-test-uninspected",
        action="store_true",
        help="Required explicit declaration; absence is an error.",
    )
    return parser.parse_args()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    load_principal_protocol(protocol_path)
    if not re.fullmatch(r"[0-9a-f]{40}", args.aggregation_code_commit):
        raise ValueError("aggregation code commit must be a 40-character Git hash")
    if not args.declare_principal_test_uninspected:
        raise ValueError("the untouched principal-test declaration is required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "protocol_sha256": sha256_file(protocol_path),
        "aggregation_code_commit": args.aggregation_code_commit,
        "principal_test_outcomes_inspected": False,
        "analysis_plan": _file_entry(args.analysis_plan.resolve()),
        "collector_checkpoints": _checkpoint_entries(args.collector_checkpoint),
        "final_checkpoints": _checkpoint_entries(args.final_checkpoint),
        "family_datasets": _family_json_entries(args.family_dataset_manifest),
        "shadow_budget_audits": _family_json_entries(args.shadow_budget_audit),
        "validation_episode_files": _identity_json_entries(args.validation_result),
        "accounting_results": _identity_json_entries(args.accounting_result),
        "cpu_latency_results": _identity_json_entries(args.cpu_latency_result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _file_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _checkpoint_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for raw_path in paths:
        path = raw_path.resolve()
        identities = []
        for family_id in (
            "feed_forward",
            "finite_stack_10",
            "structured_k0",
            "structured_k1",
            "structured_k2",
            "structured_k4",
            "gru_n64",
        ):
            try:
                policy = load_principal_policy(family_id, path)
                identities.append((family_id, policy.metadata))
            except (KeyError, TypeError, ValueError):
                continue
        if len(identities) != 1:
            raise ValueError(f"could not identify checkpoint family exactly once: {path}")
        family_id, metadata = identities[0]
        entries.append(
            {
                **_file_entry(path),
                "family_id": family_id,
                "training_seed": int(metadata["training_seed"]),
            }
        )
    return sorted(entries, key=lambda value: (value["family_id"], value["training_seed"]))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON evidence must be an object: {path}")
    return value


def _family_json_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for raw_path in paths:
        path = raw_path.resolve()
        value = _read_json(path)
        entries.append({**_file_entry(path), "family_id": str(value["family_id"])})
    return sorted(entries, key=lambda value: value["family_id"])


def _identity_json_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for raw_path in paths:
        path = raw_path.resolve()
        value = _read_json(path)
        entries.append(
            {
                **_file_entry(path),
                "family_id": str(value["family_id"]),
                "training_seed": int(value["training_seed"]),
            }
        )
    return sorted(entries, key=lambda value: (value["family_id"], value["training_seed"]))


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
