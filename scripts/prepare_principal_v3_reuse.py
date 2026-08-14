#!/usr/bin/env python3
"""Freeze the exact sealed V2 inputs reused by final-training attempt V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REUSED_PREFIXES = ("collectors/", "shadow/", "audits/", "datasets/")
FAMILIES = (
    "feed_forward",
    "finite_stack_10",
    "structured_k0",
    "structured_k1",
    "structured_k2",
    "structured_k4",
    "gru_n64",
)
SEEDS = (14303, 14304, 14305, 14306, 14307)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-attempt", type=Path, required=True)
    parser.add_argument("--experiment-host-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-source-seal-sha256", required=True)
    parser.add_argument("--expected-source-code-commit", required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--orchestration-code-commit", required=True)
    parser.add_argument("--container-image-digest", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _parse_seal(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        if len(digest) != 64 or separator != "  " or not relative:
            raise ValueError(f"invalid source seal line: {line!r}")
        if relative in entries:
            raise ValueError(f"duplicate source seal entry: {relative}")
        entries[relative] = digest
    return entries


def _container_to_host(path: str, experiment_host_root: Path) -> Path:
    container_path = Path(path)
    try:
        relative = container_path.relative_to("/experiments")
    except ValueError as error:
        raise ValueError(f"shard is outside /experiments: {path}") from error
    return experiment_host_root / relative


def _identity_digest(entries: list[dict[str, Any]]) -> str:
    lines = [
        f"{entry['sha256']}  {entry['bytes']}  {entry['file']}"
        for entry in entries
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_attempt.resolve()
    experiment_root = args.experiment_host_root.resolve()
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(source.glob("principal_test*")):
        raise RuntimeError("principal_test exists in the source attempt")

    seal_path = source / "orchestrator" / "seal_manifest.sha256"
    source_seal_sha256 = sha256_file(seal_path)
    if source_seal_sha256 != args.expected_source_seal_sha256:
        raise ValueError("source attempt seal hash differs")
    sealed_failure_path = source / "orchestrator" / "SEALED_FAILED.json"
    sealed_failure = _load(sealed_failure_path)
    if sealed_failure.get("code_commit") != args.expected_source_code_commit:
        raise ValueError("source sealed-failure commit differs")

    seal = _parse_seal(seal_path)
    selected = {
        relative: digest
        for relative, digest in seal.items()
        if relative.startswith(REUSED_PREFIXES)
    }
    if len(selected) != 504:
        raise ValueError(f"expected 504 reused entries, found {len(selected)}")
    selected_lines = [
        f"{selected[relative]}  {relative}" for relative in sorted(selected)
    ]
    selected_text = "\n".join(selected_lines) + "\n"
    selected_sha256 = hashlib.sha256(selected_text.encode()).hexdigest()

    categories: dict[str, dict[str, Any]] = {}
    selected_bytes = 0
    for category in ("collectors", "shadow", "audits", "datasets"):
        category_lines = []
        category_bytes = 0
        for relative in sorted(selected):
            if not relative.startswith(f"{category}/"):
                continue
            path = source / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            observed = sha256_file(path)
            if observed != selected[relative]:
                raise ValueError(f"source hash mismatch: {relative}")
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise PermissionError(f"reused input is writable: {relative}")
            size = path.stat().st_size
            category_bytes += size
            category_lines.append(f"{observed}  {relative}")
        category_text = "\n".join(category_lines) + "\n"
        categories[category] = {
            "files": len(category_lines),
            "bytes": category_bytes,
            "manifest_sha256": hashlib.sha256(
                category_text.encode()
            ).hexdigest(),
        }
        selected_bytes += category_bytes

    if selected_bytes != 565_708_536:
        raise ValueError(f"unexpected reused byte count: {selected_bytes}")
    expected_categories = {
        "collectors": (140, "14aa8e86c727c139f1aaf747f82791b44c1a67cd50a976f6176ab0f89644b538"),
        "shadow": (350, "62264041a001db32530ece6d5a7d24ccdff02fafff9ad08b0b40cf5d82bfa3b8"),
        "audits": (7, "d9c8adb5f33327aeaddbb256cdbb29aea8ff28e0c3f6d8f73980751c4becd11b"),
        "datasets": (7, "5caf6efcb1a1e5a84321315312e2cda3775599ca52e23773f6a9e67390de830b"),
    }
    for category, (count, digest) in expected_categories.items():
        if categories[category]["files"] != count:
            raise ValueError(f"unexpected {category} file count")
        if categories[category]["manifest_sha256"] != digest:
            raise ValueError(f"unexpected {category} selected manifest hash")
    if selected_sha256 != (
        "c3a408686bd7bac4d90f58f0f68fbdb56624bf7953f8ebdea7700fe7353813ca"
    ):
        raise ValueError("combined reused manifest hash differs")

    collector_count = 0
    shadow_episode_count = 0
    audit_episode_count = 0
    dataset_records = []
    observed_shard_hashes: dict[Path, str] = {}
    for family in FAMILIES:
        audit_path = source / "audits" / f"{family}.json"
        audit = _load(audit_path)
        if audit.get("status") != "completed" or audit.get("decision") != "GO":
            raise ValueError(f"shadow audit is not GO: {family}")
        if audit.get("code_commit") != args.expected_source_code_commit:
            raise ValueError(f"shadow audit commit differs: {family}")
        if int(audit.get("realised_episode_budget", -1)) != 20_000:
            raise ValueError(f"shadow audit budget differs: {family}")
        audit_episode_count += int(audit["realised_episode_budget"])

        for seed in SEEDS:
            collector_directory = source / "collectors" / family / str(seed)
            collector = _load(collector_directory / "result.json")
            if collector.get("status") != "completed":
                raise ValueError(f"collector is incomplete: {family}/{seed}")
            if collector.get("code_commit") != args.expected_source_code_commit:
                raise ValueError(f"collector commit differs: {family}/{seed}")
            checkpoint = collector_directory / "checkpoint.npz"
            if sha256_file(checkpoint) != collector.get("checkpoint_sha256"):
                raise ValueError(f"collector checkpoint differs: {family}/{seed}")
            collector_count += 1

            shadow_directory = source / "shadow" / family / str(seed)
            shadow = _load(shadow_directory / "manifest.json")
            if shadow.get("status") != "completed":
                raise ValueError(f"shadow partition is incomplete: {family}/{seed}")
            if shadow.get("code_commit") != args.expected_source_code_commit:
                raise ValueError(f"shadow commit differs: {family}/{seed}")
            if int(shadow.get("episode_count", -1)) != 4_000:
                raise ValueError(f"shadow partition budget differs: {family}/{seed}")
            if shadow.get("student_checkpoint_sha256") != collector.get(
                "checkpoint_sha256"
            ):
                raise ValueError(f"shadow collector link differs: {family}/{seed}")
            shadow_episode_count += int(shadow["episode_count"])

        dataset_path = source / "datasets" / family / "manifest.json"
        dataset = _load(dataset_path)
        if dataset.get("status") != "completed":
            raise ValueError(f"family dataset is incomplete: {family}")
        if dataset.get("code_commit") != args.expected_source_code_commit:
            raise ValueError(f"family dataset commit differs: {family}")
        if int(dataset.get("episode_count", -1)) != 40_000:
            raise ValueError(f"family dataset size differs: {family}")
        if len(dataset.get("shards", [])) != 80:
            raise ValueError(f"family dataset shard count differs: {family}")

        source_manifests: dict[str, Path] = {}
        referenced_bytes = 0
        for entry in dataset["shards"]:
            shard_path = _container_to_host(entry["file"], experiment_root)
            observed = observed_shard_hashes.get(shard_path)
            if observed is None:
                observed = sha256_file(shard_path)
                observed_shard_hashes[shard_path] = observed
            if observed != entry["sha256"]:
                raise ValueError(f"dataset shard hash mismatch: {entry['file']}")
            if shard_path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"dataset shard byte count differs: {entry['file']}")
            referenced_bytes += int(entry["bytes"])
            manifest_path = shard_path.parent.parent / "manifest.json"
            source_manifests.setdefault(str(entry["source"]), manifest_path)
            if source_manifests[str(entry["source"])] != manifest_path:
                raise ValueError("one dataset source spans multiple manifests")
        for source_id, declared_hash in dataset["source_manifests"].items():
            manifest_path = source_manifests.get(source_id)
            if manifest_path is None or sha256_file(manifest_path) != declared_hash:
                raise ValueError(f"dataset source manifest differs: {family}/{source_id}")
        dataset_records.append(
            {
                "family_id": family,
                "manifest_sha256": sha256_file(dataset_path),
                "episodes": int(dataset["episode_count"]),
                "transitions": int(dataset["transition_count"]),
                "referenced_shards": len(dataset["shards"]),
                "referenced_bytes": referenced_bytes,
                "referenced_shard_identity_sha256": _identity_digest(
                    dataset["shards"]
                ),
                "source_manifest_sha256": dict(dataset["source_manifests"]),
            }
        )

    if collector_count != 35 or shadow_episode_count != 140_000:
        raise ValueError("collector or shadow totals differ")
    if audit_episode_count != 140_000:
        raise ValueError("audited shadow total differs")

    output.mkdir(parents=True)
    selected_path = output / "v2-inputs.sha256"
    selected_path.write_text(selected_text)
    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "source_attempt": str(source),
        "source_attempt_code_commit": args.expected_source_code_commit,
        "source_attempt_seal_manifest_sha256": source_seal_sha256,
        "source_sealed_failure_sha256": sha256_file(sealed_failure_path),
        "training_code_commit": args.training_code_commit,
        "orchestration_code_commit": args.orchestration_code_commit,
        "container_image_digest": args.container_image_digest,
        "principal_test_absent": True,
        "selected_manifest": {
            "path": str(selected_path),
            "sha256": selected_sha256,
            "files": len(selected),
            "bytes": selected_bytes,
        },
        "categories": categories,
        "collector_models": collector_count,
        "shadow_partitions": 35,
        "shadow_episodes": shadow_episode_count,
        "shadow_audits": 7,
        "shadow_audited_episodes": audit_episode_count,
        "family_datasets": dataset_records,
        "aggregation_code_commit": args.expected_source_code_commit,
        "reuse_justification": (
            "The corrected gate changes only post-training export verification; "
            "collection, shadow labelling, audit and aggregation inputs are reused "
            "without modification by their sealed hashes."
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["selected_manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
