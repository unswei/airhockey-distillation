"""Fail-closed evidence gate for opening the principal test split."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from airhockey_distill.principal_sweep import (
    complete_shadow_schedule_sha256,
    evaluation_schedule,
    evaluation_schedule_sha256,
    load_principal_protocol,
    principal_family_spec,
    sha256_file,
    shadow_schedule_sha256,
    validate_shadow_partition_shards,
    validate_principal_dataset_manifest,
)
from airhockey_distill.principal_efficiency import policy_accounting
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS, load_principal_policy

IDENTITY_EVIDENCE = (
    "collector_checkpoints",
    "final_checkpoints",
    "validation_episode_files",
    "accounting_results",
    "cpu_latency_results",
)
FAMILY_EVIDENCE = ("family_datasets", "shadow_budget_audits")
EVIDENCE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "protocol_sha256",
        "aggregation_code_commit",
        "principal_test_outcomes_inspected",
        "analysis_plan",
        *IDENTITY_EVIDENCE,
        *FAMILY_EVIDENCE,
    }
)


def evaluate_test_release(
    protocol_path: str | Path,
    evidence_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return ``GO`` only when every predeclared artefact checks out."""

    resolved_protocol_path = Path(protocol_path).resolve()
    protocol = load_principal_protocol(resolved_protocol_path)
    raw_evidence_path = Path(evidence_manifest_path)
    evidence_path = raw_evidence_path.resolve()
    failures: list[str] = []
    manifest: dict[str, Any] = {}
    manifest_hash: str | None = None
    if not evidence_path.is_file():
        failures.append(f"evidence manifest does not exist: {evidence_path}")
    else:
        try:
            manifest_hash = sha256_file(evidence_path)
            loaded = json.loads(evidence_path.read_text())
            if not isinstance(loaded, dict):
                raise TypeError("evidence manifest must be a JSON object")
            manifest = loaded
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            failures.append(f"could not read evidence manifest: {error}")

    if manifest:
        try:
            _check_manifest_contract(
                protocol,
                resolved_protocol_path,
                evidence_path,
                manifest,
                failures,
            )
        except Exception as error:
            failures.append(
                "unexpected evidence-validation failure; test remains closed: "
                f"{type(error).__name__}: {error}"
            )
    test_config = Path(protocol["evaluation"]["test"]["distribution_config"])
    if sha256_file(test_config) != protocol["provenance"][
        "principal_test_config_sha256"
    ]:
        failures.append("principal test distribution config hash mismatch")
    test_schedule_hash = (
        principal_test_schedule_sha256(protocol) if not failures else None
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "decision": "GO" if not failures else "NO_GO",
        "closed_action": protocol["test_release_gate"][
            "action_if_any_check_fails"
        ],
        "protocol": str(resolved_protocol_path),
        "protocol_sha256": sha256_file(resolved_protocol_path),
        "immutable_protocol_sha256": protocol["immutable_protocol_sha256"],
        "evidence_manifest": str(evidence_path),
        "evidence_manifest_sha256": manifest_hash,
        "principal_test_config_sha256": protocol["provenance"][
            "principal_test_config_sha256"
        ],
        "principal_test_schedule_sha256": test_schedule_hash,
        "all_release_conditions_passed": not failures,
        "observed_evidence_counts": {
            name: len(manifest.get(name, []))
            if isinstance(manifest.get(name), list)
            else None
            for name in (*IDENTITY_EVIDENCE, *FAMILY_EVIDENCE)
        },
        "failures": failures,
        "declaration_note": (
            "The no-prior-inspection declaration is required and hash-frozen, "
            "but cannot be independently established from repository files."
        ),
    }


def validate_go_release_report(
    protocol_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Re-run a frozen release manifest before allowing test evaluation."""

    resolved_protocol = Path(protocol_path).resolve()
    resolved_report = Path(report_path).resolve()
    report = json.loads(resolved_report.read_text())
    if report.get("decision") != "GO" or report.get("status") != "completed":
        raise ValueError("principal test release report is not GO")
    if report.get("protocol_sha256") != sha256_file(resolved_protocol):
        raise ValueError("release report protocol hash mismatch")
    evidence_path = Path(str(report.get("evidence_manifest", ""))).resolve()
    if not evidence_path.is_file() or report.get("evidence_manifest_sha256") != (
        sha256_file(evidence_path)
    ):
        raise ValueError("release report evidence manifest is missing or changed")
    repeated = evaluate_test_release(resolved_protocol, evidence_path)
    if repeated["decision"] != "GO":
        raise ValueError("principal release evidence no longer passes")
    for name in (
        "immutable_protocol_sha256",
        "principal_test_config_sha256",
        "principal_test_schedule_sha256",
    ):
        if report.get(name) != repeated[name]:
            raise ValueError(f"release report {name} mismatch")
    return report


def principal_test_schedule(protocol: dict[str, Any]) -> tuple[tuple[Any, int], ...]:
    """Build the test schedule for the evaluator after release validation.

    This is an internal constructor, not an authorisation boundary. External
    callers must use ``validate_go_release_report`` first; the principal
    evaluator enforces that order.
    """

    specification = protocol["evaluation"]["test"]
    from airhockey_distill.envs import load_direct_launch_distribution

    distribution = load_direct_launch_distribution(
        specification["distribution_config"]
    )
    shots = distribution.generate(specification["split"])
    if len(shots) != int(specification["expected_shots"]):
        raise ValueError("principal test distribution has the wrong shot count")
    blackouts = tuple(
        int(value)
        for value in specification["core_blackout_steps"]
        + specification["extrapolation_blackout_steps"]
    )
    schedule = tuple((shot, blackout) for shot in shots for blackout in blackouts)
    if len(schedule) != int(specification["expected_total_episodes_per_student_seed"]):
        raise ValueError("principal test schedule has the wrong episode count")
    return schedule


def principal_test_schedule_sha256(protocol: dict[str, Any]) -> str:
    records = [
        {"shot_id": generated.shot.shot_id, "blackout_steps": blackout}
        for generated, blackout in principal_test_schedule(protocol)
    ]
    import hashlib

    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _check_manifest_contract(
    protocol: dict[str, Any],
    protocol_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    failures: list[str],
) -> None:
    expected_identities = {
        (family_id, int(seed))
        for family_id in PRINCIPAL_FAMILY_IDS
        for seed in protocol["matched_seeds"]["training"]
    }
    if int(manifest.get("schema_version", -1)) != 1:
        failures.append("evidence manifest schema_version must be 1")
    unknown_keys = set(manifest) - EVIDENCE_MANIFEST_KEYS
    missing_keys = EVIDENCE_MANIFEST_KEYS - set(manifest)
    if unknown_keys:
        failures.append(
            "evidence manifest has undeclared fields: "
            + ", ".join(sorted(unknown_keys))
        )
    if missing_keys:
        failures.append(
            "evidence manifest is missing fields: "
            + ", ".join(sorted(missing_keys))
        )
    if manifest.get("status") != "frozen":
        failures.append("evidence manifest status must be frozen")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        failures.append("evidence manifest protocol hash mismatch")
    if manifest.get("principal_test_outcomes_inspected") is not False:
        failures.append("principal test outcomes must be declared uninspected")
    code_commit = str(manifest.get("aggregation_code_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        failures.append("aggregation code commit must be a frozen 40-character hash")

    analysis_plan = manifest.get("analysis_plan")
    if not isinstance(analysis_plan, dict):
        failures.append("analysis plan evidence is missing")
    else:
        _check_file_hash(
            manifest_path,
            analysis_plan,
            "analysis plan",
            failures,
            lambda path, _: path
            == (protocol_path.parents[2] / "docs/principal_sweep_v1.md").resolve(),
        )

    indexed: dict[str, dict[tuple[str, int], tuple[dict[str, Any], Path, Any]]] = {}
    for category in IDENTITY_EVIDENCE:
        indexed[category] = _indexed_evidence(
            manifest_path,
            manifest.get(category),
            category,
            expected_identities,
            failures,
        )
    family_indexed: dict[str, dict[str, tuple[dict[str, Any], Path, Any]]] = {}
    for category in FAMILY_EVIDENCE:
        family_indexed[category] = _family_evidence(
            manifest_path,
            manifest.get(category),
            category,
            failures,
        )

    for identity, (entry, path, _) in indexed["collector_checkpoints"].items():
        _check_checkpoint(
            protocol, protocol_path, identity, entry, path, "collector", failures
        )
    final_hashes: dict[tuple[str, int], str] = {}
    final_paths: dict[tuple[str, int], Path] = {}
    for identity, (entry, path, _) in indexed["final_checkpoints"].items():
        _check_checkpoint(
            protocol, protocol_path, identity, entry, path, "final", failures
        )
        final_hashes[identity] = str(entry.get("sha256"))
        final_paths[identity] = path
    dataset_hashes: dict[str, str] = {}
    dataset_values: dict[str, dict[str, Any]] = {}
    for family_id, (entry, path, value) in family_indexed["family_datasets"].items():
        if not isinstance(value, dict):
            failures.append(f"family dataset {family_id} is not JSON")
            continue
        try:
            validate_principal_dataset_manifest(
                protocol,
                value,
                str(entry["sha256"]),
                family_id=family_id,
                stage="final",
            )
            if value.get("code_commit") != code_commit:
                raise ValueError("aggregation code commit mismatch")
            _check_dataset_shards(path.parent, value)
            dataset_hashes[family_id] = str(entry["sha256"])
            dataset_values[family_id] = value
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"family dataset {family_id}: {error}")
    collector_hashes = {
        identity: str(entry.get("sha256"))
        for identity, (entry, _, _) in indexed["collector_checkpoints"].items()
    }
    for family_id, (_, _, value) in family_indexed["shadow_budget_audits"].items():
        _check_shadow_audit(
            protocol,
            protocol_path,
            family_id,
            value,
            collector_hashes,
            dataset_values.get(family_id),
            failures,
        )
    for identity, (_, path, _) in indexed["final_checkpoints"].items():
        try:
            if load_principal_policy(identity[0], path).metadata.get(
                "dataset_manifest_sha256"
            ) != dataset_hashes.get(identity[0]):
                raise ValueError("final checkpoint uses a different family dataset")
        except (TypeError, ValueError) as error:
            failures.append(
                f"final checkpoint {identity[0]} seed {identity[1]}: {error}"
            )
    for identity, (_, _, value) in indexed["validation_episode_files"].items():
        _check_validation(
            protocol,
            protocol_path,
            identity,
            value,
            final_hashes.get(identity),
            failures,
        )
    for category in ("accounting_results", "cpu_latency_results"):
        for identity, (_, _, value) in indexed[category].items():
            _check_efficiency(
                protocol,
                protocol_path,
                category,
                identity,
                value,
                final_hashes.get(identity),
                final_paths.get(identity),
                failures,
            )


def _indexed_evidence(
    manifest_path: Path,
    raw_entries: Any,
    category: str,
    expected: set[tuple[str, int]],
    failures: list[str],
) -> dict[tuple[str, int], tuple[dict[str, Any], Path, Any]]:
    if not isinstance(raw_entries, list):
        failures.append(f"{category} must be a list of 35 frozen files")
        return {}
    result: dict[tuple[str, int], tuple[dict[str, Any], Path, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            failures.append(f"{category} contains a non-object entry")
            continue
        try:
            identity = (str(entry["family_id"]), int(entry["training_seed"]))
        except (KeyError, TypeError, ValueError):
            failures.append(f"{category} entry has no valid family/seed identity")
            continue
        if identity in result:
            failures.append(f"{category} duplicates {identity[0]} seed {identity[1]}")
            continue
        checked = _read_frozen_file(manifest_path, entry, category, failures)
        if checked is not None:
            result[identity] = (entry, *checked)
    missing = expected - set(result)
    extra = set(result) - expected
    if missing:
        failures.append(f"{category} is missing {len(missing)} family/seed entries")
    if extra:
        failures.append(f"{category} has {len(extra)} unexpected family/seed entries")
    return result


def _family_evidence(
    manifest_path: Path,
    raw_entries: Any,
    category: str,
    failures: list[str],
) -> dict[str, tuple[dict[str, Any], Path, Any]]:
    if not isinstance(raw_entries, list):
        failures.append(f"{category} must be a list of seven frozen files")
        return {}
    result: dict[str, tuple[dict[str, Any], Path, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict) or "family_id" not in entry:
            failures.append(f"{category} entry has no family identity")
            continue
        family_id = str(entry["family_id"])
        if family_id in result:
            failures.append(f"{category} duplicates {family_id}")
            continue
        checked = _read_frozen_file(manifest_path, entry, category, failures)
        if checked is not None:
            result[family_id] = (entry, *checked)
    missing = set(PRINCIPAL_FAMILY_IDS) - set(result)
    extra = set(result) - set(PRINCIPAL_FAMILY_IDS)
    if missing:
        failures.append(f"{category} is missing {len(missing)} families")
    if extra:
        failures.append(f"{category} has {len(extra)} unexpected families")
    return result


def _read_frozen_file(
    manifest_path: Path,
    entry: dict[str, Any],
    label: str,
    failures: list[str],
) -> tuple[Path, Any] | None:
    try:
        raw_path = Path(str(entry["path"]))
        path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        path = path.resolve()
        if sha256_file(path) != str(entry["sha256"]):
            raise ValueError("hash mismatch")
        value: Any = None
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text())
        return path, value
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, ValueError) as error:
        failures.append(f"{label} evidence could not be frozen: {error}")
        return None


def _check_file_hash(
    manifest_path: Path,
    entry: dict[str, Any],
    label: str,
    failures: list[str],
    validator: Callable[[Path, Any], bool] | None = None,
) -> None:
    checked = _read_frozen_file(manifest_path, entry, label, failures)
    if checked is not None and validator is not None and not validator(*checked):
        failures.append(f"{label} is not the canonical principal sweep analysis plan")


def _check_checkpoint(
    protocol: dict[str, Any],
    protocol_path: Path,
    identity: tuple[str, int],
    entry: dict[str, Any],
    path: Path,
    stage: str,
    failures: list[str],
) -> None:
    family_id, seed = identity
    try:
        principal_family_spec(protocol, family_id)
        policy = load_principal_policy(family_id, path)
        metadata = policy.metadata
        if metadata.get("student_id") != family_id:
            raise ValueError("family mismatch")
        if int(metadata.get("training_seed", -1)) != seed:
            raise ValueError("training seed mismatch")
        if metadata.get("training_stage") != stage:
            raise ValueError(f"checkpoint is not from the {stage} stage")
        if metadata.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("protocol hash mismatch")
        if stage == "collector" and metadata.get("dataset_manifest_sha256") != (
            protocol["base_teacher_dataset"]["manifest_sha256"]
        ):
            raise ValueError("collector did not use the common base dataset")
        if str(entry.get("sha256")) != sha256_file(path):
            raise ValueError("frozen checkpoint hash changed")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{stage} checkpoint {family_id} seed {seed}: {error}")


def _check_shadow_audit(
    protocol: dict[str, Any],
    protocol_path: Path,
    family_id: str,
    value: Any,
    collector_hashes: dict[tuple[str, int], str],
    dataset_manifest: dict[str, Any] | None,
    failures: list[str],
) -> None:
    try:
        if not isinstance(value, dict):
            raise ValueError("audit is not JSON")
        if value.get("decision") != "GO" or value.get("status") != "completed":
            raise ValueError("audit did not pass")
        if value.get("family_id") != family_id:
            raise ValueError("family mismatch")
        if value.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("protocol hash mismatch")
        if value.get("shadow_schedule_sha256") != complete_shadow_schedule_sha256(protocol):
            raise ValueError("schedule hash mismatch")
        if int(value.get("realised_episode_budget", -1)) != int(
            protocol["shadow_labelling"]["episodes_per_family"]
        ):
            raise ValueError("episode budget mismatch")
        partitions = value.get("partitions", [])
        if len(partitions) != len(protocol["matched_seeds"]["training"]):
            raise ValueError("audit does not contain five partitions")
        if bool(value.get("adaptive_top_up_used")):
            raise ValueError("adaptive top-up was used")
        if sum(int(entry.get("episode_count", -1)) for entry in partitions) != int(
            value["realised_episode_budget"]
        ):
            raise ValueError("partition episode counts do not sum to the budget")
        if sum(
            int(entry.get("teacher_query_count", -1)) for entry in partitions
        ) != int(value.get("realised_teacher_query_count", -1)):
            raise ValueError("partition query counts do not sum to the audit total")
        expected_seeds = {
            int(seed) for seed in protocol["matched_seeds"]["training"]
        }
        observed_seeds = {
            int(entry.get("collector_training_seed", -1)) for entry in partitions
        }
        if observed_seeds != expected_seeds:
            raise ValueError("partition collector seeds are incomplete")
        if dataset_manifest is None:
            raise ValueError("matching family dataset is absent")
        sources = dataset_manifest.get("source_manifests", {})
        for entry in partitions:
            seed = int(entry["collector_training_seed"])
            if entry.get("student_checkpoint_sha256") != collector_hashes.get(
                (family_id, seed)
            ):
                raise ValueError(
                    f"seed {seed} partition uses another collector checkpoint"
                )
            if entry.get("teacher_checkpoint_sha256") != protocol["teacher"][
                "actor_sha256"
            ]:
                raise ValueError(f"seed {seed} partition uses another teacher")
            if entry.get("manifest_sha256") != sources.get(f"shadow_seed_{seed}"):
                raise ValueError(
                    f"seed {seed} partition is not in the family aggregate"
                )
            partition_manifest_path = Path(str(entry.get("manifest", ""))).resolve()
            if sha256_file(partition_manifest_path) != entry.get("manifest_sha256"):
                raise ValueError(f"seed {seed} partition manifest hash mismatch")
            partition_manifest = json.loads(partition_manifest_path.read_text())
            if partition_manifest.get("status") != "completed":
                raise ValueError(f"seed {seed} partition is incomplete")
            if partition_manifest.get("family_id") != family_id:
                raise ValueError(f"seed {seed} partition family mismatch")
            if int(partition_manifest.get("collector_training_seed", -1)) != seed:
                raise ValueError(f"seed {seed} partition seed mismatch")
            if partition_manifest.get("protocol_sha256") != sha256_file(protocol_path):
                raise ValueError(f"seed {seed} partition protocol mismatch")
            if partition_manifest.get("shadow_schedule_sha256") != (
                shadow_schedule_sha256(protocol, seed)
            ):
                raise ValueError(f"seed {seed} partition schedule mismatch")
            counts = validate_shadow_partition_shards(
                protocol,
                seed,
                partition_manifest_path.parent,
                partition_manifest,
            )
            if counts["episode_count"] != int(entry.get("episode_count", -1)) or (
                counts["teacher_query_count"]
                != int(entry.get("teacher_query_count", -1))
            ):
                raise ValueError(f"seed {seed} partition audit counts changed")
    except (TypeError, ValueError) as error:
        failures.append(f"shadow budget audit {family_id}: {error}")


def _check_dataset_shards(directory: Path, manifest: dict[str, Any]) -> None:
    episode_total = 0
    transition_total = 0
    for entry in manifest.get("shards", []):
        raw_path = Path(str(entry["file"]))
        path = raw_path if raw_path.is_absolute() else directory / raw_path
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {path}")
        episode_total += int(entry["episodes"])
        transition_total += int(entry["transitions"])
    if episode_total != int(manifest.get("episode_count", -1)):
        raise ValueError("dataset shard episode counts do not match the manifest")
    if transition_total != int(manifest.get("transition_count", -1)):
        raise ValueError("dataset shard transition counts do not match the manifest")


def _check_validation(
    protocol: dict[str, Any],
    protocol_path: Path,
    identity: tuple[str, int],
    value: Any,
    checkpoint_hash: str | None,
    failures: list[str],
) -> None:
    family_id, seed = identity
    try:
        if not isinstance(value, dict):
            raise ValueError("validation evidence is not JSON")
        if value.get("status") != "completed":
            raise ValueError("validation did not complete")
        if (value.get("family_id"), int(value.get("training_seed", -1))) != identity:
            raise ValueError("family/seed mismatch")
        if value.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("protocol hash mismatch")
        if value.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("validation uses a different final checkpoint")
        if value.get("evaluation_split") != protocol["evaluation"]["validation"]["split"]:
            raise ValueError("file is not from the validation split")
        if value.get("evaluation_schedule_sha256") != evaluation_schedule_sha256(protocol):
            raise ValueError("validation schedule hash mismatch")
        episodes = value.get("episodes")
        schedule = evaluation_schedule(protocol)
        if not isinstance(episodes, list) or len(episodes) != len(schedule):
            raise ValueError("validation episode rows are incomplete")
        for index, (episode, (generated, blackout)) in enumerate(zip(episodes, schedule, strict=True)):
            if (
                int(episode.get("schedule_index", -1)) != index
                or episode.get("shot_id") != generated.shot.shot_id
                or int(episode.get("blackout_steps", -1)) != blackout
            ):
                raise ValueError(f"validation episode {index} is off schedule")
    except (TypeError, ValueError) as error:
        failures.append(f"validation {family_id} seed {seed}: {error}")


def _check_efficiency(
    protocol: dict[str, Any],
    protocol_path: Path,
    category: str,
    identity: tuple[str, int],
    value: Any,
    checkpoint_hash: str | None,
    checkpoint_path: Path | None,
    failures: list[str],
) -> None:
    family_id, seed = identity
    try:
        if not isinstance(value, dict):
            raise ValueError("evidence is not JSON")
        if value.get("decision") != "GO" or value.get("status") != "completed":
            raise ValueError("measurement did not pass")
        if (value.get("family_id"), int(value.get("training_seed", -1))) != identity:
            raise ValueError("family/seed mismatch")
        if value.get("protocol_sha256") != sha256_file(protocol_path):
            raise ValueError("protocol hash mismatch")
        if value.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("measurement uses a different final checkpoint")
        if category == "accounting_results":
            if not value.get("checks") or not all(value["checks"].values()):
                raise ValueError("accounting checks are incomplete or failed")
            family = principal_family_spec(protocol, family_id)
            if int(value.get("total_trainable_parameters", -1)) != int(
                family["expected_total_parameters"]
            ) or int(value.get("core_parameters", -1)) != int(
                family["expected_core_parameters"]
            ):
                raise ValueError("parameter accounting disagrees with the predeclaration")
            if checkpoint_path is None:
                raise ValueError("matching final checkpoint is absent")
            expected = policy_accounting(
                protocol,
                family_id,
                load_principal_policy(family_id, checkpoint_path),
            )
            for name in (
                "total_trainable_parameters",
                "core_parameters",
                "exported_tensor_element_counts",
                "exported_parameter_bytes",
                "recurrent_memory_bytes",
                "total_policy_carry_bytes",
                "carry_arrays",
                "multiply_adds_per_step",
                "multiply_add_definition",
            ):
                if value.get(name) != expected[name]:
                    raise ValueError(f"accounting field {name} differs from the checkpoint")
        else:
            specification = protocol["measurements"]["cpu_latency"]
            runtime = value.get("runtime", {})
            measured = value.get("measurements", {})
            if runtime.get("decision") != "GO":
                raise ValueError("latency runtime contract failed")
            if runtime.get("hostname", "").split(".", 1)[0] != specification["host"]:
                raise ValueError("latency was not measured on Marvin")
            expected_cpu_words = set(
                re.findall(r"[a-z0-9]+", str(specification["cpu"]).lower())
            )
            observed_cpu_words = set(
                re.findall(r"[a-z0-9]+", str(runtime.get("cpu_model", "")).lower())
            )
            if not expected_cpu_words <= observed_cpu_words:
                raise ValueError("latency CPU model differs from the predeclaration")
            if runtime.get("logical_cpu_affinity") != [
                int(specification["pinned_logical_cpu"])
            ]:
                raise ValueError("latency process was not pinned to logical CPU 0")
            expected_environment = {
                str(name): str(expected)
                for name, expected in specification["environment_variables"].items()
            }
            if runtime.get("environment_variables") != expected_environment:
                raise ValueError("latency thread environment differs from the predeclaration")
            contract = value.get("runtime_contract", {})
            if (
                contract.get("runtime") != specification["runtime"]
                or int(contract.get("processes", -1)) != int(specification["processes"])
                or int(contract.get("threads", -1)) != int(specification["threads"])
                or contract.get("include") != specification["include"]
                or contract.get("exclude") != specification["exclude"]
                or contract.get("container_digest")
                != protocol["provenance"]["container_digest"]
            ):
                raise ValueError("latency runtime contract differs from the predeclaration")
            if int(measured.get("warmup_calls", -1)) != int(
                specification["warmup_calls_per_checkpoint"]
            ) or int(measured.get("timed_calls_per_repetition", -1)) != int(
                specification["timed_calls_per_repetition"]
            ) or int(measured.get("repetitions", -1)) != int(
                specification["repetitions_per_checkpoint"]
            ):
                raise ValueError("latency call counts differ from the predeclaration")
            for name in ("median_microseconds", "p95_microseconds"):
                if float(measured.get(name, -1.0)) < 0.0:
                    raise ValueError(f"latency {name} is absent")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{category} {family_id} seed {seed}: {error}")
