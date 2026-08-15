import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airhockey_distill.principal_release import (
    _check_efficiency,
    _check_mixed_provenance,
    _check_validation,
    evaluate_test_release,
    principal_test_schedule,
    validate_go_release_report,
)
from airhockey_distill.principal_sweep import load_principal_protocol, sha256_file
from airhockey_distill.students import (
    PRINCIPAL_FAMILY_IDS,
    initialise_feed_forward_parameters,
    save_principal_checkpoint,
)
from scripts.evaluate_principal_student import run as evaluate_student


PROTOCOL = Path("configs/experiments/principal_sweep_execution_v1.yaml")


def test_release_gate_is_no_go_when_manifest_is_absent(tmp_path):
    result = evaluate_test_release(PROTOCOL, tmp_path / "missing.json")

    assert result["decision"] == "NO_GO"
    assert result["closed_action"] == "keep_principal_test_closed"
    assert result["failures"]
    assert result["principal_test_schedule_sha256"] is None


def test_release_gate_is_no_go_for_the_unfilled_template():
    result = evaluate_test_release(
        PROTOCOL,
        "configs/experiments/principal_test_release_evidence_template_v1.json",
    )

    assert result["decision"] == "NO_GO"
    assert any("schema_version must be 2" in value for value in result["failures"])
    assert any("status must be frozen" in value for value in result["failures"])
    assert any("provenance_manifests" in value for value in result["failures"])
    assert any("35" in value for value in result["failures"])


def test_release_evidence_helpers_accept_the_manifest_call_contract(tmp_path):
    failures = []

    _check_validation({}, tmp_path, ("feed_forward", 14303), None, None, failures)
    _check_efficiency(
        {},
        tmp_path,
        "accounting_results",
        ("feed_forward", 14303),
        None,
        None,
        tmp_path / "checkpoint.npz",
        failures,
    )

    assert len(failures) == 2


def test_principal_test_schedule_is_not_the_validation_schedule():
    protocol = load_principal_protocol(PROTOCOL)
    schedule = principal_test_schedule(protocol)

    assert len(schedule) == 1350
    assert [blackout for _, blackout in schedule[:6]] == [0, 5, 10, 15, 20, 25]


def test_test_evaluation_rejects_a_missing_release_report(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    save_principal_checkpoint(
        "feed_forward",
        checkpoint,
        initialise_feed_forward_parameters(14303),
        {
            "student_id": "feed_forward",
            "training_seed": 14303,
            "training_stage": "final",
            "protocol_sha256": sha256_file(PROTOCOL),
        },
    )
    args = type(
        "Args",
        (),
        {
            "protocol": PROTOCOL,
            "family": "feed_forward",
            "seed": 14303,
            "checkpoint": checkpoint,
            "split": "test",
            "release_report": None,
            "output": tmp_path / "result.json",
            "code_commit": "0" * 40,
        },
    )()

    with pytest.raises(ValueError, match="requires a GO release report"):
        evaluate_student(args)


def test_complete_frozen_manifest_can_go_and_a_changed_hash_closes_it(
    monkeypatch, tmp_path
):
    protocol = load_principal_protocol(PROTOCOL)
    identities = [
        (family_id, int(seed))
        for family_id in PRINCIPAL_FAMILY_IDS
        for seed in protocol["matched_seeds"]["training"]
    ]
    blob = tmp_path / "checkpoint.npz"
    blob.write_bytes(b"frozen-checkpoint")
    common = tmp_path / "evidence.json"
    common.write_text(json.dumps({"code_commit": "a" * 40}) + "\n")
    dataset_hash = sha256_file(common)

    def file_entry(path):
        return {"path": str(path), "sha256": sha256_file(path)}

    identity_entries = [
        {
            **file_entry(common),
            "family_id": family_id,
            "training_seed": seed,
        }
        for family_id, seed in identities
    ]
    checkpoint_entries = [
        {
            **file_entry(blob),
            "family_id": family_id,
            "training_seed": seed,
        }
        for family_id, seed in identities
    ]
    family_entries = [
        {**file_entry(common), "family_id": family_id}
        for family_id in PRINCIPAL_FAMILY_IDS
    ]
    manifest = {
        "schema_version": 2,
        "status": "frozen",
        "protocol_sha256": sha256_file(PROTOCOL),
        "aggregation_code_commit": "a" * 40,
        "principal_test_outcomes_inspected": False,
        "analysis_plan": file_entry(Path("docs/principal_sweep_v1.md").resolve()),
        "provenance_manifests": {
            "v3_final_training": file_entry(common),
            "v3_measurements": file_entry(common),
            "v4_structured_optimisation": file_entry(common),
        },
        "collector_checkpoints": checkpoint_entries,
        "final_checkpoints": checkpoint_entries,
        "family_datasets": family_entries,
        "shadow_budget_audits": family_entries,
        "validation_episode_files": identity_entries,
        "accounting_results": identity_entries,
        "cpu_latency_results": identity_entries,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(
        "airhockey_distill.principal_release._check_checkpoint",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release.validate_principal_dataset_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release._check_dataset_shards",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release._check_shadow_audit",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release._check_validation",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release._check_efficiency",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release._check_mixed_provenance",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release.load_principal_policy",
        lambda *args: SimpleNamespace(
            metadata={"dataset_manifest_sha256": dataset_hash}
        ),
    )

    result = evaluate_test_release(PROTOCOL, manifest_path)
    assert result["decision"] == "GO"
    assert result["all_release_conditions_passed"]
    assert result["observed_evidence_counts"] == {
        "collector_checkpoints": 35,
        "final_checkpoints": 35,
        "validation_episode_files": 35,
        "accounting_results": 35,
        "cpu_latency_results": 35,
        "family_datasets": 7,
        "shadow_budget_audits": 7,
    }
    assert result["observed_provenance_manifests"] == [
        "v3_final_training",
        "v3_measurements",
        "v4_structured_optimisation",
    ]
    assert result["principal_test_schedule_sha256"]

    manifest["final_checkpoints"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    changed = evaluate_test_release(PROTOCOL, manifest_path)
    assert changed["decision"] == "NO_GO"
    assert not changed["all_release_conditions_passed"]
    assert changed["principal_test_schedule_sha256"] is None


def test_go_report_is_rechecked_against_its_manifest(monkeypatch, tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n")
    protocol_hash = sha256_file(PROTOCOL)
    report = {
        "status": "completed",
        "decision": "GO",
        "protocol_sha256": protocol_hash,
        "evidence_manifest": str(evidence),
        "evidence_manifest_sha256": sha256_file(evidence),
        "immutable_protocol_sha256": "immutable",
        "principal_test_config_sha256": "test-config",
        "principal_test_schedule_sha256": "schedule",
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(
        "airhockey_distill.principal_release.evaluate_test_release",
        lambda protocol, manifest: {
            "decision": "NO_GO",
            "immutable_protocol_sha256": "immutable",
            "principal_test_config_sha256": "test-config",
            "principal_test_schedule_sha256": "schedule",
        },
    )

    with pytest.raises(ValueError, match="no longer passes"):
        validate_go_release_report(PROTOCOL, report_path)


def test_mixed_v3_v4_provenance_binds_selected_results_and_fails_closed(
    monkeypatch, tmp_path
):
    identities = {("feed_forward", 14303), ("structured_k2", 14303)}
    structured_identity = ("structured_k2", 14303)
    training_commit = "a" * 40
    measurement_commit = "b" * 40
    optimisation_commit = "c" * 40
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("protocol\n")
    protocol_hash = sha256_file(protocol_path)
    source_v2_seal = "d" * 64
    final_paths = {}
    final_hashes = {}
    for family_id, seed in identities:
        checkpoint = tmp_path / f"{family_id}-{seed}.npz"
        checkpoint.write_bytes(family_id.encode())
        final_paths[(family_id, seed)] = checkpoint
        final_hashes[(family_id, seed)] = sha256_file(checkpoint)

    selected_hashes = {
        category: {
            identity: sha256_file(
                _write_bytes(
                    tmp_path / f"{category}-{identity[0]}.json",
                    f"{category}-{identity[0]}".encode(),
                )
            )
            for identity in identities
        }
        for category in (
            "validation_episode_files",
            "accounting_results",
            "cpu_latency_results",
        )
    }
    indexed = {
        category: {
            identity: (
                {"sha256": selected_hashes[category][identity]},
                tmp_path / f"{category}-{identity[0]}.json",
                (
                    {
                        "code_commit": (
                            optimisation_commit
                            if category == "cpu_latency_results"
                            or (
                                category == "validation_episode_files"
                                and identity == structured_identity
                            )
                            else measurement_commit
                        ),
                        **(
                            {
                                "runtime": {
                                    "inference_implementation": (
                                        "native_pairwise_float32_v1"
                                        if identity == structured_identity
                                        else "numpy_float32_v1"
                                    )
                                }
                            }
                            if category == "validation_episode_files"
                            else {}
                        ),
                        **(
                            {
                                "runtime_contract": {
                                    "inference_implementation": (
                                        "native_pairwise_float32_v1"
                                        if identity == structured_identity
                                        else "numpy_float32_v1"
                                    )
                                }
                            }
                            if category == "cpu_latency_results"
                            else {}
                        ),
                    }
                ),
            )
            for identity in identities
        }
        for category in selected_hashes
    }

    final_manifest = {
        "status": "completed",
        "final_models": len(identities),
        "principal_test_opened": False,
        "training_code_commit": training_commit,
        "models": [
            {
                "family_id": family_id,
                "training_seed": seed,
                "checkpoint_sha256": final_hashes[(family_id, seed)],
            }
            for family_id, seed in sorted(identities)
        ],
    }
    final_manifest_path = _write_json(tmp_path / "final.json", final_manifest)
    final_entry = {
        "path": str(final_manifest_path),
        "sha256": sha256_file(final_manifest_path),
    }
    v3_measurements = {
        "status": "completed",
        "decision": "GO",
        "principal_test_opened": False,
        "protocol_sha256": protocol_hash,
        "validation_schedule_sha256": "validation-schedule",
        "measurement_code_commit": measurement_commit,
        "source_v2_seal_sha256": source_v2_seal,
        "final_training_manifest_sha256": final_entry["sha256"],
        "models": [
            {
                "family_id": family_id,
                "training_seed": seed,
                "checkpoint_sha256": final_hashes[(family_id, seed)],
                "validation_sha256": selected_hashes[
                    "validation_episode_files"
                ][(family_id, seed)],
                "accounting_sha256": selected_hashes["accounting_results"][
                    (family_id, seed)
                ],
            }
            for family_id, seed in sorted(identities)
        ],
    }
    v3_path = _write_json(tmp_path / "v3.json", v3_measurements)
    v3_entry = {"path": str(v3_path), "sha256": sha256_file(v3_path)}

    v4_root = tmp_path / "v4"
    native_directory = v4_root / "native"
    verification_directory = v4_root / "verification"
    native_directory.mkdir(parents=True)
    verification_directory.mkdir()
    binary = _write_bytes(
        native_directory / "_airhockey_pairwise_float32.test.so",
        b"native",
    )
    build = {
        "implementation": "native_pairwise_float32_v1",
        "code_commit": optimisation_commit,
        "binary_sha256": sha256_file(binary),
    }
    build_path = _write_json(native_directory / "build.json", build)
    native_verification = {
        "decision": "GO",
        "code_commit": optimisation_commit,
        "loaded_binary_sha256": sha256_file(binary),
        "linear_kernel": {"bit_exact": True},
        "checkpoints": [
            {
                "family_id": structured_identity[0],
                "training_seed": structured_identity[1],
                "checkpoint_sha256": final_hashes[structured_identity],
                "passed": True,
                "action_bit_exact": True,
                "carry_bit_exact": True,
                "action_maximum_absolute_error": 0.0,
                "carry_maximum_absolute_error": 0.0,
            }
        ],
    }
    verification_path = _write_json(
        verification_directory / "native-kernel.json",
        native_verification,
    )
    v4 = {
        "status": "completed",
        "decision": "GO",
        "principal_test_opened": False,
        "protocol_sha256": protocol_hash,
        "validation_schedule_sha256": "validation-schedule",
        "optimisation_code_commit": optimisation_commit,
        "source_v2_seal_sha256": source_v2_seal,
        "source_v3_final_manifest_sha256": final_entry["sha256"],
        "source_v3_measurement_manifest_sha256": v3_entry["sha256"],
        "native_build_manifest_sha256": sha256_file(build_path),
        "native_binary_sha256": sha256_file(binary),
        "native_verification_sha256": sha256_file(verification_path),
        "native_linear_bit_exact": True,
        "native_verified_structured_checkpoints": 1,
        "structured_validation_reruns": 1,
        "structured_validation_episode_rows_identical_to_v3": True,
        "reused_v3_validation_results": 1,
        "reused_v3_accounting_results": 2,
        "new_isolated_cpu_latency_results": 2,
        "models": [
            {
                "family_id": family_id,
                "training_seed": seed,
                "checkpoint_sha256": final_hashes[(family_id, seed)],
                "validation_sha256": selected_hashes[
                    "validation_episode_files"
                ][(family_id, seed)],
                "accounting_sha256": selected_hashes["accounting_results"][
                    (family_id, seed)
                ],
                "cpu_latency_sha256": selected_hashes["cpu_latency_results"][
                    (family_id, seed)
                ],
                "validation_source": (
                    "V4 native rerun"
                    if (family_id, seed) == structured_identity
                    else "V3 unchanged family reuse"
                ),
                "inference_implementation": (
                    "native_pairwise_float32_v1"
                    if (family_id, seed) == structured_identity
                    else "numpy_float32_v1"
                ),
            }
            for family_id, seed in sorted(identities)
        ],
    }
    v4_path = _write_json(v4_root / "manifest.json", v4)
    provenance = {
        "v3_final_training": (final_entry, final_manifest_path, final_manifest),
        "v3_measurements": (v3_entry, v3_path, v3_measurements),
        "v4_structured_optimisation": (
            {"path": str(v4_path), "sha256": sha256_file(v4_path)},
            v4_path,
            v4,
        ),
    }
    monkeypatch.setattr(
        "airhockey_distill.principal_release.evaluation_schedule_sha256",
        lambda protocol: "validation-schedule",
    )
    monkeypatch.setattr(
        "airhockey_distill.principal_release.load_principal_policy",
        lambda *args: SimpleNamespace(metadata={"code_commit": training_commit}),
    )

    failures = []
    _check_mixed_provenance(
        {},
        protocol_path,
        provenance,
        indexed,
        identities,
        final_hashes,
        final_paths,
        failures,
    )
    assert failures == []

    indexed["cpu_latency_results"][structured_identity][2]["code_commit"] = "e" * 40
    _check_mixed_provenance(
        {},
        protocol_path,
        provenance,
        indexed,
        identities,
        final_hashes,
        final_paths,
        failures,
    )
    assert any("latency commit differs" in failure for failure in failures)


def _write_bytes(path, value):
    path.write_bytes(value)
    return path


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path
