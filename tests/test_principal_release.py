import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airhockey_distill.principal_release import (
    _check_efficiency,
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
    assert any("status must be frozen" in value for value in result["failures"])
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
        "schema_version": 1,
        "status": "frozen",
        "protocol_sha256": sha256_file(PROTOCOL),
        "aggregation_code_commit": "a" * 40,
        "principal_test_outcomes_inspected": False,
        "analysis_plan": file_entry(Path("docs/principal_sweep_v1.md").resolve()),
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
