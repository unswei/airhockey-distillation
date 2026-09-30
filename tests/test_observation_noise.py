from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from airhockey_distill.envs.observation_noise import NoisyPuckObservationAdapter, standard_noise
from airhockey_distill.envs.policy_interface import PublicObservationAdapter
from scripts.evaluate_observation_noise import setup
from scripts.analyse_observation_noise import analyse_records, validate_record


def test_zero_noise_is_bit_exact_and_input_is_unchanged():
    raw = np.linspace(-1.2, 1.2, 20, dtype=np.float32)
    before = raw.copy()
    adapter = NoisyPuckObservationAdapter(standard_noise("a", 42, 126), 0, [1.8847, .9747])
    for visible in (True, False):
        np.testing.assert_array_equal(adapter.adapt(raw, puck_visible=visible),
                                      PublicObservationAdapter().adapt(raw, puck_visible=visible))
    np.testing.assert_array_equal(raw, before)


def test_physical_scale_masking_clipping_and_repeated_reads():
    raw = np.zeros(20, dtype=np.float32)
    adapter = NoisyPuckObservationAdapter([[1., -1.], [2., 3.]], 5, [2., 1.])
    expected = np.zeros(19, dtype=np.float32)
    expected[16:19] = [.005, -.01, 1.]
    np.testing.assert_array_equal(adapter.adapt(raw, puck_visible=True), expected)
    np.testing.assert_array_equal(adapter.adapt(raw, puck_visible=True), expected)
    np.testing.assert_array_equal(adapter.adapt(raw, puck_visible=False), np.zeros(19))
    adapter.observation_step = 1
    np.testing.assert_allclose(adapter.adapt(raw, puck_visible=True)[16:18], [.01, .03])
    raw[16] = 1.2
    assert adapter.adapt(raw, puck_visible=True)[16] == 1
    adapter.observation_step = 2
    with pytest.raises(ValueError, match="index"):
        adapter.adapt(raw, puck_visible=True)


def test_noise_is_order_independent_and_scaled_across_levels():
    trace = standard_noise("alias-a", 42, 126)
    np.testing.assert_array_equal(trace, standard_noise("alias-a", 42, 126))
    np.testing.assert_array_equal(trace[:3], standard_noise("alias-a", 42, 3))
    assert not np.array_equal(trace, standard_noise("alias-b", 42, 126))
    one = NoisyPuckObservationAdapter(trace, 1, [2, 1]).adapt(np.zeros(20), puck_visible=True)
    five = NoisyPuckObservationAdapter(trace, 5, [2, 1]).adapt(np.zeros(20), puck_visible=True)
    np.testing.assert_allclose(five[16:18], 5 * one[16:18], rtol=1e-6)


@pytest.mark.parametrize("sigma,ranges", [(-1, [1, 1]), (float("nan"), [1, 1]), (1, [0, 1]), (1, [1])])
def test_invalid_noise_inputs_rejected(sigma, ranges):
    with pytest.raises(ValueError):
        NoisyPuckObservationAdapter([[0, 0]], sigma, ranges)


def test_fresh_shots_same_distribution_and_complete_predeclared_grid():
    cfg, _, shots = setup()
    original = yaml.safe_load(open("configs/env/direct_launch_v3_principal_test_v1.yaml"))
    extension = yaml.safe_load(open(cfg["distribution_config"]))
    assert original["sampling"] == extension["sampling"]
    assert original["geometry"] == extension["geometry"]
    assert original["splits"]["principal_test"]["seed"] != extension["splits"]["noise_test"]["seed"]
    assert len(shots) == 225
    assert len({s.alias_family_id for s in shots if s.alias_family_id}) == 90
    assert sum(s.alias_family_id is None for s in shots) == 45
    assert cfg["noise_std_mm"] == [0, 1, 5]
    assert cfg["blackout_steps"] == [0, 20]
    assert len(shots) * len(cfg["noise_std_mm"]) * len(cfg["blackout_steps"]) * (1 + 3 * 5) == 21600


def synthetic_records():
    cfg, _, shots = setup()
    cfg = deepcopy(cfg)
    cfg["analysis"]["bootstrap_replicates"] = 100
    opening = {"config": cfg, "episodes_per_policy": 1350,
               "shots": [s.as_dict() for s in shots], "checkpoints": {}}
    records = {}
    for family in ["teacher", *cfg["families"]]:
        for seed in ([None] if family == "teacher" else cfg["training_seeds"]):
            name = "teacher" if seed is None else f"{family}-{seed}"
            opening["checkpoints"][name] = {"sha256": name}
            records[name] = {"status": "completed", "classification": cfg["classification"],
                "family_id": family, "training_seed": seed, "opening_sha256": "opening",
                "checkpoint_sha256": name,
                "episodes": [{"shot_id": s.shot.shot_id, "alias_family_id": s.alias_family_id,
                    "launch_region": s.launch_region, "target_region": s.target_region,
                    "blackout_steps": b, "noise_std_mm": n, "steps": 25,
                    "outcome": "goal_conceded" if n == 5 and family != "teacher" else "returned"}
                    for s in shots for b in cfg["blackout_steps"] for n in cfg["noise_std_mm"]]}
    return opening, records


def test_paired_noise_analysis_has_correct_sign_and_no_teacher_replication():
    opening, records = synthetic_records()
    result = analyse_records(opening, "opening", records)
    assert result["episode_rows"] == 21600
    cell = result["groups"]["all"]["by_blackout_ms"]["400"]["5"]
    assert cell["policies"]["teacher"]["episodes"] == 225
    assert cell["policies"]["structured_k0"]["episodes"] == 1125
    assert cell["noise_minus_clean"]["structured_k0"] == {
        "estimate_points": -100., "pointwise_95_interval_points": [-100., -100.]}
    assert cell["paired_policy_contrasts"]["structured_k0_minus_teacher"]["estimate_points"] == -100.
    assert cell["paired_policy_contrasts"]["structured_k0_minus_gru_n64"]["pointwise_95_interval_points"] == [0., 0.]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "checkpoint", "opening", "noise", "outcome"])
def test_noise_analysis_rejects_incomplete_or_mismatched_evidence(fault):
    opening, records = synthetic_records()
    record = records["teacher"]
    if fault == "missing":
        record["episodes"].pop()
    elif fault == "duplicate":
        record["episodes"][0] = record["episodes"][1]
    elif fault in ("checkpoint", "opening"):
        record[f"{fault}_sha256"] = "wrong"
    elif fault == "noise":
        record["episodes"][0]["noise_std_mm"] = 999
    else:
        record["episodes"][0]["outcome"] = "rollout_limit"
    with pytest.raises(ValueError):
        validate_record(record, opening, "opening", "teacher", None)


def test_saved_noise_results_reproduce_from_public_episode_archive():
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "results/observation_noise_v1.json").read_text())
    bundle = json.loads(gzip.decompress((root / "results/observation_noise_v1_episodes.json.gz").read_bytes()))
    opening = json.loads(bundle["opening_json"])
    opening_hash = hashlib.sha256(bundle["opening_json"].encode()).hexdigest()
    assert opening_hash == saved["provenance"]["opening_sha256"]
    for relative, expected in opening["source_sha256"].items():
        # The manuscript audit was extended after inference to distinguish
        # Table 1 from the new Table 2; it is not an inference dependency.
        if relative == "scripts/audit_submission_numbers.py":
            continue
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    assert saved["provenance"]["analysis_script_sha256"] == hashlib.sha256(
        (root / "scripts/analyse_observation_noise.py").read_bytes()).hexdigest()
    for name, text in bundle["result_json"].items():
        assert hashlib.sha256(text.encode()).hexdigest() == saved["provenance"]["raw_results_sha256"][name]
    recomputed = analyse_records(opening, opening_hash, {n: json.loads(t) for n, t in bundle["result_json"].items()})
    assert recomputed == {k: v for k, v in saved.items() if k != "provenance"}


def test_principal_paper_audit_ignores_the_new_gru_row(monkeypatch):
    import importlib

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    audit = importlib.import_module("audit_submission_numbers")
    principal = r"\begin{table*}\label{tab:complete-comparison}GRU-64 & principal\end{table*}"
    noise = r"\begin{table}\label{tab:observation-noise}GRU-64 & noise\end{table}"
    assert audit.comparison_table_source(principal + noise) == principal
    with pytest.raises(ValueError):
        audit.comparison_table_source(noise)
