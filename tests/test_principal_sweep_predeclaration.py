from pathlib import Path
import hashlib

import yaml

from airhockey_distill.envs import load_direct_launch_distribution


CONFIG_PATH = Path("configs/experiments/principal_sweep_v1.yaml")


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_principal_sweep_has_all_seven_families_and_five_matched_seeds():
    config = _config()

    assert [family["id"] for family in config["families"]] == [
        "feed_forward",
        "finite_stack_10",
        "structured_k0",
        "structured_k1",
        "structured_k2",
        "structured_k4",
        "gru_n64",
    ]
    assert config["matched_seeds"]["training"] == [
        14303,
        14304,
        14305,
        14306,
        14307,
    ]
    assert config["test_release_gate"]["required_final_student_checkpoints"] == 35


def test_shadow_budget_is_equal_paired_and_family_specific():
    shadow = _config()["shadow_labelling"]

    assert shadow["rounds"] == 1
    assert shadow["episodes_per_family"] == 20000
    assert shadow["collectors_per_family"] == 5
    assert shadow["episodes_per_collector"] == 4000
    assert shadow["collector_training_seeds"] == [
        14303,
        14304,
        14305,
        14306,
        14307,
    ]
    assert "same incoming shot and blackout" in shadow["paired_schedule"]
    assert "Never mix shadow episodes across families" in shadow["family_pooling"]
    assert shadow["weak_family_rule"].startswith("no_extra_rounds")
    assert shadow["loss_weighting"] == "equal_total_weight_per_complete_episode"


def test_validation_and_new_principal_test_are_fixed_and_paired():
    evaluation = _config()["evaluation"]
    validation_distribution = load_direct_launch_distribution(
        evaluation["validation"]["distribution_config"]
    )
    test_distribution = load_direct_launch_distribution(
        evaluation["test"]["distribution_config"]
    )

    assert validation_distribution.expected_shot_count("validation") == 225
    assert test_distribution.expected_shot_count("principal_test") == 225
    assert test_distribution.split("principal_test").seed == 5303
    assert evaluation["validation"]["blackout_steps"] == [0, 5, 10, 15, 20]
    assert evaluation["test"]["core_blackout_steps"] == [0, 5, 10, 15, 20]
    assert evaluation["test"]["extrapolation_blackout_steps"] == [25]
    assert evaluation["pairing"]["same_shot_and_blackout_across_families"]


def test_test_release_is_fail_closed_and_preliminary_results_are_excluded():
    config = _config()
    release = config["test_release_gate"]
    provenance = config["provenance"]

    assert release["action_if_any_check_fails"] == "keep_principal_test_closed"
    assert release["after_release"].startswith("no_retraining")
    assert release["require_test_outcomes_never_previously_inspected"]
    assert provenance["motivating_result_excluded_from_principal_statistics"]
    assert provenance["existing_structured_k2_pilot_excluded_from_principal_statistics"]


def test_parameter_and_state_accounting_is_predeclared():
    config = _config()
    families = {family["id"]: family for family in config["families"]}

    for rank in (0, 1, 2, 4):
        family = families[f"structured_k{rank}"]
        assert family["expected_total_parameters"] == 12002 + 163 * rank
        assert family["expected_core_parameters"] == 2304 + 163 * rank
    assert families["gru_n64"]["expected_total_parameters"] == 28898
    assert families["finite_stack_10"]["persistent_state_float32_values"] == 30
    assert config["measurements"]["cpu_latency"]["host"] == "marvin"
    assert config["measurements"]["cpu_latency"]["threads"] == 1
    assert config["measurements"]["cpu_latency"]["warmup_calls_per_checkpoint"] >= 10000
    assert config["measurements"]["cpu_latency"]["timed_calls_per_repetition"] >= 10000


def test_principal_test_config_is_hash_bound():
    config = _config()
    path = Path(config["evaluation"]["test"]["distribution_config"])

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        config["provenance"]["principal_test_config_sha256"]
    )
