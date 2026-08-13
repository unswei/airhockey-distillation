from pathlib import Path

import numpy as np
import pytest
import yaml

from airhockey_distill.students import GRURecurrentPolicy, initialise_gru_parameters
from airhockey_distill.students.gru_torch import GRURecurrentModule
from scripts.train_gru_student import (
    validate_gru_full_training_config,
    verify_gru_export,
)
from scripts.train_structured_student import pad_episode_split


def test_gru_pilot_matches_structured_training_budget_and_split():
    gru = yaml.safe_load(
        Path(
            "configs/student/gru_n64_full_pilot_seed_14303_v1.yaml"
        ).read_text()
    )
    structured = yaml.safe_load(
        Path("configs/student/structured_n64_k2_shadow_round1_v1.yaml").read_text()
    )

    validate_gru_full_training_config(gru)

    assert gru["dataset"] == structured["dataset"]
    assert gru["training"] == structured["training"]
    assert gru["export"] == structured["export"]
    assert gru["policy"] == {
        "config": "configs/student/gru_n64.yaml",
        "id": "gru_n64",
        "state_dimension": 64,
    }
    assert gru["pilot_scope"]["classification"] == (
        "engineering_pilot_not_principal_family_comparison"
    )
    assert gru["provenance"]["aggregate_manifest_sha256"] == (
        "e67438a24da12cfc1aed92f6df64253f82ce000cb52c3085d28468007ebf0d0d"
    )


def test_gru_full_config_rejects_masked_prefix_and_principal_claim():
    config = yaml.safe_load(
        Path(
            "configs/student/gru_n64_full_pilot_seed_14303_v1.yaml"
        ).read_text()
    )

    config["training"]["burn_in_steps"] = 16
    config["training"]["loss_steps"] = 48
    with pytest.raises(ValueError, match="every valid step"):
        validate_gru_full_training_config(config)

    config["training"]["burn_in_steps"] = 0
    config["training"]["loss_steps"] = 64
    config["pilot_scope"]["classification"] = "principal_comparison"
    with pytest.raises(ValueError, match="engineering-only"):
        validate_gru_full_training_config(config)


def test_gru_full_export_verifies_actions_states_and_exact_reload(tmp_path):
    module = GRURecurrentModule(seed=14303)
    parameters = module.export_numpy_parameters()
    exported = GRURecurrentPolicy(parameters, {})
    restored = GRURecurrentPolicy(parameters, {})
    episodes = []
    rng = np.random.default_rng(14303)
    for episode_index in range(3):
        length = 70
        previous_actions = rng.uniform(-1, 1, size=(length, 2)).astype(np.float32)
        previous_actions[0] = 0.0
        episodes.append(
            {
                "episode_index": episode_index,
                "observations": rng.normal(size=(length, 19)).astype(np.float32),
                "previous_actions": previous_actions,
                "teacher_actions": np.zeros((length, 2), dtype=np.float32),
                "puck_visible": np.arange(length) < 20,
            }
        )
    split = pad_episode_split(
        episodes,
        maximum_episode_steps=128,
        burn_in_steps=0,
    )

    action_error, state_error, exact = verify_gru_export(
        module, exported, restored, split, episode_count=3
    )

    assert action_error <= 2e-6
    assert state_error <= 2e-6
    assert exact
