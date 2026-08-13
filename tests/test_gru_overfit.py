import pytest

from scripts.train_gru_tiny_overfit import validate_gru_overfit_config


def test_gru_overfit_config_requires_all_valid_step_loss():
    config = {
        "policy": {"id": "gru_n64", "state_dimension": 64},
        "dataset": {"episodes": 16},
        "training": {
            "training_episode_count": 1,
            "batch_size": 1,
            "burn_in_steps": 0,
            "target": "deterministic_teacher_action_mean",
        },
    }

    validate_gru_overfit_config(config)
    config["training"]["burn_in_steps"] = 16
    with pytest.raises(ValueError, match="every valid step"):
        validate_gru_overfit_config(config)
