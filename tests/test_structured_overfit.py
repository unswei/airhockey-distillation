import numpy as np

from scripts.collect_teacher_dataset import (
    DETERMINISTIC_ACTION_SEMANTICS,
    SAMPLED_ACTION_SEMANTICS,
    teacher_action_semantics,
)
from scripts.train_structured_tiny_overfit import (
    evaluate_overfit_gate,
    pad_complete_episodes,
)


def test_teacher_target_semantics_do_not_relabel_sampled_actions():
    assert teacher_action_semantics(False) == SAMPLED_ACTION_SEMANTICS
    assert teacher_action_semantics(True) == DETERMINISTIC_ACTION_SEMANTICS
    assert SAMPLED_ACTION_SEMANTICS != DETERMINISTIC_ACTION_SEMANTICS


def test_complete_episode_padding_masks_burn_in_and_padding():
    episodes = [_episode(20, 1.0), _episode(23, -1.0)]

    padded = pad_complete_episodes(
        episodes,
        sequence_length=32,
        burn_in_steps=16,
    )

    assert padded["observations"].shape == (2, 32, 19)
    assert padded["valid_mask"].sum(axis=1).tolist() == [20, 23]
    assert padded["loss_mask"].sum(axis=1).tolist() == [4, 7]
    assert not np.any(padded["loss_mask"][:, :16])
    assert not np.any(padded["loss_mask"][0, 20:])
    np.testing.assert_array_equal(
        padded["observations"][0, :20], episodes[0]["observations"]
    )
    np.testing.assert_array_equal(
        padded["previous_actions"][1, :23], episodes[1]["previous_actions"]
    )


def test_tiny_overfit_gate_requires_every_predeclared_check():
    gate = {
        "maximum_training_action_mse": 1e-4,
        "minimum_fractional_loss_reduction": 0.99,
        "maximum_export_absolute_error": 5e-6,
        "require_exact_checkpoint_reload": True,
    }

    passing = evaluate_overfit_gate(
        initial_mse=0.2,
        final_mse=5e-5,
        export_error=1e-6,
        reload_exact=True,
        gate=gate,
    )
    failing = evaluate_overfit_gate(
        initial_mse=0.2,
        final_mse=2e-4,
        export_error=1e-4,
        reload_exact=False,
        gate=gate,
    )

    assert all(check["passed"] for check in passing)
    assert [check["check_id"] for check in failing if not check["passed"]] == [
        "near_zero_training_action_mse",
        "numpy_export_agreement",
        "exact_checkpoint_reload",
    ]


def _episode(length: int, value: float):
    previous_actions = np.full((length, 2), value, dtype=np.float32)
    previous_actions[0] = 0.0
    return {
        "observations": np.full((length, 19), value, dtype=np.float32),
        "previous_actions": previous_actions,
        "teacher_actions": np.full((length, 2), value, dtype=np.float32),
    }
