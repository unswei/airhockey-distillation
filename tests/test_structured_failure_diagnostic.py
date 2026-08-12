import numpy as np

from scripts.diagnose_structured_student import (
    history_examples,
    phase_for_step,
)


def test_phase_partition_exposes_live_masked_control_steps():
    phases = {
        "locked": (0, 5),
        "live_but_loss_masked": (5, 16),
        "loss_bearing": (16, 128),
    }

    assert phase_for_step(4, phases) == "locked"
    assert phase_for_step(5, phases) == "live_but_loss_masked"
    assert phase_for_step(15, phases) == "live_but_loss_masked"
    assert phase_for_step(16, phases) == "loss_bearing"


def test_history_examples_left_pad_without_future_leakage():
    observations = np.arange(4 * 19, dtype=np.float32).reshape(4, 19)
    previous_actions = np.arange(8, dtype=np.float32).reshape(4, 2)
    teacher_actions = -previous_actions
    features, targets, steps = history_examples(
        [
            {
                "observations": observations,
                "previous_actions": previous_actions,
                "teacher_actions": teacher_actions,
            }
        ],
        history_steps=3,
    )

    combined = np.concatenate((observations, previous_actions), axis=1)
    np.testing.assert_array_equal(features[0, : 2 * 21], 0.0)
    np.testing.assert_array_equal(features[0, 2 * 21 : 3 * 21], combined[0])
    np.testing.assert_array_equal(features[0, -3:], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(features[3, : 3 * 21], combined[1:4].reshape(-1))
    np.testing.assert_array_equal(targets, teacher_actions)
    np.testing.assert_array_equal(steps, np.arange(4))
