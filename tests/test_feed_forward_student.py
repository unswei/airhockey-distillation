import numpy as np
import pytest

from airhockey_distill.students import (
    PARAMETER_SHAPES,
    FeedForwardPolicy,
    initialise_feed_forward_parameters,
    save_feed_forward_checkpoint,
    save_principal_feed_forward_checkpoint,
)


def test_feed_forward_checkpoint_round_trip_reproduces_actions(tmp_path):
    rng = np.random.default_rng(12)
    parameters = {
        name: rng.normal(0.0, 0.1, shape).astype(np.float32)
        for name, shape in PARAMETER_SHAPES.items()
    }
    policy = FeedForwardPolicy(parameters, {"seed": 12})
    observations = rng.normal(size=(7, 19)).astype(np.float32)
    expected = policy.action(observations)
    path = tmp_path / "feed_forward.npz"

    save_feed_forward_checkpoint(path, parameters, {"seed": 12})
    restored = FeedForwardPolicy.load(path)

    np.testing.assert_array_equal(restored.action(observations), expected)
    assert restored.metadata == {"seed": 12}
    assert restored.parameter_count == 5602


def test_feed_forward_zero_parameters_produce_zero_action():
    policy = FeedForwardPolicy(
        {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in PARAMETER_SHAPES.items()
        },
        {},
    )

    np.testing.assert_array_equal(
        policy.action(np.ones(19, dtype=np.float32)),
        np.zeros(2, dtype=np.float32),
    )


def test_feed_forward_rejects_wrong_observation_shape():
    policy = FeedForwardPolicy(
        {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in PARAMETER_SHAPES.items()
        },
        {},
    )

    with pytest.raises(ValueError, match="shape"):
        policy.action(np.zeros(18, dtype=np.float32))


def test_principal_feed_forward_initialisation_and_checkpoint_are_self_describing(
    tmp_path,
):
    parameters = initialise_feed_forward_parameters(13)
    path = tmp_path / "principal-feed-forward.npz"

    save_principal_feed_forward_checkpoint(
        path,
        parameters,
        {"student_id": "feed_forward"},
    )
    restored = FeedForwardPolicy.load(path)

    assert restored.metadata["policy"] == "feed_forward"
    assert restored.metadata["observation_dimension"] == 19
    assert restored.metadata["action_output"] == "tanh_mean"
    assert restored.parameter_count == 5602
