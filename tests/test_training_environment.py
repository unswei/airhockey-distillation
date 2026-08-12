from pathlib import Path

import pytest

from airhockey_distill.envs import DirectLaunchTrainingEnv

from .fakes import FakeDirectLaunchBackend

CONFIG = Path("configs/env/direct_launch_v2.yaml")


def test_training_reset_samples_only_train_shots_and_configured_blackouts() -> None:
    environment = DirectLaunchTrainingEnv(
        backend=FakeDirectLaunchBackend(),
        distribution_config=CONFIG,
        split="train",
        sampling_seed=17,
        minimum_blackout_steps=0,
        maximum_blackout_steps=20,
    )
    try:
        sampled = []
        for _ in range(30):
            _, info = environment.reset()
            sampled.append((info["shot_id"], environment.blackout.length_steps))
            assert set(info) == {
                "shot_id",
                "observation_step",
                "puck_visible",
                "action_locked",
            }

        assert all(shot_id.startswith("direct_launch_v2:train:") for shot_id, _ in sampled)
        assert all(0 <= length <= 20 for _, length in sampled)
        assert len({shot_id for shot_id, _ in sampled}) > 1
        assert len({length for _, length in sampled}) > 1
    finally:
        environment.close()


def test_training_sampling_replays_from_the_same_reset_seed() -> None:
    first = DirectLaunchTrainingEnv(
        backend=FakeDirectLaunchBackend(),
        distribution_config=CONFIG,
        sampling_seed=1,
    )
    second = DirectLaunchTrainingEnv(
        backend=FakeDirectLaunchBackend(),
        distribution_config=CONFIG,
        sampling_seed=999,
    )
    try:
        first_info = first.reset(seed=42)[1]
        second_info = second.reset(seed=42)[1]

        assert first_info["shot_id"] == second_info["shot_id"]
        assert first.blackout == second.blackout
    finally:
        first.close()
        second.close()


def test_training_can_sample_a_predeclared_weighted_blackout_set() -> None:
    environment = DirectLaunchTrainingEnv(
        backend=FakeDirectLaunchBackend(),
        distribution_config=CONFIG,
        sampling_seed=23,
        blackout_lengths=(0, 5, 20),
        blackout_probabilities=(0.8, 0.1, 0.1),
    )
    try:
        lengths = []
        for _ in range(200):
            environment.reset()
            lengths.append(environment.blackout.length_steps)

        assert set(lengths) == {0, 5, 20}
        assert lengths.count(0) > 120
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("lengths", "probabilities", "message"),
    [
        ((0, 0), None, "unique"),
        ((0, 5), (1.0,), "match"),
        ((0, 5), (0.7, 0.2), "sum"),
        ((0, 5), (1.1, -0.1), "positive"),
    ],
)
def test_training_rejects_invalid_weighted_blackout_sets(
    lengths, probabilities, message
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectLaunchTrainingEnv(
            backend=FakeDirectLaunchBackend(),
            distribution_config=CONFIG,
            blackout_lengths=lengths,
            blackout_probabilities=probabilities,
        )
