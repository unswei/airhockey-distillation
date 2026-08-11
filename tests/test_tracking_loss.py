import pytest

from airhockey_distill.envs.tracking_loss import BlackoutSchedule


def test_blackout_interval_is_half_open_and_deterministic() -> None:
    schedule = BlackoutSchedule(start_observation_step=5, length_steps=3)

    assert [schedule.is_visible(step) for step in range(10)] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert schedule.stop_observation_step == 8


def test_zero_length_blackout_is_always_visible() -> None:
    schedule = BlackoutSchedule(start_observation_step=5, length_steps=0)
    assert all(schedule.is_visible(step) for step in range(20))


@pytest.mark.parametrize(
    ("start", "length"),
    ((-1, 2), (2, -1)),
)
def test_negative_schedule_values_are_rejected(start: int, length: int) -> None:
    with pytest.raises(ValueError):
        BlackoutSchedule(start_observation_step=start, length_steps=length)
