from pathlib import Path

import pytest

from airhockey_distill.envs import DefenceRewardTracker, load_defence_reward

CONFIG = Path("configs/reward/defend_shot_v1.yaml")


def test_reward_config_covers_every_outcome() -> None:
    specification = load_defence_reward(CONFIG)

    assert specification.reward_id == "defend_shot_v1"
    assert specification.first_contact == 0.2
    assert specification.terminal_reward("goal_conceded") == -1.0
    assert specification.terminal_reward("returned") == 1.0
    assert specification.terminal_reward("timeout_without_contact") == 0.0


def test_first_contact_is_awarded_once() -> None:
    tracker = DefenceRewardTracker(load_defence_reward(CONFIG))

    assert tracker.observe(puck_mallet_contact=True, outcome=None) == 0.2
    assert tracker.observe(puck_mallet_contact=True, outcome=None) == 0.0
    assert tracker.observe(puck_mallet_contact=False, outcome="returned") == 1.0


def test_contact_and_terminal_reward_can_share_a_step() -> None:
    tracker = DefenceRewardTracker(load_defence_reward(CONFIG))

    assert tracker.observe(puck_mallet_contact=True, outcome="returned") == 1.2


@pytest.mark.parametrize(
    ("outcome", "expected"),
    (
        ("goal_conceded", -1.0),
        ("returned", 1.0),
        ("arrested", 1.0),
        ("safe_deflection", 1.0),
        ("missed_goal_without_contact", 0.0),
        ("timeout_after_contact", 0.0),
        ("timeout_without_contact", 0.0),
    ),
)
def test_terminal_outcome_rewards(outcome: str, expected: float) -> None:
    tracker = DefenceRewardTracker(load_defence_reward(CONFIG))

    assert tracker.observe(puck_mallet_contact=False, outcome=outcome) == expected


def test_reset_allows_contact_reward_in_the_next_episode() -> None:
    tracker = DefenceRewardTracker(load_defence_reward(CONFIG))
    tracker.observe(puck_mallet_contact=True, outcome=None)

    tracker.reset()

    assert tracker.observe(puck_mallet_contact=True, outcome=None) == 0.2
