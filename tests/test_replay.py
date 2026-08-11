import numpy as np

from airhockey_distill.envs.defend_shot import DefendShotTrackingLoss
from airhockey_distill.envs.shot import DEFAULT_DIRECT_LAUNCH_SHOT
from airhockey_distill.envs.tracking_loss import BlackoutSchedule
from airhockey_distill.evaluation import (
    FixedCentreController,
    assert_equivalent_replay,
    rollout_public_controller,
)
from tests.fakes import FakeDirectLaunchBackend


def test_same_shot_replays_the_same_public_trajectory() -> None:
    backend = FakeDirectLaunchBackend(terminal_step=8)
    environment = DefendShotTrackingLoss(
        backend,
        blackout=BlackoutSchedule(start_observation_step=5, length_steps=2),
    )
    controller = FixedCentreController(environment.ee_workspace_xy)

    first = rollout_public_controller(
        environment, controller, DEFAULT_DIRECT_LAUNCH_SHOT
    )
    first_actions = tuple(action.copy() for action in backend.actions)
    second = rollout_public_controller(
        environment, controller, DEFAULT_DIRECT_LAUNCH_SHOT
    )

    assert_equivalent_replay(first, second)
    assert first.signature() == second.signature()
    assert first.visibility[5:7] == (False, False)
    assert first.visibility[:5] == (True,) * 5
    for action in (*first_actions, *backend.actions):
        assert action.shape == (6,)
        np.testing.assert_array_equal(action[2:], np.zeros(4))
