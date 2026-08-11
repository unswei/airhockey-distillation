import numpy as np

from airhockey_distill.envs.defend_shot import (
    DefendShotTrackingLoss,
    public_info_has_privileged_state,
)
from airhockey_distill.envs.policy_interface import PUCK_POSITION_XY_SLICE
from airhockey_distill.envs.shot import DEFAULT_DIRECT_LAUNCH_SHOT
from airhockey_distill.envs.tracking_loss import BlackoutSchedule
from airhockey_distill.evaluation import InactiveController, rollout_public_controller
from tests.fakes import FakeDirectLaunchBackend


def test_public_reset_and_step_do_not_expose_privileged_state() -> None:
    backend = FakeDirectLaunchBackend(terminal_step=2)
    environment = DefendShotTrackingLoss(
        backend,
        blackout=BlackoutSchedule(start_observation_step=0, length_steps=3),
    )

    observation, info = environment.reset(shot=DEFAULT_DIRECT_LAUNCH_SHOT)
    assert not public_info_has_privileged_state(info)
    assert set(info) == {"shot_id", "observation_step", "puck_visible"}
    np.testing.assert_array_equal(observation[PUCK_POSITION_XY_SLICE], (0.0, 0.0))

    next_observation, _, _, _, next_info = environment.step((0.0, 0.0))
    assert not public_info_has_privileged_state(next_info)
    np.testing.assert_array_equal(next_observation[PUCK_POSITION_XY_SLICE], (0.0, 0.0))


def test_public_rollout_never_requests_explicit_privileged_state() -> None:
    class PublicOnlyEnvironment(DefendShotTrackingLoss):
        def privileged_state(self):  # type: ignore[no-untyped-def]
            raise AssertionError("public controller requested privileged state")

    environment = PublicOnlyEnvironment(FakeDirectLaunchBackend(terminal_step=3))
    trace = rollout_public_controller(
        environment, InactiveController(), DEFAULT_DIRECT_LAUNCH_SHOT
    )

    assert trace.steps == 3
    assert trace.outcome == "upstream_terminal"
