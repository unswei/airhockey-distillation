import numpy as np
import pytest

from airhockey_distill.envs import (
    DEFAULT_DIRECT_LAUNCH_SHOT,
    BlackoutSchedule,
    DefendShotTrackingLoss,
    MujocoDirectLaunchBackend,
)
from airhockey_distill.evaluation import (
    FixedCentreController,
    PrivilegedInterceptController,
    assert_equivalent_replay,
    rollout_privileged_controller,
    rollout_public_controller,
)


@pytest.mark.integration
def test_direct_launch_replays_in_pinned_mujoco_environment() -> None:
    pytest.importorskip("air_hockey_challenge")
    environment = DefendShotTrackingLoss(
        MujocoDirectLaunchBackend(),
        blackout=BlackoutSchedule(start_observation_step=5, length_steps=3),
        timeout_steps=20,
    )
    try:
        observation, info = environment.reset(shot=DEFAULT_DIRECT_LAUNCH_SHOT)
        state = environment.privileged_state()

        assert observation.shape == (19,)
        assert environment.observation_space.shape == (19,)
        assert environment.action_space.shape == (2,)
        assert info["puck_visible"] is True
        np.testing.assert_allclose(
            state.puck_position_table_xy,
            DEFAULT_DIRECT_LAUNCH_SHOT.position_table_xy,
        )
        np.testing.assert_allclose(
            state.puck_velocity_table_xy,
            DEFAULT_DIRECT_LAUNCH_SHOT.velocity_table_xy,
        )

        controller = FixedCentreController(environment.ee_workspace_xy)
        first = rollout_public_controller(
            environment, controller, DEFAULT_DIRECT_LAUNCH_SHOT
        )
        second = rollout_public_controller(
            environment, controller, DEFAULT_DIRECT_LAUNCH_SHOT
        )

        assert_equivalent_replay(first, second)
        assert first.visibility[5:8] == (False, False, False)
        assert first.steps == 20
        assert first.truncated is True
    finally:
        environment.close()


@pytest.mark.integration
def test_contact_aware_privileged_return_terminates_before_timeout() -> None:
    pytest.importorskip("air_hockey_challenge")
    environment = DefendShotTrackingLoss(MujocoDirectLaunchBackend())
    try:
        controller = PrivilegedInterceptController(environment.ee_workspace_xy)

        trace = rollout_privileged_controller(
            environment,
            controller,
            DEFAULT_DIRECT_LAUNCH_SHOT,
        )

        assert trace.outcome == "returned"
        assert trace.terminated is True
        assert trace.truncated is False
        assert trace.steps < environment.timeout_steps
    finally:
        environment.close()
