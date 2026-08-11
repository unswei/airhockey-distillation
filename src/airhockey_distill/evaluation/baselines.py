"""Non-learning controls for validating the minimal defence task."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from airhockey_distill.envs.defend_shot import PrivilegedState
from airhockey_distill.envs.policy_interface import (
    END_EFFECTOR_XY_SLICE,
    PUBLIC_OBSERVATION_DIM,
    normalise_planar_position,
)


def _public_observation(value: ArrayLike) -> NDArray[np.float32]:
    observation = np.asarray(value, dtype=np.float32)
    if observation.shape != (PUBLIC_OBSERVATION_DIM,):
        raise ValueError(
            f"public observation must have shape ({PUBLIC_OBSERVATION_DIM},)"
        )
    return observation


@dataclass(frozen=True)
class InactiveController:
    """Command the currently observed mallet position, producing no target motion."""

    name: str = "inactive"

    def act(self, observation: ArrayLike) -> NDArray[np.float32]:
        return _public_observation(observation)[END_EFFECTOR_XY_SLICE].copy()


@dataclass(frozen=True)
class FixedCentreController:
    """Hold the mallet at the fixed neutral centre of the defensive workspace."""

    workspace_xy: ArrayLike = field(repr=False)
    centre_position_robot_xy: tuple[float, float] = (0.65, 0.0)
    name: str = "fixed_centre"

    def act(self, observation: ArrayLike) -> NDArray[np.float32]:
        _public_observation(observation)
        return normalise_planar_position(
            self.centre_position_robot_xy, self.workspace_xy
        )


@dataclass(frozen=True)
class PrivilegedInterceptController:
    """Move to a constant-x intercept computed from true puck state."""

    workspace_xy: ArrayLike = field(repr=False)
    intercept_x_robot: float = 0.75
    name: str = "privileged_intercept"

    def act(
        self,
        observation: ArrayLike,
        privileged_state: PrivilegedState,
    ) -> NDArray[np.float32]:
        _public_observation(observation)
        puck_x, puck_y = privileged_state.puck_position_robot_xy
        puck_vx, puck_vy = privileged_state.puck_velocity_robot_xy

        intercept_y = puck_y
        if puck_vx < -1e-6:
            time_to_intercept = (self.intercept_x_robot - puck_x) / puck_vx
            if time_to_intercept >= 0.0:
                intercept_y = puck_y + time_to_intercept * puck_vy

        return normalise_planar_position(
            (self.intercept_x_robot, intercept_y), self.workspace_xy
        )
