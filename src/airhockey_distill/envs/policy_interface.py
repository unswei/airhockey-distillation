"""Public observation and action adapters shared by every learned policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

UPSTREAM_OBSERVATION_DIM = 20
PUBLIC_OBSERVATION_DIM = 19
PUBLIC_ACTION_DIM = 2
UPSTREAM_ACTION_DIM = 6

JOINT_POSITION_SLICE = slice(0, 7)
JOINT_VELOCITY_SLICE = slice(7, 14)
END_EFFECTOR_XY_SLICE = slice(14, 16)
PUCK_POSITION_XY_SLICE = slice(16, 18)
UPSTREAM_PUCK_VELOCITY_XY_SLICE = slice(18, 20)
PUCK_VISIBLE_INDEX = 18


def _vector(value: ArrayLike, size: int, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


class PublicObservationAdapter:
    """Remove puck velocity and apply the deterministic visibility mask."""

    def adapt(
        self, upstream_observation: ArrayLike, *, puck_visible: bool
    ) -> NDArray[np.float32]:
        upstream = _vector(
            upstream_observation,
            UPSTREAM_OBSERVATION_DIM,
            "upstream observation",
        )
        public = np.empty(PUBLIC_OBSERVATION_DIM, dtype=np.float32)
        public[:18] = upstream[:18]
        if not puck_visible:
            public[PUCK_POSITION_XY_SLICE] = 0.0
        public[PUCK_VISIBLE_INDEX] = float(puck_visible)
        return public


@dataclass(frozen=True)
class PlanarActionAdapter:
    """Expand a planar target with fixed mid-range impedance commands."""

    fixed_stiffness_normalised: tuple[float, float] = (0.0, 0.0)
    fixed_damping_normalised: tuple[float, float] = (0.0, 0.0)

    def adapt(self, planar_action: ArrayLike) -> NDArray[np.float32]:
        target = np.clip(
            _vector(planar_action, PUBLIC_ACTION_DIM, "planar action"),
            -1.0,
            1.0,
        )
        full_action = np.asarray(
            (
                *target,
                *self.fixed_stiffness_normalised,
                *self.fixed_damping_normalised,
            ),
            dtype=np.float32,
        )
        if full_action.shape != (UPSTREAM_ACTION_DIM,):
            raise ValueError("fixed impedance commands must each contain two values")
        if np.any(np.abs(full_action[2:]) > 1.0):
            raise ValueError("fixed impedance commands must lie in [-1, 1]")
        return full_action


def normalise_planar_position(
    position_xy: ArrayLike, workspace_xy: ArrayLike
) -> NDArray[np.float32]:
    """Map a physical planar position into the upstream action range."""

    position = _vector(position_xy, 2, "planar position")
    workspace = np.asarray(workspace_xy, dtype=np.float32)
    if workspace.shape != (2, 2):
        raise ValueError("workspace must have shape (2, 2)")
    low = workspace[:, 0]
    high = workspace[:, 1]
    if np.any(high <= low):
        raise ValueError("workspace upper bounds must exceed lower bounds")
    return np.clip(2.0 * (position - low) / (high - low) - 1.0, -1.0, 1.0)


def denormalise_planar_position(
    action_xy: ArrayLike, workspace_xy: ArrayLike
) -> NDArray[np.float32]:
    """Map a normalised target into the physical planar workspace."""

    action = np.clip(_vector(action_xy, 2, "planar action"), -1.0, 1.0)
    workspace = np.asarray(workspace_xy, dtype=np.float32)
    if workspace.shape != (2, 2):
        raise ValueError("workspace must have shape (2, 2)")
    return workspace[:, 0] + 0.5 * (action + 1.0) * (workspace[:, 1] - workspace[:, 0])
