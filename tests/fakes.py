"""Deterministic backend used by tests that do not require MuJoCo."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from airhockey_distill.envs.defend_shot import (
    BackendSnapshot,
    PrivilegedState,
    TableGeometry,
)
from airhockey_distill.envs.shot import ShotSpec


class FakeDirectLaunchBackend:
    def __init__(self, *, terminal_step: int = 6) -> None:
        self.ee_workspace_xy = np.asarray(
            ((0.59665, 1.31), (-0.46585, 0.46585)), dtype=np.float32
        )
        self.table_geometry = TableGeometry(1.948, 1.038, 0.25)
        self.terminal_step = terminal_step
        self.step_index = 0
        self.shot: ShotSpec | None = None
        self.actions: list[NDArray[np.float32]] = []
        self.closed = False

    def reset(self, shot: ShotSpec) -> BackendSnapshot:
        self.shot = shot
        self.step_index = 0
        self.actions.clear()
        return self._snapshot()

    def step(self, upstream_action: NDArray[np.float32]) -> BackendSnapshot:
        action = np.asarray(upstream_action, dtype=np.float32)
        if action.shape != (6,):
            raise ValueError("fake backend expects the six-dimensional upstream action")
        self.actions.append(action.copy())
        self.step_index += 1
        return self._snapshot(terminated=self.step_index >= self.terminal_step)

    def close(self) -> None:
        self.closed = True

    def _snapshot(self, *, terminated: bool = False) -> BackendSnapshot:
        if self.shot is None:
            raise RuntimeError("fake backend has not been reset")
        dt = 0.02 * self.step_index
        table_x = self.shot.position_table_xy[0] + self.shot.velocity_table_xy[0] * dt
        table_y = self.shot.position_table_xy[1] + self.shot.velocity_table_xy[1] * dt
        robot_x = table_x + 1.51
        robot_y = table_y

        upstream = np.zeros(20, dtype=np.float32)
        upstream[:7] = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
        upstream[7:14] = 0.01 * self.step_index
        upstream[14:16] = (-0.85, 0.0)
        upstream[16:18] = (0.4 - 0.01 * self.step_index, 0.2)
        upstream[18:20] = self.shot.velocity_table_xy

        return BackendSnapshot(
            upstream_policy_observation=upstream,
            privileged_state=PrivilegedState(
                observation_step=self.step_index,
                puck_position_table_xy=(table_x, table_y),
                puck_velocity_table_xy=self.shot.velocity_table_xy,
                puck_position_robot_xy=(robot_x, robot_y),
                puck_velocity_robot_xy=self.shot.velocity_table_xy,
            ),
            terminated=terminated,
        )
