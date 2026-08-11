"""Minimal single-shot tracking-loss task around the pinned MuJoCo stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .policy_interface import PlanarActionAdapter, PublicObservationAdapter
from .shot import DEFAULT_DIRECT_LAUNCH_SHOT, ShotSpec
from .tracking_loss import BlackoutSchedule


@dataclass(frozen=True)
class TableGeometry:
    length: float
    width: float
    goal_width: float


@dataclass(frozen=True)
class PrivilegedState:
    """Simulator state available only to controls and evaluation code."""

    observation_step: int
    puck_position_table_xy: tuple[float, float]
    puck_velocity_table_xy: tuple[float, float]
    puck_position_robot_xy: tuple[float, float]
    puck_velocity_robot_xy: tuple[float, float]

    @property
    def puck_speed(self) -> float:
        return float(np.linalg.norm(self.puck_velocity_table_xy))


@dataclass(frozen=True)
class BackendSnapshot:
    upstream_policy_observation: NDArray[np.float32]
    privileged_state: PrivilegedState
    reward: float = 0.0
    terminated: bool = False


class DirectLaunchBackend(Protocol):
    """Backend boundary used to test policy semantics without MuJoCo."""

    ee_workspace_xy: NDArray[np.float32]
    table_geometry: TableGeometry

    def reset(self, shot: ShotSpec) -> BackendSnapshot: ...

    def step(self, upstream_action: NDArray[np.float32]) -> BackendSnapshot: ...

    def close(self) -> None: ...


class MujocoDirectLaunchBackend:
    """Pinned upstream single-robot defender with a deterministic launch."""

    def __init__(self, *, upstream_horizon_steps: int = 500) -> None:
        # Keep imports lazy so pure interface tests do not require the simulator.
        from air_hockey_challenge.environments.iiwas.env_single import (
            AirHockeySingle,
        )
        from air_hockey_challenge.environments.position_control_wrapper import (
            IiwaPositionDefend,
        )
        from drl_air_hockey.agents.spacer_agent import SpaceRAgent

        class DeterministicIiwaPositionDefend(IiwaPositionDefend):
            def __init__(self, direct_shot: ShotSpec, **kwargs: Any) -> None:
                self.direct_shot = direct_shot
                super().__init__(**kwargs)

            def setup(self, observation: NDArray[np.float64]) -> None:
                shot = self.direct_shot
                self._write_data("puck_x_pos", shot.position_table_xy[0])
                self._write_data("puck_y_pos", shot.position_table_xy[1])
                self._write_data("puck_yaw_pos", shot.yaw_position)
                self._write_data("puck_x_vel", shot.velocity_table_xy[0])
                self._write_data("puck_y_vel", shot.velocity_table_xy[1])
                self._write_data("puck_yaw_vel", shot.yaw_velocity)
                AirHockeySingle.setup(self, observation)

        self._environment = DeterministicIiwaPositionDefend(
            direct_shot=DEFAULT_DIRECT_LAUNCH_SHOT,
            interpolation_order=-1,
            horizon=upstream_horizon_steps,
        )
        self._agent = SpaceRAgent(
            env_info=self._environment.env_info,
            agent_id=1,
            train=True,
        )
        self.ee_workspace_xy = np.asarray(
            self._agent.ee_table_minmax, dtype=np.float32
        ).copy()
        table = self._environment.env_info["table"]
        self.table_geometry = TableGeometry(
            length=float(table["length"]),
            width=float(table["width"]),
            goal_width=float(table["goal_width"]),
        )
        self._observation_step = 0

    def reset(self, shot: ShotSpec) -> BackendSnapshot:
        self._environment.direct_shot = shot
        raw_observation = np.asarray(self._environment.reset())
        self._agent.reset()
        self._observation_step = 0
        return self._snapshot(raw_observation)

    def step(self, upstream_action: NDArray[np.float32]) -> BackendSnapshot:
        low_level_action = self._agent.process_raw_act(
            np.asarray(upstream_action, dtype=np.float32)
        )
        raw_observation, reward, terminated, _ = self._environment.step(
            low_level_action
        )
        self._observation_step += 1
        return self._snapshot(
            np.asarray(raw_observation),
            reward=float(reward),
            terminated=bool(terminated),
        )

    def close(self) -> None:
        stop = getattr(self._environment, "stop", None)
        if stop is not None:
            stop()

    def _snapshot(
        self,
        raw_observation: NDArray[np.floating[Any]],
        *,
        reward: float = 0.0,
        terminated: bool = False,
    ) -> BackendSnapshot:
        policy_observation = np.asarray(
            self._agent.process_raw_obs(raw_observation), dtype=np.float32
        )
        state = PrivilegedState(
            observation_step=self._observation_step,
            puck_position_table_xy=(
                self._read_scalar("puck_x_pos"),
                self._read_scalar("puck_y_pos"),
            ),
            puck_velocity_table_xy=(
                self._read_scalar("puck_x_vel"),
                self._read_scalar("puck_y_vel"),
            ),
            puck_position_robot_xy=(
                float(raw_observation[0]),
                float(raw_observation[1]),
            ),
            puck_velocity_robot_xy=(
                float(raw_observation[3]),
                float(raw_observation[4]),
            ),
        )
        return BackendSnapshot(
            upstream_policy_observation=policy_observation,
            privileged_state=state,
            reward=reward,
            terminated=terminated,
        )

    def _read_scalar(self, name: str) -> float:
        return float(np.asarray(self._environment._read_data(name)).reshape(-1)[0])


class DefendShotTrackingLoss(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """One direct-launched shot with one deterministic observation blackout."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        backend: DirectLaunchBackend | None = None,
        *,
        blackout: BlackoutSchedule | None = None,
        timeout_steps: int = 125,
        observation_adapter: PublicObservationAdapter | None = None,
        action_adapter: PlanarActionAdapter | None = None,
    ) -> None:
        if timeout_steps <= 0:
            raise ValueError("timeout_steps must be positive")
        self.backend = backend if backend is not None else MujocoDirectLaunchBackend()
        self.blackout = blackout if blackout is not None else BlackoutSchedule()
        self.timeout_steps = timeout_steps
        self.observation_adapter = observation_adapter or PublicObservationAdapter()
        self.action_adapter = action_adapter or PlanarActionAdapter()
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(19,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self._snapshot: BackendSnapshot | None = None
        self._shot: ShotSpec | None = None
        self._observation_step = 0
        self._episode_done = False

    @property
    def ee_workspace_xy(self) -> NDArray[np.float32]:
        return np.asarray(self.backend.ee_workspace_xy, dtype=np.float32).copy()

    @property
    def table_geometry(self) -> TableGeometry:
        return self.backend.table_geometry

    def reset(
        self,
        *,
        shot: ShotSpec = DEFAULT_DIRECT_LAUNCH_SHOT,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._shot = shot
        self._observation_step = 0
        self._episode_done = False
        self._snapshot = self.backend.reset(shot)
        return self._public_observation(), self._public_info()

    def step(
        self, planar_action: ArrayLike
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._snapshot is None or self._shot is None:
            raise RuntimeError("reset must be called before step")
        if self._episode_done:
            raise RuntimeError("reset must be called after an episode ends")

        upstream_action = self.action_adapter.adapt(planar_action)
        self._snapshot = self.backend.step(upstream_action)
        self._observation_step += 1
        terminated = self._snapshot.terminated
        truncated = self._observation_step >= self.timeout_steps and not terminated
        self._episode_done = terminated or truncated

        info = self._public_info()
        if self._episode_done:
            info["outcome"] = self._classify_outcome(truncated=truncated)

        return (
            self._public_observation(),
            self._snapshot.reward,
            terminated,
            truncated,
            info,
        )

    def privileged_state(self) -> PrivilegedState:
        """Return ground truth through an explicit evaluation-only boundary."""

        if self._snapshot is None:
            raise RuntimeError("reset must be called before requesting state")
        return self._snapshot.privileged_state

    def close(self) -> None:
        self.backend.close()

    def _public_observation(self) -> NDArray[np.float32]:
        if self._snapshot is None:
            raise RuntimeError("environment has not been reset")
        return self.observation_adapter.adapt(
            self._snapshot.upstream_policy_observation,
            puck_visible=self.blackout.is_visible(self._observation_step),
        )

    def _public_info(self) -> dict[str, Any]:
        if self._shot is None:
            raise RuntimeError("environment has not been reset")
        return {
            "shot_id": self._shot.shot_id,
            "observation_step": self._observation_step,
            "puck_visible": self.blackout.is_visible(self._observation_step),
        }

    def _classify_outcome(self, *, truncated: bool) -> str:
        if self._snapshot is None:
            raise RuntimeError("environment has not been reset")
        state = self._snapshot.privileged_state
        puck_x, puck_y = state.puck_position_table_xy
        puck_vx, _ = state.puck_velocity_table_xy
        table = self.table_geometry

        if puck_x < -table.length / 2 and abs(puck_y) <= table.goal_width / 2:
            return "goal_conceded"
        if puck_x > 0.0 and puck_vx > 0.0:
            return "cleared"
        if state.puck_speed < 0.1:
            return "arrested"
        if puck_x > -0.8 and puck_vx > 0.1:
            return "returned"
        if truncated:
            return "timeout"
        if abs(puck_x) > table.length / 2 or abs(puck_y) > table.width / 2:
            return "out_of_bounds_non_goal"
        return "upstream_terminal"


def public_info_has_privileged_state(info: Mapping[str, Any]) -> bool:
    """Conservative key check used by tests and rollout validation."""

    privileged_fragments = ("position", "velocity", "state", "ground_truth", "puck_")
    allowed_puck_key = "puck_visible"
    return any(
        key != allowed_puck_key
        and any(fragment in key.lower() for fragment in privileged_fragments)
        for key in info
    )
