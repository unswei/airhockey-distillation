"""Deterministic public and privileged baseline rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from airhockey_distill.envs.defend_shot import (
    DefendShotTrackingLoss,
    PrivilegedState,
    public_info_has_privileged_state,
)
from airhockey_distill.envs.shot import ShotSpec


class PublicController(Protocol):
    name: str

    def act(self, observation: ArrayLike) -> NDArray[np.float32]: ...


class PrivilegedController(Protocol):
    name: str

    def act(
        self, observation: ArrayLike, privileged_state: PrivilegedState
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class EpisodeTrace:
    controller_name: str
    shot_id: str
    observations: tuple[NDArray[np.float32], ...]
    actions: tuple[NDArray[np.float32], ...]
    visibility: tuple[bool, ...]
    total_reward: float
    terminated: bool
    truncated: bool
    outcome: str

    @property
    def steps(self) -> int:
        return len(self.actions)

    def signature(self) -> str:
        """Hash the public trajectory for exact replay comparisons."""

        digest = sha256()
        digest.update(self.shot_id.encode())
        digest.update(np.asarray(self.visibility, dtype=np.uint8).tobytes())
        for observation in self.observations:
            digest.update(np.asarray(observation, dtype=np.float32).tobytes())
        for action in self.actions:
            digest.update(np.asarray(action, dtype=np.float32).tobytes())
        digest.update(np.float64(self.total_reward).tobytes())
        digest.update(self.outcome.encode())
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "controller": self.controller_name,
            "shot_id": self.shot_id,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "outcome": self.outcome,
            "public_trajectory_sha256": self.signature(),
        }


def rollout_public_controller(
    environment: DefendShotTrackingLoss,
    controller: PublicController,
    shot: ShotSpec,
) -> EpisodeTrace:
    """Roll out a controller that only receives the public observation."""

    observation, info = environment.reset(shot=shot)
    _check_public_info(info)
    observations = [observation.copy()]
    actions: list[NDArray[np.float32]] = []
    visibility = [bool(info["puck_visible"])]
    total_reward = 0.0
    terminated = False
    truncated = False
    outcome = "rollout_limit"

    while not (terminated or truncated):
        action = np.asarray(controller.act(observation), dtype=np.float32)
        observation, reward, terminated, truncated, info = environment.step(action)
        _check_public_info(info)
        actions.append(action.copy())
        observations.append(observation.copy())
        visibility.append(bool(info["puck_visible"]))
        total_reward += float(reward)
        if terminated or truncated:
            outcome = str(info["outcome"])

    return EpisodeTrace(
        controller_name=controller.name,
        shot_id=shot.shot_id,
        observations=tuple(observations),
        actions=tuple(actions),
        visibility=tuple(visibility),
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        outcome=outcome,
    )


def rollout_privileged_controller(
    environment: DefendShotTrackingLoss,
    controller: PrivilegedController,
    shot: ShotSpec,
) -> EpisodeTrace:
    """Roll out the explicitly privileged task-validity control."""

    observation, info = environment.reset(shot=shot)
    _check_public_info(info)
    observations = [observation.copy()]
    actions: list[NDArray[np.float32]] = []
    visibility = [bool(info["puck_visible"])]
    total_reward = 0.0
    terminated = False
    truncated = False
    outcome = "rollout_limit"

    while not (terminated or truncated):
        action = np.asarray(
            controller.act(observation, environment.privileged_state()),
            dtype=np.float32,
        )
        observation, reward, terminated, truncated, info = environment.step(action)
        _check_public_info(info)
        actions.append(action.copy())
        observations.append(observation.copy())
        visibility.append(bool(info["puck_visible"]))
        total_reward += float(reward)
        if terminated or truncated:
            outcome = str(info["outcome"])

    return EpisodeTrace(
        controller_name=controller.name,
        shot_id=shot.shot_id,
        observations=tuple(observations),
        actions=tuple(actions),
        visibility=tuple(visibility),
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        outcome=outcome,
    )


def assert_equivalent_replay(first: EpisodeTrace, second: EpisodeTrace) -> None:
    if first.signature() != second.signature():
        raise AssertionError(
            "replayed public trajectories differ: "
            f"{first.signature()} != {second.signature()}"
        )


def _check_public_info(info: dict[str, Any]) -> None:
    if public_info_has_privileged_state(info):
        raise RuntimeError(f"public info contains privileged state: {sorted(info)}")
