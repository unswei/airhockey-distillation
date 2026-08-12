"""Training-time sampling around the public direct-launch environment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from .defend_shot import DefendShotTrackingLoss, DirectLaunchBackend
from .reward import DefenceRewardTracker, load_defence_reward
from .shot_distribution import load_direct_launch_distribution
from .tracking_loss import BlackoutSchedule


class DirectLaunchTrainingEnv(DefendShotTrackingLoss):
    """Sample versioned shots and blackout lengths without policy leakage."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        distribution_config: str | Path = "configs/env/direct_launch_v2.yaml",
        reward_config: str | Path = "configs/reward/defend_shot_v1.yaml",
        split: str = "train",
        sampling_seed: int = 5201,
        blackout_start_observation_step: int = 5,
        minimum_blackout_steps: int = 0,
        maximum_blackout_steps: int = 20,
        blackout_lengths: Sequence[int] | None = None,
        blackout_probabilities: Sequence[float] | None = None,
        action_lock_steps: int = 0,
        backend: DirectLaunchBackend | None = None,
        render_mode: str | None = None,
    ) -> None:
        if sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative")
        if minimum_blackout_steps < 0:
            raise ValueError("minimum_blackout_steps must be non-negative")
        if maximum_blackout_steps < minimum_blackout_steps:
            raise ValueError(
                "maximum_blackout_steps must be at least minimum_blackout_steps"
            )
        if blackout_lengths is None and blackout_probabilities is not None:
            raise ValueError("blackout probabilities require explicit lengths")
        self.blackout_lengths: tuple[int, ...] | None = None
        self.blackout_probabilities: tuple[float, ...] | None = None
        if blackout_lengths is not None:
            lengths = tuple(int(value) for value in blackout_lengths)
            if not lengths or any(value < 0 for value in lengths):
                raise ValueError(
                    "blackout lengths must be non-empty and non-negative"
                )
            if len(set(lengths)) != len(lengths):
                raise ValueError("blackout lengths must be unique")
            self.blackout_lengths = lengths
            if blackout_probabilities is not None:
                probabilities = np.asarray(blackout_probabilities, dtype=np.float64)
                if probabilities.shape != (len(lengths),):
                    raise ValueError("blackout probabilities must match lengths")
                if not np.all(np.isfinite(probabilities)) or np.any(
                    probabilities <= 0.0
                ):
                    raise ValueError(
                        "blackout probabilities must be finite and positive"
                    )
                if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-9):
                    raise ValueError("blackout probabilities must sum to one")
                self.blackout_probabilities = tuple(
                    float(value) for value in probabilities
                )
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render mode {render_mode!r}")

        self.distribution = load_direct_launch_distribution(distribution_config)
        self.distribution_split = split
        self.generated_shots = self.distribution.generate(split)
        self.sampling_seed = sampling_seed
        self.blackout_start_observation_step = blackout_start_observation_step
        self.minimum_blackout_steps = minimum_blackout_steps
        self.maximum_blackout_steps = maximum_blackout_steps
        self.render_mode = render_mode
        self._sampling_rng = np.random.default_rng(sampling_seed)

        super().__init__(
            backend=backend,
            blackout=BlackoutSchedule(
                start_observation_step=blackout_start_observation_step,
                length_steps=minimum_blackout_steps,
            ),
            reward_tracker=DefenceRewardTracker(load_defence_reward(reward_config)),
            action_lock_steps=action_lock_steps,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if seed is not None:
            if seed < 0:
                raise ValueError("reset seed must be non-negative")
            self._sampling_rng = np.random.default_rng(seed)

        shot_index = int(self._sampling_rng.integers(len(self.generated_shots)))
        if self.blackout_lengths is None:
            blackout_length = int(
                self._sampling_rng.integers(
                    self.minimum_blackout_steps,
                    self.maximum_blackout_steps + 1,
                )
            )
        else:
            blackout_length = int(
                self._sampling_rng.choice(
                    self.blackout_lengths,
                    p=self.blackout_probabilities,
                )
            )
        self.blackout = BlackoutSchedule(
            start_observation_step=self.blackout_start_observation_step,
            length_steps=blackout_length,
        )
        generated = self.generated_shots[shot_index]
        observation, info = super().reset(
            shot=generated.shot,
            seed=seed,
            options=options,
        )
        return observation, info

    def render(self) -> np.ndarray:
        raise NotImplementedError("the vector-only teacher smoke run does not render")
