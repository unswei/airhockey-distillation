"""Training-time sampling around the public direct-launch environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .defend_shot import DefendShotTrackingLoss, DirectLaunchBackend
from .shot_distribution import load_direct_launch_distribution
from .tracking_loss import BlackoutSchedule


class DirectLaunchTrainingEnv(DefendShotTrackingLoss):
    """Sample versioned shots and blackout lengths without policy leakage."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        distribution_config: str | Path = "configs/env/direct_launch_v2.yaml",
        split: str = "train",
        sampling_seed: int = 5201,
        blackout_start_observation_step: int = 5,
        minimum_blackout_steps: int = 0,
        maximum_blackout_steps: int = 20,
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
        blackout_length = int(
            self._sampling_rng.integers(
                self.minimum_blackout_steps,
                self.maximum_blackout_steps + 1,
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
