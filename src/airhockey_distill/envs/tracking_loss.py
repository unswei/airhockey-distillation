"""Deterministic puck-visibility schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlackoutSchedule:
    """One half-open blackout interval over policy observation steps."""

    start_observation_step: int = 5
    length_steps: int = 10

    def __post_init__(self) -> None:
        if self.start_observation_step < 0:
            raise ValueError("blackout start must be non-negative")
        if self.length_steps < 0:
            raise ValueError("blackout length must be non-negative")

    @property
    def stop_observation_step(self) -> int:
        return self.start_observation_step + self.length_steps

    def is_visible(self, observation_step: int) -> bool:
        if observation_step < 0:
            raise ValueError("observation step must be non-negative")
        return not (
            self.start_observation_step <= observation_step < self.stop_observation_step
        )
