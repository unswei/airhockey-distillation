"""Versioned direct-launch shot definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ShotSpec:
    """One deterministic puck launch in the upstream table frame."""

    shot_id: str
    position_table_xy: tuple[float, float]
    velocity_table_xy: tuple[float, float]
    yaw_position: float = 0.0
    yaw_velocity: float = 0.0

    def __post_init__(self) -> None:
        values = (
            *self.position_table_xy,
            *self.velocity_table_xy,
            self.yaw_position,
            self.yaw_velocity,
        )
        if not self.shot_id:
            raise ValueError("shot_id must not be empty")
        if not all(isfinite(value) for value in values):
            raise ValueError("shot values must be finite")
        if self.velocity_table_xy[0] >= 0.0:
            raise ValueError("a defending shot must travel towards negative table x")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable shot record."""

        return asdict(self)


DEFAULT_DIRECT_LAUNCH_SHOT = ShotSpec(
    shot_id="near_post_right_v1",
    position_table_xy=(0.55, 0.11),
    velocity_table_xy=(-1.6, 0.0),
)
