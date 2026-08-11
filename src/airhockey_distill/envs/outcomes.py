"""Contact-aware terminal outcomes for the single-shot defence task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PuckState(Protocol):
    puck_position_table_xy: tuple[float, float]
    puck_velocity_table_xy: tuple[float, float]

    @property
    def puck_speed(self) -> float: ...


class TableGeometryLike(Protocol):
    length: float
    width: float
    goal_width: float


@dataclass(frozen=True)
class OutcomeThresholds:
    return_plane_table_x: float = -0.55
    minimum_return_velocity_x: float = 0.15
    arrest_speed: float = 0.1
    arrest_confirmation_steps: int = 3

    def __post_init__(self) -> None:
        if self.minimum_return_velocity_x <= 0.0:
            raise ValueError("minimum return velocity must be positive")
        if self.arrest_speed <= 0.0:
            raise ValueError("arrest speed must be positive")
        if self.arrest_confirmation_steps <= 0:
            raise ValueError("arrest confirmation steps must be positive")


class ContactAwareOutcomeTracker:
    """Classify conclusive outcomes without exposing contact to the policy."""

    def __init__(self, thresholds: OutcomeThresholds | None = None) -> None:
        self.thresholds = thresholds or OutcomeThresholds()
        self.reset()

    def reset(self) -> None:
        self.contact_occurred = False
        self.first_contact_step: int | None = None
        self._arrest_steps = 0

    def observe(
        self,
        *,
        state: PuckState,
        geometry: TableGeometryLike,
        observation_step: int,
        puck_mallet_contact: bool,
        upstream_terminal: bool,
    ) -> str | None:
        if puck_mallet_contact:
            self.contact_occurred = True
            if self.first_contact_step is None:
                self.first_contact_step = observation_step

        puck_x, puck_y = state.puck_position_table_xy
        puck_vx, _ = state.puck_velocity_table_xy
        beyond_goal_plane = puck_x < -geometry.length / 2.0
        outside_goal = abs(puck_y) > geometry.goal_width / 2.0
        outside_side = abs(puck_y) > geometry.width / 2.0

        if beyond_goal_plane and not outside_goal:
            return "goal_conceded"

        if self.contact_occurred:
            if (
                puck_x >= self.thresholds.return_plane_table_x
                and puck_vx >= self.thresholds.minimum_return_velocity_x
            ):
                return "returned"

            if state.puck_speed <= self.thresholds.arrest_speed:
                self._arrest_steps += 1
                if (
                    self._arrest_steps >= self.thresholds.arrest_confirmation_steps
                    or upstream_terminal
                ):
                    return "arrested"
            else:
                self._arrest_steps = 0

            if (beyond_goal_plane and outside_goal) or outside_side:
                return "safe_deflection"
        elif beyond_goal_plane or outside_side:
            return "missed_goal_without_contact"

        if upstream_terminal:
            return (
                "upstream_terminal_after_contact"
                if self.contact_occurred
                else "upstream_terminal_without_contact"
            )
        return None

    def timeout_outcome(self) -> str:
        return (
            "timeout_after_contact"
            if self.contact_occurred
            else "timeout_without_contact"
        )
