"""Versioned, contact-aware reward for teacher training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

KNOWN_OUTCOMES = frozenset(
    {
        "goal_conceded",
        "returned",
        "arrested",
        "safe_deflection",
        "missed_goal_without_contact",
        "timeout_after_contact",
        "timeout_without_contact",
        "upstream_terminal_after_contact",
        "upstream_terminal_without_contact",
    }
)


@dataclass(frozen=True)
class DefenceRewardSpecification:
    reward_id: str
    first_contact: float
    terminal_outcomes: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.reward_id:
            raise ValueError("reward_id must not be empty")
        if not np.isfinite(self.first_contact):
            raise ValueError("first-contact reward must be finite")
        names = [name for name, _ in self.terminal_outcomes]
        if len(names) != len(set(names)):
            raise ValueError("terminal outcome reward names must be unique")
        if set(names) != KNOWN_OUTCOMES:
            raise ValueError(
                "terminal outcome rewards must exactly cover known outcomes"
            )
        if not np.isfinite([value for _, value in self.terminal_outcomes]).all():
            raise ValueError("terminal outcome rewards must be finite")

    def terminal_reward(self, outcome: str) -> float:
        try:
            return dict(self.terminal_outcomes)[outcome]
        except KeyError as error:
            raise KeyError(f"unknown terminal outcome {outcome!r}") from error


DEFAULT_DEFENCE_REWARD_SPECIFICATION = DefenceRewardSpecification(
    reward_id="defend_shot_v1",
    first_contact=0.2,
    terminal_outcomes=(
        ("goal_conceded", -1.0),
        ("returned", 1.0),
        ("arrested", 1.0),
        ("safe_deflection", 1.0),
        ("missed_goal_without_contact", 0.0),
        ("timeout_after_contact", 0.0),
        ("timeout_without_contact", 0.0),
        ("upstream_terminal_after_contact", 0.0),
        ("upstream_terminal_without_contact", 0.0),
    ),
)


class DefenceRewardTracker:
    """Award first contact once and score only explicit terminal outcomes."""

    def __init__(
        self,
        specification: DefenceRewardSpecification | None = None,
    ) -> None:
        self.specification = specification or DEFAULT_DEFENCE_REWARD_SPECIFICATION
        self.reset()

    def reset(self) -> None:
        self.contact_reward_awarded = False

    def observe(
        self,
        *,
        puck_mallet_contact: bool,
        outcome: str | None,
    ) -> float:
        reward = 0.0
        if puck_mallet_contact and not self.contact_reward_awarded:
            reward += self.specification.first_contact
            self.contact_reward_awarded = True
        if outcome is not None:
            reward += self.specification.terminal_reward(outcome)
        return float(reward)


def load_defence_reward(path: str | Path) -> DefenceRewardSpecification:
    with Path(path).open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("reward config must be a mapping")
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("unsupported reward schema version")
    rewards = _mapping(raw, "rewards")
    terminal = _mapping(rewards, "terminal_outcomes")
    return DefenceRewardSpecification(
        reward_id=str(raw["reward_id"]),
        first_contact=float(rewards["first_contact"]),
        terminal_outcomes=tuple(
            (str(outcome), float(value)) for outcome, value in terminal.items()
        ),
    )


def _mapping(mapping: Any, key: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), dict):
        raise TypeError(f"{key} must be a mapping")
    return mapping[key]
