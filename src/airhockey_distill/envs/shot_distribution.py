"""Versioned, deterministic direct-launch shot distributions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import atan2, degrees, hypot, isclose
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .shot import ShotSpec


@dataclass(frozen=True)
class NumericRange:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.minimum, self.maximum)).all():
            raise ValueError("range bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("range maximum must exceed minimum")

    def interpolate(self, unit_value: float) -> float:
        if unit_value < 0.0 or unit_value > 1.0:
            raise ValueError("unit value must lie in [0, 1]")
        return self.minimum + unit_value * (self.maximum - self.minimum)


@dataclass(frozen=True)
class DistributionSplit:
    seed: int
    shots_per_region_pair: int
    purpose: str

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("split seed must be non-negative")
        if self.shots_per_region_pair <= 0:
            raise ValueError("shots_per_region_pair must be positive")
        if not self.purpose:
            raise ValueError("split purpose must not be empty")


@dataclass(frozen=True)
class GeneratedShot:
    shot: ShotSpec
    distribution_id: str
    split: str
    launch_region: str
    target_region: str
    target_goal_y: float
    nominal_approach_time_seconds: float
    nominal_speed: float
    approach_angle_degrees: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "shot": self.shot.as_dict(),
            "distribution_id": self.distribution_id,
            "split": self.split,
            "launch_region": self.launch_region,
            "target_region": self.target_region,
            "target_goal_y": self.target_goal_y,
            "nominal_approach_time_seconds": self.nominal_approach_time_seconds,
            "nominal_speed": self.nominal_speed,
            "approach_angle_degrees": self.approach_angle_degrees,
        }


@dataclass(frozen=True)
class DirectLaunchDistribution:
    schema_version: int
    distribution_id: str
    table_length: float
    table_width: float
    goal_width: float
    defending_goal_plane_x: float
    defender_approach_plane_x: float
    launch_x: NumericRange
    ballistic_approach_time_seconds: NumericRange
    realised_approach_time_target_seconds: NumericRange
    lateral_launch_regions: tuple[tuple[str, NumericRange], ...]
    goal_target_regions: tuple[tuple[str, NumericRange], ...]
    splits: tuple[tuple[str, DistributionSplit], ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema version {self.schema_version}")
        if not self.distribution_id:
            raise ValueError("distribution_id must not be empty")
        if min(self.table_length, self.table_width, self.goal_width) <= 0.0:
            raise ValueError("table dimensions must be positive")
        if not isclose(
            self.defending_goal_plane_x,
            -self.table_length / 2.0,
            abs_tol=1e-9,
        ):
            raise ValueError("defending goal plane must equal -table_length / 2")
        if not (
            self.defending_goal_plane_x
            < self.defender_approach_plane_x
            < self.launch_x.minimum
        ):
            raise ValueError("approach plane must lie between the goal and launches")
        if len(self.lateral_launch_regions) < 3:
            raise ValueError("at least three lateral launch regions are required")
        if len(self.goal_target_regions) < 3:
            raise ValueError("at least three goal target regions are required")
        _require_unique_names(self.lateral_launch_regions, "launch region")
        _require_unique_names(self.goal_target_regions, "target region")
        _require_unique_names(self.splits, "split")

        table_half_width = self.table_width / 2.0
        for _, region in self.lateral_launch_regions:
            if (
                region.minimum <= -table_half_width
                or region.maximum >= table_half_width
            ):
                raise ValueError("launch regions must lie inside the table")
        goal_half_width = self.goal_width / 2.0
        for _, region in self.goal_target_regions:
            if region.minimum <= -goal_half_width or region.maximum >= goal_half_width:
                raise ValueError("target regions must lie inside the goal")

    def split(self, split_name: str) -> DistributionSplit:
        try:
            return dict(self.splits)[split_name]
        except KeyError as error:
            raise KeyError(f"unknown distribution split {split_name!r}") from error

    def expected_shot_count(self, split_name: str) -> int:
        split = self.split(split_name)
        return (
            len(self.lateral_launch_regions)
            * len(self.goal_target_regions)
            * split.shots_per_region_pair
        )

    def generate(self, split_name: str) -> tuple[GeneratedShot, ...]:
        split = self.split(split_name)
        rng = np.random.default_rng(split.seed)
        generated: list[GeneratedShot] = []

        for launch_name, launch_y_range in self.lateral_launch_regions:
            for target_name, target_y_range in self.goal_target_regions:
                samples = _latin_hypercube(
                    rng,
                    rows=split.shots_per_region_pair,
                    columns=4,
                )
                for index, values in enumerate(samples):
                    launch_x = _stable_float(
                        self.launch_x.interpolate(float(values[0]))
                    )
                    launch_y = _stable_float(
                        launch_y_range.interpolate(float(values[1]))
                    )
                    target_y = _stable_float(
                        target_y_range.interpolate(float(values[2]))
                    )
                    approach_time = _stable_float(
                        self.ballistic_approach_time_seconds.interpolate(
                            float(values[3])
                        )
                    )
                    velocity_x = _stable_float(
                        (self.defender_approach_plane_x - launch_x) / approach_time
                    )
                    time_to_goal = (self.defending_goal_plane_x - launch_x) / velocity_x
                    velocity_y = _stable_float((target_y - launch_y) / time_to_goal)
                    shot_id = (
                        f"{self.distribution_id}:{split_name}:{launch_name}:"
                        f"{target_name}:{index:03d}"
                    )
                    generated.append(
                        GeneratedShot(
                            shot=ShotSpec(
                                shot_id=shot_id,
                                position_table_xy=(launch_x, launch_y),
                                velocity_table_xy=(velocity_x, velocity_y),
                            ),
                            distribution_id=self.distribution_id,
                            split=split_name,
                            launch_region=launch_name,
                            target_region=target_name,
                            target_goal_y=target_y,
                            nominal_approach_time_seconds=approach_time,
                            nominal_speed=_stable_float(hypot(velocity_x, velocity_y)),
                            approach_angle_degrees=_stable_float(
                                degrees(atan2(velocity_y, -velocity_x))
                            ),
                        )
                    )

        order = rng.permutation(len(generated))
        return tuple(generated[int(index)] for index in order)


def load_direct_launch_distribution(path: str | Path) -> DirectLaunchDistribution:
    config_path = Path(path)
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("distribution config must be a mapping")
    if raw.get("launch_mode") != "direct_launch":
        raise ValueError("distribution launch_mode must be direct_launch")

    geometry = _mapping(raw, "geometry")
    sampling = _mapping(raw, "sampling")
    splits = _mapping(raw, "splits")
    return DirectLaunchDistribution(
        schema_version=int(raw["schema_version"]),
        distribution_id=str(raw["distribution_id"]),
        table_length=float(geometry["table_length"]),
        table_width=float(geometry["table_width"]),
        goal_width=float(geometry["goal_width"]),
        defending_goal_plane_x=float(geometry["defending_goal_plane_x"]),
        defender_approach_plane_x=float(geometry["defender_approach_plane_x"]),
        launch_x=_numeric_range(sampling["launch_x"], "launch_x"),
        ballistic_approach_time_seconds=_numeric_range(
            sampling["ballistic_approach_time_seconds"],
            "ballistic_approach_time_seconds",
        ),
        realised_approach_time_target_seconds=_numeric_range(
            sampling["realised_approach_time_target_seconds"],
            "realised_approach_time_target_seconds",
        ),
        lateral_launch_regions=_named_ranges(
            sampling["lateral_launch_regions"], "lateral_launch_regions"
        ),
        goal_target_regions=_named_ranges(
            sampling["goal_target_regions"], "goal_target_regions"
        ),
        splits=tuple(
            (
                str(name),
                DistributionSplit(
                    seed=int(_mapping(splits, name)["seed"]),
                    shots_per_region_pair=int(
                        _mapping(splits, name)["shots_per_region_pair"]
                    ),
                    purpose=str(_mapping(splits, name)["purpose"]),
                ),
            )
            for name in splits
        ),
    )


def manifest_sha256(shots: tuple[GeneratedShot, ...]) -> str:
    canonical = dumps(
        [shot.as_dict() for shot in shots],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(canonical).hexdigest()


def summarise_distribution(shots: tuple[GeneratedShot, ...]) -> dict[str, Any]:
    if not shots:
        raise ValueError("cannot summarise an empty distribution")
    launch_counts = Counter(shot.launch_region for shot in shots)
    target_counts = Counter(shot.target_region for shot in shots)
    pair_counts = Counter(
        f"{shot.launch_region}->{shot.target_region}" for shot in shots
    )
    return {
        "count": len(shots),
        "distinct_shot_ids": len({shot.shot.shot_id for shot in shots}),
        "manifest_sha256": manifest_sha256(shots),
        "launch_region_counts": dict(sorted(launch_counts.items())),
        "target_region_counts": dict(sorted(target_counts.items())),
        "region_pair_counts": dict(sorted(pair_counts.items())),
        "nominal_approach_time_seconds": _minimum_maximum(
            shot.nominal_approach_time_seconds for shot in shots
        ),
        "nominal_speed": _minimum_maximum(shot.nominal_speed for shot in shots),
        "approach_angle_degrees": _minimum_maximum(
            shot.approach_angle_degrees for shot in shots
        ),
    }


def _latin_hypercube(
    rng: np.random.Generator,
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    samples = np.empty((rows, columns), dtype=np.float64)
    for column in range(columns):
        strata = (np.arange(rows) + rng.random(rows)) / rows
        samples[:, column] = strata[rng.permutation(rows)]
    return samples


def _mapping(mapping: Any, key: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), dict):
        raise TypeError(f"{key} must be a mapping")
    return mapping[key]


def _numeric_range(raw: Any, name: str) -> NumericRange:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{name} must contain two bounds")
    return NumericRange(float(raw[0]), float(raw[1]))


def _named_ranges(raw: Any, name: str) -> tuple[tuple[str, NumericRange], ...]:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping")
    return tuple(
        (str(region_name), _numeric_range(bounds, str(region_name)))
        for region_name, bounds in raw.items()
    )


def _require_unique_names(items: tuple[tuple[str, Any], ...], label: str) -> None:
    names = [name for name, _ in items]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} names must be unique")


def _minimum_maximum(values: Any) -> dict[str, float]:
    sequence = tuple(float(value) for value in values)
    return {"minimum": min(sequence), "maximum": max(sequence)}


def _stable_float(value: float) -> float:
    """Quantise manifest values so hashes are stable across supported hosts."""

    return round(float(value), 12)
