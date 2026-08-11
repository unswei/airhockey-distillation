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
    purpose: str
    shots_per_region_pair: int | None = None
    shots_per_target_region: tuple[tuple[str, int], ...] = ()
    shots_per_launch_target_pair: tuple[tuple[str, int], ...] = ()
    alias_families: int | None = None
    support_shots_per_launch_target_pair: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("split seed must be non-negative")
        if not self.purpose:
            raise ValueError("split purpose must not be empty")
        specified_count_modes = sum(
            (
                self.shots_per_region_pair is not None,
                bool(self.shots_per_target_region),
                bool(self.shots_per_launch_target_pair),
                self.alias_families is not None,
            )
        )
        if specified_count_modes != 1:
            raise ValueError("split must define exactly one shot-count mode")
        if self.shots_per_region_pair is not None and self.shots_per_region_pair <= 0:
            raise ValueError("shots_per_region_pair must be positive")
        if self.shots_per_target_region:
            _require_unique_names(
                self.shots_per_target_region,
                "shots_per_target_region",
            )
            if any(count <= 0 for _, count in self.shots_per_target_region):
                raise ValueError("shots_per_target_region counts must be positive")
        if self.shots_per_launch_target_pair:
            _require_unique_names(
                self.shots_per_launch_target_pair,
                "shots_per_launch_target_pair",
            )
            if any(count <= 0 for _, count in self.shots_per_launch_target_pair):
                raise ValueError("shots_per_launch_target_pair counts must be positive")
        if self.alias_families is not None:
            if self.alias_families <= 0:
                raise ValueError("alias_families must be positive")
            if not self.support_shots_per_launch_target_pair:
                raise ValueError(
                    "an aliased split must define support shots for every pair"
                )
            _require_unique_names(
                self.support_shots_per_launch_target_pair,
                "support_shots_per_launch_target_pair",
            )
            if any(count < 0 for _, count in self.support_shots_per_launch_target_pair):
                raise ValueError("support shot counts must be non-negative")

    def shots_for_pair(self, launch_name: str, target_name: str) -> int:
        if self.alias_families is not None:
            pair_name = f"{launch_name}->{target_name}"
            try:
                return dict(self.support_shots_per_launch_target_pair)[pair_name]
            except KeyError as error:
                raise KeyError(
                    f"split has no support count for pair {pair_name!r}"
                ) from error
        if self.shots_per_region_pair is not None:
            return self.shots_per_region_pair
        if self.shots_per_target_region:
            try:
                return dict(self.shots_per_target_region)[target_name]
            except KeyError as error:
                raise KeyError(
                    f"split has no count for target {target_name!r}"
                ) from error
        pair_name = f"{launch_name}->{target_name}"
        try:
            return dict(self.shots_per_launch_target_pair)[pair_name]
        except KeyError as error:
            raise KeyError(f"split has no count for pair {pair_name!r}") from error


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
    alias_family_id: str | None = None
    nominal_blackout_start_table_xy: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
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
        if self.alias_family_id is not None:
            result["alias_family_id"] = self.alias_family_id
            result["nominal_blackout_start_table_xy"] = (
                self.nominal_blackout_start_table_xy
            )
        return result


@dataclass(frozen=True)
class ObservationAliasingSpec:
    """Nominal rendezvous shared by shots needing different defences."""

    blackout_start_observation_step: int
    policy_step_seconds: float
    launch_region: str
    rendezvous_x: NumericRange
    rendezvous_y: float
    longitudinal_speed: NumericRange
    target_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.blackout_start_observation_step <= 1:
            raise ValueError("aliasing needs at least two visible observation steps")
        if self.policy_step_seconds <= 0.0:
            raise ValueError("policy_step_seconds must be positive")
        if not np.isfinite(self.rendezvous_y):
            raise ValueError("rendezvous_y must be finite")
        if len(self.target_regions) < 2:
            raise ValueError("aliasing needs at least two target regions")
        if len(set(self.target_regions)) != len(self.target_regions):
            raise ValueError("alias target regions must be unique")

    @property
    def rendezvous_time_seconds(self) -> float:
        return self.blackout_start_observation_step * self.policy_step_seconds


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
    observation_aliasing: ObservationAliasingSpec | None = None

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
        target_names = {name for name, _ in self.goal_target_regions}
        launch_names = {name for name, _ in self.lateral_launch_regions}
        pair_names = {
            f"{launch_name}->{target_name}"
            for launch_name in launch_names
            for target_name in target_names
        }
        if self.observation_aliasing is not None:
            aliasing = self.observation_aliasing
            if aliasing.launch_region not in launch_names:
                raise ValueError("alias launch region must be declared")
            if not set(aliasing.target_regions).issubset(target_names):
                raise ValueError("alias target regions must be declared")
            if not (
                self.defender_approach_plane_x
                < aliasing.rendezvous_x.minimum
                < aliasing.rendezvous_x.maximum
                < self.launch_x.maximum
            ):
                raise ValueError(
                    "alias rendezvous x must lie after the launch and before approach"
                )
        for _, split in self.splits:
            if split.shots_per_target_region:
                split_target_names = {name for name, _ in split.shots_per_target_region}
                if split_target_names != target_names:
                    raise ValueError(
                        "shots_per_target_region names must exactly match goal target "
                        "region names"
                    )
            if split.shots_per_launch_target_pair:
                split_pair_names = {
                    name for name, _ in split.shots_per_launch_target_pair
                }
                if split_pair_names != pair_names:
                    raise ValueError(
                        "shots_per_launch_target_pair names must exactly match all "
                        "launch/target region pairs"
                    )
            if split.alias_families is not None:
                if self.observation_aliasing is None:
                    raise ValueError("aliased split requires observation_aliasing")
                split_pair_names = {
                    name for name, _ in split.support_shots_per_launch_target_pair
                }
                if split_pair_names != pair_names:
                    raise ValueError(
                        "support shot names must exactly match all launch/target pairs"
                    )
            elif self.observation_aliasing is not None:
                raise ValueError(
                    "every split of an aliased distribution must use alias families"
                )

    def split(self, split_name: str) -> DistributionSplit:
        try:
            return dict(self.splits)[split_name]
        except KeyError as error:
            raise KeyError(f"unknown distribution split {split_name!r}") from error

    def expected_shot_count(self, split_name: str) -> int:
        split = self.split(split_name)
        alias_count = 0
        if split.alias_families is not None:
            if self.observation_aliasing is None:
                raise RuntimeError("aliased split has no alias specification")
            alias_count = split.alias_families * len(
                self.observation_aliasing.target_regions
            )
        return (
            sum(
                split.shots_for_pair(launch_name, target_name)
                for launch_name, _ in self.lateral_launch_regions
                for target_name, _ in self.goal_target_regions
            )
            + alias_count
        )

    def generate(self, split_name: str) -> tuple[GeneratedShot, ...]:
        split = self.split(split_name)
        rng = np.random.default_rng(split.seed)
        if split.alias_families is not None:
            return self._generate_observation_aliased(split_name, split, rng)
        generated: list[GeneratedShot] = []

        for launch_name, launch_y_range in self.lateral_launch_regions:
            for target_name, target_y_range in self.goal_target_regions:
                samples = _latin_hypercube(
                    rng,
                    rows=split.shots_for_pair(launch_name, target_name),
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

    def _generate_observation_aliased(
        self,
        split_name: str,
        split: DistributionSplit,
        rng: np.random.Generator,
    ) -> tuple[GeneratedShot, ...]:
        aliasing = self.observation_aliasing
        if aliasing is None or split.alias_families is None:
            raise RuntimeError("observation aliasing is not configured")
        launch_ranges = dict(self.lateral_launch_regions)
        target_ranges = dict(self.goal_target_regions)
        generated: list[GeneratedShot] = []
        samples = _latin_hypercube(
            rng,
            rows=split.alias_families,
            columns=4,
        )
        rendezvous_time = aliasing.rendezvous_time_seconds
        for family_index, values in enumerate(samples):
            rendezvous_x = _stable_float(
                aliasing.rendezvous_x.interpolate(float(values[0]))
            )
            speed_x = _stable_float(
                aliasing.longitudinal_speed.interpolate(float(values[1]))
            )
            velocity_x = -speed_x
            launch_x = _stable_float(rendezvous_x + speed_x * rendezvous_time)
            if not self.launch_x.minimum <= launch_x <= self.launch_x.maximum:
                raise ValueError(f"alias launch x {launch_x} lies outside launch_x")
            target_unit = float(values[2])
            family_id = f"{self.distribution_id}:{split_name}:alias:{family_index:03d}"
            for target_index, target_name in enumerate(aliasing.target_regions):
                target_range = target_ranges[target_name]
                unit = target_unit if target_index % 2 == 0 else 1.0 - target_unit
                target_y = _stable_float(target_range.interpolate(unit))
                time_from_rendezvous_to_goal = (
                    self.defending_goal_plane_x - rendezvous_x
                ) / velocity_x
                velocity_y = _stable_float(
                    (target_y - aliasing.rendezvous_y) / time_from_rendezvous_to_goal
                )
                launch_y = _stable_float(
                    aliasing.rendezvous_y - velocity_y * rendezvous_time
                )
                launch_range = launch_ranges[aliasing.launch_region]
                if not launch_range.minimum <= launch_y <= launch_range.maximum:
                    raise ValueError(
                        f"alias launch y {launch_y} lies outside "
                        f"{aliasing.launch_region}"
                    )
                shot_id = f"{family_id}:{target_name}"
                approach_time = _stable_float(
                    (self.defender_approach_plane_x - launch_x) / velocity_x
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
                        launch_region=aliasing.launch_region,
                        target_region=target_name,
                        target_goal_y=target_y,
                        nominal_approach_time_seconds=approach_time,
                        nominal_speed=_stable_float(hypot(velocity_x, velocity_y)),
                        approach_angle_degrees=_stable_float(
                            degrees(atan2(velocity_y, -velocity_x))
                        ),
                        alias_family_id=family_id,
                        nominal_blackout_start_table_xy=(
                            rendezvous_x,
                            aliasing.rendezvous_y,
                        ),
                    )
                )

        for launch_name, launch_y_range in self.lateral_launch_regions:
            for target_name, target_y_range in self.goal_target_regions:
                count = split.shots_for_pair(launch_name, target_name)
                if count == 0:
                    continue
                support_samples = _latin_hypercube(rng, rows=count, columns=4)
                for index, values in enumerate(support_samples):
                    generated.append(
                        self._make_standard_shot(
                            split_name=split_name,
                            launch_name=launch_name,
                            launch_y_range=launch_y_range,
                            target_name=target_name,
                            target_y_range=target_y_range,
                            index=index,
                            values=values,
                            id_component="support",
                        )
                    )

        order = rng.permutation(len(generated))
        return tuple(generated[int(index)] for index in order)

    def _make_standard_shot(
        self,
        *,
        split_name: str,
        launch_name: str,
        launch_y_range: NumericRange,
        target_name: str,
        target_y_range: NumericRange,
        index: int,
        values: np.ndarray,
        id_component: str,
    ) -> GeneratedShot:
        launch_x = _stable_float(self.launch_x.interpolate(float(values[0])))
        launch_y = _stable_float(launch_y_range.interpolate(float(values[1])))
        target_y = _stable_float(target_y_range.interpolate(float(values[2])))
        approach_time = _stable_float(
            self.ballistic_approach_time_seconds.interpolate(float(values[3]))
        )
        velocity_x = _stable_float(
            (self.defender_approach_plane_x - launch_x) / approach_time
        )
        time_to_goal = (self.defending_goal_plane_x - launch_x) / velocity_x
        velocity_y = _stable_float((target_y - launch_y) / time_to_goal)
        shot_id = (
            f"{self.distribution_id}:{split_name}:{id_component}:"
            f"{launch_name}:{target_name}:{index:03d}"
        )
        return GeneratedShot(
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
    split_definitions: list[tuple[str, DistributionSplit]] = []
    for name in splits:
        split_raw = _mapping(splits, name)
        uniform_count = split_raw.get("shots_per_region_pair")
        target_counts = split_raw.get("shots_per_target_region")
        pair_counts = split_raw.get("shots_per_launch_target_pair")
        alias_families = split_raw.get("alias_families")
        support_counts = split_raw.get("support_shots_per_launch_target_pair")
        if target_counts is not None and not isinstance(target_counts, dict):
            raise TypeError("shots_per_target_region must be a mapping")
        if pair_counts is not None and not isinstance(pair_counts, dict):
            raise TypeError("shots_per_launch_target_pair must be a mapping")
        if support_counts is not None and not isinstance(support_counts, dict):
            raise TypeError("support_shots_per_launch_target_pair must be a mapping")
        split_definitions.append(
            (
                str(name),
                DistributionSplit(
                    seed=int(split_raw["seed"]),
                    purpose=str(split_raw["purpose"]),
                    shots_per_region_pair=(
                        int(uniform_count) if uniform_count is not None else None
                    ),
                    shots_per_target_region=(
                        tuple(
                            (str(target_name), int(count))
                            for target_name, count in target_counts.items()
                        )
                        if target_counts is not None
                        else ()
                    ),
                    shots_per_launch_target_pair=(
                        tuple(
                            (str(pair_name), int(count))
                            for pair_name, count in pair_counts.items()
                        )
                        if pair_counts is not None
                        else ()
                    ),
                    alias_families=(
                        int(alias_families) if alias_families is not None else None
                    ),
                    support_shots_per_launch_target_pair=(
                        tuple(
                            (str(pair_name), int(count))
                            for pair_name, count in support_counts.items()
                        )
                        if support_counts is not None
                        else ()
                    ),
                ),
            )
        )
    aliasing_raw = sampling.get("observation_aliasing")
    if aliasing_raw is not None and not isinstance(aliasing_raw, dict):
        raise TypeError("observation_aliasing must be a mapping")
    observation_aliasing = None
    if aliasing_raw is not None:
        raw_targets = aliasing_raw.get("target_regions")
        if not isinstance(raw_targets, list):
            raise TypeError("alias target_regions must be a list")
        observation_aliasing = ObservationAliasingSpec(
            blackout_start_observation_step=int(
                aliasing_raw["blackout_start_observation_step"]
            ),
            policy_step_seconds=float(aliasing_raw["policy_step_seconds"]),
            launch_region=str(aliasing_raw["launch_region"]),
            rendezvous_x=_numeric_range(
                aliasing_raw["rendezvous_x"], "alias rendezvous_x"
            ),
            rendezvous_y=float(aliasing_raw["rendezvous_y"]),
            longitudinal_speed=_numeric_range(
                aliasing_raw["longitudinal_speed"],
                "alias longitudinal_speed",
            ),
            target_regions=tuple(str(value) for value in raw_targets),
        )
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
        splits=tuple(split_definitions),
        observation_aliasing=observation_aliasing,
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
    summary = {
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
    alias_shots = [shot for shot in shots if shot.alias_family_id is not None]
    if alias_shots:
        summary["alias_shot_count"] = len(alias_shots)
        summary["alias_family_count"] = len(
            {shot.alias_family_id for shot in alias_shots}
        )
    return summary


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
