from pathlib import Path

import numpy as np

from airhockey_distill.envs import (
    load_direct_launch_distribution,
    manifest_sha256,
    summarise_distribution,
)

V1_CONFIG = Path("configs/env/direct_launch_v1.yaml")
V2_CONFIG = Path("configs/env/direct_launch_v2.yaml")
V3_CONFIG = Path("configs/env/direct_launch_v3.yaml")


def test_calibration_split_has_balanced_216_distinct_shots() -> None:
    distribution = load_direct_launch_distribution(V1_CONFIG)
    shots = distribution.generate("calibration")
    summary = summarise_distribution(shots)

    assert len(shots) == 216
    assert summary["distinct_shot_ids"] == 216
    assert summary["manifest_sha256"] == (
        "82986091f72a3e51cde803ed73527e060bb0d0a89fda6d08e82eb810abf8724e"
    )
    assert set(summary["launch_region_counts"].values()) == {72}
    assert set(summary["target_region_counts"].values()) == {72}
    assert set(summary["region_pair_counts"].values()) == {24}


def test_distribution_covers_centre_posts_regions_angles_and_times() -> None:
    distribution = load_direct_launch_distribution(V1_CONFIG)
    shots = distribution.generate("calibration")
    launch_regions = {shot.launch_region for shot in shots}
    target_regions = {shot.target_region for shot in shots}
    times = np.asarray([shot.nominal_approach_time_seconds for shot in shots])
    angles = np.asarray([shot.approach_angle_degrees for shot in shots])

    assert launch_regions == {"left", "centre", "right"}
    assert target_regions == {"near_post_left", "goal_centre", "near_post_right"}
    assert times.min() >= 0.36
    assert times.max() <= 0.75
    assert times.min() < 0.38
    assert times.max() > 0.73
    assert distribution.realised_approach_time_target_seconds.minimum == 0.4
    assert distribution.realised_approach_time_target_seconds.maximum == 1.0
    assert angles.min() < -5.0
    assert angles.max() > 5.0
    assert len(np.unique(np.round(angles, decimals=3))) > 200


def test_velocity_reaches_approach_plane_on_time_and_goal_target_on_line() -> None:
    distribution = load_direct_launch_distribution(V1_CONFIG)
    for generated in distribution.generate("calibration"):
        position_x, position_y = generated.shot.position_table_xy
        velocity_x, velocity_y = generated.shot.velocity_table_xy
        approach_time = (
            distribution.defender_approach_plane_x - position_x
        ) / velocity_x
        time_to_goal = (distribution.defending_goal_plane_x - position_x) / velocity_x
        projected_goal_y = position_y + time_to_goal * velocity_y

        assert np.isclose(approach_time, generated.nominal_approach_time_seconds)
        assert np.isclose(projected_goal_y, generated.target_goal_y)


def test_split_manifests_are_deterministic_and_independent() -> None:
    distribution = load_direct_launch_distribution(V1_CONFIG)
    first = distribution.generate("calibration")
    second = distribution.generate("calibration")
    validation = distribution.generate("validation")

    assert manifest_sha256(first) == manifest_sha256(second)
    assert [shot.as_dict() for shot in first] == [shot.as_dict() for shot in second]
    assert manifest_sha256(first) != manifest_sha256(validation)
    assert distribution.expected_shot_count("train") == 900
    assert distribution.expected_shot_count("validation") == 225
    assert distribution.expected_shot_count("test") == 225


def test_v2_calibration_preserves_coverage_but_emphasises_near_posts() -> None:
    distribution = load_direct_launch_distribution(V2_CONFIG)
    shots = distribution.generate("calibration")
    summary = summarise_distribution(shots)

    assert len(shots) == 216
    assert summary["distinct_shot_ids"] == 216
    assert summary["manifest_sha256"] == (
        "6e2b61f71135c23c2c1d459c8d90c65e8a153cc645986103d95c67b2e0c2733c"
    )
    assert set(summary["launch_region_counts"].values()) == {72}
    assert summary["target_region_counts"] == {
        "goal_centre": 12,
        "near_post_left": 102,
        "near_post_right": 102,
    }
    assert set(summary["region_pair_counts"].values()) == {2, 4, 34, 66}
    assert distribution.expected_shot_count("train") == 900
    assert distribution.expected_shot_count("validation") == 225
    assert distribution.expected_shot_count("test") == 225


def test_v3_calibration_contains_paired_alias_families_and_support_shots() -> None:
    distribution = load_direct_launch_distribution(V3_CONFIG)
    shots = distribution.generate("calibration")
    summary = summarise_distribution(shots)
    aliases = [shot for shot in shots if shot.alias_family_id is not None]

    assert len(shots) == 216
    assert len(aliases) == 180
    assert len({shot.alias_family_id for shot in aliases}) == 90
    assert summary["alias_shot_count"] == 180
    assert summary["alias_family_count"] == 90
    assert summary["manifest_sha256"] == (
        "359b093d97d213d52764ad553b9c81478296d8600a9edd7379edc005bcd88c9a"
    )
    assert summary["launch_region_counts"] == {
        "centre": 180,
        "left": 18,
        "right": 18,
    }
    assert summary["target_region_counts"] == {
        "goal_centre": 8,
        "near_post_left": 104,
        "near_post_right": 104,
    }
    assert distribution.expected_shot_count("train") == 900
    assert distribution.expected_shot_count("validation") == 225
    assert distribution.expected_shot_count("test") == 225


def test_v3_alias_pairs_share_nominal_blackout_position_but_diverge_to_posts() -> None:
    distribution = load_direct_launch_distribution(V3_CONFIG)
    aliasing = distribution.observation_aliasing
    assert aliasing is not None
    families: dict[str, list[object]] = {}
    for generated in distribution.generate("calibration"):
        if generated.alias_family_id is not None:
            families.setdefault(generated.alias_family_id, []).append(generated)

    for family in families.values():
        assert len(family) == 2
        first, second = family
        assert {first.target_region, second.target_region} == {
            "near_post_left",
            "near_post_right",
        }
        positions = []
        for generated in family:
            initial = np.asarray(generated.shot.position_table_xy)
            velocity = np.asarray(generated.shot.velocity_table_xy)
            positions.append(initial + velocity * aliasing.rendezvous_time_seconds)
        np.testing.assert_allclose(positions[0], positions[1], atol=1e-11)
        np.testing.assert_allclose(
            positions[0], first.nominal_blackout_start_table_xy, atol=1e-11
        )
        assert abs(first.target_goal_y - second.target_goal_y) >= 0.176
        assert np.sign(first.shot.velocity_table_xy[1]) != np.sign(
            second.shot.velocity_table_xy[1]
        )
