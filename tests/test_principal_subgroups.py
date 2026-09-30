from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyse_principal_subgroups import (
    bootstrap_rates, bootstrap_weights, index_episodes, membership, sampling_design,
)


def episodes():
    return [
        {"shot_id": shot, "alias_family_id": alias, "launch_region": launch,
         "target_region": target, "blackout_steps": blackout, "outcome": "returned"}
        for shot, alias, launch, target in (
            ("a-left", "a", "centre", "near_post_left"),
            ("a-right", "a", "centre", "near_post_right"),
            ("s-left", None, "left", "near_post_left"),
            ("s-right", None, "right", "near_post_right"),
        )
        for blackout in (0, 20)
    ]


def test_index_is_order_independent_and_retains_alias_pair_as_one_unit():
    indexed, metadata = index_episodes(episodes(), (0, 20))
    assert (indexed, metadata) == index_episodes(list(reversed(episodes())), (0, 20))
    units, strata = sampling_design(metadata)
    assert units == ["a", "s-left", "s-right"]
    assert strata["alias"] == [0]
    assert sum(membership(m, "alias") for m in metadata.values()) == 2
    assert sum(membership(m, "support") for m in metadata.values()) == 2
    assert sum(membership(m, "alias_left") for m in metadata.values()) == 1


@pytest.mark.parametrize("fault", ["duplicate", "missing_horizon", "metadata", "pair", "horizon"])
def test_invalid_pairing_fails_closed(fault):
    rows = deepcopy(episodes())
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing_horizon":
        rows.pop()
    elif fault == "metadata":
        rows[0]["launch_region"] = "left"
    elif fault == "pair":
        rows = [r for r in rows if r["shot_id"] != "a-right"]
    else:
        rows[0]["blackout_steps"] = 99
    with pytest.raises(ValueError):
        index_episodes(rows, (0, 20))


def test_stratified_draws_retain_sizes_and_are_reproducible():
    strata = {"alias": [0, 1], "support": [2, 3, 4]}
    seeds, units = bootstrap_weights(5, strata, 100, 123)
    again = bootstrap_weights(5, strata, 100, 123)
    np.testing.assert_array_equal(seeds.sum(axis=1), 5)
    np.testing.assert_array_equal(units[:, :2].sum(axis=1), 2)
    np.testing.assert_array_equal(units[:, 2:].sum(axis=1), 3)
    np.testing.assert_array_equal(seeds, again[0])
    np.testing.assert_array_equal(units, again[1])


def test_alias_members_and_comparisons_share_draws():
    seeds, units = bootstrap_weights(2, {"alias": [0, 1]}, 100, 123)
    # Each alias family has one save and one failure for every training seed.
    both = bootstrap_rates(np.ones((2, 2)), np.array([2, 2]), seeds, units)
    left = bootstrap_rates(np.ones((2, 2)), np.array([1, 1]), seeds, units)
    right = bootstrap_rates(np.zeros((2, 2)), np.array([1, 1]), seeds, units)
    np.testing.assert_array_equal(both, 0.5)
    np.testing.assert_array_equal(left - right, 1.0)
    np.testing.assert_array_equal(both - both, 0.0)


def test_vectorised_bootstrap_matches_explicit_seed_and_unit_draws():
    sums = np.array([[2, 0, 1], [0, 1, 1]], dtype=float)
    counts = np.array([2, 2, 1])
    seed_draws = [[0, 0], [0, 1], [1, 1]]
    unit_draws = [[0, 0, 2], [0, 1, 2], [1, 1, 2]]
    seed_weights = np.array([np.bincount(d, minlength=2) for d in seed_draws])
    unit_weights = np.array([np.bincount(d, minlength=3) for d in unit_draws])
    expected = [sums[np.ix_(s, u)].sum() / (len(s) * counts[u].sum())
                for s, u in zip(seed_draws, unit_draws, strict=True)]
    observed = bootstrap_rates(sums, counts, seed_weights, unit_weights)
    np.testing.assert_allclose(observed, expected)


def test_teacher_has_no_artificial_training_seed_variation():
    _, units = bootstrap_weights(5, {"alias": [0, 1]}, 100, 123)
    samples = bootstrap_rates(np.ones((1, 2)), np.array([2, 2]), np.ones((100, 1)), units)
    np.testing.assert_array_equal(samples, 0.5)


def test_empty_subgroup_and_nonpartitioning_strata_are_rejected():
    with pytest.raises(ValueError, match="empty subgroup"):
        bootstrap_rates(np.zeros((1, 1)), np.zeros(1), np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="partition"):
        bootstrap_weights(5, {"a": [0], "b": [0]}, 10, 123)


def test_saved_subgroups_match_analysis_source_and_frozen_principal_rates():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/principal_subgroups_posthoc_v1.json").read_text())
    principal_path = root / "results/principal_sweep_v1_statistics.json"
    principal = json.loads(principal_path.read_text())
    provenance = result["provenance"]
    assert provenance["analysis_script_sha256"] == hashlib.sha256(
        (root / "scripts/analyse_principal_subgroups.py").read_bytes()
    ).hexdigest()
    assert provenance["principal_statistics_sha256"] == hashlib.sha256(principal_path.read_bytes()).hexdigest()
    assert provenance["episode_rows"] == 48_600
    assert provenance["seed_horizon_rates_reconciled"] == 216
    assert len(provenance["test_files_sha256"]) == 36
    assert result["groups"]["alias"]["shots_per_policy"] == 180
    assert result["groups"]["support"]["shots_per_policy"] == 45
    for ms in (0, 100, 200, 300, 400, 500):
        groups = {g: result["groups"][g]["by_blackout_ms"][str(ms)]["policies"]
                  for g in ("alias", "support")}
        for family in groups["alias"]:
            observed = sum(result["groups"][g]["shots_per_policy"] * groups[g][family]["save_rate_percent"]
                           for g in groups) / 225
            expected = (principal["teacher_save_rates"][str(ms // 20)] if family == "teacher" else
                        principal["student_save_rates"][family][str(ms // 20)]) * 100
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)
            for group in groups:
                values = groups[group][family]
                np.testing.assert_allclose(values["save_rate_percent"],
                                           values["saves"] / values["episode_rows"] * 100)
