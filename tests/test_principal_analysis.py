import numpy as np

from scripts.analyse_principal_sweep import cluster_bootstrap_rates


def test_cluster_bootstrap_keeps_seed_and_alias_unit_draws_paired():
    sums = np.asarray([[2.0, 0.0], [0.0, 1.0]])
    counts = np.asarray([2, 1])
    seed_draws = np.asarray([[0, 0], [1, 1]])
    unit_draws = np.asarray([[0, 0], [1, 1]])

    observed = cluster_bootstrap_rates(sums, counts, seed_draws, unit_draws)

    assert np.array_equal(observed, np.asarray([1.0, 1.0]))
