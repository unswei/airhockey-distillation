import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airhockey_distill.students import FeedForwardPolicy
from airhockey_distill.students.feed_forward_torch import FeedForwardModule


def test_feed_forward_torch_matches_numpy_export():
    rng = np.random.default_rng(14)
    module = FeedForwardModule(seed=14)
    policy = FeedForwardPolicy(module.export_numpy_parameters(), {})
    observations = rng.normal(size=(3, 11, 19)).astype(np.float32)

    with torch.no_grad():
        observed = module(torch.from_numpy(observations)).numpy()
    expected = policy.action(observations.reshape(-1, 19)).reshape(3, 11, 2)

    np.testing.assert_allclose(observed, expected, atol=1e-6)
    assert module.parameter_count == 5602
