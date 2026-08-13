import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airhockey_distill.students import (
    FiniteStackPolicy,
    save_finite_stack_checkpoint,
)
from airhockey_distill.students.finite_stack_torch import FiniteStackModule


def test_torch_finite_stack_matches_numpy_runtime():
    rng = np.random.default_rng(71)
    module = FiniteStackModule(seed=71)
    numpy_policy = FiniteStackPolicy(module.export_numpy_parameters(), {})
    observations = rng.normal(size=(3, 19)).astype(np.float32)
    histories = rng.normal(size=(3, 30)).astype(np.float32)

    with torch.no_grad():
        torch_actions, torch_histories = module.forward_step(
            torch.from_numpy(observations),
            torch.from_numpy(histories),
        )
    numpy_actions, numpy_histories = numpy_policy.step(observations, histories)

    np.testing.assert_allclose(torch_actions.numpy(), numpy_actions, atol=1e-6)
    np.testing.assert_array_equal(torch_histories.numpy(), numpy_histories)
    assert module.parameter_count == 7330


def test_torch_sequence_matches_numpy_sequence():
    rng = np.random.default_rng(72)
    module = FiniteStackModule(seed=72)
    numpy_policy = FiniteStackPolicy(module.export_numpy_parameters(), {})
    observations = rng.normal(size=(2, 13, 19)).astype(np.float32)
    initial_history = rng.normal(size=(2, 30)).astype(np.float32)

    with torch.no_grad():
        torch_actions, torch_histories = module.forward_sequence(
            torch.from_numpy(observations),
            torch.from_numpy(initial_history),
        )
    expected_actions = []
    expected_histories = []
    for row in range(2):
        actions, histories = numpy_policy.sequence(
            observations[row],
            initial_history[row],
        )
        expected_actions.append(actions)
        expected_histories.append(histories)

    np.testing.assert_allclose(
        torch_actions.numpy(),
        np.stack(expected_actions),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        torch_histories.numpy(),
        np.stack(expected_histories),
    )


def test_torch_export_checkpoint_reload_is_exact(tmp_path):
    rng = np.random.default_rng(75)
    module = FiniteStackModule(seed=75)
    observations = rng.normal(size=(19, 19)).astype(np.float32)
    parameters = module.export_numpy_parameters()
    exported = FiniteStackPolicy(parameters, {})
    expected_actions, expected_histories = exported.sequence(observations)
    checkpoint = tmp_path / "finite_stack_10.npz"

    save_finite_stack_checkpoint(
        checkpoint,
        parameters,
        {"training_seed": 75},
    )
    restored = FiniteStackPolicy.load(checkpoint)
    observed_actions, observed_histories = restored.sequence(observations)

    np.testing.assert_array_equal(observed_actions, expected_actions)
    np.testing.assert_array_equal(observed_histories, expected_histories)


def test_sequence_loss_backpropagates_through_finite_stack_network():
    generator = torch.Generator().manual_seed(73)
    module = FiniteStackModule(seed=73)
    observations = torch.randn((2, 12, 19), generator=generator)
    targets = torch.tanh(torch.randn((2, 12, 2), generator=generator))

    predictions, histories = module.forward_sequence(observations)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    assert predictions.shape == (2, 12, 2)
    assert histories.shape == (2, 12, 30)
    for name in ("encoder_0_weight", "action_output_weight"):
        gradient = getattr(module, name).grad
        assert gradient is not None
        assert torch.all(torch.isfinite(gradient))
        assert torch.any(gradient != 0.0)


def test_torch_module_rejects_wrong_history_shape():
    module = FiniteStackModule(seed=74)

    with pytest.raises(ValueError, match="previous history"):
        module.forward_step(torch.zeros(2, 19), torch.zeros(2, 29))
