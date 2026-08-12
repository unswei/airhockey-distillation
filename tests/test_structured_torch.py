import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airhockey_distill.students import StructuredRecurrentPolicy
from airhockey_distill.students.structured_torch import StructuredRecurrentModule


def test_torch_student_matches_numpy_runtime():
    rng = np.random.default_rng(41)
    module = StructuredRecurrentModule(seed=41)
    numpy_policy = StructuredRecurrentPolicy(module.export_numpy_parameters(), {})
    observations = rng.normal(size=(3, 19)).astype(np.float32)
    previous_actions = rng.normal(size=(3, 2)).astype(np.float32)
    previous_states = rng.normal(size=(3, 64)).astype(np.float32)

    with torch.no_grad():
        torch_actions, torch_states = module.forward_step(
            torch.from_numpy(observations),
            torch.from_numpy(previous_actions),
            torch.from_numpy(previous_states),
        )
    numpy_actions, numpy_states = numpy_policy.step(
        observations, previous_actions, previous_states
    )

    np.testing.assert_allclose(torch_actions.numpy(), numpy_actions, atol=4e-7)
    np.testing.assert_allclose(torch_states.numpy(), numpy_states, atol=4e-7)
    assert module.parameter_count == 12328


def test_sequence_loss_backpropagates_through_structured_recurrence():
    generator = torch.Generator().manual_seed(42)
    module = StructuredRecurrentModule(seed=42)
    observations = torch.randn((2, 8, 19), generator=generator)
    previous_actions = torch.randn((2, 8, 2), generator=generator)
    targets = torch.tanh(torch.randn((2, 8, 2), generator=generator))

    predictions, states = module.forward_sequence(observations, previous_actions)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    assert predictions.shape == (2, 8, 2)
    assert states.shape == (2, 8, 64)
    for name in (
        "recurrence_alpha",
        "innovation_output_weight",
        "innovation_state_weight",
        "action_output_weight",
    ):
        gradient = getattr(module, name).grad
        assert gradient is not None
        assert torch.all(torch.isfinite(gradient))
        assert torch.any(gradient != 0.0)
