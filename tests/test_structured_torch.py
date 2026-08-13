import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airhockey_distill.students import StructuredRecurrentPolicy
from airhockey_distill.students.structured_torch import StructuredRecurrentModule


@pytest.mark.parametrize("rank", (0, 1, 2, 4))
def test_torch_student_matches_numpy_runtime(rank):
    rng = np.random.default_rng(41)
    module = StructuredRecurrentModule(seed=41, innovation_rank=rank)
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

    np.testing.assert_allclose(torch_actions.numpy(), numpy_actions, atol=1e-6)
    np.testing.assert_allclose(torch_states.numpy(), numpy_states, atol=1e-6)
    assert module.parameter_count == 12002 + 163 * rank


def test_k0_torch_module_has_no_nonlinear_recurrent_branch():
    module = StructuredRecurrentModule(seed=43, innovation_rank=0)

    assert module.innovation_rank == 0
    assert not any(
        name.startswith("innovation_") for name, _ in module.named_parameters()
    )


@pytest.mark.parametrize("rank", (0, 1, 2, 4))
def test_torch_module_infers_rank_from_exported_parameters(rank):
    source = StructuredRecurrentModule(seed=44, innovation_rank=rank)

    restored = StructuredRecurrentModule(
        parameters=source.export_numpy_parameters(),
    )

    assert restored.innovation_rank == rank
    assert restored.parameter_count == source.parameter_count


@pytest.mark.parametrize("rank", (0, 1, 2, 4))
def test_sequence_loss_backpropagates_through_structured_recurrence(rank):
    generator = torch.Generator().manual_seed(42)
    module = StructuredRecurrentModule(seed=42, innovation_rank=rank)
    observations = torch.randn((2, 8, 19), generator=generator)
    previous_actions = torch.randn((2, 8, 2), generator=generator)
    targets = torch.tanh(torch.randn((2, 8, 2), generator=generator))

    predictions, states = module.forward_sequence(observations, previous_actions)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    assert predictions.shape == (2, 8, 2)
    assert states.shape == (2, 8, 64)
    names = ["recurrence_alpha", "action_output_weight"]
    if rank:
        names.extend(("innovation_output_weight", "innovation_state_weight"))
    for name in names:
        gradient = getattr(module, name).grad
        assert gradient is not None
        assert torch.all(torch.isfinite(gradient))
        assert torch.any(gradient != 0.0)
