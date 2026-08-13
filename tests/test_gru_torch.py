import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airhockey_distill.students import GRURecurrentPolicy
from airhockey_distill.students.gru_torch import GRURecurrentModule


def test_torch_gru_student_matches_numpy_runtime():
    rng = np.random.default_rng(55)
    module = GRURecurrentModule(seed=55)
    numpy_policy = GRURecurrentPolicy(module.export_numpy_parameters(), {})
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

    np.testing.assert_allclose(torch_actions.numpy(), numpy_actions, atol=5e-7)
    np.testing.assert_allclose(torch_states.numpy(), numpy_states, atol=5e-7)
    assert module.parameter_count == 28898


def test_gru_gate_equations_match_torch_gru_cell():
    rng = np.random.default_rng(56)
    module = GRURecurrentModule(seed=56)
    gru_cell = torch.nn.GRUCell(34, 64)
    with torch.no_grad():
        gru_cell.weight_ih.copy_(module.gru_input_weight)
        gru_cell.weight_hh.copy_(module.gru_hidden_weight)
        gru_cell.bias_ih.copy_(module.gru_input_bias)
        gru_cell.bias_hh.copy_(module.gru_hidden_bias)
    observations = torch.from_numpy(rng.normal(size=(5, 19)).astype(np.float32))
    previous_actions = torch.from_numpy(
        rng.normal(size=(5, 2)).astype(np.float32)
    )
    previous_states = torch.from_numpy(
        rng.normal(size=(5, 64)).astype(np.float32)
    )

    with torch.no_grad():
        encoded = torch.nn.functional.silu(
            torch.nn.functional.linear(
                observations, module.encoder_0_weight, module.encoder_0_bias
            )
        )
        encoded = torch.nn.functional.silu(
            torch.nn.functional.linear(
                encoded, module.encoder_1_weight, module.encoder_1_bias
            )
        )
        expected = gru_cell(
            torch.cat((encoded, previous_actions), dim=-1), previous_states
        )
        _, observed = module.forward_step(
            observations, previous_actions, previous_states
        )

    torch.testing.assert_close(observed, expected, atol=2e-7, rtol=1e-6)


def test_sequence_loss_backpropagates_through_gru_recurrence():
    generator = torch.Generator().manual_seed(57)
    module = GRURecurrentModule(seed=57)
    observations = torch.randn((2, 8, 19), generator=generator)
    previous_actions = torch.randn((2, 8, 2), generator=generator)
    targets = torch.tanh(torch.randn((2, 8, 2), generator=generator))

    predictions, states = module.forward_sequence(observations, previous_actions)
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    assert predictions.shape == (2, 8, 2)
    assert states.shape == (2, 8, 64)
    for name in (
        "gru_input_weight",
        "gru_hidden_weight",
        "action_output_weight",
    ):
        gradient = getattr(module, name).grad
        assert gradient is not None
        assert torch.all(torch.isfinite(gradient))
        assert torch.any(gradient != 0.0)
