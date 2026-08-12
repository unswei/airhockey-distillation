import numpy as np
import pytest

from airhockey_distill.students import (
    StructuredRecurrentPolicy,
    initialise_structured_parameters,
)


def test_automatic_differentiation_confirms_rank_two_innovation():
    torch = pytest.importorskip("torch")
    policy = StructuredRecurrentPolicy(initialise_structured_parameters(21), {})
    rng = np.random.default_rng(21)
    observation = rng.normal(size=19).astype(np.float32)
    previous_action = rng.normal(size=2).astype(np.float32)
    previous_state = rng.normal(size=64).astype(np.float32)
    encoded = policy._encode_batch(observation[None, :])[0]
    parameters = {
        name: torch.as_tensor(value.astype(np.float64))
        for name, value in policy.parameters.items()
    }
    encoded_tensor = torch.as_tensor(encoded.astype(np.float64))
    action_tensor = torch.as_tensor(previous_action.astype(np.float64))
    state_tensor = torch.as_tensor(previous_state.astype(np.float64))

    def update(state):
        linear = (
            torch.tanh(parameters["recurrence_alpha"]) * state
            + parameters["recurrence_input_weight"] @ encoded_tensor
            + parameters["recurrence_action_weight"] @ action_tensor
            + parameters["recurrence_bias"]
        )
        innovation = torch.tanh(
            parameters["innovation_state_weight"] @ state
            + parameters["innovation_input_weight"] @ encoded_tensor
            + parameters["innovation_action_weight"] @ action_tensor
            + parameters["innovation_bias"]
        )
        return linear + parameters["innovation_output_weight"] @ innovation

    autodiff_jacobian = torch.autograd.functional.jacobian(
        update, state_tensor
    ).numpy()
    analytical_jacobian = policy.recurrent_jacobian(
        observation, previous_action, previous_state
    )
    np.testing.assert_allclose(
        analytical_jacobian, autodiff_jacobian, atol=3e-7, rtol=1e-6
    )

    fixed_linear_jacobian = np.diag(
        np.tanh(parameters["recurrence_alpha"].numpy())
    )
    nonlinear_departure = autodiff_jacobian - fixed_linear_jacobian
    singular_values = np.linalg.svd(nonlinear_departure, compute_uv=False)

    assert np.linalg.matrix_rank(nonlinear_departure, tol=1e-7) <= 2
    assert singular_values[2] < 1e-12
