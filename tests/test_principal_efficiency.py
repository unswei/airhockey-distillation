from types import SimpleNamespace

import numpy as np
import pytest

from airhockey_distill.principal_efficiency import (
    benchmark_policy_calls,
    policy_accounting,
    validate_latency_runtime,
)
from airhockey_distill.principal_sweep import load_principal_protocol
from airhockey_distill.students import (
    PRINCIPAL_FAMILY_IDS,
    initialise_feed_forward_parameters,
    initialise_finite_stack_parameters,
    initialise_gru_parameters,
    initialise_structured_parameters,
    principal_policy_from_parameters,
)


def _policy(family_id):
    if family_id == "feed_forward":
        parameters = initialise_feed_forward_parameters(14303)
    elif family_id == "finite_stack_10":
        parameters = initialise_finite_stack_parameters(14303)
    elif family_id == "gru_n64":
        parameters = initialise_gru_parameters(14303)
    else:
        parameters = initialise_structured_parameters(
            14303, innovation_rank=int(family_id.removeprefix("structured_k"))
        )
    return principal_policy_from_parameters(family_id, parameters, {})


@pytest.mark.parametrize("family_id", PRINCIPAL_FAMILY_IDS)
def test_accounting_matches_predeclared_parameters_state_and_multiply_adds(family_id):
    protocol = load_principal_protocol(
        "configs/experiments/principal_sweep_execution_v1.yaml"
    )
    expected = {
        "feed_forward": (0, 0, 5440),
        "finite_stack_10": (0, 120, 7168),
        "structured_k0": (256, 264, 11776),
        "structured_k1": (256, 264, 11938),
        "structured_k2": (256, 264, 12100),
        "structured_k4": (256, 264, 12424),
        "gru_n64": (256, 264, 28352),
    }

    result = policy_accounting(protocol, family_id, _policy(family_id))

    assert result["decision"] == "GO"
    assert all(result["checks"].values())
    assert (
        result["recurrent_memory_bytes"],
        result["total_policy_carry_bytes"],
        result["multiply_adds_per_step"],
    ) == expected[family_id]
    assert sum(result["exported_tensor_element_counts"].values()) == result[
        "total_trainable_parameters"
    ]


def test_latency_benchmark_measures_complete_calls_and_repetition_means():
    class Policy:
        def initial_carry(self):
            return 0

        def act(self, observation, carry):
            assert observation.shape == (19,)
            return np.zeros(2, dtype=np.float32), carry + 1

    ticks = iter((100, 5100, 6000, 16000))
    result = benchmark_policy_calls(
        Policy(),
        warmup_calls=2,
        timed_calls_per_repetition=5,
        repetitions=2,
        clock=lambda: next(ticks),
    )

    assert result["total_timed_calls"] == 10
    assert result["repetition_mean_microseconds"] == [1.0, 2.0]
    assert result["median_microseconds"] == 1.5
    assert result["p95_microseconds"] == pytest.approx(1.95)
    assert result["p95_definition"] == (
        "95th percentile across the configured repetition means"
    )


def test_principal_policy_reports_concrete_inference_implementation():
    assert _policy("feed_forward").inference_implementation == "numpy_float32_v1"
    assert (
        _policy("structured_k2").inference_implementation
        == "numpy_pairwise_float32_v1"
    )


def test_latency_runtime_fails_closed_off_marvin(monkeypatch):
    specification = {
        "host": "marvin",
        "cpu": "Intel Core Ultra 9 285",
        "pinned_logical_cpu": 0,
        "environment_variables": {"OMP_NUM_THREADS": 1},
    }
    monkeypatch.setattr("platform.node", lambda: "workstation")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "airhockey_distill.principal_efficiency._cpu_model_name",
        lambda: "Apple M4",
    )
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    result = validate_latency_runtime(specification)

    assert result["decision"] == "NO_GO"
    assert len(result["failures"]) >= 4
