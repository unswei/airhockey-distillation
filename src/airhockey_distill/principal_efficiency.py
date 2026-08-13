"""Canonical efficiency measurements for exported principal policies."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np

from airhockey_distill.principal_sweep import (
    principal_family_spec,
    sha256_file,
    validate_training_seed,
)
from airhockey_distill.students import load_principal_policy


def load_final_principal_policy(
    protocol: dict[str, Any],
    protocol_path: Path,
    family_id: str,
    seed: int,
    checkpoint: Path,
) -> Any:
    """Load a hashable final checkpoint and enforce principal provenance."""

    principal_family_spec(protocol, family_id)
    validate_training_seed(protocol, seed)
    policy = load_principal_policy(family_id, checkpoint)
    if policy.metadata.get("student_id") != family_id:
        raise ValueError("checkpoint family mismatch")
    if int(policy.metadata.get("training_seed", -1)) != seed:
        raise ValueError("checkpoint seed mismatch")
    if policy.metadata.get("training_stage") != "final":
        raise ValueError("efficiency measurement requires a final checkpoint")
    if policy.metadata.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("checkpoint protocol hash mismatch")
    return policy


def policy_accounting(
    protocol: dict[str, Any],
    family_id: str,
    policy: Any,
) -> dict[str, Any]:
    """Count exported tensors and batch-one runtime state without estimates."""

    family = principal_family_spec(protocol, family_id)
    parameters = {
        str(name): np.asarray(value)
        for name, value in policy.policy.parameters.items()
    }
    if not all(value.dtype == np.float32 for value in parameters.values()):
        raise ValueError("all exported policy tensors must be float32")
    tensor_counts = {
        name: int(value.size) for name, value in sorted(parameters.items())
    }
    total_parameters = sum(tensor_counts.values())
    core_parameters = int(policy.core_parameter_count)
    carry = policy.initial_carry()
    carry_arrays = _named_arrays(carry)
    total_carry_bytes = sum(int(value.nbytes) for value in carry_arrays.values())
    recurrent_memory_bytes = sum(
        int(value.nbytes)
        for name, value in carry_arrays.items()
        if name.rsplit(".", 1)[-1] == "memory"
    )
    if family_id == "feed_forward":
        expected_carry_values = 0
    elif family_id == "finite_stack_10":
        expected_carry_values = int(family["persistent_state_float32_values"])
    else:
        expected_carry_values = int(family["state_dimension"]) + int(
            protocol["matched_architecture"]["previous_action_dimension"]
        )
    expected_memory_values = (
        int(family.get("state_dimension", 0))
        if family_id.startswith("structured_") or family_id == "gru_n64"
        else 0
    )
    multiply_adds = sum(
        int(value.size)
        for name, value in parameters.items()
        if name.endswith("_weight") or name == "recurrence_alpha"
    )
    checks = {
        "exported_tensor_count_matches_policy": total_parameters
        == int(policy.parameter_count),
        "total_parameters_match_predeclaration": total_parameters
        == int(family["expected_total_parameters"]),
        "core_parameters_match_predeclaration": core_parameters
        == int(family["expected_core_parameters"]),
        "total_carry_matches_architecture": total_carry_bytes
        == expected_carry_values * 4,
        "recurrent_memory_matches_architecture": recurrent_memory_bytes
        == expected_memory_values * 4,
        "all_exported_tensors_are_float32": all(
            value.dtype == np.float32 for value in parameters.values()
        ),
        "all_carry_arrays_are_float32": all(
            value.dtype == np.float32 for value in carry_arrays.values()
        ),
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "total_trainable_parameters": total_parameters,
        "core_parameters": core_parameters,
        "exported_tensor_element_counts": tensor_counts,
        "exported_parameter_bytes": sum(
            int(value.nbytes) for value in parameters.values()
        ),
        "recurrent_memory_bytes": recurrent_memory_bytes,
        "total_policy_carry_bytes": total_carry_bytes,
        "carry_arrays": {
            name: {"shape": list(value.shape), "bytes": int(value.nbytes)}
            for name, value in sorted(carry_arrays.items())
        },
        "multiply_adds_per_step": multiply_adds,
        "multiply_add_definition": (
            "one per exported dense-weight element plus one per structured "
            "diagonal recurrence-alpha element; biases, activations, gates and "
            "history copies are excluded"
        ),
    }


def validate_latency_runtime(specification: dict[str, Any]) -> dict[str, Any]:
    """Enforce the predeclared Marvin single-thread benchmark conditions."""

    failures: list[str] = []
    hostname = platform.node()
    if hostname.split(".", 1)[0] != str(specification["host"]):
        failures.append(
            f"hostname is {hostname!r}; run the container with --hostname {specification['host']}"
        )
    if not platform.system().lower().startswith("linux"):
        failures.append("CPU benchmark must run under Linux")
    model_name = _cpu_model_name()
    expected_words = re.findall(r"[a-z0-9]+", str(specification["cpu"]).lower())
    observed_words = set(re.findall(r"[a-z0-9]+", model_name.lower()))
    if not all(word in observed_words for word in expected_words):
        failures.append(f"CPU model {model_name!r} does not match the protocol")
    environment = {
        str(name): os.environ.get(str(name))
        for name in specification["environment_variables"]
    }
    for name, expected in specification["environment_variables"].items():
        if environment[str(name)] != str(expected):
            failures.append(f"{name} must be set to {expected} before process start")
    pinned_cpu = int(specification["pinned_logical_cpu"])
    if not hasattr(os, "sched_setaffinity"):
        failures.append("logical CPU affinity is unavailable")
        affinity: list[int] = []
    else:
        try:
            os.sched_setaffinity(0, {pinned_cpu})
            affinity = sorted(int(value) for value in os.sched_getaffinity(0))
            if affinity != [pinned_cpu]:
                failures.append("process affinity is not the one predeclared CPU")
        except OSError as error:
            affinity = []
            failures.append(f"could not pin logical CPU {pinned_cpu}: {error}")
    governor, frequency_khz = _cpu_frequency_state(pinned_cpu)
    if governor is None:
        failures.append("CPU governor could not be recorded")
    if frequency_khz is None:
        failures.append("CPU frequency state could not be recorded")
    return {
        "decision": "GO" if not failures else "NO_GO",
        "failures": failures,
        "hostname": hostname,
        "platform": platform.platform(),
        "cpu_model": model_name,
        "logical_cpu_affinity": affinity,
        "environment_variables": environment,
        "cpu_governor": governor,
        "cpu_frequency_khz_at_start": frequency_khz,
    }


def benchmark_policy_calls(
    policy: Any,
    *,
    warmup_calls: int,
    timed_calls_per_repetition: int,
    repetitions: int,
    clock: Callable[[], int] = perf_counter_ns,
) -> dict[str, Any]:
    """Benchmark complete batch-one policy steps using repetition means."""

    if min(warmup_calls, timed_calls_per_repetition, repetitions) <= 0:
        raise ValueError("latency benchmark counts must be positive")
    observation = np.linspace(-0.5, 0.5, 19, dtype=np.float32)
    observation[-3:] = 0.0
    carry = policy.initial_carry()
    for _ in range(warmup_calls):
        _, carry = policy.act(observation, carry)
    repetition_ns_per_call = []
    for _ in range(repetitions):
        start = clock()
        for _ in range(timed_calls_per_repetition):
            _, carry = policy.act(observation, carry)
        elapsed = clock() - start
        repetition_ns_per_call.append(elapsed / timed_calls_per_repetition)
    microseconds = np.asarray(repetition_ns_per_call, dtype=np.float64) / 1000.0
    return {
        "warmup_calls": warmup_calls,
        "timed_calls_per_repetition": timed_calls_per_repetition,
        "repetitions": repetitions,
        "total_timed_calls": timed_calls_per_repetition * repetitions,
        "clock": "perf_counter_ns",
        "input": observation.tolist(),
        "repetition_mean_microseconds": [float(value) for value in microseconds],
        "median_microseconds": float(np.median(microseconds)),
        "p95_microseconds": float(np.quantile(microseconds, 0.95)),
        "p95_definition": "95th percentile across the ten repetition means",
    }


def _named_arrays(value: Any, prefix: str = "carry") -> dict[str, np.ndarray[Any, Any]]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        return {prefix: value}
    if hasattr(value, "__dataclass_fields__"):
        result: dict[str, np.ndarray[Any, Any]] = {}
        for name in value.__dataclass_fields__:
            result.update(_named_arrays(getattr(value, name), f"{prefix}.{name}"))
        return result
    raise TypeError(f"unsupported policy carry type {type(value).__name__}")


def _cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor()
    for line in cpuinfo.read_text().splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _cpu_frequency_state(cpu: int) -> tuple[str | None, int | None]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    governor_path = root / "scaling_governor"
    frequency_paths = (root / "scaling_cur_freq", root / "cpuinfo_cur_freq")
    governor = governor_path.read_text().strip() if governor_path.is_file() else None
    frequency = next(
        (
            int(path.read_text().strip())
            for path in frequency_paths
            if path.is_file()
        ),
        None,
    )
    return governor, frequency
