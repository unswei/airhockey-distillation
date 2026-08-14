#!/usr/bin/env python3
"""Verify the native structured kernel against canonical NumPy arithmetic."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import _airhockey_pairwise_float32 as native_pairwise
from airhockey_distill.principal_sweep import sha256_file
from airhockey_distill.students import load_principal_policy
from airhockey_distill.students.structured import (
    NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION,
)


LINEAR_SHAPES = (
    (64, 19),
    (64, 64),
    (64, 32),
    (64, 4),
    (64, 2),
    (64, 1),
    (64, 96),
    (4, 64),
    (4, 32),
    (4, 2),
    (2, 64),
    (1, 64),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("FAMILY", "SEED", "PATH"),
        required=True,
    )
    parser.add_argument("--kernel-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--linear-trials", type=int, default=256)
    parser.add_argument("--policy-steps", type=int, default=1000)
    return parser.parse_args()


def canonical_linear(
    value: np.ndarray[Any, np.dtype[np.float32]],
    weight: np.ndarray[Any, np.dtype[np.float32]],
    bias: np.ndarray[Any, np.dtype[np.float32]] | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    terms = value[None, :] * weight
    while terms.shape[-1] > 1:
        pair_count = terms.shape[-1] // 2
        reduced = (
            terms[:, : 2 * pair_count : 2]
            + terms[:, 1 : 2 * pair_count : 2]
        )
        if terms.shape[-1] % 2:
            reduced = np.concatenate((reduced, terms[:, -1:]), axis=-1)
        terms = reduced
    result = terms[:, 0]
    return result if bias is None else result + bias


def verify_linear_kernel(trials: int) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("linear trials must be positive")
    rng = np.random.default_rng(20260815)
    comparisons = 0
    maximum_absolute_error = 0.0
    exact = True
    for output_width, input_width in LINEAR_SHAPES:
        output = np.empty(output_width, dtype=np.float32)
        for trial in range(trials):
            value = rng.normal(size=input_width).astype(np.float32)
            weight = rng.normal(size=(output_width, input_width)).astype(np.float32)
            bias = (
                None
                if trial % 2
                else rng.normal(size=output_width).astype(np.float32)
            )
            native_pairwise.linear_into(value, weight, bias, output)
            expected = canonical_linear(value, weight, bias)
            error = float(np.max(np.abs(output - expected)))
            maximum_absolute_error = max(maximum_absolute_error, error)
            exact = exact and bool(
                np.array_equal(output.view(np.uint32), expected.view(np.uint32))
            )
            comparisons += output_width
    return {
        "shapes": [list(shape) for shape in LINEAR_SHAPES],
        "trials_per_shape": trials,
        "scalar_comparisons": comparisons,
        "maximum_absolute_error": maximum_absolute_error,
        "bit_exact": exact,
    }


def verify_checkpoint(
    family: str,
    seed: int,
    checkpoint: Path,
    steps: int,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("policy steps must be positive")
    policy = load_principal_policy(family, checkpoint)
    implementation = getattr(
        policy.policy,
        "batch_one_inference_implementation",
        None,
    )
    rng = np.random.default_rng(920_000 + seed)
    carry = policy.initial_carry()
    reference_state = carry.memory.copy()
    reference_previous_action = carry.previous_action.copy()
    action_maximum_absolute_error = 0.0
    carry_maximum_absolute_error = 0.0
    action_bit_exact = True
    carry_bit_exact = True
    for _ in range(steps):
        observation = rng.normal(size=19).astype(np.float32)
        expected_action, expected_state = policy.policy.step(
            observation,
            reference_previous_action,
            reference_state,
        )
        action, carry = policy.act(observation, carry)
        action_maximum_absolute_error = max(
            action_maximum_absolute_error,
            float(np.max(np.abs(action - expected_action))),
        )
        carry_maximum_absolute_error = max(
            carry_maximum_absolute_error,
            float(np.max(np.abs(carry.memory - expected_state))),
        )
        action_bit_exact = action_bit_exact and bool(
            np.array_equal(action.view(np.uint32), expected_action.view(np.uint32))
        )
        carry_bit_exact = carry_bit_exact and bool(
            np.array_equal(
                carry.memory.view(np.uint32),
                expected_state.view(np.uint32),
            )
        )
        reference_state = expected_state
        reference_previous_action = expected_action
    return {
        "family_id": family,
        "training_seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "steps": steps,
        "implementation": implementation,
        "action_maximum_absolute_error": action_maximum_absolute_error,
        "carry_maximum_absolute_error": carry_maximum_absolute_error,
        "action_bit_exact": action_bit_exact,
        "carry_bit_exact": carry_bit_exact,
        "passed": (
            implementation == NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION
            and action_bit_exact
            and carry_bit_exact
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    kernel_manifest_path = args.kernel_manifest.resolve()
    kernel_manifest = json.loads(kernel_manifest_path.read_text())
    loaded_binary = Path(native_pairwise.__file__).resolve()
    binary_hash = sha256_file(loaded_binary)
    manifest_matches = (
        kernel_manifest.get("implementation")
        == NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION
        and kernel_manifest.get("code_commit") == args.code_commit
        and kernel_manifest.get("binary_sha256") == binary_hash
    )
    linear = verify_linear_kernel(args.linear_trials)
    checkpoints = [
        verify_checkpoint(
            family,
            int(seed),
            Path(path).resolve(),
            args.policy_steps,
        )
        for family, seed, path in args.checkpoint
    ]
    decision = (
        "GO"
        if manifest_matches
        and linear["bit_exact"]
        and all(record["passed"] for record in checkpoints)
        else "NO_GO"
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "decision": decision,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "implementation": NATIVE_PAIRWISE_FLOAT32_IMPLEMENTATION,
        "kernel_manifest": str(kernel_manifest_path),
        "kernel_manifest_sha256": sha256_file(kernel_manifest_path),
        "loaded_binary": str(loaded_binary),
        "loaded_binary_sha256": binary_hash,
        "kernel_manifest_matches": manifest_matches,
        "linear_kernel": linear,
        "checkpoints": checkpoints,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if decision != "GO":
        raise RuntimeError("structured native-kernel verification failed")
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
