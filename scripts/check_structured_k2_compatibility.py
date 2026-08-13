#!/usr/bin/env python3
"""Verify bit-exact inference for the frozen successful k=2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from airhockey_distill.students import StructuredRecurrentPolicy

CHECKPOINT_SHA256 = "c6e5556690bf6215c94da32cd07286487f581b1ae67ff5b117501ad68b4ee9e1"
ACTION_SHA256 = "2563e7ed46eb8217e6b40be62e2332eb33f61fc56df92c50761723c14e1bdd62"
STATE_SHA256 = "87571ac63055583b6af0ffaa8d423c856710324a785062f37733f6cd24814271"
SEQUENCE_SEED = 88103
SEQUENCE_STEPS = 257


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def verify(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    rng = np.random.default_rng(SEQUENCE_SEED)
    observations = rng.normal(size=(SEQUENCE_STEPS, 19)).astype(np.float32)
    previous_actions = rng.uniform(
        -1.0,
        1.0,
        size=(SEQUENCE_STEPS, 2),
    ).astype(np.float32)
    previous_actions[0] = 0.0
    initial_state = rng.normal(size=64).astype(np.float32)

    policy = StructuredRecurrentPolicy.load(checkpoint)
    actions, states = policy.teacher_forced_sequence(
        observations,
        previous_actions,
        initial_state,
    )
    observed = {
        "checkpoint_sha256": _sha256_file(checkpoint),
        "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "state_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
    }
    expected = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "action_sha256": ACTION_SHA256,
        "state_sha256": STATE_SHA256,
    }
    checks = {
        name: observed[name] == expected[name]
        for name in expected
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "innovation_rank": policy.innovation_rank,
        "sequence_seed": SEQUENCE_SEED,
        "sequence_steps": SEQUENCE_STEPS,
        "observed": observed,
        "expected": expected,
        "checks": checks,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = verify(parse_args().checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
