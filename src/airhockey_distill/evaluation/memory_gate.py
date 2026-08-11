"""Paired Stage B gate for evidence that blackout performance needs memory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def evaluate_memory_gate(
    teacher_episodes: Sequence[Mapping[str, Any]],
    feed_forward_episodes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare exact paired episodes and apply the predeclared memory criteria."""

    teacher = _index(teacher_episodes)
    student = _index(feed_forward_episodes)
    if set(teacher) != set(student):
        missing_teacher = sorted(set(student) - set(teacher))
        missing_student = sorted(set(teacher) - set(student))
        raise ValueError(
            "teacher and feed-forward episode keys differ: "
            f"missing_teacher={missing_teacher[:3]}, missing_feed_forward={missing_student[:3]}"
        )

    blackout_values = sorted({key[1] for key in teacher})
    bootstrap_samples = int(config["bootstrap_samples"])
    seed = int(config["bootstrap_seed"])
    rng = np.random.default_rng(seed)
    by_blackout: dict[str, Any] = {}
    for blackout in blackout_values:
        keys = sorted(key for key in teacher if key[1] == blackout)
        teacher_saved = np.asarray(
            [teacher[key]["outcome"] in SAVE_OUTCOMES for key in keys],
            dtype=np.float64,
        )
        student_saved = np.asarray(
            [student[key]["outcome"] in SAVE_OUTCOMES for key in keys],
            dtype=np.float64,
        )
        differences = teacher_saved - student_saved
        bootstrap = differences[
            rng.integers(
                0, len(differences), size=(bootstrap_samples, len(differences))
            )
        ].mean(axis=1)
        by_blackout[str(blackout)] = {
            "episodes": len(keys),
            "teacher_save_rate": float(teacher_saved.mean()),
            "feed_forward_save_rate": float(student_saved.mean()),
            "paired_teacher_advantage": float(differences.mean()),
            "paired_teacher_advantage_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "teacher_only_saves": int(
                np.sum((teacher_saved == 1) & (student_saved == 0))
            ),
            "feed_forward_only_saves": int(
                np.sum((teacher_saved == 0) & (student_saved == 1))
            ),
        }

    no_blackout = by_blackout["0"]
    long_blackout = by_blackout[str(int(config["long_blackout_steps"]))]
    advantage_growth = (
        long_blackout["paired_teacher_advantage"]
        - no_blackout["paired_teacher_advantage"]
    )
    checks = [
        _check(
            "credible_visible_feed_forward_baseline",
            no_blackout["feed_forward_save_rate"]
            >= float(config["minimum_feed_forward_no_blackout_save_rate"]),
            no_blackout["feed_forward_save_rate"],
            f">= {config['minimum_feed_forward_no_blackout_save_rate']}",
        ),
        _check(
            "comparable_no_blackout_performance",
            no_blackout["paired_teacher_advantage"]
            <= float(config["maximum_no_blackout_teacher_advantage"]),
            no_blackout["paired_teacher_advantage"],
            f"<= {config['maximum_no_blackout_teacher_advantage']}",
        ),
        _check(
            "material_long_blackout_advantage",
            long_blackout["paired_teacher_advantage"]
            >= float(config["minimum_long_blackout_teacher_advantage"]),
            long_blackout["paired_teacher_advantage"],
            f">= {config['minimum_long_blackout_teacher_advantage']}",
        ),
        _check(
            "advantage_grows_with_blackout",
            advantage_growth >= float(config["minimum_advantage_growth"]),
            advantage_growth,
            f">= {config['minimum_advantage_growth']}",
        ),
    ]
    if bool(config["require_positive_long_advantage_ci_lower"]):
        checks.append(
            _check(
                "positive_long_blackout_advantage_ci",
                long_blackout["paired_teacher_advantage_ci95"][0] > 0.0,
                long_blackout["paired_teacher_advantage_ci95"][0],
                "> 0.0",
            )
        )
    blocking = [check["check_id"] for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "decision": "GO" if not blocking else "NO_GO",
        "interpretation": (
            "paired evidence supports a memory requirement"
            if not blocking
            else "paired evidence does not yet establish a memory requirement"
        ),
        "paired_episode_count": len(teacher),
        "by_blackout_steps": by_blackout,
        "advantage_growth_0_to_long_blackout": float(advantage_growth),
        "checks": checks,
        "blocking_checks": blocking,
    }


def _index(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for episode in episodes:
        key = (str(episode["shot_id"]), int(episode["blackout_steps"]))
        if key in indexed:
            raise ValueError(f"duplicate paired episode key {key}")
        indexed[key] = episode
    if not indexed:
        raise ValueError("paired evaluation must not be empty")
    return indexed


def _check(
    check_id: str, passed: bool, observed: float, required: str
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "observed": float(observed),
        "required": required,
    }
