"""DreamerV3 training and checkpoint utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable


def build_training_arguments(
    profile: dict[str, Any],
    *,
    environment_id: str,
    logdir: Path,
) -> list[str]:
    """Translate a versioned project profile into pinned Dreamer flags."""

    arguments = [
        "--configs",
        str(profile["model_preset"]),
        "--task",
        f"gymnasium_{environment_id}",
        "--logdir",
        str(logdir),
        "--seed",
        str(profile["seed"]),
        "--batch_size",
        str(profile["batch_size"]),
        "--batch_length",
        str(profile["batch_length"]),
        "--report_length",
        str(profile["report_length"]),
        "--replay.size",
        str(profile["replay_size"]),
        "--run.steps",
        str(profile["steps"]),
        "--run.train_ratio",
        str(profile["train_ratio"]),
        "--run.envs",
        str(profile.get("envs", 1)),
        "--run.log_every",
        str(profile.get("log_every", 120)),
        "--run.report_every",
        str(profile.get("report_every", 300)),
        "--run.save_every",
        str(profile.get("save_every", 900)),
        "--run.debug",
        _dreamer_boolean(bool(profile.get("debug", True))),
        "--jax.platform",
        "cuda",
        "--jax.prealloc",
        "False",
        "--logger.outputs",
        "jsonl",
        "--errfile",
        "True",
    ]
    replay_fractions = profile.get("replay_fractions")
    if replay_fractions is not None:
        for name in ("uniform", "priority", "recency"):
            arguments.extend(
                [f"--replay.fracs.{name}", str(replay_fractions[name])]
            )
    return arguments


class RetainingCheckpointFactory:
    """Increase upstream retention and expose its training checkpoint."""

    def __init__(self, constructor: Callable[..., Any], keep: int):
        if keep < 1:
            raise ValueError("checkpoint retention must be positive")
        self._constructor = constructor
        self._keep = keep
        self.training_checkpoint: Any | None = None

    def __call__(
        self,
        directory: Any = None,
        keep: int = 1,
        step: Any = None,
        write: bool = True,
    ) -> Any:
        is_training = directory is not None and Path(str(directory)).name == "ckpt"
        checkpoint = self._constructor(
            directory,
            keep=self._keep if is_training else keep,
            step=step,
            write=write,
        )
        if is_training:
            self.training_checkpoint = checkpoint
        return checkpoint

    def save_final(self) -> None:
        if self.training_checkpoint is None:
            raise RuntimeError("Dreamer did not create its training checkpoint")
        self.training_checkpoint.save()


def list_complete_checkpoints(directory: Path) -> tuple[tuple[int, Path], ...]:
    """Return complete checkpoints ordered by their stored environment step."""

    checkpoints: dict[int, Path] = {}
    if not directory.exists():
        return ()
    for path in directory.iterdir():
        if not path.is_dir() or not (path / "done").exists():
            continue
        step = read_checkpoint_step(path)
        previous = checkpoints.get(step)
        if previous is None or path.name > previous.name:
            checkpoints[step] = path
    return tuple(sorted(checkpoints.items()))


def read_checkpoint_step(checkpoint: Path) -> int:
    """Read the integer step stored by the trusted pinned training process."""

    value = pickle.loads((checkpoint / "step.pkl").read_bytes())
    if not isinstance(value, int) or value < 0:
        raise TypeError(f"invalid checkpoint step in {checkpoint}")
    return value


def select_best_validation_result(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select by save rate, then return, then the earlier checkpoint."""

    checkpoint_results = [
        result for result in results if result.get("policy") == "checkpoint"
    ]
    if not checkpoint_results:
        raise ValueError("no checkpoint validation results")
    return max(
        checkpoint_results,
        key=lambda result: (
            float(result["summary"]["save_rate"]),
            float(result["summary"]["mean_score"]),
            -int(result["checkpoint_step"]),
        ),
    )


def _dreamer_boolean(value: bool) -> str:
    return "True" if value else "False"
