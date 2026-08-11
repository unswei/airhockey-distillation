import pickle
from pathlib import Path

import pytest

from airhockey_distill.teachers import (
    RetainingCheckpointFactory,
    StepCheckpointClockFactory,
    build_training_arguments,
    list_complete_checkpoints,
    select_best_validation_result,
)


def test_training_arguments_keep_environment_and_logging_independent():
    profile = {
        "model_preset": "size1m",
        "seed": 7,
        "batch_size": 16,
        "batch_length": 64,
        "report_length": 32,
        "replay_size": 2_000_000,
        "steps": 1_000_000,
        "train_ratio": 80.0,
        "envs": 1,
        "log_every": 120,
        "report_every": 600,
        "save_every": 1800,
        "debug": True,
        "replay_fractions": {
            "uniform": 0.2,
            "priority": 0.6,
            "recency": 0.2,
        },
    }

    arguments = build_training_arguments(
        profile,
        environment_id="Test-v0",
        logdir=Path("/tmp/test-logdir"),
    )

    assert _value(arguments, "--run.envs") == "1"
    assert _value(arguments, "--run.log_every") == "120"
    assert _value(arguments, "--run.train_ratio") == "80.0"
    assert _value(arguments, "--replay.fracs.priority") == "0.6"


def test_retaining_checkpoint_factory_retains_and_finalises_training_checkpoint():
    created = []

    class FakeCheckpoint:
        def __init__(self, directory, *, keep, step, write):
            self.directory = directory
            self.keep = keep
            self.step = step
            self.write = write
            self.saved = 0
            created.append(self)

        def save(self):
            self.saved += 1

    factory = RetainingCheckpointFactory(FakeCheckpoint, keep=17)
    training = factory("/run/dreamer/ckpt")
    loader = factory()
    factory.save_final()

    assert training.keep == 17
    assert training.saved == 1
    assert loader.keep == 1
    assert created == [training, loader]


def test_retaining_checkpoint_factory_requires_a_training_checkpoint():
    factory = RetainingCheckpointFactory(lambda *args, **kwargs: object(), keep=2)
    with pytest.raises(RuntimeError, match="did not create"):
        factory.save_final()


def test_complete_checkpoints_are_ordered_and_duplicate_steps_are_collapsed(tmp_path):
    _checkpoint(tmp_path / "2026A", 20)
    _checkpoint(tmp_path / "2026B", 20)
    _checkpoint(tmp_path / "first", 10)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()

    checkpoints = list_complete_checkpoints(tmp_path)

    assert checkpoints == (
        (10, tmp_path / "first"),
        (20, tmp_path / "2026B"),
    )


def test_step_checkpoint_clock_replaces_only_the_save_clock():
    wall_clocks = []

    def make_wall_clock(every):
        wall_clocks.append(every)
        return f"wall-{every}"

    factory = StepCheckpointClockFactory(
        make_wall_clock,
        save_every_seconds=300,
        checkpoint_every_steps=20_000,
    )

    assert factory(30) == "wall-30"
    save_clock = factory(300)
    assert save_clock(10) is False
    assert save_clock(19_990) is False
    assert save_clock(20_000) is True
    assert save_clock(20_010) is False
    assert save_clock(40_000) is True
    assert wall_clocks == [30]


def test_step_checkpoint_clock_resumes_at_the_next_boundary():
    factory = StepCheckpointClockFactory(
        lambda every: None,
        save_every_seconds=300,
        checkpoint_every_steps=20_000,
    )

    save_clock = factory(300)

    assert save_clock(60_010) is False
    assert save_clock(79_990) is False
    assert save_clock(80_000) is True


def test_validation_selection_uses_save_rate_then_return_then_earlier_step():
    results = [
        _result("untrained", None, 0.9, 2.0),
        _result("checkpoint", 20_000, 0.6, 0.2),
        _result("checkpoint", 40_000, 0.7, 0.1),
        _result("checkpoint", 60_000, 0.7, 0.3),
        _result("checkpoint", 80_000, 0.7, 0.3),
    ]

    selected = select_best_validation_result(results)

    assert selected["checkpoint_step"] == 60_000


def test_validation_selection_requires_a_checkpoint():
    with pytest.raises(ValueError, match="no checkpoint"):
        select_best_validation_result([_result("untrained", None, 0.1, -1.0)])


def _value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def _result(policy: str, step: int | None, save_rate: float, score: float):
    return {
        "policy": policy,
        "checkpoint_step": step,
        "summary": {"save_rate": save_rate, "mean_score": score},
    }


def _checkpoint(path: Path, step: int) -> None:
    path.mkdir()
    (path / "done").touch()
    (path / "step.pkl").write_bytes(pickle.dumps(step))
