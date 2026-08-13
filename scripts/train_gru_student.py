#!/usr/bin/env python3
"""Train one matched GRU-64 student on deterministic teacher means."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from airhockey_distill.students import GRURecurrentPolicy, save_gru_checkpoint

if __package__:
    from scripts.train_structured_student import (
        EpisodeSplit,
        evaluate_split,
        load_episode_splits,
        split_residue_mapping,
        train_epoch,
        validate_deterministic_manifest,
    )
else:
    from train_structured_student import (
        EpisodeSplit,
        evaluate_split,
        load_episode_splits,
        split_residue_mapping,
        train_epoch,
        validate_deterministic_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from airhockey_distill.students.gru_torch import GRURecurrentModule

    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    validate_gru_full_training_config(config)
    training = config["training"]
    export_config = config["export"]

    dataset_directory = args.dataset.resolve()
    manifest_path = dataset_directory / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    expected_manifest_hash = config["provenance"]["aggregate_manifest_sha256"]
    if manifest_hash != expected_manifest_hash:
        raise ValueError("aggregate dataset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    validate_deterministic_manifest(manifest, config["dataset"])
    splits = load_episode_splits(
        dataset_directory,
        manifest,
        config["dataset"],
        maximum_episode_steps=int(training["maximum_episode_steps"]),
        burn_in_steps=int(training["burn_in_steps"]),
    )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    seed = int(training["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(
        max(1, min(int(training["torch_threads"]), os.cpu_count() or 1))
    )
    module = GRURecurrentModule(seed=seed)
    optimiser = torch.optim.AdamW(
        module.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)

    batch_size = int(training["batch_size"])
    sequence_length = int(training["sequence_length"])
    maximum_epochs = int(training["maximum_epochs"])
    patience_epochs = int(training["patience_epochs"])
    minimum_improvement = float(training["minimum_validation_improvement"])
    gradient_clip = float(training["gradient_clip_global_norm"])

    initial_metrics = {
        name: evaluate_split(module, split, batch_size, sequence_length)
        for name, split in splits.items()
        if name != "internal_test"
    }
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    patience = 0
    epochs_completed = 0
    metrics_path = output / "metrics.jsonl"
    started = perf_counter()
    with metrics_path.open("w") as metrics_stream:
        _write_metric(
            metrics_stream,
            {
                "epoch": 0,
                "train": initial_metrics["train"],
                "validation": initial_metrics["validation"],
            },
        )
        for epoch in range(1, maximum_epochs + 1):
            online_training_mse = train_epoch(
                module,
                optimiser,
                splits["train"],
                batch_size=batch_size,
                sequence_length=sequence_length,
                gradient_clip=gradient_clip,
                generator=generator,
            )
            validation = evaluate_split(
                module, splits["validation"], batch_size, sequence_length
            )
            _write_metric(
                metrics_stream,
                {
                    "epoch": epoch,
                    "online_train_action_mse": online_training_mse,
                    "validation": validation,
                },
            )
            epochs_completed = epoch
            validation_mse = float(validation["action_mse"])
            if validation_mse < best_validation - minimum_improvement:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = deepcopy(module.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= patience_epochs:
                    break

    if best_state is None:
        raise RuntimeError("training did not produce a selected checkpoint")
    module.load_state_dict(best_state)
    module.eval()
    selected_metrics = {
        name: evaluate_split(module, split, batch_size, sequence_length)
        for name, split in splits.items()
    }

    checkpoint_path = output / "checkpoint.npz"
    checkpoint_metadata = {
        "schema_version": 1,
        "student_id": config["policy"]["id"],
        "training_seed": seed,
        "code_commit": args.code_commit,
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": manifest_hash,
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "selected_epoch": best_epoch,
        "validation_action_mse": best_validation,
        "pilot_scope": config["pilot_scope"]["classification"],
    }
    exported_parameters = module.export_numpy_parameters()
    save_gru_checkpoint(checkpoint_path, exported_parameters, checkpoint_metadata)
    exported = GRURecurrentPolicy(exported_parameters, checkpoint_metadata)
    restored = GRURecurrentPolicy.load(checkpoint_path)
    action_error, state_error, reload_exact = verify_gru_export(
        module,
        exported,
        restored,
        splits["validation"],
        episode_count=int(export_config["verification_episodes"]),
    )
    export_error = max(action_error, state_error)
    if export_error > float(export_config["maximum_absolute_error"]):
        raise RuntimeError("exported NumPy GRU does not match PyTorch")
    if bool(export_config["require_exact_checkpoint_reload"]) and not reload_exact:
        raise RuntimeError("GRU checkpoint reload is not exact")

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "config_sha256": _sha256(config_path),
        "pilot_scope": config["pilot_scope"],
        "dataset": str(dataset_directory),
        "dataset_manifest_sha256": manifest_hash,
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "student_input_fields": ["observations", "previous_actions"],
        "split_episode_counts": {
            name: split.episode_count for name, split in splits.items()
        },
        "split_transition_counts": {
            name: split.transition_count for name, split in splits.items()
        },
        "training_seed": seed,
        "selected_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "selection_metric": training["selection_metric"],
        "initial_metrics": initial_metrics,
        "selected_metrics": selected_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "export_action_maximum_absolute_error": action_error,
        "export_state_maximum_absolute_error": state_error,
        "export_maximum_absolute_error": export_error,
        "checkpoint_reload_exact": reload_exact,
        "parameter_count": exported.parameter_count,
        "recurrent_parameter_count": exported.recurrent_parameter_count,
        "duration_seconds": perf_counter() - started,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def validate_gru_full_training_config(config: dict[str, Any]) -> None:
    """Validate the frozen GRU engineering-pilot protocol."""

    if config["policy"]["id"] != "gru_n64":
        raise ValueError("full GRU pilot requires policy gru_n64")
    if int(config["policy"]["state_dimension"]) != 64:
        raise ValueError("full GRU pilot requires state dimension 64")
    if config["pilot_scope"]["classification"] != (
        "engineering_pilot_not_principal_family_comparison"
    ):
        raise ValueError("GRU pilot must retain its engineering-only scope")
    training = config["training"]
    if training["target"] != "deterministic_teacher_action_mean":
        raise ValueError("full GRU target must be deterministic teacher means")
    if training["reset_state"] != "episode_boundary_only":
        raise ValueError("GRU state may reset only at episode boundaries")
    sequence_length = int(training["sequence_length"])
    burn_in_steps = int(training["burn_in_steps"])
    if sequence_length != burn_in_steps + int(training["loss_steps"]):
        raise ValueError("sequence length must equal burn-in plus loss suffix")
    if burn_in_steps != 0:
        raise ValueError("GRU pilot requires loss on every valid step")
    if int(training["maximum_episode_steps"]) % sequence_length:
        raise ValueError("maximum episode steps must divide into complete chunks")
    split_residue_mapping(config["dataset"])


def verify_gru_export(
    module: Any,
    exported: GRURecurrentPolicy,
    restored: GRURecurrentPolicy,
    split: EpisodeSplit,
    *,
    episode_count: int,
) -> tuple[float, float, bool]:
    """Compare PyTorch actions/states and exact framework-neutral reload."""

    import torch

    action_error = 0.0
    state_error = 0.0
    reload_exact = True
    module.eval()
    for index in range(min(episode_count, split.episode_count)):
        length = int(split.episode_lengths[index])
        observations = split.observations[index, :length]
        previous_actions = split.previous_actions[index, :length]
        with torch.no_grad():
            torch_actions, torch_states = module.forward_sequence(
                torch.from_numpy(observations[None, :]),
                torch.from_numpy(previous_actions[None, :]),
            )
        exported_actions, exported_states = exported.teacher_forced_sequence(
            observations, previous_actions
        )
        restored_actions, restored_states = restored.teacher_forced_sequence(
            observations, previous_actions
        )
        action_error = max(
            action_error,
            float(np.max(np.abs(torch_actions.numpy()[0] - exported_actions))),
        )
        state_error = max(
            state_error,
            float(np.max(np.abs(torch_states.numpy()[0] - exported_states))),
        )
        reload_exact = reload_exact and np.array_equal(
            restored_actions, exported_actions
        ) and np.array_equal(restored_states, exported_states)
    return action_error, state_error, reload_exact


def _write_metric(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    print(json.dumps(record, sort_keys=True), flush=True)


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
