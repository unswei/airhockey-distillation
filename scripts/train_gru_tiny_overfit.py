#!/usr/bin/env python3
"""Overfit the matched GRU-64 on a tiny deterministic-mean dataset."""

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
from scripts.train_structured_tiny_overfit import (
    _load_complete_episodes,
    _validate_manifest,
    _write_metric,
    evaluate_overfit_gate,
    masked_action_mse,
    pad_complete_episodes,
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
    training = config["training"]
    gate = config["overfit_gate"]
    dataset_config = config["dataset"]
    validate_gru_overfit_config(config)

    dataset_directory = args.dataset.resolve()
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, dataset_config)
    source_episodes = _load_complete_episodes(dataset_directory, manifest)
    episodes = source_episodes[: int(training["training_episode_count"])]
    padded = pad_complete_episodes(
        episodes,
        sequence_length=int(training["sequence_length"]),
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
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    module = GRURecurrentModule(seed=seed)
    optimiser = torch.optim.AdamW(
        module.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    observations = torch.from_numpy(padded["observations"])
    previous_actions = torch.from_numpy(padded["previous_actions"])
    targets = torch.from_numpy(padded["teacher_actions"])
    loss_mask = torch.from_numpy(padded["loss_mask"])

    started = perf_counter()
    initial_mse = _evaluate_mse(
        module, observations, previous_actions, targets, loss_mask
    )
    best_mse = initial_mse
    best_epoch = 0
    best_state = deepcopy(module.state_dict())
    epochs_completed = 0
    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("w") as metrics_stream:
        _write_metric(metrics_stream, 0, initial_mse)
        for epoch in range(1, int(training["maximum_epochs"]) + 1):
            module.train()
            predictions, _ = module.forward_sequence(observations, previous_actions)
            loss = masked_action_mse(predictions, targets, loss_mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(), float(training["gradient_clip_global_norm"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite gradient norm during GRU overfit")
            optimiser.step()
            epochs_completed = epoch
            if epoch % int(training["log_every_epochs"]) and epoch != int(
                training["maximum_epochs"]
            ):
                continue
            current_mse = _evaluate_mse(
                module, observations, previous_actions, targets, loss_mask
            )
            _write_metric(metrics_stream, epoch, current_mse)
            if current_mse < best_mse:
                best_mse = current_mse
                best_epoch = epoch
                best_state = deepcopy(module.state_dict())
            if current_mse <= float(gate["maximum_training_action_mse"]):
                break

    module.load_state_dict(best_state)
    module.eval()
    checkpoint_path = output / "checkpoint.npz"
    checkpoint_metadata = {
        "training_seed": seed,
        "code_commit": args.code_commit,
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(manifest_path),
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "selected_epoch": best_epoch,
        "training_action_mse": best_mse,
    }
    exported_parameters = module.export_numpy_parameters()
    save_gru_checkpoint(checkpoint_path, exported_parameters, checkpoint_metadata)
    exported = GRURecurrentPolicy(exported_parameters, checkpoint_metadata)
    restored = GRURecurrentPolicy.load(checkpoint_path)
    action_error, state_error, reload_exact = _verify_export(
        module, exported, restored, episodes
    )
    export_error = max(action_error, state_error)
    reduction = 1.0 - best_mse / initial_mse if initial_mse > 0.0 else 0.0
    checks = evaluate_overfit_gate(
        initial_mse=initial_mse,
        final_mse=best_mse,
        export_error=export_error,
        reload_exact=reload_exact,
        gate=gate,
    )
    blocking = [check["check_id"] for check in checks if not check["passed"]]
    result = {
        "schema_version": 1,
        "status": "completed",
        "decision": "GO" if not blocking else "NO_GO",
        "blocking_checks": blocking,
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "config_sha256": _sha256(config_path),
        "dataset": str(dataset_directory),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "student_input_fields": ["observations", "previous_actions"],
        "source_episode_count": len(source_episodes),
        "training_episode_count": len(episodes),
        "episode_lengths": [len(episode["observations"]) for episode in episodes],
        "loss_bearing_action_values": int(np.sum(padded["loss_mask"])) * 2,
        "initial_training_action_mse": initial_mse,
        "selected_training_action_mse": best_mse,
        "fractional_loss_reduction": reduction,
        "selected_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "export_action_maximum_absolute_error": action_error,
        "export_state_maximum_absolute_error": state_error,
        "export_maximum_absolute_error": export_error,
        "checkpoint_reload_exact": reload_exact,
        "parameter_count": exported.parameter_count,
        "recurrent_parameter_count": exported.recurrent_parameter_count,
        "checks": checks,
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


def validate_gru_overfit_config(config: dict[str, Any]) -> None:
    """Reject architecture, target and loss-mask changes before training."""

    policy = config["policy"]
    if policy["id"] != "gru_n64" or int(policy["state_dimension"]) != 64:
        raise ValueError("tiny GRU overfit requires the matched GRU-64 policy")
    training = config["training"]
    episode_count = int(training["training_episode_count"])
    if not 0 < episode_count <= int(config["dataset"]["episodes"]):
        raise ValueError("training episode count must lie inside the dataset")
    if int(training["batch_size"]) != episode_count:
        raise ValueError("tiny overfit uses all selected episodes in one batch")
    if int(training["burn_in_steps"]) != 0:
        raise ValueError("GRU engineering gate requires loss on every valid step")
    if training["target"] != "deterministic_teacher_action_mean":
        raise ValueError("tiny overfit target must be deterministic teacher means")


def _evaluate_mse(module: Any, observations: Any, previous_actions: Any, targets: Any, loss_mask: Any) -> float:
    import torch

    module.eval()
    with torch.no_grad():
        predictions, _ = module.forward_sequence(observations, previous_actions)
        return float(masked_action_mse(predictions, targets, loss_mask))


def _verify_export(
    module: Any,
    exported: GRURecurrentPolicy,
    restored: GRURecurrentPolicy,
    episodes: list[dict[str, np.ndarray[Any, Any]]],
) -> tuple[float, float, bool]:
    import torch

    action_error = 0.0
    state_error = 0.0
    reload_exact = True
    module.eval()
    for episode in episodes:
        observations = episode["observations"]
        previous_actions = episode["previous_actions"]
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
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
