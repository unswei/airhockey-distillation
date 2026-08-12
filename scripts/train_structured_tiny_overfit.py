#!/usr/bin/env python3
"""Overfit the n=64, k=2 student on a tiny deterministic-mean dataset."""

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

from airhockey_distill.students import (
    StructuredRecurrentPolicy,
    save_structured_checkpoint,
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

    from airhockey_distill.students.structured_torch import (
        StructuredRecurrentModule,
    )

    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    training = config["training"]
    gate = config["overfit_gate"]
    dataset_config = config["dataset"]
    _validate_config(config)

    dataset_directory = args.dataset.resolve()
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, dataset_config)
    source_episodes = _load_complete_episodes(dataset_directory, manifest)
    training_episode_count = int(
        training.get("training_episode_count", len(source_episodes))
    )
    if not 0 < training_episode_count <= len(source_episodes):
        raise ValueError("training episode count must lie inside the dataset")
    episodes = source_episodes[:training_episode_count]
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
    module = StructuredRecurrentModule(seed=seed)
    optimiser = torch.optim.AdamW(
        module.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    observations = torch.from_numpy(padded["observations"])
    previous_actions = torch.from_numpy(padded["previous_actions"])
    targets = torch.from_numpy(padded["teacher_actions"])
    loss_mask = torch.from_numpy(padded["loss_mask"])
    maximum_epochs = int(training["maximum_epochs"])
    log_every = int(training["log_every_epochs"])
    gradient_clip = float(training["gradient_clip_global_norm"])
    required_mse = float(gate["maximum_training_action_mse"])

    started = perf_counter()
    initial_mse = _evaluate_mse(
        module, observations, previous_actions, targets, loss_mask
    )
    best_mse = initial_mse
    best_epoch = 0
    best_state = deepcopy(module.state_dict())
    metrics_path = output / "metrics.jsonl"
    epochs_completed = 0
    with metrics_path.open("w") as metrics_stream:
        _write_metric(metrics_stream, 0, initial_mse)
        for epoch in range(1, maximum_epochs + 1):
            module.train()
            predictions, _ = module.forward_sequence(observations, previous_actions)
            loss = masked_action_mse(predictions, targets, loss_mask)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(), gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite gradient norm during tiny overfit")
            optimiser.step()
            epochs_completed = epoch

            should_evaluate = epoch % log_every == 0 or epoch == maximum_epochs
            if not should_evaluate:
                continue
            current_mse = _evaluate_mse(
                module, observations, previous_actions, targets, loss_mask
            )
            _write_metric(metrics_stream, epoch, current_mse)
            if current_mse < best_mse:
                best_mse = current_mse
                best_epoch = epoch
                best_state = deepcopy(module.state_dict())
            if current_mse <= required_mse:
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
    save_structured_checkpoint(
        checkpoint_path, exported_parameters, checkpoint_metadata
    )
    exported = StructuredRecurrentPolicy(exported_parameters, checkpoint_metadata)
    restored = StructuredRecurrentPolicy.load(checkpoint_path)
    export_error, reload_exact = _verify_export(
        module,
        exported,
        restored,
        episodes,
    )
    fractional_reduction = (
        1.0 - best_mse / initial_mse if initial_mse > 0.0 else 0.0
    )
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
        "fractional_loss_reduction": fractional_reduction,
        "selected_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
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


def pad_complete_episodes(
    episodes: list[dict[str, np.ndarray[Any, Any]]],
    *,
    sequence_length: int,
    burn_in_steps: int,
) -> dict[str, np.ndarray[Any, Any]]:
    """Pad complete episodes; mask padding and the initial burn-in from loss."""

    if not episodes:
        raise ValueError("tiny overfit dataset must contain episodes")
    if not 0 <= burn_in_steps < sequence_length:
        raise ValueError("burn-in must lie inside the sequence length")
    observations = np.zeros((len(episodes), sequence_length, 19), dtype=np.float32)
    previous_actions = np.zeros(
        (len(episodes), sequence_length, 2), dtype=np.float32
    )
    teacher_actions = np.zeros((len(episodes), sequence_length, 2), dtype=np.float32)
    valid_mask = np.zeros((len(episodes), sequence_length), dtype=bool)
    loss_mask = np.zeros((len(episodes), sequence_length), dtype=bool)
    for index, episode in enumerate(episodes):
        length = len(episode["observations"])
        if length > sequence_length:
            raise ValueError(
                f"episode length {length} exceeds sequence length {sequence_length}"
            )
        if length <= burn_in_steps:
            raise ValueError("episode contains no loss-bearing suffix")
        observations[index, :length] = episode["observations"]
        previous_actions[index, :length] = episode["previous_actions"]
        teacher_actions[index, :length] = episode["teacher_actions"]
        valid_mask[index, :length] = True
        loss_mask[index, burn_in_steps:length] = True
    return {
        "observations": observations,
        "previous_actions": previous_actions,
        "teacher_actions": teacher_actions,
        "valid_mask": valid_mask,
        "loss_mask": loss_mask,
    }


def masked_action_mse(predictions: Any, targets: Any, mask: Any) -> Any:
    expanded = mask.unsqueeze(-1).expand_as(predictions)
    if not bool(torch_any(expanded)):
        raise ValueError("loss mask must select at least one action value")
    return ((predictions - targets) ** 2)[expanded].mean()


def torch_any(value: Any) -> Any:
    """Small indirection keeps the padding helper importable without PyTorch."""

    import torch

    return torch.any(value)


def evaluate_overfit_gate(
    *,
    initial_mse: float,
    final_mse: float,
    export_error: float,
    reload_exact: bool,
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pure gate helper used by tests and external diagnostics."""

    reduction = 1.0 - final_mse / initial_mse if initial_mse > 0.0 else 0.0
    return [
        _check(
            "near_zero_training_action_mse",
            final_mse,
            f"<= {gate['maximum_training_action_mse']}",
            final_mse <= float(gate["maximum_training_action_mse"]),
        ),
        _check(
            "material_fractional_loss_reduction",
            reduction,
            f">= {gate['minimum_fractional_loss_reduction']}",
            reduction >= float(gate["minimum_fractional_loss_reduction"]),
        ),
        _check(
            "numpy_export_agreement",
            export_error,
            f"<= {gate['maximum_export_absolute_error']}",
            export_error <= float(gate["maximum_export_absolute_error"]),
        ),
        _check(
            "exact_checkpoint_reload",
            reload_exact,
            str(bool(gate["require_exact_checkpoint_reload"])).lower(),
            reload_exact or not bool(gate["require_exact_checkpoint_reload"]),
        ),
    ]


def _evaluate_mse(
    module: Any,
    observations: Any,
    previous_actions: Any,
    targets: Any,
    loss_mask: Any,
) -> float:
    import torch

    module.eval()
    with torch.no_grad():
        predictions, _ = module.forward_sequence(observations, previous_actions)
        return float(masked_action_mse(predictions, targets, loss_mask))


def _verify_export(
    module: Any,
    exported: StructuredRecurrentPolicy,
    restored: StructuredRecurrentPolicy,
    episodes: list[dict[str, np.ndarray[Any, Any]]],
) -> tuple[float, bool]:
    import torch

    maximum_error = 0.0
    reload_exact = True
    module.eval()
    for episode in episodes:
        observations = episode["observations"]
        previous_actions = episode["previous_actions"]
        with torch.no_grad():
            torch_actions, _ = module.forward_sequence(
                torch.from_numpy(observations[None, :]),
                torch.from_numpy(previous_actions[None, :]),
            )
        exported_actions, exported_states = exported.teacher_forced_sequence(
            observations, previous_actions
        )
        restored_actions, restored_states = restored.teacher_forced_sequence(
            observations, previous_actions
        )
        maximum_error = max(
            maximum_error,
            float(
                np.max(
                    np.abs(
                        torch_actions.numpy()[0]
                        - exported_actions
                    )
                )
            ),
        )
        reload_exact = reload_exact and np.array_equal(
            restored_actions, exported_actions
        ) and np.array_equal(restored_states, exported_states)
    return maximum_error, reload_exact


def _load_complete_episodes(
    dataset_directory: Path,
    manifest: dict[str, Any],
) -> list[dict[str, np.ndarray[Any, Any]]]:
    episodes = []
    required_fields = ("observations", "previous_actions", "teacher_actions")
    for shard_entry in manifest["shards"]:
        shard_path = dataset_directory / shard_entry["file"]
        if _sha256(shard_path) != shard_entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            if int(shard["dataset_schema_version"]) != 2:
                raise ValueError(f"unsupported dataset schema: {shard_path}")
            if str(shard["teacher_action_semantics"]) != manifest[
                "teacher_action_semantics"
            ]:
                raise ValueError(f"shard action semantics mismatch: {shard_path}")
            offsets = shard["episode_offsets"]
            fields = {
                name: np.asarray(shard[name], dtype=np.float32)
                for name in required_fields
            }
            for local_index in range(len(offsets) - 1):
                selection = slice(
                    int(offsets[local_index]), int(offsets[local_index + 1])
                )
                episode = {
                    name: np.ascontiguousarray(value[selection])
                    for name, value in fields.items()
                }
                _validate_episode(episode)
                episodes.append(episode)
    if len(episodes) != int(manifest["episode_count"]):
        raise ValueError("dataset manifest episode count does not match shards")
    return episodes


def _validate_episode(episode: dict[str, np.ndarray[Any, Any]]) -> None:
    length = len(episode["observations"])
    if episode["observations"].shape != (length, 19):
        raise ValueError("episode observations must have shape (time, 19)")
    for name in ("previous_actions", "teacher_actions"):
        if episode[name].shape != (length, 2):
            raise ValueError(f"episode {name} must have shape (time, 2)")
    if not all(np.all(np.isfinite(value)) for value in episode.values()):
        raise ValueError("episode contains non-finite public training values")
    if not np.array_equal(episode["previous_actions"][0], np.zeros(2)):
        raise ValueError("previous action must reset to zero at episode boundary")
    if np.any(np.abs(episode["teacher_actions"]) > 1.0):
        raise ValueError("teacher actions must lie in the public action range")


def _validate_config(config: dict[str, Any]) -> None:
    if int(config["policy"]["state_dimension"]) != 64:
        raise ValueError("tiny overfit requires state dimension 64")
    if int(config["policy"]["innovation_rank"]) != 2:
        raise ValueError("tiny overfit requires innovation rank 2")
    training = config["training"]
    training_episode_count = int(
        training.get("training_episode_count", config["dataset"]["episodes"])
    )
    if not 0 < training_episode_count <= int(config["dataset"]["episodes"]):
        raise ValueError("training episode count must lie inside the dataset")
    if int(training["batch_size"]) != training_episode_count:
        raise ValueError("tiny overfit uses all complete episodes in one batch")
    if training["target"] != "deterministic_teacher_action_mean":
        raise ValueError("tiny overfit target must be deterministic teacher means")


def _validate_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if manifest["status"] != "completed":
        raise ValueError("teacher dataset is not complete")
    if manifest["dataset_id"] != config["id"]:
        raise ValueError("teacher dataset id does not match tiny-overfit config")
    if manifest["teacher_action_semantics"] != config["teacher_action_semantics"]:
        raise ValueError("teacher dataset does not contain deterministic mean targets")
    if not bool(manifest.get("deterministic_inference")):
        raise ValueError("teacher dataset inference was not deterministic")
    if int(manifest["episode_count"]) != int(config["episodes"]):
        raise ValueError("teacher dataset episode count does not match config")


def _check(
    check_id: str,
    observed: Any,
    required: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def _write_metric(stream: Any, epoch: int, action_mse: float) -> None:
    record = {"epoch": epoch, "training_action_mse": action_mse}
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
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
