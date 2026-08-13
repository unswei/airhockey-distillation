#!/usr/bin/env python3
"""Train one structured recurrent student on deterministic teacher means."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from copy import deepcopy
from dataclasses import dataclass
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

DETERMINISTIC_ACTION_SEMANTICS = (
    "deterministic_actor_mean_after_public_adapter_clip"
)


@dataclass(frozen=True)
class EpisodeSplit:
    """Padded complete episodes from one deterministic episode split."""

    observations: np.ndarray[Any, Any]
    previous_actions: np.ndarray[Any, Any]
    teacher_actions: np.ndarray[Any, Any]
    puck_visible: np.ndarray[Any, Any]
    valid_mask: np.ndarray[Any, Any]
    loss_mask: np.ndarray[Any, Any]
    episode_indices: np.ndarray[Any, Any]
    episode_lengths: np.ndarray[Any, Any]

    @property
    def episode_count(self) -> int:
        return int(len(self.episode_indices))

    @property
    def transition_count(self) -> int:
        return int(np.sum(self.valid_mask))


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
    validate_full_training_config(config)
    training = config["training"]
    export_config = config["export"]

    dataset_directory = args.dataset.resolve()
    manifest_path = dataset_directory / "manifest.json"
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
    innovation_rank = int(config["policy"]["innovation_rank"])
    module = StructuredRecurrentModule(
        seed=seed,
        innovation_rank=innovation_rank,
    )
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
                module,
                splits["validation"],
                batch_size,
                sequence_length,
            )
            record = {
                "epoch": epoch,
                "online_train_action_mse": online_training_mse,
                "validation": validation,
            }
            _write_metric(metrics_stream, record)
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
        "innovation_rank": innovation_rank,
        "code_commit": args.code_commit,
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(manifest_path),
        "teacher_action_semantics": manifest["teacher_action_semantics"],
        "selected_epoch": best_epoch,
        "validation_action_mse": best_validation,
    }
    exported_parameters = module.export_numpy_parameters()
    save_structured_checkpoint(
        checkpoint_path, exported_parameters, checkpoint_metadata
    )
    exported = StructuredRecurrentPolicy(exported_parameters, checkpoint_metadata)
    restored = StructuredRecurrentPolicy.load(checkpoint_path)
    export_error, reload_exact = verify_export(
        module,
        exported,
        restored,
        splits["validation"],
        episode_count=int(export_config["verification_episodes"]),
    )
    if export_error > float(export_config["maximum_absolute_error"]):
        raise RuntimeError("exported NumPy student does not match PyTorch")
    if bool(export_config["require_exact_checkpoint_reload"]) and not reload_exact:
        raise RuntimeError("structured checkpoint reload is not exact")

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "config_sha256": _sha256(config_path),
        "dataset": str(dataset_directory),
        "dataset_manifest_sha256": _sha256(manifest_path),
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


def train_epoch(
    module: Any,
    optimiser: Any,
    split: EpisodeSplit,
    *,
    batch_size: int,
    sequence_length: int,
    gradient_clip: float,
    generator: Any,
) -> float:
    """Run deterministic episode batches with state carried across chunks."""

    import torch

    module.train()
    permutation = torch.randperm(split.episode_count, generator=generator).numpy()
    squared_error_sum = 0.0
    action_value_count = 0
    for start in range(0, split.episode_count, batch_size):
        indices = permutation[start : start + batch_size]
        carry = None
        for chunk in chunk_slices(split.observations.shape[1], sequence_length):
            observations = torch.from_numpy(split.observations[indices, chunk])
            previous_actions = torch.from_numpy(
                split.previous_actions[indices, chunk]
            )
            targets = torch.from_numpy(split.teacher_actions[indices, chunk])
            loss_mask = torch.from_numpy(split.loss_mask[indices, chunk])
            predictions, states = module.forward_sequence(
                observations, previous_actions, carry
            )
            carry = states[:, -1].detach()
            expanded = loss_mask.unsqueeze(-1).expand_as(predictions)
            if not bool(torch.any(expanded)):
                continue
            squared_errors = (predictions - targets) ** 2
            loss = squared_errors[expanded].mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(), gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite gradient norm during training")
            optimiser.step()
            selected = squared_errors.detach()[expanded]
            squared_error_sum += float(selected.sum())
            action_value_count += int(selected.numel())
    if not action_value_count:
        raise RuntimeError("training split has no loss-bearing actions")
    return squared_error_sum / action_value_count


def evaluate_split(
    module: Any,
    split: EpisodeSplit,
    batch_size: int,
    sequence_length: int,
) -> dict[str, float]:
    import torch

    totals = {
        "all": [0.0, 0],
        "visible": [0.0, 0],
        "hidden": [0.0, 0],
    }
    module.eval()
    with torch.no_grad():
        for start in range(0, split.episode_count, batch_size):
            indices = np.arange(start, min(start + batch_size, split.episode_count))
            carry = None
            for chunk in chunk_slices(
                split.observations.shape[1], sequence_length
            ):
                observations = torch.from_numpy(split.observations[indices, chunk])
                previous_actions = torch.from_numpy(
                    split.previous_actions[indices, chunk]
                )
                targets = torch.from_numpy(split.teacher_actions[indices, chunk])
                predictions, states = module.forward_sequence(
                    observations, previous_actions, carry
                )
                carry = states[:, -1]
                errors = (predictions - targets) ** 2
                loss_mask = torch.from_numpy(split.loss_mask[indices, chunk])
                visibility = torch.from_numpy(split.puck_visible[indices, chunk])
                for name, mask in (
                    ("all", loss_mask),
                    ("visible", loss_mask & visibility),
                    ("hidden", loss_mask & ~visibility),
                ):
                    expanded = mask.unsqueeze(-1).expand_as(errors)
                    if bool(torch.any(expanded)):
                        selected = errors[expanded]
                        totals[name][0] += float(selected.sum())
                        totals[name][1] += int(selected.numel())
    if not totals["all"][1]:
        raise RuntimeError("evaluation split has no loss-bearing actions")
    return {
        "action_mse": totals["all"][0] / totals["all"][1],
        "visible_action_mse": _safe_mean(*totals["visible"]),
        "hidden_action_mse": _safe_mean(*totals["hidden"]),
        "loss_bearing_action_values": totals["all"][1],
    }


def chunk_slices(total_steps: int, sequence_length: int) -> tuple[slice, ...]:
    if total_steps <= 0 or sequence_length <= 0:
        raise ValueError("step counts must be positive")
    return tuple(
        slice(start, min(start + sequence_length, total_steps))
        for start in range(0, total_steps, sequence_length)
    )


def load_episode_splits(
    dataset_directory: Path,
    manifest: dict[str, Any],
    dataset_config: dict[str, Any],
    *,
    maximum_episode_steps: int,
    burn_in_steps: int,
) -> dict[str, EpisodeSplit]:
    residue_to_split = split_residue_mapping(dataset_config)
    fields = (
        "observations",
        "previous_actions",
        "teacher_actions",
        "puck_visible",
    )
    collected: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "internal_test": [],
    }
    for shard_number, shard_entry in enumerate(manifest["shards"], start=1):
        shard_path = dataset_directory / shard_entry["file"]
        if _sha256(shard_path) != shard_entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            if int(shard["dataset_schema_version"]) != 2:
                raise ValueError(f"unsupported dataset schema: {shard_path}")
            if str(shard["teacher_action_semantics"]) != (
                DETERMINISTIC_ACTION_SEMANTICS
            ):
                raise ValueError("dataset shard does not contain teacher means")
            offsets = shard["episode_offsets"]
            indices = shard["episode_indices"]
            loaded = {name: np.asarray(shard[name]) for name in fields}
            for local_index, episode_index_value in enumerate(indices):
                episode_index = int(episode_index_value)
                selection = slice(
                    int(offsets[local_index]), int(offsets[local_index + 1])
                )
                episode = {
                    name: np.ascontiguousarray(value[selection])
                    for name, value in loaded.items()
                }
                validate_training_episode(
                    episode, maximum_episode_steps=maximum_episode_steps
                )
                episode["episode_index"] = episode_index
                collected[residue_to_split[episode_index % len(residue_to_split)]].append(
                    episode
                )
        if shard_number % 10 == 0 or shard_number == len(manifest["shards"]):
            print(
                json.dumps(
                    {
                        "dataset_shards_loaded": shard_number,
                        "dataset_shards_total": len(manifest["shards"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    splits = {
        name: pad_episode_split(
            episodes,
            maximum_episode_steps=maximum_episode_steps,
            burn_in_steps=burn_in_steps,
        )
        for name, episodes in collected.items()
    }
    expected_episodes = int(manifest["episode_count"])
    if sum(split.episode_count for split in splits.values()) != expected_episodes:
        raise ValueError("episode splits do not cover the dataset")
    return splits


def split_residue_mapping(config: dict[str, Any]) -> dict[int, str]:
    modulus = int(config["episode_split_modulus"])
    groups = {
        "train": config["train_residues"],
        "validation": config["validation_residues"],
        "internal_test": config["internal_test_residues"],
    }
    mapping: dict[int, str] = {}
    for name, values in groups.items():
        for raw_value in values:
            value = int(raw_value)
            if value < 0 or value >= modulus or value in mapping:
                raise ValueError("dataset split residues must be disjoint and valid")
            mapping[value] = name
    if set(mapping) != set(range(modulus)):
        raise ValueError("dataset split residues must cover the modulus")
    return mapping


def pad_episode_split(
    episodes: list[dict[str, Any]],
    *,
    maximum_episode_steps: int,
    burn_in_steps: int,
) -> EpisodeSplit:
    if not episodes:
        raise ValueError("dataset split must contain episodes")
    if not 0 <= burn_in_steps < maximum_episode_steps:
        raise ValueError("burn-in must lie inside the maximum episode length")
    count = len(episodes)
    observations = np.zeros((count, maximum_episode_steps, 19), dtype=np.float32)
    previous_actions = np.zeros((count, maximum_episode_steps, 2), dtype=np.float32)
    teacher_actions = np.zeros((count, maximum_episode_steps, 2), dtype=np.float32)
    puck_visible = np.zeros((count, maximum_episode_steps), dtype=bool)
    valid_mask = np.zeros((count, maximum_episode_steps), dtype=bool)
    loss_mask = np.zeros((count, maximum_episode_steps), dtype=bool)
    episode_indices = np.empty(count, dtype=np.int64)
    episode_lengths = np.empty(count, dtype=np.int16)
    for row, episode in enumerate(episodes):
        length = len(episode["observations"])
        if length <= burn_in_steps:
            raise ValueError("episode contains no loss-bearing suffix")
        observations[row, :length] = episode["observations"]
        previous_actions[row, :length] = episode["previous_actions"]
        teacher_actions[row, :length] = episode["teacher_actions"]
        puck_visible[row, :length] = episode["puck_visible"]
        valid_mask[row, :length] = True
        loss_mask[row, burn_in_steps:length] = True
        episode_indices[row] = int(episode["episode_index"])
        episode_lengths[row] = length
    return EpisodeSplit(
        observations=observations,
        previous_actions=previous_actions,
        teacher_actions=teacher_actions,
        puck_visible=puck_visible,
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        episode_indices=episode_indices,
        episode_lengths=episode_lengths,
    )


def validate_training_episode(
    episode: dict[str, Any], *, maximum_episode_steps: int
) -> None:
    length = len(episode["observations"])
    if not 0 < length <= maximum_episode_steps:
        raise ValueError("episode length lies outside the configured bound")
    if episode["observations"].shape != (length, 19):
        raise ValueError("episode observations must have shape (time, 19)")
    for name in ("previous_actions", "teacher_actions"):
        if episode[name].shape != (length, 2):
            raise ValueError(f"episode {name} must have shape (time, 2)")
    if episode["puck_visible"].shape != (length,):
        raise ValueError("episode visibility must have shape (time,)")
    public_values = (
        episode["observations"],
        episode["previous_actions"],
        episode["teacher_actions"],
    )
    if not all(np.all(np.isfinite(value)) for value in public_values):
        raise ValueError("episode contains non-finite public training values")
    if not np.array_equal(episode["previous_actions"][0], np.zeros(2)):
        raise ValueError("previous action must reset at the episode boundary")
    if np.any(np.abs(episode["teacher_actions"]) > 1.0):
        raise ValueError("teacher actions must lie in the public action range")


def validate_full_training_config(config: dict[str, Any]) -> None:
    if int(config["policy"]["state_dimension"]) != 64:
        raise ValueError("full training requires state dimension 64")
    if int(config["policy"]["innovation_rank"]) not in (0, 1, 2, 4):
        raise ValueError("full training requires innovation rank 0, 1, 2 or 4")
    training = config["training"]
    if training["target"] != "deterministic_teacher_action_mean":
        raise ValueError("full training target must be deterministic teacher means")
    if training["reset_state"] != "episode_boundary_only":
        raise ValueError("recurrent state may reset only at episode boundaries")
    sequence_length = int(training["sequence_length"])
    burn_in_steps = int(training["burn_in_steps"])
    if sequence_length != burn_in_steps + int(training["loss_steps"]):
        raise ValueError("sequence length must equal burn-in plus loss suffix")
    if int(training["maximum_episode_steps"]) % sequence_length:
        raise ValueError("maximum episode steps must be divisible by sequence length")
    split_residue_mapping(config["dataset"])


def validate_deterministic_manifest(
    manifest: dict[str, Any], dataset_config: dict[str, Any]
) -> None:
    if manifest["status"] != "completed":
        raise ValueError("teacher dataset is not complete")
    if manifest["dataset_id"] != dataset_config["id"]:
        raise ValueError("teacher dataset id does not match the training config")
    expected = dataset_config["teacher_action_semantics"]
    if expected != DETERMINISTIC_ACTION_SEMANTICS:
        raise ValueError("training config does not request deterministic means")
    if manifest["teacher_action_semantics"] != expected:
        raise ValueError("teacher dataset does not contain deterministic means")
    if not bool(manifest.get("deterministic_inference")):
        raise ValueError("teacher dataset inference was not deterministic")
    if int(manifest["episode_count"]) != int(dataset_config["episodes"]):
        raise ValueError("teacher dataset episode count does not match config")


def verify_export(
    module: Any,
    exported: StructuredRecurrentPolicy,
    restored: StructuredRecurrentPolicy,
    split: EpisodeSplit,
    *,
    episode_count: int,
) -> tuple[float, bool]:
    import torch

    maximum_error = 0.0
    reload_exact = True
    module.eval()
    for index in range(min(episode_count, split.episode_count)):
        length = int(split.episode_lengths[index])
        observations = split.observations[index, :length]
        previous_actions = split.previous_actions[index, :length]
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
            float(np.max(np.abs(torch_actions.numpy()[0] - exported_actions))),
        )
        reload_exact = reload_exact and np.array_equal(
            restored_actions, exported_actions
        ) and np.array_equal(restored_states, exported_states)
    return maximum_error, reload_exact


def _safe_mean(total: float, count: int) -> float:
    return total / count if count else float("nan")


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
