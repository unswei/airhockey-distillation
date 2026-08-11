#!/usr/bin/env python3
"""Train the observation-only Stage B student from teacher demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from airhockey_distill.students import (
    FeedForwardPolicy,
    save_feed_forward_checkpoint,
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
    from torch import nn

    config = _load_mapping(args.config)
    training = config["training"]
    dataset_config = config["dataset"]
    dataset_directory = args.dataset.resolve()
    dataset_manifest_path = dataset_directory / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    if dataset_manifest["status"] != "completed":
        raise ValueError("teacher dataset is not complete")
    if dataset_manifest["dataset_id"] != dataset_config["id"]:
        raise ValueError("teacher dataset id does not match student config")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    splits = _load_splits(dataset_directory, dataset_manifest, dataset_config)
    seed = int(training["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_0 = nn.Linear(19, 64)
            self.encoder_1 = nn.Linear(64, 32)
            self.action_hidden = nn.Linear(32, 64)
            self.action_output = nn.Linear(64, 2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = torch.nn.functional.silu(self.encoder_0(value))
            value = torch.nn.functional.silu(self.encoder_1(value))
            value = torch.nn.functional.silu(self.action_hidden(value))
            return torch.tanh(self.action_output(value))

    network = Network()
    optimiser = torch.optim.AdamW(
        network.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    observations = torch.from_numpy(splits["train"]["observations"])
    targets = torch.from_numpy(splits["train"]["teacher_actions"])
    batch_size = int(training["batch_size"])
    maximum_epochs = int(training["maximum_epochs"])
    patience_epochs = int(training["patience_epochs"])
    gradient_clip = float(training["gradient_clip_global_norm"])
    generator = torch.Generator().manual_seed(seed)

    metrics_path = output / "metrics.jsonl"
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    started = perf_counter()
    with metrics_path.open("w") as metrics_stream:
        for epoch in range(1, maximum_epochs + 1):
            network.train()
            permutation = torch.randperm(len(observations), generator=generator)
            total_squared_error = 0.0
            total_elements = 0
            for start in range(0, len(observations), batch_size):
                indices = permutation[start : start + batch_size]
                prediction = network(observations[indices])
                loss = torch.mean((prediction - targets[indices]) ** 2)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
                optimiser.step()
                total_squared_error += float(loss) * prediction.numel()
                total_elements += prediction.numel()

            validation = _torch_metrics(network, splits["validation"], batch_size)
            record = {
                "epoch": epoch,
                "train_action_mse": total_squared_error / total_elements,
                "validation": validation,
            }
            metrics_stream.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_stream.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation["action_mse"] < best_validation - 1e-9:
                best_validation = validation["action_mse"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in network.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= patience_epochs:
                    break

    if best_state is None:
        raise RuntimeError("training did not produce a selected checkpoint")
    network.load_state_dict(best_state)
    network.eval()
    state = network.state_dict()
    parameters = {
        "encoder_0_weight": state["encoder_0.weight"].numpy(),
        "encoder_0_bias": state["encoder_0.bias"].numpy(),
        "encoder_1_weight": state["encoder_1.weight"].numpy(),
        "encoder_1_bias": state["encoder_1.bias"].numpy(),
        "action_hidden_weight": state["action_hidden.weight"].numpy(),
        "action_hidden_bias": state["action_hidden.bias"].numpy(),
        "action_output_weight": state["action_output.weight"].numpy(),
        "action_output_bias": state["action_output.bias"].numpy(),
    }
    checkpoint_path = output / "checkpoint.npz"
    checkpoint_metadata = {
        "schema_version": 1,
        "policy": "feed_forward_stage_b",
        "observation_dimension": 19,
        "action_dimension": 2,
        "selected_epoch": best_epoch,
        "validation_action_mse": best_validation,
        "training_seed": seed,
        "code_commit": args.code_commit,
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
    }
    save_feed_forward_checkpoint(checkpoint_path, parameters, checkpoint_metadata)

    exported = FeedForwardPolicy.load(checkpoint_path)
    sample = splits["validation"]["observations"][:128]
    with torch.no_grad():
        torch_actions = network(torch.from_numpy(sample)).numpy()
    numpy_actions = exported.action(sample)
    export_max_absolute_error = float(np.max(np.abs(numpy_actions - torch_actions)))
    if not np.allclose(numpy_actions, torch_actions, atol=5e-6, rtol=1e-6):
        raise RuntimeError("exported NumPy policy does not match the trained network")

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "dataset": str(dataset_directory),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "export_verification_max_absolute_error": export_max_absolute_error,
        "parameter_count": exported.parameter_count,
        "selected_epoch": best_epoch,
        "epochs_completed": epoch,
        "selection_metric": training["selection_metric"],
        "split_transition_counts": {
            name: len(values["observations"]) for name, values in splits.items()
        },
        "metrics": {
            name: _torch_metrics(network, values, batch_size)
            for name, values in splits.items()
        },
        "duration_seconds": perf_counter() - started,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _load_splits(
    dataset_directory: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, np.ndarray[Any, Any]]]:
    modulus = int(config["episode_split_modulus"])
    residues = {
        "train": {int(value) for value in config["train_residues"]},
        "validation": {int(value) for value in config["validation_residues"]},
        "test": {int(value) for value in config["test_residues"]},
    }
    if set.union(*residues.values()) != set(range(modulus)):
        raise ValueError("episode split residues must cover the modulus")
    if sum(len(values) for values in residues.values()) != modulus:
        raise ValueError("episode split residues must not overlap")

    collected: dict[str, dict[str, list[np.ndarray[Any, Any]]]] = {
        name: {"observations": [], "teacher_actions": [], "puck_visible": []}
        for name in residues
    }
    for shard_number, shard_entry in enumerate(manifest["shards"], start=1):
        shard_path = dataset_directory / shard_entry["file"]
        if _sha256(shard_path) != shard_entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            if int(shard["dataset_schema_version"]) != 2:
                raise ValueError(f"unsupported dataset shard schema: {shard_path}")
            actions = shard["teacher_actions"]
            if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.0):
                raise ValueError(f"invalid executed teacher actions: {shard_path}")
            offsets = shard["episode_offsets"]
            indices = shard["episode_indices"]
            loaded_fields = {
                field: np.asarray(shard[field])
                for field in next(iter(collected.values()))
            }
            for local_index, episode_index in enumerate(indices):
                residue = int(episode_index) % modulus
                split = next(
                    name for name, values in residues.items() if residue in values
                )
                selection = slice(
                    int(offsets[local_index]), int(offsets[local_index + 1])
                )
                for field in collected[split]:
                    collected[split][field].append(loaded_fields[field][selection])
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

    result: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for split, fields in collected.items():
        result[split] = {
            field: np.ascontiguousarray(np.concatenate(values, axis=0))
            for field, values in fields.items()
        }
        if not len(result[split]["observations"]):
            raise ValueError(f"dataset split is empty: {split}")
    return result


def _torch_metrics(
    network: Any, split: dict[str, np.ndarray[Any, Any]], batch_size: int
) -> dict[str, float]:
    import torch

    observations = split["observations"]
    targets = split["teacher_actions"]
    squared_errors = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            prediction = network(
                torch.from_numpy(observations[start : start + batch_size])
            )
            target = torch.from_numpy(targets[start : start + batch_size])
            squared_errors.append(((prediction - target) ** 2).numpy())
    errors = np.concatenate(squared_errors, axis=0)
    visible = split["puck_visible"].astype(bool)
    metrics = {"action_mse": float(np.mean(errors))}
    for name, mask in (("visible", visible), ("hidden", ~visible)):
        metrics[f"{name}_action_mse"] = (
            float(np.mean(errors[mask])) if np.any(mask) else float("nan")
        )
    return metrics


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
