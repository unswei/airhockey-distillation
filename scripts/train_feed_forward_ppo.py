#!/usr/bin/env python3
"""Train a strictly memoryless PPO baseline directly on defence reward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from airhockey_distill.envs import DirectLaunchTrainingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    profile = config[args.profile]
    _validate_seed(profile, args.seed)
    _validate_runtime_version(config, stable_baselines3.__version__)

    output = args.output.resolve()
    binding_path = output / "binding.json"
    result_path = output / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        _validate_existing_binding(binding_path, _binding(args, config_path))
        return result
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = _binding(args, config_path)
    if binding_path.exists():
        _validate_existing_binding(binding_path, binding)
    else:
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    resolved_config_path = output / "config.yaml"
    if not resolved_config_path.exists():
        resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=True))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    task = config["task"]
    distribution = config["training_distribution"]
    env_count = int(profile["parallel_envs"])

    def make_environment(index: int):
        def factory():
            environment = DirectLaunchTrainingEnv(
                distribution_config=task["distribution_config"],
                reward_config=task["reward_config"],
                split=task["training_split"],
                sampling_seed=(
                    int(distribution["sampling_seed"])
                    + args.seed * env_count
                    + index
                ),
                blackout_start_observation_step=int(
                    task["blackout_start_observation_step"]
                ),
                blackout_lengths=tuple(
                    int(value) for value in distribution["blackout_steps"]
                ),
                blackout_probabilities=tuple(
                    float(value)
                    for value in distribution["blackout_probabilities"]
                ),
                action_lock_steps=int(task["action_lock_steps"]),
            )
            return Monitor(environment)

        return factory

    vector_environment = DummyVecEnv(
        [make_environment(index) for index in range(env_count)]
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    latest = _latest_checkpoint(checkpoints)
    started = perf_counter()
    try:
        if latest is None:
            activation = _activation(config["policy"]["activation"], torch)
            model = PPO(
                "MlpPolicy",
                vector_environment,
                learning_rate=float(profile["learning_rate"]),
                n_steps=int(profile["rollout_steps_per_env"]),
                batch_size=int(profile["batch_size"]),
                n_epochs=int(profile["epochs_per_update"]),
                gamma=float(profile["gamma"]),
                gae_lambda=float(profile["gae_lambda"]),
                clip_range=float(profile["clip_range"]),
                ent_coef=float(profile["entropy_coefficient"]),
                vf_coef=float(profile["value_coefficient"]),
                max_grad_norm=float(profile["maximum_gradient_norm"]),
                policy_kwargs={
                    "activation_fn": activation,
                    "net_arch": {
                        "pi": [
                            int(value)
                            for value in config["policy"]["policy_hidden_widths"]
                        ],
                        "vf": [
                            int(value)
                            for value in config["policy"]["value_hidden_widths"]
                        ],
                    },
                },
                seed=args.seed,
                device="cpu",
                verbose=1,
            )
            reset_num_timesteps = True
        else:
            model = PPO.load(latest, env=vector_environment, device="cpu")
            reset_num_timesteps = False

        sampling_seed = (
            int(distribution["sampling_seed"]) + args.seed * env_count
        )
        vector_environment.seed(sampling_seed)
        model.set_logger(
            configure(str(output / "logs"), ["stdout", "csv", "json"])
        )
        requested_steps = int(profile["total_timesteps"])
        remaining_steps = max(0, requested_steps - int(model.num_timesteps))
        if remaining_steps:
            save_frequency = max(
                1, int(profile["checkpoint_every_steps"]) // env_count
            )
            callback = CheckpointCallback(
                save_freq=save_frequency,
                save_path=str(checkpoints),
                name_prefix="ppo",
            )
            model.learn(
                total_timesteps=remaining_steps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=False,
            )
        checkpoint_path = output / "model.zip"
        model.save(checkpoint_path)

        actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters())
        actor_parameters.extend(model.policy.action_net.parameters())
        result = {
            "schema_version": 1,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "policy": "feed_forward_ppo",
            "profile": args.profile,
            "training_seed": args.seed,
            "code_commit": args.code_commit,
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "requested_timesteps": requested_steps,
            "completed_timesteps": int(model.num_timesteps),
            "vector_environment_sampling_seed": sampling_seed,
            "actor_parameter_count": sum(
                parameter.numel() for parameter in actor_parameters
            ),
            "training_parameter_count": sum(
                parameter.numel() for parameter in model.policy.parameters()
            ),
            "duration_seconds_this_invocation": perf_counter() - started,
            "runtime": {
                "hostname": platform.node(),
                "python": sys.version,
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "stable_baselines3": stable_baselines3.__version__,
            },
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        vector_environment.close()


def _binding(
    args: argparse.Namespace,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_sha256": _sha256(config_path),
        "profile": args.profile,
        "seed": args.seed,
        "code_commit": args.code_commit,
    }


def _validate_existing_binding(path: Path, expected: dict[str, Any]) -> None:
    if json.loads(path.read_text()) != expected:
        raise ValueError("existing PPO run binding does not match this invocation")


def _latest_checkpoint(directory: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("ppo_*_steps.zip"):
        match = re.fullmatch(r"ppo_(\d+)_steps\.zip", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def _activation(name: str, torch: Any) -> Any:
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"unsupported PPO activation {name!r}")
    return activations[name]


def _validate_seed(profile: dict[str, Any], seed: int) -> None:
    allowed = tuple(int(value) for value in profile["seeds"])
    if seed not in allowed:
        raise ValueError(f"seed {seed} is not predeclared for this profile")


def _validate_runtime_version(config: dict[str, Any], actual: str) -> None:
    expected = str(config["provenance"]["stable_baselines3_version"])
    if actual != expected:
        raise RuntimeError(
            f"Stable-Baselines3 version mismatch: expected {expected}, got {actual}"
        )


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
