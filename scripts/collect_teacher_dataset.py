#!/usr/bin/env python3
"""Collect resumable, sharded Dreamer demonstrations for Stage B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import gymnasium
import mujoco
import numpy as np
import yaml
from ruamel import yaml as ruamel_yaml

from airhockey_distill.envs import DirectLaunchTrainingEnv

ENVIRONMENT_ID = "AirHockeyDefendShotDataset-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_mapping(args.config)
    teacher_config = _load_mapping(args.teacher_config)
    dataset = config["dataset"]
    profile = teacher_config["full"]
    checkpoint = args.checkpoint.resolve()
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    frozen_manifest = json.loads(manifest_path.read_text())
    if int(frozen_manifest["checkpoint_step"]) != int(
        config["provenance"]["teacher_selection_step"]
    ):
        raise ValueError("frozen teacher step does not match student config")

    output = args.output.resolve()
    shards_directory = output / "shards"
    output.mkdir(parents=True, exist_ok=True)
    shards_directory.mkdir(exist_ok=True)
    resolved_config_path = output / "config.yaml"
    if not resolved_config_path.exists():
        resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=True))

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=dataset["distribution_config"],
            reward_config=dataset["reward_config"],
            split=dataset["distribution_split"],
            sampling_seed=int(dataset["sampling_seed"]),
            blackout_start_observation_step=int(
                dataset["blackout_start_observation_step"]
            ),
            minimum_blackout_steps=int(dataset["minimum_blackout_steps"]),
            maximum_blackout_steps=int(dataset["maximum_blackout_steps"]),
            **kwargs,
        )

    gymnasium.register(id=ENVIRONMENT_ID, entry_point=make_environment)

    import elements
    import jax
    import portal
    from dreamerv3 import agent as dreamer_agent
    from dreamerv3 import main as dreamer_main
    from embodied.envs import from_gymnasium

    from_gymnasium.elements = elements
    upstream = ruamel_yaml.YAML(typ="safe").load(
        (Path(dreamer_agent.__file__).parent / "configs.yaml").read_text()
    )
    dreamer_config = elements.Config(upstream["defaults"])
    dreamer_config = dreamer_config.update(upstream[str(profile["model_preset"])])
    dreamer_config = dreamer_config.update(
        {
            "task": f"gymnasium_{ENVIRONMENT_ID}",
            "logdir": str(output / "runtime"),
            "seed": int(profile["seed"]),
            "batch_size": int(profile["batch_size"]),
            "batch_length": int(profile["batch_length"]),
            "report_length": int(profile["report_length"]),
            "jax.platform": "cuda",
            "jax.prealloc": False,
            "run.envs": 1,
            "run.debug": True,
        }
    )
    portal.setup(
        errfile=False,
        clientkw={"logging_color": "cyan"},
        serverkw={"logging_color": "cyan"},
        ipv6=False,
    )
    agent = dreamer_main.make_agent(dreamer_config)
    loader = elements.Checkpoint()
    loader.agent = agent
    loader.load(checkpoint, keys=["agent"])

    episode_count = int(dataset["episodes"])
    episodes_per_shard = int(dataset["episodes_per_shard"])
    if episode_count <= 0 or episodes_per_shard <= 0:
        raise ValueError("dataset episode counts must be positive")
    if episode_count % episodes_per_shard:
        raise ValueError("episodes must be divisible by episodes_per_shard")

    environment = make_environment()
    started = perf_counter()
    try:
        for first_episode in range(0, episode_count, episodes_per_shard):
            shard_index = first_episode // episodes_per_shard
            shard_path = shards_directory / f"shard-{shard_index:04d}.npz"
            if shard_path.exists():
                _validate_shard(shard_path, first_episode, episodes_per_shard)
                continue
            payload = _collect_shard(
                agent,
                environment,
                first_episode=first_episode,
                episode_count=episodes_per_shard,
                sampling_seed=int(dataset["sampling_seed"]),
            )
            temporary = shard_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, shard_path)
            print(
                json.dumps(
                    {
                        "shard": shard_index,
                        "episodes_collected": first_episode + episodes_per_shard,
                        "transitions": int(payload["observations"].shape[0]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        environment.close()

    shards = []
    outcome_counts: Counter[str] = Counter()
    transition_count = 0
    for shard_path in sorted(shards_directory.glob("shard-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            transitions = int(shard["observations"].shape[0])
            transition_count += transitions
            outcome_counts.update(str(value) for value in shard["episode_outcomes"])
            first_episode = int(shard["episode_indices"][0])
            shard_episodes = int(shard["episode_indices"].shape[0])
        shards.append(
            {
                "file": str(shard_path.relative_to(output)),
                "first_episode": first_episode,
                "episodes": shard_episodes,
                "transitions": transitions,
                "bytes": shard_path.stat().st_size,
                "sha256": _sha256(shard_path),
            }
        )
    if sum(item["episodes"] for item in shards) != episode_count:
        raise RuntimeError("dataset is incomplete after collection")

    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset["id"],
        "dataset_schema_version": 2,
        "teacher_action_semantics": "executed_after_public_adapter_clip",
        "code_commit": args.code_commit,
        "config_sha256": _sha256(args.config.resolve()),
        "frozen_teacher": {
            "path": str(checkpoint),
            "manifest_sha256": _sha256(manifest_path),
            "checkpoint_step": frozen_manifest["checkpoint_step"],
            "agent_sha256": frozen_manifest["files"]["agent.pkl"]["sha256"],
        },
        "episode_count": episode_count,
        "transition_count": transition_count,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "shards": shards,
        "duration_seconds_this_invocation": perf_counter() - started,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "mujoco": mujoco.__version__,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _collect_shard(
    agent: Any,
    environment: DirectLaunchTrainingEnv,
    *,
    first_episode: int,
    episode_count: int,
    sampling_seed: int,
) -> dict[str, np.ndarray[Any, Any]]:
    observations: list[np.ndarray[Any, Any]] = []
    previous_actions: list[np.ndarray[Any, Any]] = []
    teacher_raw_actions: list[np.ndarray[Any, Any]] = []
    teacher_actions: list[np.ndarray[Any, Any]] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    visible: list[bool] = []
    privileged_positions: list[tuple[float, float]] = []
    privileged_velocities: list[tuple[float, float]] = []
    episode_offsets = [0]
    episode_indices: list[int] = []
    shot_ids: list[str] = []
    blackout_steps: list[int] = []
    outcomes: list[str] = []

    for episode_index in range(first_episode, first_episode + episode_count):
        observation, info = environment.reset(seed=sampling_seed + episode_index)
        carry = agent.init_policy(1)
        reward = 0.0
        previous_action = np.zeros(2, dtype=np.float32)
        is_first = True
        outcome = "rollout_limit"
        while True:
            state = environment.privileged_state()
            policy_observation = {
                "image": np.asarray([observation], dtype=np.float32),
                "reward": np.asarray([reward], dtype=np.float32),
                "is_first": np.asarray([is_first], dtype=bool),
                "is_last": np.asarray([False], dtype=bool),
                "is_terminal": np.asarray([False], dtype=bool),
            }
            carry, actions, _ = agent.policy(carry, policy_observation, mode="eval")
            teacher_raw_action = np.asarray(actions["action"], dtype=np.float32)[0]
            teacher_action = np.clip(teacher_raw_action, -1.0, 1.0).astype(np.float32)

            observations.append(np.asarray(observation, dtype=np.float32))
            previous_actions.append(previous_action)
            teacher_raw_actions.append(teacher_raw_action)
            teacher_actions.append(teacher_action)
            visible.append(bool(info["puck_visible"]))
            privileged_positions.append(state.puck_position_robot_xy)
            privileged_velocities.append(state.puck_velocity_robot_xy)

            observation, reward, terminated, truncated, info = environment.step(
                teacher_action
            )
            rewards.append(float(reward))
            terminals.append(bool(terminated or truncated))
            previous_action = teacher_action
            is_first = False
            if terminated or truncated:
                outcome = str(info["outcome"])
                break

        episode_indices.append(episode_index)
        shot_ids.append(str(info["shot_id"]))
        blackout_steps.append(int(environment.blackout.length_steps))
        outcomes.append(outcome)
        episode_offsets.append(len(observations))

    return {
        "dataset_schema_version": np.asarray(2, dtype=np.int64),
        "teacher_action_semantics": np.asarray("executed_after_public_adapter_clip"),
        "observations": np.asarray(observations, dtype=np.float32),
        "previous_actions": np.asarray(previous_actions, dtype=np.float32),
        "teacher_raw_actions": np.asarray(teacher_raw_actions, dtype=np.float32),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=bool),
        "puck_visible": np.asarray(visible, dtype=bool),
        "privileged_puck_position_robot_xy": np.asarray(
            privileged_positions, dtype=np.float32
        ),
        "privileged_puck_velocity_robot_xy": np.asarray(
            privileged_velocities, dtype=np.float32
        ),
        "episode_offsets": np.asarray(episode_offsets, dtype=np.int64),
        "episode_indices": np.asarray(episode_indices, dtype=np.int64),
        "episode_shot_ids": np.asarray(shot_ids),
        "episode_blackout_steps": np.asarray(blackout_steps, dtype=np.int16),
        "episode_outcomes": np.asarray(outcomes),
    }


def _validate_shard(path: Path, first_episode: int, episode_count: int) -> None:
    with np.load(path, allow_pickle=False) as shard:
        if int(shard["dataset_schema_version"]) != 2:
            raise ValueError(f"existing shard has an incompatible schema: {path}")
        if str(shard["teacher_action_semantics"]) != (
            "executed_after_public_adapter_clip"
        ):
            raise ValueError(
                f"existing shard has incompatible action semantics: {path}"
            )
        indices = shard["episode_indices"]
        expected = np.arange(first_episode, first_episode + episode_count)
        if not np.array_equal(indices, expected):
            raise ValueError(f"existing shard has unexpected episodes: {path}")
        if shard["episode_offsets"].shape != (episode_count + 1,):
            raise ValueError(f"existing shard has invalid offsets: {path}")
        if int(shard["episode_offsets"][-1]) != shard["observations"].shape[0]:
            raise ValueError(f"existing shard has invalid transition count: {path}")
        actions = shard["teacher_actions"]
        if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.0):
            raise ValueError(f"existing shard has invalid executed actions: {path}")


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
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "episode_count",
                    "transition_count",
                    "outcome_counts",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
