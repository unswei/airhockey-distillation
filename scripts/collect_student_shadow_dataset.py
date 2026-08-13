#!/usr/bin/env python3
"""Collect deterministic teacher labels on student-controlled trajectories."""

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
import numpy as np
import yaml

from airhockey_distill.envs import DirectLaunchTrainingEnv
from airhockey_distill.students import StructuredRecurrentPolicy
from airhockey_distill.teachers import enable_deterministic_dreamer_inference

ENVIRONMENT_ID = "AirHockeyStructuredStudentShadowDataset-v0"
DETERMINISTIC_ACTION_SEMANTICS = "deterministic_actor_mean_after_public_adapter_clip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    config = _load_mapping(args.config.resolve())
    teacher_config = _load_mapping(args.teacher_config.resolve())
    collection = config["shadow_collection"]
    _validate_config(
        config,
        args.student_checkpoint.resolve(),
        args.teacher_checkpoint.resolve(),
    )
    output = args.output.resolve()
    shards_directory = output / "shards"
    output.mkdir(parents=True, exist_ok=True)
    shards_directory.mkdir(exist_ok=True)
    resolved_config_path = output / "config.yaml"
    if not resolved_config_path.exists():
        resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=True))

    teacher, jax_version, devices = _load_teacher(
        config,
        teacher_config,
        args.teacher_checkpoint.resolve(),
        output,
    )
    student = StructuredRecurrentPolicy.load(args.student_checkpoint.resolve())
    environment = _make_environment(collection)
    episode_count = int(collection["episodes"])
    episodes_per_shard = int(collection["episodes_per_shard"])
    first_index = int(collection["first_episode_index"])
    if episode_count <= 0 or episodes_per_shard <= 0:
        raise ValueError("episode counts must be positive")
    if episode_count % episodes_per_shard:
        raise ValueError("episodes must be divisible by episodes per shard")

    started = perf_counter()
    try:
        for first_offset in range(0, episode_count, episodes_per_shard):
            shard_index = first_offset // episodes_per_shard
            shard_path = shards_directory / f"shard-{shard_index:04d}.npz"
            expected_first_index = first_index + first_offset
            if shard_path.exists():
                _validate_shard(shard_path, expected_first_index, episodes_per_shard)
                continue
            payload = _collect_shard(
                teacher,
                student,
                environment,
                first_episode_index=expected_first_index,
                first_collection_offset=first_offset,
                episode_count=episodes_per_shard,
                sampling_seed=int(collection["sampling_seed"]),
            )
            temporary = shard_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, shard_path)
            print(
                json.dumps(
                    {
                        "shard": shard_index,
                        "episodes_collected": first_offset + episodes_per_shard,
                        "transitions": int(payload["observations"].shape[0]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        environment.close()

    shards = []
    outcomes: Counter[str] = Counter()
    transition_count = 0
    for shard_path in sorted(shards_directory.glob("shard-*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            transitions = int(shard["observations"].shape[0])
            transition_count += transitions
            outcomes.update(str(value) for value in shard["episode_outcomes"])
            shard_first = int(shard["episode_indices"][0])
            shard_episodes = int(shard["episode_indices"].shape[0])
        shards.append(
            {
                "file": str(shard_path.relative_to(output)),
                "first_episode": shard_first,
                "episodes": shard_episodes,
                "transitions": transitions,
                "bytes": shard_path.stat().st_size,
                "sha256": _sha256(shard_path),
            }
        )
    if sum(item["episodes"] for item in shards) != episode_count:
        raise RuntimeError("shadow dataset is incomplete")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": collection["id"],
        "dataset_schema_version": 2,
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "rollout_controller": collection["rollout_controller"],
        "teacher_previous_action": collection["teacher_previous_action"],
        "code_commit": args.code_commit,
        "config_sha256": _sha256(args.config.resolve()),
        "teacher_checkpoint_sha256": _sha256(
            args.teacher_checkpoint.resolve() / "agent.pkl"
        ),
        "student_checkpoint_sha256": _sha256(args.student_checkpoint.resolve()),
        "episode_count": episode_count,
        "first_episode_index": first_index,
        "transition_count": transition_count,
        "outcome_counts": dict(sorted(outcomes.items())),
        "shards": shards,
        "duration_seconds_this_invocation": perf_counter() - started,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax_version,
            "jax_devices": devices,
            "mujoco": mujoco.__version__,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _collect_shard(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    environment: DirectLaunchTrainingEnv,
    *,
    first_episode_index: int,
    first_collection_offset: int,
    episode_count: int,
    sampling_seed: int,
) -> dict[str, np.ndarray[Any, Any]]:
    observations = []
    previous_actions = []
    teacher_actions = []
    student_actions = []
    rewards = []
    terminals = []
    visible = []
    offsets = [0]
    episode_indices = []
    shot_ids = []
    blackout_steps = []
    outcomes = []
    for local_index in range(episode_count):
        episode_index = first_episode_index + local_index
        collection_offset = first_collection_offset + local_index
        observation, info = environment.reset(seed=sampling_seed + collection_offset)
        teacher_carry = teacher.init_policy(1)
        student_carry = student.initial_carry()
        previous_command = np.zeros(2, dtype=np.float32)
        reward = 0.0
        step = 0
        outcome = "rollout_limit"
        while True:
            teacher_carry = _replace_teacher_previous_action(
                teacher_carry, previous_command
            )
            policy_observation = {
                "image": np.asarray([observation], dtype=np.float32),
                "reward": np.asarray([reward], dtype=np.float32),
                "is_first": np.asarray([step == 0], dtype=bool),
                "is_last": np.asarray([False], dtype=bool),
                "is_terminal": np.asarray([False], dtype=bool),
            }
            teacher_carry, teacher_output, _ = teacher.policy(
                teacher_carry, policy_observation, mode="eval"
            )
            teacher_action = np.clip(
                np.asarray(teacher_output["action"], dtype=np.float32)[0],
                -1.0,
                1.0,
            )
            student_action, student_carry = student.act(observation, student_carry)
            observations.append(np.asarray(observation, dtype=np.float32))
            previous_actions.append(previous_command.copy())
            teacher_actions.append(teacher_action)
            student_actions.append(student_action)
            visible.append(bool(info["puck_visible"]))
            observation, reward, terminated, truncated, info = environment.step(
                student_action
            )
            rewards.append(float(reward))
            terminals.append(bool(terminated or truncated))
            previous_command = np.asarray(student_action, dtype=np.float32)
            step += 1
            if terminated or truncated:
                outcome = str(info["outcome"])
                break
        episode_indices.append(episode_index)
        shot_ids.append(str(info["shot_id"]))
        blackout_steps.append(int(environment.blackout.length_steps))
        outcomes.append(outcome)
        offsets.append(len(observations))
    return {
        "dataset_schema_version": np.asarray(2, dtype=np.int64),
        "teacher_action_semantics": np.asarray(DETERMINISTIC_ACTION_SEMANTICS),
        "observations": np.asarray(observations, dtype=np.float32),
        "previous_actions": np.asarray(previous_actions, dtype=np.float32),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.float32),
        "behaviour_actions": np.asarray(student_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=bool),
        "puck_visible": np.asarray(visible, dtype=bool),
        "episode_offsets": np.asarray(offsets, dtype=np.int64),
        "episode_indices": np.asarray(episode_indices, dtype=np.int64),
        "episode_shot_ids": np.asarray(shot_ids),
        "episode_blackout_steps": np.asarray(blackout_steps, dtype=np.int16),
        "episode_outcomes": np.asarray(outcomes),
    }


def _replace_teacher_previous_action(carry: Any, action: np.ndarray[Any, Any]) -> Any:
    import jax

    previous = dict(carry[3])
    previous["action"] = jax.device_put(np.asarray([action], dtype=np.float32))
    return (*carry[:3], previous)


def _make_environment(config: dict[str, Any]) -> DirectLaunchTrainingEnv:
    return DirectLaunchTrainingEnv(
        distribution_config=config["distribution_config"],
        reward_config=config["reward_config"],
        split=config["distribution_split"],
        sampling_seed=int(config["sampling_seed"]),
        blackout_start_observation_step=int(config["blackout_start_observation_step"]),
        minimum_blackout_steps=int(config["minimum_blackout_steps"]),
        maximum_blackout_steps=int(config["maximum_blackout_steps"]),
        action_lock_steps=int(config["action_lock_steps"]),
    )


def _load_teacher(
    config: dict[str, Any],
    teacher_config: dict[str, Any],
    checkpoint: Path,
    output: Path,
) -> tuple[Any, str, list[str]]:
    from ruamel import yaml as ruamel_yaml

    collection = config["shadow_collection"]

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=collection["distribution_config"],
            reward_config=collection["reward_config"],
            split=collection["distribution_split"],
            sampling_seed=int(collection["sampling_seed"]),
            blackout_start_observation_step=int(
                collection["blackout_start_observation_step"]
            ),
            minimum_blackout_steps=int(collection["minimum_blackout_steps"]),
            maximum_blackout_steps=int(collection["maximum_blackout_steps"]),
            action_lock_steps=int(collection["action_lock_steps"]),
            **kwargs,
        )

    gymnasium.register(id=ENVIRONMENT_ID, entry_point=make_environment)
    import elements
    import jax
    import portal
    from dreamerv3 import agent as dreamer_agent
    from dreamerv3 import main as dreamer_main
    from embodied.envs import from_gymnasium
    from embodied.jax import outs as embodied_outs

    enable_deterministic_dreamer_inference(embodied_outs.Agg)
    from_gymnasium.elements = elements
    upstream = ruamel_yaml.YAML(typ="safe").load(
        (Path(dreamer_agent.__file__).parent / "configs.yaml").read_text()
    )
    profile = teacher_config["full"]
    dreamer_config = elements.Config(upstream["defaults"])
    dreamer_config = dreamer_config.update(upstream[str(profile["model_preset"])])
    dreamer_config = dreamer_config.update(
        {
            "task": f"gymnasium_{ENVIRONMENT_ID}",
            "logdir": str(output / "teacher-runtime"),
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
    teacher = dreamer_main.make_agent(dreamer_config)
    loader = elements.Checkpoint()
    loader.agent = teacher
    loader.load(checkpoint, keys=["agent"])
    return teacher, jax.__version__, [str(device) for device in jax.devices()]


def _validate_config(
    config: dict[str, Any], student_checkpoint: Path, teacher_checkpoint: Path
) -> None:
    collection = config["shadow_collection"]
    if collection["teacher_action_semantics"] != DETERMINISTIC_ACTION_SEMANTICS:
        raise ValueError("shadow collection must use deterministic teacher means")
    if collection["teacher_previous_action"] != ("previous_student_requested_command"):
        raise ValueError("teacher must consume the student command history")
    if _sha256(student_checkpoint) != config["behaviour_policy"]["checkpoint_sha256"]:
        raise ValueError("behaviour checkpoint hash mismatch")
    if (
        _sha256(teacher_checkpoint / "agent.pkl")
        != config["provenance"]["teacher_actor_sha256"]
    ):
        raise ValueError("teacher checkpoint hash mismatch")


def _validate_shard(path: Path, first_episode: int, episode_count: int) -> None:
    with np.load(path, allow_pickle=False) as shard:
        expected = np.arange(first_episode, first_episode + episode_count)
        if not np.array_equal(shard["episode_indices"], expected):
            raise ValueError(f"existing shard has unexpected episodes: {path}")
        if str(shard["teacher_action_semantics"]) != DETERMINISTIC_ACTION_SEMANTICS:
            raise ValueError(f"existing shard has incompatible targets: {path}")
        offsets = shard["episode_offsets"]
        for local_index in range(episode_count):
            start = int(offsets[local_index])
            stop = int(offsets[local_index + 1])
            if not np.array_equal(
                shard["previous_actions"][start + 1 : stop],
                shard["behaviour_actions"][start : stop - 1],
            ):
                raise ValueError(f"existing shard has invalid command history: {path}")


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
