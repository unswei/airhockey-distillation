#!/usr/bin/env python3
"""Collect one matched principal-family shadow partition."""

from __future__ import annotations

import argparse
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

from airhockey_distill.envs import DirectLaunchTrainingEnv
from airhockey_distill.principal_sweep import (
    DETERMINISTIC_ACTION_SEMANTICS,
    load_principal_protocol,
    principal_family_spec,
    sha256_file,
    shadow_schedule_records,
    shadow_schedule_sha256,
    validate_training_seed,
)
from airhockey_distill.students import load_principal_policy
from airhockey_distill.teachers import enable_deterministic_dreamer_inference

ENVIRONMENT_ID = "AirHockeyPrincipalShadowDataset-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    protocol_path = args.protocol.resolve()
    protocol = load_principal_protocol(protocol_path)
    principal_family_spec(protocol, args.family)
    validate_training_seed(protocol, args.seed)
    teacher_config = _load_mapping(args.teacher_config.resolve())
    if args.teacher_config.resolve() != Path(
        protocol["shadow_labelling"]["teacher_config"]
    ).resolve() or sha256_file(args.teacher_config.resolve()) != (
        protocol["shadow_labelling"]["teacher_config_sha256"]
    ):
        raise ValueError("teacher config does not match the execution specification")
    student_checkpoint = args.student_checkpoint.resolve()
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    policy = load_principal_policy(args.family, student_checkpoint)
    _validate_checkpoints(
        protocol,
        policy.metadata,
        args.family,
        args.seed,
        student_checkpoint,
        teacher_checkpoint,
    )
    schedule = shadow_schedule_records(protocol, args.seed)
    output = args.output.resolve()
    shards_directory = output / "shards"
    output.mkdir(parents=True, exist_ok=True)
    shards_directory.mkdir(exist_ok=True)
    resolved = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "family_id": args.family,
        "collector_training_seed": args.seed,
        "shadow_schedule_sha256": shadow_schedule_sha256(protocol, args.seed),
        "student_checkpoint_sha256": sha256_file(student_checkpoint),
    }
    resolved_path = output / "resolved_run.json"
    if resolved_path.exists():
        if json.loads(resolved_path.read_text()) != resolved:
            raise ValueError("existing shadow output has a different resolved run")
    else:
        resolved_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")

    teacher, jax_version, devices = _load_teacher(
        protocol,
        teacher_config,
        teacher_checkpoint,
        output,
    )
    environment = _make_environment(protocol)
    episodes_per_shard = int(protocol["shadow_labelling"]["episodes_per_shard"])
    if len(schedule) % episodes_per_shard:
        raise ValueError("collector partition must divide into complete shards")
    started = perf_counter()
    try:
        for first in range(0, len(schedule), episodes_per_shard):
            records = schedule[first : first + episodes_per_shard]
            shard_index = first // episodes_per_shard
            shard_path = shards_directory / f"shard-{shard_index:04d}.npz"
            if shard_path.exists():
                validate_shadow_shard(shard_path, records)
                continue
            payload = collect_shadow_shard(
                teacher,
                policy,
                environment,
                records,
            )
            temporary = shard_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, shard_path)
            print(
                json.dumps(
                    {
                        "family_id": args.family,
                        "training_seed": args.seed,
                        "shard": shard_index,
                        "episodes_collected": first + len(records),
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
            shard_transitions = int(shard["observations"].shape[0])
            transition_count += shard_transitions
            outcomes.update(str(value) for value in shard["episode_outcomes"])
            episode_indices = np.asarray(shard["episode_indices"])
        shards.append(
            {
                "file": str(shard_path.relative_to(output)),
                "first_episode": int(episode_indices[0]),
                "episodes": int(len(episode_indices)),
                "transitions": shard_transitions,
                "bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )
    if sum(int(value["episodes"]) for value in shards) != len(schedule):
        raise RuntimeError("principal shadow partition is incomplete")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": f"principal_{args.family}_shadow_v1_seed_{args.seed}",
        "dataset_schema_version": 2,
        "family_id": args.family,
        "collector_training_seed": args.seed,
        "teacher_action_semantics": DETERMINISTIC_ACTION_SEMANTICS,
        "deterministic_inference": True,
        "rollout_controller": "family_and_seed_specific_base_only_checkpoint",
        "teacher_previous_action": "previous_behaviour_policy_requested_command",
        "code_commit": args.code_commit,
        "protocol_sha256": sha256_file(protocol_path),
        "shadow_schedule_sha256": shadow_schedule_sha256(protocol, args.seed),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint / "agent.pkl"),
        "student_checkpoint_sha256": sha256_file(student_checkpoint),
        "episode_count": len(schedule),
        "first_collection_offset": schedule[0]["collection_offset"],
        "last_collection_offset": schedule[-1]["collection_offset"],
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


def collect_shadow_shard(
    teacher: Any,
    policy: Any,
    environment: DirectLaunchTrainingEnv,
    records: list[dict[str, int]],
) -> dict[str, np.ndarray[Any, Any]]:
    observations = []
    previous_actions = []
    teacher_actions = []
    behaviour_actions = []
    rewards = []
    terminals = []
    visible = []
    offsets = [0]
    episode_indices = []
    collection_offsets = []
    reset_seeds = []
    shot_ids = []
    blackout_steps = []
    outcomes = []
    for record in records:
        observation, info = environment.reset(seed=record["reset_seed"])
        if str(info["shot_id"]) != record["shot_id"] or int(
            environment.blackout.length_steps
        ) != int(record["blackout_steps"]):
            raise RuntimeError("environment reset disagrees with the paired schedule")
        teacher_carry = teacher.init_policy(1)
        policy_carry = policy.initial_carry()
        previous_command = np.zeros(2, dtype=np.float32)
        reward = 0.0
        step = 0
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
                teacher_carry,
                policy_observation,
                mode="eval",
            )
            teacher_action = np.clip(
                np.asarray(teacher_output["action"], dtype=np.float32)[0],
                -1.0,
                1.0,
            )
            behaviour_action, policy_carry = policy.act(observation, policy_carry)
            observations.append(np.asarray(observation, dtype=np.float32))
            previous_actions.append(previous_command.copy())
            teacher_actions.append(teacher_action)
            behaviour_actions.append(behaviour_action)
            visible.append(bool(info["puck_visible"]))
            observation, reward, terminated, truncated, info = environment.step(
                behaviour_action
            )
            rewards.append(float(reward))
            terminals.append(bool(terminated or truncated))
            previous_command = np.asarray(behaviour_action, dtype=np.float32)
            step += 1
            if terminated or truncated:
                break
        episode_indices.append(record["episode_index"])
        collection_offsets.append(record["collection_offset"])
        reset_seeds.append(record["reset_seed"])
        shot_ids.append(str(info["shot_id"]))
        blackout_steps.append(int(environment.blackout.length_steps))
        outcomes.append(str(info["outcome"]))
        offsets.append(len(observations))
    return {
        "dataset_schema_version": np.asarray(2, dtype=np.int64),
        "teacher_action_semantics": np.asarray(DETERMINISTIC_ACTION_SEMANTICS),
        "observations": np.asarray(observations, dtype=np.float32),
        "previous_actions": np.asarray(previous_actions, dtype=np.float32),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.float32),
        "behaviour_actions": np.asarray(behaviour_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=bool),
        "puck_visible": np.asarray(visible, dtype=bool),
        "episode_offsets": np.asarray(offsets, dtype=np.int64),
        "episode_indices": np.asarray(episode_indices, dtype=np.int64),
        "episode_collection_offsets": np.asarray(collection_offsets, dtype=np.int64),
        "episode_reset_seeds": np.asarray(reset_seeds, dtype=np.int64),
        "episode_shot_ids": np.asarray(shot_ids),
        "episode_blackout_steps": np.asarray(blackout_steps, dtype=np.int16),
        "episode_outcomes": np.asarray(outcomes),
    }


def validate_shadow_shard(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with np.load(path, allow_pickle=False) as shard:
        for name in ("episode_indices", "episode_collection_offsets", "episode_reset_seeds"):
            expected_name = {
                "episode_indices": "episode_index",
                "episode_collection_offsets": "collection_offset",
                "episode_reset_seeds": "reset_seed",
            }[name]
            expected = np.asarray([value[expected_name] for value in records])
            if not np.array_equal(shard[name], expected):
                raise ValueError(f"existing shard has the wrong paired schedule: {path}")
        if str(shard["teacher_action_semantics"]) != DETERMINISTIC_ACTION_SEMANTICS:
            raise ValueError("existing shard does not contain deterministic means")
        if not np.array_equal(
            shard["episode_shot_ids"],
            np.asarray([value["shot_id"] for value in records]),
        ) or not np.array_equal(
            shard["episode_blackout_steps"],
            np.asarray([value["blackout_steps"] for value in records]),
        ):
            raise ValueError("existing shard has the wrong realised episode schedule")
        offsets = shard["episode_offsets"]
        for local_index in range(len(records)):
            first = int(offsets[local_index])
            last = int(offsets[local_index + 1])
            if not np.array_equal(
                shard["previous_actions"][first + 1 : last],
                shard["behaviour_actions"][first : last - 1],
            ):
                raise ValueError("existing shard has invalid behaviour command history")


def _validate_checkpoints(
    protocol: dict[str, Any],
    metadata: dict[str, Any],
    family_id: str,
    seed: int,
    student_checkpoint: Path,
    teacher_checkpoint: Path,
) -> None:
    if metadata.get("student_id") != family_id:
        raise ValueError("behaviour checkpoint family mismatch")
    if int(metadata.get("training_seed", -1)) != seed:
        raise ValueError("behaviour checkpoint seed mismatch")
    if metadata.get("training_stage") != "collector":
        raise ValueError("shadow behaviour checkpoint must be a collector checkpoint")
    if metadata.get("protocol_sha256") != protocol.get("resolved_sha256"):
        raise ValueError("behaviour checkpoint protocol hash mismatch")
    if metadata.get("dataset_manifest_sha256") != (
        protocol["base_teacher_dataset"]["manifest_sha256"]
    ):
        raise ValueError("behaviour checkpoint was not trained on the common base data")
    if sha256_file(teacher_checkpoint / "agent.pkl") != (
        protocol["teacher"]["actor_sha256"]
    ):
        raise ValueError("teacher checkpoint hash mismatch")
    if not student_checkpoint.is_file():
        raise ValueError("student checkpoint does not exist")


def _make_environment(protocol: dict[str, Any]) -> DirectLaunchTrainingEnv:
    shadow = protocol["shadow_labelling"]
    return DirectLaunchTrainingEnv(
        distribution_config=shadow["distribution_config"],
        reward_config=shadow["reward_config"],
        split=shadow["distribution_split"],
        sampling_seed=int(shadow["collection_sampling_seed"]),
        blackout_start_observation_step=int(
            shadow["blackout_start_observation_step"]
        ),
        minimum_blackout_steps=int(shadow["minimum_blackout_steps"]),
        maximum_blackout_steps=int(shadow["maximum_blackout_steps"]),
        action_lock_steps=int(shadow["action_lock_steps"]),
    )


def _load_teacher(
    protocol: dict[str, Any],
    teacher_config: dict[str, Any],
    checkpoint: Path,
    output: Path,
) -> tuple[Any, str, list[str]]:
    from ruamel import yaml as ruamel_yaml

    shadow = protocol["shadow_labelling"]

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=shadow["distribution_config"],
            reward_config=shadow["reward_config"],
            split=shadow["distribution_split"],
            sampling_seed=int(shadow["collection_sampling_seed"]),
            blackout_start_observation_step=int(
                shadow["blackout_start_observation_step"]
            ),
            minimum_blackout_steps=int(shadow["minimum_blackout_steps"]),
            maximum_blackout_steps=int(shadow["maximum_blackout_steps"]),
            action_lock_steps=int(shadow["action_lock_steps"]),
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
    config = elements.Config(upstream["defaults"])
    config = config.update(upstream[str(profile["model_preset"])])
    config = config.update(
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
    teacher = dreamer_main.make_agent(config)
    loader = elements.Checkpoint()
    loader.agent = teacher
    loader.load(checkpoint, keys=["agent"])
    return teacher, jax.__version__, [str(device) for device in jax.devices()]


def _replace_teacher_previous_action(carry: Any, action: np.ndarray) -> Any:
    import jax

    previous = dict(carry[3])
    previous["action"] = jax.device_put(np.asarray([action], dtype=np.float32))
    return (*carry[:3], previous)


def _load_mapping(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return value


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
