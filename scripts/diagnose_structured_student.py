#!/usr/bin/env python3
"""Diagnose offline and closed-loop failure of one structured student."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scipy.spatial import cKDTree

from airhockey_distill.envs import (
    BlackoutSchedule,
    DefenceRewardTracker,
    DefendShotTrackingLoss,
    DirectLaunchTrainingEnv,
    load_defence_reward,
    load_direct_launch_distribution,
)
from airhockey_distill.students import StructuredRecurrentPolicy
from airhockey_distill.teachers import enable_deterministic_dreamer_inference

ENVIRONMENT_ID = "AirHockeyStructuredStudentDiagnostic-v0"
SAVE_OUTCOMES = frozenset({"returned", "arrested", "safe_deflection"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_mapping(config_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))

    dataset_directory = args.dataset.resolve()
    dataset_manifest_path = dataset_directory / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    _validate_bindings(
        config,
        dataset_manifest,
        dataset_manifest_path,
        args.student_checkpoint.resolve(),
    )
    student = StructuredRecurrentPolicy.load(args.student_checkpoint.resolve())
    phase_data = load_no_blackout_episodes(
        dataset_directory, dataset_manifest, config["dataset"]
    )
    offline = offline_diagnostic(
        student,
        phase_data,
        config["phase_diagnostic"],
    )

    teacher, jax_version, jax_devices = _load_teacher(config, output, args)
    closed_loop, rollout_histories = closed_loop_diagnostic(
        teacher,
        student,
        config["closed_loop_diagnostic"],
    )
    neighbourhood = neighbourhood_diagnostic(
        phase_data,
        rollout_histories,
        config["target_neighbourhood_diagnostic"],
    )

    findings = classify_findings(offline, closed_loop, neighbourhood)
    result = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": args.code_commit,
        "config_sha256": _sha256(config_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "student_checkpoint_sha256": _sha256(args.student_checkpoint.resolve()),
        "teacher_action_semantics": dataset_manifest["teacher_action_semantics"],
        "offline_action_diagnostic": offline,
        "closed_loop_diagnostic": closed_loop,
        "target_neighbourhood_diagnostic": neighbourhood,
        "findings": findings,
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version,
            "jax": jax_version,
            "jax_devices": jax_devices,
            "mujoco": mujoco.__version__,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def load_no_blackout_episodes(
    dataset_directory: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    split_map = {
        int(value): "train" for value in config["train_residues"]
    }
    split_map.update(
        {int(value): "validation" for value in config["validation_residues"]}
    )
    modulus = int(config["episode_split_modulus"])
    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for shard_entry in manifest["shards"]:
        shard_path = dataset_directory / shard_entry["file"]
        if _sha256(shard_path) != shard_entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            offsets = shard["episode_offsets"]
            indices = shard["episode_indices"]
            blackouts = shard["episode_blackout_steps"]
            for local_index, episode_index_raw in enumerate(indices):
                if int(blackouts[local_index]) != int(config["no_blackout_steps"]):
                    continue
                episode_index = int(episode_index_raw)
                split = split_map.get(episode_index % modulus)
                if split is None:
                    continue
                selection = slice(
                    int(offsets[local_index]), int(offsets[local_index + 1])
                )
                result[split].append(
                    {
                        "episode_index": episode_index,
                        "observations": np.asarray(
                            shard["observations"][selection], dtype=np.float32
                        ),
                        "previous_actions": np.asarray(
                            shard["previous_actions"][selection], dtype=np.float32
                        ),
                        "teacher_actions": np.asarray(
                            shard["teacher_actions"][selection], dtype=np.float32
                        ),
                    }
                )
    if not result["train"] or not result["validation"]:
        raise ValueError("no-blackout diagnostic splits must be non-empty")
    return result


def offline_diagnostic(
    student: StructuredRecurrentPolicy,
    episodes_by_split: dict[str, list[dict[str, Any]]],
    phase_config: dict[str, Any],
) -> dict[str, Any]:
    phases = {
        name: (int(bounds[0]), int(bounds[1]))
        for name, bounds in phase_config["phases"].items()
    }
    totals: dict[str, dict[str, list[float | int]]] = {
        mode: {name: [0.0, 0] for name in (*phases, "overall")}
        for mode in (
            "teacher_forced_previous_actions",
            "recursive_student_previous_actions",
            "teacher_prefix_through_burn_in",
        )
    }
    target_totals = {name: [0.0, 0, np.zeros(2)] for name in (*phases, "overall")}
    for episode in episodes_by_split["validation"]:
        observations = episode["observations"]
        previous_actions = episode["previous_actions"]
        targets = episode["teacher_actions"]
        teacher_forced, _ = student.teacher_forced_sequence(
            observations, previous_actions
        )
        recursive = _fixed_observation_rollout(
            student, observations, targets, prefix_steps=0
        )
        prefix = _fixed_observation_rollout(
            student,
            observations,
            targets,
            prefix_steps=int(phase_config["burn_in_steps"]),
        )
        for step in range(len(observations)):
            phase = phase_for_step(step, phases)
            for mode, predictions in (
                ("teacher_forced_previous_actions", teacher_forced),
                ("recursive_student_previous_actions", recursive),
                ("teacher_prefix_through_burn_in", prefix),
            ):
                _add_squared_error(totals[mode][phase], predictions[step], targets[step])
                _add_squared_error(totals[mode]["overall"], predictions[step], targets[step])
            for name in (phase, "overall"):
                target_totals[name][0] += float(np.sum(targets[step] ** 2))
                target_totals[name][1] += 2
                target_totals[name][2] += targets[step]
    target_baselines = {}
    for name, (_, count, action_sum) in target_totals.items():
        mean = action_sum / (count / 2)
        squared_error = 0.0
        action_values = 0
        start, stop = phases.get(name, (0, 10**9))
        for episode in episodes_by_split["validation"]:
            target = episode["teacher_actions"][start:stop]
            squared_error += float(np.sum((target - mean) ** 2))
            action_values += int(target.size)
        target_baselines[name] = {
            "constant_mean_action": mean.tolist(),
            "constant_mean_action_mse": squared_error / action_values,
            "action_values": action_values,
        }
    return {
        "validation_no_blackout_episodes": len(episodes_by_split["validation"]),
        "phase_bounds": {name: list(bounds) for name, bounds in phases.items()},
        "action_mse": {
            mode: {name: total / count for name, (total, count) in values.items()}
            for mode, values in totals.items()
        },
        "target_baselines": target_baselines,
    }


def _fixed_observation_rollout(
    student: StructuredRecurrentPolicy,
    observations: np.ndarray[Any, Any],
    teacher_actions: np.ndarray[Any, Any],
    *,
    prefix_steps: int,
) -> np.ndarray[Any, Any]:
    state = np.zeros(64, dtype=np.float32)
    previous = np.zeros(2, dtype=np.float32)
    predictions = []
    for step, observation in enumerate(observations):
        action, state = student.step(observation, previous, state)
        predictions.append(action)
        previous = teacher_actions[step] if step < prefix_steps else action
    return np.asarray(predictions, dtype=np.float32)


def phase_for_step(step: int, phases: dict[str, tuple[int, int]]) -> str:
    matches = [name for name, (start, stop) in phases.items() if start <= step < stop]
    if len(matches) != 1:
        raise ValueError(f"step {step} belongs to {len(matches)} phases")
    return matches[0]


def closed_loop_diagnostic(
    teacher: Any,
    student: StructuredRecurrentPolicy,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    distribution = load_direct_launch_distribution(config["distribution_config"])
    generated = distribution.generate(config["distribution_split"])
    selected = _select_evenly(generated, int(config["shot_count"]))
    reward_specification = load_defence_reward(config["reward_config"])
    environment = DefendShotTrackingLoss(
        reward_tracker=DefenceRewardTracker(reward_specification),
        action_lock_steps=int(config["action_lock_steps"]),
    )
    variants = tuple(str(value) for value in config["variants"])
    records: list[dict[str, Any]] = []
    standard_histories: list[dict[str, Any]] = []
    started = perf_counter()
    try:
        for generated_shot in selected:
            for variant in variants:
                environment.blackout = BlackoutSchedule(
                    start_observation_step=5,
                    length_steps=int(config["blackout_steps"]),
                )
                observation, _ = environment.reset(shot=generated_shot.shot)
                teacher_carry = teacher.init_policy(1)
                student_state = np.zeros(64, dtype=np.float32)
                previous_command = np.zeros(2, dtype=np.float32)
                previous_teacher_action = np.zeros(2, dtype=np.float32)
                reward = 0.0
                score = 0.0
                steps = 0
                outcome = "rollout_limit"
                observations = []
                previous_commands = []
                student_actions = []
                teacher_actions = []
                while steps < environment.timeout_steps:
                    teacher_carry = _replace_teacher_previous_action(
                        teacher_carry, previous_command
                    )
                    policy_observation = {
                        "image": np.asarray([observation], dtype=np.float32),
                        "reward": np.asarray([reward], dtype=np.float32),
                        "is_first": np.asarray([steps == 0], dtype=bool),
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
                    student_previous = (
                        previous_teacher_action
                        if variant == "oracle_teacher_previous_action"
                        else previous_command
                    )
                    student_action, student_state = student.step(
                        observation, student_previous, student_state
                    )
                    if variant == "teacher_reference":
                        command = teacher_action
                    elif (
                        variant == "teacher_controlled_prefix_16" and steps < 16
                    ):
                        command = teacher_action
                    else:
                        command = student_action
                    observations.append(np.asarray(observation, dtype=np.float32))
                    previous_commands.append(previous_command.copy())
                    student_actions.append(student_action)
                    teacher_actions.append(teacher_action)
                    observation, reward, terminated, truncated, info = environment.step(
                        command
                    )
                    score += float(reward)
                    previous_command = np.asarray(command, dtype=np.float32)
                    previous_teacher_action = teacher_action
                    steps += 1
                    if terminated or truncated:
                        outcome = str(info["outcome"])
                        break
                student_array = np.asarray(student_actions, dtype=np.float32)
                teacher_array = np.asarray(teacher_actions, dtype=np.float32)
                records.append(
                    {
                        "variant": variant,
                        "shot_id": generated_shot.shot.shot_id,
                        "outcome": outcome,
                        "score": score,
                        "steps": steps,
                        "shadow_teacher_action_mse": float(
                            np.mean((student_array - teacher_array) ** 2)
                        ),
                    }
                )
                if variant == "standard_student":
                    standard_histories.append(
                        {
                            "observations": np.asarray(observations, dtype=np.float32),
                            "previous_actions": np.asarray(
                                previous_commands, dtype=np.float32
                            ),
                            "teacher_actions": teacher_array,
                        }
                    )
    finally:
        environment.close()
    summaries = {
        variant: _summarise_records(
            [record for record in records if record["variant"] == variant]
        )
        for variant in variants
    }
    return {
        "duration_seconds": perf_counter() - started,
        "summaries": summaries,
        "episodes": records,
    }, standard_histories


def _replace_teacher_previous_action(carry: Any, action: np.ndarray[Any, Any]) -> Any:
    import jax

    previous = dict(carry[3])
    previous["action"] = jax.device_put(np.asarray([action], dtype=np.float32))
    return (*carry[:3], previous)


def neighbourhood_diagnostic(
    episodes_by_split: dict[str, list[dict[str, Any]]],
    rollout_episodes: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    history_steps = int(config["history_steps"])
    train_x, train_y, _ = history_examples(
        episodes_by_split["train"], history_steps=history_steps
    )
    validation_x, validation_y, validation_steps = history_examples(
        episodes_by_split["validation"], history_steps=history_steps
    )
    rollout_x, rollout_y, rollout_steps = history_examples(
        rollout_episodes, history_steps=history_steps
    )
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    active = scale >= float(config["standardisation_minimum_scale"])
    if not np.any(active):
        raise RuntimeError("history features contain no varying dimensions")
    train_z = (train_x[:, active] - mean[active]) / scale[active]
    tree = cKDTree(train_z)
    neighbours = int(config["neighbours"])
    validation = _query_neighbourhood(
        tree,
        train_y,
        validation_x,
        validation_y,
        validation_steps,
        mean,
        scale,
        active,
        neighbours,
        int(config["query_chunk_size"]),
    )
    rollout = _query_neighbourhood(
        tree,
        train_y,
        rollout_x,
        rollout_y,
        rollout_steps,
        mean,
        scale,
        active,
        neighbours,
        int(config["query_chunk_size"]),
    )
    return {
        "history_steps": history_steps,
        "active_standardised_dimensions": int(np.sum(active)),
        "training_examples": len(train_x),
        "validation_examples": len(validation_x),
        "student_rollout_examples": len(rollout_x),
        "on_policy_validation": validation,
        "student_closed_loop": rollout,
        "student_to_on_policy_distance_ratio": {
            key: _safe_ratio(
                rollout["nearest_standardised_distance"][key],
                validation["nearest_standardised_distance"][key],
            )
            for key in ("median", "p95")
        },
    }


def history_examples(
    episodes: list[dict[str, Any]], *, history_steps: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    features = []
    targets = []
    steps = []
    width = 21
    for episode in episodes:
        values = np.concatenate(
            (episode["observations"], episode["previous_actions"]), axis=1
        )
        for step in range(len(values)):
            history = np.zeros((history_steps, width), dtype=np.float32)
            validity = np.zeros(history_steps, dtype=np.float32)
            start = max(0, step - history_steps + 1)
            count = step - start + 1
            history[-count:] = values[start : step + 1]
            validity[-count:] = 1.0
            features.append(np.concatenate((history.reshape(-1), validity)))
            targets.append(episode["teacher_actions"][step])
            steps.append(step)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(steps, dtype=np.int16),
    )


def _query_neighbourhood(
    tree: cKDTree,
    train_targets: np.ndarray[Any, Any],
    query_features: np.ndarray[Any, Any],
    query_targets: np.ndarray[Any, Any],
    query_steps: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    active: np.ndarray[Any, Any],
    neighbours: int,
    chunk_size: int,
) -> dict[str, Any]:
    all_distances = []
    all_indices = []
    for start in range(0, len(query_features), chunk_size):
        query = (query_features[start : start + chunk_size, active] - mean[active])
        query = query / scale[active]
        distances, indices = tree.query(query, k=neighbours, workers=1)
        all_distances.append(distances)
        all_indices.append(indices)
    distances = np.concatenate(all_distances)
    indices = np.concatenate(all_indices)
    neighbour_targets = train_targets[indices]
    nearest_mse = np.mean((neighbour_targets[:, 0] - query_targets) ** 2, axis=1)
    mean_mse = np.mean(
        (np.mean(neighbour_targets, axis=1) - query_targets) ** 2, axis=1
    )
    local_variance = np.mean(np.var(neighbour_targets, axis=1), axis=1)
    return {
        "nearest_standardised_distance": _quantiles(distances[:, 0]),
        "nearest_target_action_mse": float(np.mean(nearest_mse)),
        "neighbour_mean_target_action_mse": float(np.mean(mean_mse)),
        "local_target_variance": {
            **_quantiles(local_variance),
            "mean": float(np.mean(local_variance)),
        },
        "by_phase": {
            name: {
                "examples": int(np.sum(mask)),
                "nearest_target_action_mse": float(np.mean(nearest_mse[mask])),
                "local_target_variance_mean": float(np.mean(local_variance[mask])),
            }
            for name, mask in _step_phase_masks(query_steps).items()
            if np.any(mask)
        },
    }


def _step_phase_masks(steps: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "locked": steps < 5,
        "live_but_loss_masked": (steps >= 5) & (steps < 16),
        "loss_bearing": steps >= 16,
    }


def classify_findings(
    offline: dict[str, Any],
    closed_loop: dict[str, Any],
    neighbourhood: dict[str, Any],
) -> dict[str, Any]:
    phase = offline["action_mse"]
    standard = closed_loop["summaries"]["standard_student"]
    prefix = closed_loop["summaries"]["teacher_controlled_prefix_16"]
    oracle = closed_loop["summaries"]["oracle_teacher_previous_action"]
    ratios = neighbourhood["student_to_on_policy_distance_ratio"]
    return {
        "loss_mask_excludes_live_control": {
            "present": True,
            "evidence": "steps 5 through 15 affect control but receive no action loss",
            "live_masked_teacher_forced_mse": phase[
                "teacher_forced_previous_actions"
            ]["live_but_loss_masked"],
        },
        "previous_action_exposure": {
            "recursive_minus_teacher_forced_mse": phase[
                "recursive_student_previous_actions"
            ]["overall"]
            - phase["teacher_forced_previous_actions"]["overall"],
            "oracle_previous_action_save_rate_change": oracle["save_rate"]
            - standard["save_rate"],
        },
        "prefix_distribution_shift": {
            "teacher_prefix_save_rate_change": prefix["save_rate"]
            - standard["save_rate"],
        },
        "closed_loop_observation_shift": {
            "median_neighbour_distance_ratio": ratios["median"],
            "p95_neighbour_distance_ratio": ratios["p95"],
        },
        "target_multimodality_proxy": {
            "on_policy_local_target_variance_mean": neighbourhood[
                "on_policy_validation"
            ]["local_target_variance"]["mean"],
            "on_policy_neighbour_mean_target_mse": neighbourhood[
                "on_policy_validation"
            ]["neighbour_mean_target_action_mse"],
        },
    }


def _load_teacher(
    config: dict[str, Any], output: Path, args: argparse.Namespace
) -> tuple[Any, str, list[str]]:
    teacher_config = _load_mapping(Path(config["provenance"]["teacher_config"]))
    profile = teacher_config["full"]
    evaluation = config["closed_loop_diagnostic"]

    def make_environment(**kwargs: Any) -> DirectLaunchTrainingEnv:
        return DirectLaunchTrainingEnv(
            distribution_config=evaluation["distribution_config"],
            reward_config=evaluation["reward_config"],
            split=evaluation["distribution_split"],
            sampling_seed=12303,
            minimum_blackout_steps=0,
            maximum_blackout_steps=0,
            action_lock_steps=int(evaluation["action_lock_steps"]),
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
    loader.load(args.teacher_checkpoint.resolve(), keys=["agent"])
    return teacher, jax.__version__, [str(device) for device in jax.devices()]


def _validate_bindings(
    config: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    student_checkpoint: Path,
) -> None:
    if manifest["status"] != "completed":
        raise ValueError("dataset is not complete")
    if manifest["dataset_id"] != config["dataset"]["id"]:
        raise ValueError("dataset id mismatch")
    if _sha256(manifest_path) != config["dataset"]["manifest_sha256"]:
        raise ValueError("dataset manifest hash mismatch")
    if manifest["teacher_action_semantics"] != config["dataset"][
        "teacher_action_semantics"
    ]:
        raise ValueError("dataset target semantics mismatch")
    if _sha256(student_checkpoint) != config["student"]["checkpoint_sha256"]:
        raise ValueError("student checkpoint hash mismatch")


def _summarise_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(record["outcome"] for record in records)
    saves = sum(outcomes[name] for name in SAVE_OUTCOMES)
    return {
        "episodes": len(records),
        "save_count": saves,
        "save_rate": saves / len(records),
        "outcome_counts": dict(sorted(outcomes.items())),
        "mean_score": float(np.mean([record["score"] for record in records])),
        "shadow_teacher_action_mse": float(
            np.mean([record["shadow_teacher_action_mse"] for record in records])
        ),
    }


def _select_evenly(values: tuple[Any, ...], count: int) -> tuple[Any, ...]:
    indices = np.linspace(0, len(values) - 1, num=count, dtype=int)
    if count <= 0 or count > len(values) or len(set(indices.tolist())) != count:
        raise ValueError("invalid paired shot selection")
    return tuple(values[int(index)] for index in indices)


def _add_squared_error(
    total: list[float | int], prediction: np.ndarray[Any, Any], target: np.ndarray[Any, Any]
) -> None:
    total[0] = float(total[0]) + float(np.sum((prediction - target) ** 2))
    total[1] = int(total[1]) + int(target.size)


def _quantiles(values: np.ndarray[Any, Any]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return no ratio when exact on-policy overlap makes it undefined."""

    return numerator / denominator if denominator > 0.0 else None


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
    print(json.dumps({"status": result["status"], "findings": result["findings"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
