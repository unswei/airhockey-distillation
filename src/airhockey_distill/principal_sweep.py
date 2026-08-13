"""Shared protocol, data and schedule contracts for the principal sweep."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from airhockey_distill.envs import load_direct_launch_distribution
from airhockey_distill.students import PRINCIPAL_FAMILY_IDS

DETERMINISTIC_ACTION_SEMANTICS = (
    "deterministic_actor_mean_after_public_adapter_clip"
)


@dataclass(frozen=True)
class PrincipalEpisodeSplit:
    """Padded complete episodes shared by every principal family."""

    observations: np.ndarray[Any, Any]
    previous_actions: np.ndarray[Any, Any]
    teacher_actions: np.ndarray[Any, Any]
    puck_visible: np.ndarray[Any, Any]
    valid_mask: np.ndarray[Any, Any]
    episode_indices: np.ndarray[Any, Any]
    episode_lengths: np.ndarray[Any, Any]

    @property
    def episode_count(self) -> int:
        return int(len(self.episode_indices))

    @property
    def transition_count(self) -> int:
        return int(np.sum(self.valid_mask))


def load_principal_protocol(path: str | Path) -> dict[str, Any]:
    """Load the immutable predeclaration through its execution binding."""

    execution_path = Path(path)
    execution = yaml.safe_load(execution_path.read_text())
    if not isinstance(execution, dict):
        raise TypeError("principal execution config must be a mapping")
    protocol_path = Path(execution["protocol"]["path"])
    if sha256_file(protocol_path) != execution["protocol"]["sha256"]:
        raise ValueError("immutable principal protocol hash mismatch")
    protocol = yaml.safe_load(protocol_path.read_text())
    if not isinstance(protocol, dict):
        raise TypeError("principal protocol must be a mapping")
    protocol["execution"] = execution
    protocol["resolved_sha256"] = sha256_file(execution_path)
    protocol["immutable_protocol_sha256"] = execution["protocol"]["sha256"]
    validate_principal_protocol(protocol)
    return protocol


def validate_principal_protocol(protocol: dict[str, Any]) -> None:
    family_ids = tuple(str(value["id"]) for value in protocol["families"])
    if family_ids != PRINCIPAL_FAMILY_IDS:
        raise ValueError("principal protocol must contain the canonical seven families")
    architecture_configs = protocol["execution"]["architecture_configs"]
    if set(architecture_configs) != set(family_ids):
        raise ValueError("execution config must bind all seven architectures")
    for family_id, architecture in architecture_configs.items():
        architecture_path = Path(architecture["path"])
        if sha256_file(architecture_path) != architecture["sha256"]:
            raise ValueError(
                f"architecture config hash mismatch for {family_id}"
            )
    training = protocol["training"]
    runtime_training = protocol["execution"]["training_runtime"]
    training["torch_threads"] = runtime_training["torch_threads"]
    training["export"] = runtime_training["export"]
    if training["target"] != "deterministic_teacher_action_mean":
        raise ValueError("principal training requires deterministic teacher means")
    if training["loss"] != "action_mean_squared_error_on_every_valid_step":
        raise ValueError("principal training must supervise every valid step")
    if training["episode_weighting"] != (
        "each_complete_episode_contributes_equal_total_weight"
    ):
        raise ValueError("principal training must weight complete episodes equally")
    if int(training["maximum_episode_steps"]) % int(training["sequence_length"]):
        raise ValueError("maximum episode steps must divide into complete chunks")
    mapping = split_residue_mapping(protocol)
    matched_seeds = protocol["matched_seeds"]["training"]
    if training["initial_collector_stage"]["seeds"] != matched_seeds or (
        training["final_principal_stage"]["seeds"] != matched_seeds
    ):
        raise ValueError("both training stages must use the matched seeds")
    if training["family_specific_hyperparameter_tuning"] != "forbidden":
        raise ValueError("principal training forbids family-specific tuning")
    training_residue_count = sum(value == "train" for value in mapping.values())
    modulus = len(mapping)
    batch_size = int(training["batch_size"])
    stage_episode_counts = (
        int(protocol["base_teacher_dataset"]["episodes"]),
        int(
            training["final_principal_stage"]["data_per_family"][
                "total_episodes"
            ]
        ),
    )
    for episode_count in stage_episode_counts:
        if episode_count % modulus:
            raise ValueError("stage episode count must divide by the split modulus")
        training_episodes = episode_count * training_residue_count // modulus
        if training_episodes % batch_size:
            raise ValueError("stage training split must divide into equal batches")
    shadow = protocol["shadow_labelling"]
    shadow_runtime = protocol["execution"]["shadow_runtime"]
    for name in ("distribution_config", "reward_config", "teacher_config"):
        if sha256_file(shadow_runtime[name]) != shadow_runtime[f"{name}_sha256"]:
            raise ValueError(f"principal {name} hash mismatch")
    shadow.update(shadow_runtime)
    if shadow["collector_training_seeds"] != protocol["matched_seeds"]["training"]:
        raise ValueError("shadow collectors must use the matched training seeds")
    if int(shadow["collectors_per_family"]) != len(
        shadow["collector_training_seeds"]
    ):
        raise ValueError("shadow collector count must match collector seeds")
    if int(shadow["episodes_per_family"]) != int(
        shadow["collectors_per_family"]
    ) * int(shadow["episodes_per_collector"]):
        raise ValueError("shadow partitions must cover the per-family budget")
    if shadow["loss_weighting"] != "equal_total_weight_per_complete_episode":
        raise ValueError("shadow training must use the common episode weighting")


def principal_family_spec(
    protocol: dict[str, Any], family_id: str
) -> dict[str, Any]:
    matches = [value for value in protocol["families"] if value["id"] == family_id]
    if len(matches) != 1:
        raise ValueError(f"unknown principal family {family_id!r}")
    return dict(matches[0])


def validate_training_seed(protocol: dict[str, Any], seed: int) -> None:
    if seed not in [int(value) for value in protocol["matched_seeds"]["training"]]:
        raise ValueError("training seed is not in the matched principal seed set")


def shadow_partition(protocol: dict[str, Any], seed: int) -> tuple[int, int]:
    """Return the inclusive collection offsets assigned to one matched seed."""

    validate_training_seed(protocol, seed)
    raw = protocol["shadow_labelling"]["collection_offsets_by_seed"][str(seed)]
    first, last = int(raw[0]), int(raw[1])
    expected = int(protocol["shadow_labelling"]["episodes_per_collector"])
    if last - first + 1 != expected:
        raise ValueError("shadow collector partition has the wrong size")
    return first, last


def shadow_schedule_records(
    protocol: dict[str, Any], seed: int
) -> list[dict[str, Any]]:
    first, last = shadow_partition(protocol, seed)
    shadow = protocol["shadow_labelling"]
    sampling_seed = int(shadow["collection_sampling_seed"])
    first_episode = int(protocol["shadow_labelling"]["aggregate_episode_indices"][0])
    distribution = load_direct_launch_distribution(shadow["distribution_config"])
    shots = distribution.generate(shadow["distribution_split"])
    records = []
    for offset in range(first, last + 1):
        reset_seed = sampling_seed + offset
        rng = np.random.default_rng(reset_seed)
        shot = shots[int(rng.integers(len(shots)))]
        blackout_steps = int(
            rng.integers(
                int(shadow["minimum_blackout_steps"]),
                int(shadow["maximum_blackout_steps"]) + 1,
            )
        )
        records.append(
            {
                "collection_offset": offset,
                "episode_index": first_episode + offset,
                "reset_seed": reset_seed,
                "shot_id": shot.shot.shot_id,
                "blackout_steps": blackout_steps,
            }
        )
    return records


def shadow_schedule_sha256(protocol: dict[str, Any], seed: int) -> str:
    return _json_sha256(shadow_schedule_records(protocol, seed))


def complete_shadow_schedule_sha256(protocol: dict[str, Any]) -> str:
    records = []
    for seed in protocol["shadow_labelling"]["collector_training_seeds"]:
        records.extend(shadow_schedule_records(protocol, int(seed)))
    records.sort(key=lambda value: value["collection_offset"])
    return _json_sha256(records)


def validate_shadow_partition_shards(
    protocol: dict[str, Any],
    seed: int,
    directory: Path,
    manifest: dict[str, Any],
) -> dict[str, int]:
    """Re-open one shadow partition and validate its realised budget.

    This deliberately checks the data rather than trusting the collector
    manifest.  In addition to the paired schedule, it verifies that every
    labelled transition has a deterministic teacher mean and that the
    recurrent previous-action input is the behaviour policy's prior command.
    """

    expected = shadow_schedule_records(protocol, seed)
    observed: list[dict[str, Any]] = []
    transition_count = 0
    for entry in manifest.get("shards", []):
        raw_path = Path(entry["file"])
        path = raw_path if raw_path.is_absolute() else directory / raw_path
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"shadow shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            if int(shard["dataset_schema_version"]) != 2:
                raise ValueError("unsupported shadow shard schema")
            if str(shard["teacher_action_semantics"]) != (
                DETERMINISTIC_ACTION_SEMANTICS
            ):
                raise ValueError("shadow shard does not contain teacher means")
            observations = np.asarray(shard["observations"])
            previous_actions = np.asarray(shard["previous_actions"])
            teacher_actions = np.asarray(shard["teacher_actions"])
            behaviour_actions = np.asarray(shard["behaviour_actions"])
            rewards = np.asarray(shard["rewards"])
            terminals = np.asarray(shard["terminals"])
            puck_visible = np.asarray(shard["puck_visible"])
            transition_count += len(observations)
            if observations.shape != (len(observations), 19):
                raise ValueError("shadow observations do not have shape (steps, 19)")
            if observations.dtype != np.float32:
                raise ValueError("shadow observations are not float32")
            for name, values in (
                ("previous actions", previous_actions),
                ("teacher actions", teacher_actions),
                ("behaviour actions", behaviour_actions),
            ):
                if values.shape != (len(observations), 2):
                    raise ValueError(f"shadow {name} do not have shape (steps, 2)")
                if values.dtype != np.float32:
                    raise ValueError(f"shadow {name} are not float32")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"shadow {name} contain non-finite values")
            if rewards.shape != (len(observations),) or rewards.dtype != np.float32:
                raise ValueError("shadow rewards are not a float32 step vector")
            if terminals.shape != (len(observations),) or terminals.dtype != np.bool_:
                raise ValueError("shadow terminals are not a boolean step vector")
            if puck_visible.shape != (len(observations),) or (
                puck_visible.dtype != np.bool_
            ):
                raise ValueError("shadow visibility is not a boolean step vector")
            if not np.all(np.isfinite(rewards)):
                raise ValueError("shadow rewards contain non-finite values")
            if not np.all(np.isfinite(observations)):
                raise ValueError("shadow observations contain non-finite values")
            if np.any(np.abs(teacher_actions) > 1.0):
                raise ValueError("shadow teacher actions exceed the public range")

            offsets = np.asarray(shard["episode_offsets"], dtype=np.int64)
            episode_indices = np.asarray(shard["episode_indices"])
            if len(offsets) != len(episode_indices) + 1 or (
                len(offsets) and (int(offsets[0]) != 0 or int(offsets[-1]) != len(observations))
            ):
                raise ValueError("shadow episode offsets do not cover the shard")
            if np.any(np.diff(offsets) <= 0):
                raise ValueError("shadow partition contains an empty episode")
            for local_index in range(len(episode_indices)):
                first = int(offsets[local_index])
                last = int(offsets[local_index + 1])
                if not np.array_equal(previous_actions[first], np.zeros(2)):
                    raise ValueError("shadow previous action does not reset at episode start")
                if not np.array_equal(
                    previous_actions[first + 1 : last],
                    behaviour_actions[first : last - 1],
                ):
                    raise ValueError("shadow previous actions are not behaviour commands")
                if not bool(terminals[last - 1]) or bool(np.any(terminals[first : last - 1])):
                    raise ValueError("shadow terminal flags do not delimit the episode")

            for episode_index, offset, reset_seed, shot_id, blackout_steps in zip(
                episode_indices,
                shard["episode_collection_offsets"],
                shard["episode_reset_seeds"],
                shard["episode_shot_ids"],
                shard["episode_blackout_steps"],
                strict=True,
            ):
                observed.append(
                    {
                        "episode_index": int(episode_index),
                        "collection_offset": int(offset),
                        "reset_seed": int(reset_seed),
                        "shot_id": str(shot_id),
                        "blackout_steps": int(blackout_steps),
                    }
                )
    observed.sort(key=lambda value: value["collection_offset"])
    if observed != expected:
        raise ValueError("shadow shard contents do not match the paired schedule")
    if transition_count != int(manifest.get("transition_count", -1)):
        raise ValueError("shadow transition count disagrees with its manifest")
    return {
        "episode_count": len(observed),
        "teacher_query_count": transition_count,
    }


def evaluation_schedule(
    protocol: dict[str, Any], split_name: str = "validation"
) -> tuple[tuple[Any, int], ...]:
    """Return the canonical shot-major, blackout-minor paired schedule."""

    if split_name != "validation":
        raise ValueError("principal test remains closed; only validation is available")
    specification = protocol["evaluation"]["validation"]
    distribution = load_direct_launch_distribution(
        specification["distribution_config"]
    )
    shots = distribution.generate(specification["split"])
    if len(shots) != int(specification["expected_shots"]):
        raise ValueError("validation distribution does not match expected shot count")
    blackouts = tuple(int(value) for value in specification["blackout_steps"])
    schedule = tuple((shot, blackout) for shot in shots for blackout in blackouts)
    if len(schedule) != int(specification["expected_episodes_per_student_seed"]):
        raise ValueError("validation schedule has the wrong episode count")
    return schedule


def evaluation_schedule_sha256(
    protocol: dict[str, Any], split_name: str = "validation"
) -> str:
    records = [
        {
            "shot_id": generated.shot.shot_id,
            "blackout_steps": blackout,
        }
        for generated, blackout in evaluation_schedule(protocol, split_name)
    ]
    return _json_sha256(records)


def split_residue_mapping(protocol: dict[str, Any]) -> dict[int, str]:
    split = protocol["training"]["episode_split"]
    modulus = int(split["modulus"])
    groups = {
        "train": split["training_residues"],
        "validation": split["validation_residues"],
        "internal_test": split["internal_test_residues"],
    }
    mapping: dict[int, str] = {}
    for name, values in groups.items():
        for raw_value in values:
            value = int(raw_value)
            if value < 0 or value >= modulus or value in mapping:
                raise ValueError("episode split residues must be disjoint and valid")
            mapping[value] = name
    if set(mapping) != set(range(modulus)):
        raise ValueError("episode split residues must cover the modulus")
    return mapping


def validate_principal_dataset_manifest(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    *,
    family_id: str,
    stage: str,
) -> None:
    principal_family_spec(protocol, family_id)
    if manifest.get("status") != "completed":
        raise ValueError("principal dataset is not complete")
    if manifest.get("teacher_action_semantics") != DETERMINISTIC_ACTION_SEMANTICS:
        raise ValueError("principal dataset does not contain deterministic means")
    if not bool(manifest.get("deterministic_inference")):
        raise ValueError("principal dataset inference is not deterministic")
    base = protocol["base_teacher_dataset"]
    if stage == "collector":
        if manifest_sha256 != base["manifest_sha256"]:
            raise ValueError("collector dataset is not the frozen common base data")
        if manifest.get("dataset_id") != base["id"]:
            raise ValueError("collector dataset id does not match the common base")
        if int(manifest.get("episode_count", -1)) != int(base["episodes"]):
            raise ValueError("collector dataset episode count mismatch")
    elif stage == "final":
        expected = protocol["training"]["final_principal_stage"]["data_per_family"]
        if int(manifest.get("episode_count", -1)) != int(expected["total_episodes"]):
            raise ValueError("final family dataset episode count mismatch")
        if manifest.get("family_id") != family_id:
            raise ValueError("final dataset belongs to a different family")
        if manifest.get("protocol_sha256") != protocol.get("resolved_sha256"):
            raise ValueError("final dataset protocol hash mismatch")
        if manifest.get("source_manifests", {}).get("teacher_controlled") != (
            base["manifest_sha256"]
        ):
            raise ValueError("final dataset does not use the frozen common base")
        if manifest.get("shadow_schedule_sha256") != (
            complete_shadow_schedule_sha256(protocol)
        ):
            raise ValueError("final dataset does not use the common shadow schedule")
        expected_sources = {"teacher_controlled"} | {
            f"shadow_seed_{int(seed)}"
            for seed in protocol["matched_seeds"]["training"]
        }
        if set(manifest.get("source_manifests", {})) != expected_sources:
            raise ValueError("final dataset does not contain all five shadow partitions")
    else:
        raise ValueError("training stage must be collector or final")


def load_principal_episode_splits(
    dataset_directory: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, PrincipalEpisodeSplit]:
    mapping = split_residue_mapping(protocol)
    maximum_steps = int(protocol["training"]["maximum_episode_steps"])
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
    for entry in manifest["shards"]:
        raw_path = Path(entry["file"])
        shard_path = raw_path if raw_path.is_absolute() else dataset_directory / raw_path
        if sha256_file(shard_path) != entry["sha256"]:
            raise ValueError(f"dataset shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            if int(shard["dataset_schema_version"]) != 2:
                raise ValueError("unsupported principal dataset schema")
            if str(shard["teacher_action_semantics"]) != (
                DETERMINISTIC_ACTION_SEMANTICS
            ):
                raise ValueError("dataset shard does not contain teacher means")
            offsets = shard["episode_offsets"]
            indices = shard["episode_indices"]
            loaded = {name: np.asarray(shard[name]) for name in fields}
            for local_index, raw_episode_index in enumerate(indices):
                episode_index = int(raw_episode_index)
                selection = slice(
                    int(offsets[local_index]), int(offsets[local_index + 1])
                )
                episode = {
                    name: np.ascontiguousarray(value[selection])
                    for name, value in loaded.items()
                }
                validate_principal_episode(episode, maximum_steps)
                episode["episode_index"] = episode_index
                split_name = mapping[episode_index % len(mapping)]
                collected[split_name].append(episode)
    splits = {
        name: pad_principal_episode_split(episodes, maximum_steps)
        for name, episodes in collected.items()
    }
    if sum(value.episode_count for value in splits.values()) != int(
        manifest["episode_count"]
    ):
        raise ValueError("principal episode splits do not cover the dataset")
    return splits


def pad_principal_episode_split(
    episodes: list[dict[str, Any]], maximum_episode_steps: int
) -> PrincipalEpisodeSplit:
    if not episodes:
        raise ValueError("principal dataset split must contain episodes")
    count = len(episodes)
    observations = np.zeros((count, maximum_episode_steps, 19), dtype=np.float32)
    previous_actions = np.zeros((count, maximum_episode_steps, 2), dtype=np.float32)
    teacher_actions = np.zeros((count, maximum_episode_steps, 2), dtype=np.float32)
    puck_visible = np.zeros((count, maximum_episode_steps), dtype=bool)
    valid_mask = np.zeros((count, maximum_episode_steps), dtype=bool)
    indices = np.empty(count, dtype=np.int64)
    lengths = np.empty(count, dtype=np.int16)
    for row, episode in enumerate(episodes):
        length = len(episode["observations"])
        observations[row, :length] = episode["observations"]
        previous_actions[row, :length] = episode["previous_actions"]
        teacher_actions[row, :length] = episode["teacher_actions"]
        puck_visible[row, :length] = episode["puck_visible"]
        valid_mask[row, :length] = True
        indices[row] = int(episode["episode_index"])
        lengths[row] = length
    return PrincipalEpisodeSplit(
        observations=observations,
        previous_actions=previous_actions,
        teacher_actions=teacher_actions,
        puck_visible=puck_visible,
        valid_mask=valid_mask,
        episode_indices=indices,
        episode_lengths=lengths,
    )


def validate_principal_episode(
    episode: dict[str, Any], maximum_episode_steps: int
) -> None:
    length = len(episode["observations"])
    if not 0 < length <= maximum_episode_steps:
        raise ValueError("episode length lies outside the principal bound")
    if episode["observations"].shape != (length, 19):
        raise ValueError("episode observations must have shape (time, 19)")
    for name in ("previous_actions", "teacher_actions"):
        if episode[name].shape != (length, 2):
            raise ValueError(f"episode {name} must have shape (time, 2)")
    if episode["puck_visible"].shape != (length,):
        raise ValueError("episode visibility must have shape (time,)")
    if not all(
        np.all(np.isfinite(episode[name]))
        for name in ("observations", "previous_actions", "teacher_actions")
    ):
        raise ValueError("episode contains non-finite public values")
    if not np.array_equal(episode["previous_actions"][0], np.zeros(2)):
        raise ValueError("previous action must reset at the episode boundary")
    if np.any(np.abs(episode["teacher_actions"]) > 1.0):
        raise ValueError("teacher actions must lie in the public action range")


def episode_index_sha256(split: PrincipalEpisodeSplit) -> str:
    return hashlib.sha256(split.episode_indices.tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
