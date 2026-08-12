import hashlib
import json

import numpy as np

from scripts.audit_structured_teacher_dataset import audit_dataset


def test_audit_detects_locked_command_semantics_and_exact_repeats(tmp_path):
    dataset = tmp_path / "dataset"
    shards = dataset / "shards"
    shards.mkdir(parents=True)
    observations = np.zeros((6, 19), dtype=np.float32)
    observations[:, 14:16] = [0.25, -0.5]
    targets = np.asarray(
        [[0.75, 0.5], [0.1, 0.2], [-0.3, 0.4]] * 2,
        dtype=np.float32,
    )
    previous = np.concatenate((np.zeros((1, 2), dtype=np.float32), targets[:2]), axis=0)
    previous = np.concatenate((previous, previous), axis=0)
    shard_path = shards / "shard-0000.npz"
    np.savez_compressed(
        shard_path,
        observations=observations,
        previous_actions=previous,
        teacher_actions=targets,
        episode_offsets=np.asarray([0, 3, 6]),
        episode_shot_ids=np.asarray(["shot-a", "shot-a"]),
        episode_blackout_steps=np.asarray([0, 0]),
    )
    digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "test",
                "teacher_action_semantics": "deterministic",
                "shards": [{"file": "shards/shard-0000.npz", "sha256": digest}],
            }
        )
    )

    result = audit_dataset(dataset, action_lock_steps=1, code_commit="abc")

    previous_result = result["previous_action_semantics"]
    assert previous_result["stored_previous_equals_prior_teacher_command_all"] is True
    assert (
        previous_result["post_lock_previous_equals_prior_teacher_command_all"] is True
    )
    assert previous_result["locked_prior_command_vs_applied_hold_mse"] > 0
    target_result = result["deterministic_target_check"]
    assert target_result["repeated_shot_blackout_groups"] == 1
    assert target_result["repeated_groups_with_identical_full_inputs"] == 1
    assert target_result["repeated_groups_with_identical_targets"] == 1
    assert target_result["full_input_trajectories_with_conflicting_targets"] == 0
