from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airhockey_distill.principal_sweep import (
    PrincipalEpisodeSplit,
    load_principal_protocol,
)
from airhockey_distill.students import (
    PRINCIPAL_FAMILY_IDS,
    load_principal_policy,
    principal_policy_from_parameters,
    save_principal_checkpoint,
)
from airhockey_distill.students.principal_torch import PrincipalStudentModule
from scripts.train_principal_student import (
    evaluate_principal_split,
    train_principal_epoch,
    verify_principal_export,
)


def _split(lengths=(7, 11), seed=81):
    rng = np.random.default_rng(seed)
    maximum = 16
    count = len(lengths)
    observations = np.zeros((count, maximum, 19), dtype=np.float32)
    previous_actions = np.zeros((count, maximum, 2), dtype=np.float32)
    teacher_actions = np.zeros((count, maximum, 2), dtype=np.float32)
    visible = np.zeros((count, maximum), dtype=bool)
    valid = np.zeros((count, maximum), dtype=bool)
    for index, length in enumerate(lengths):
        observations[index, :length] = rng.normal(size=(length, 19))
        previous_actions[index, 1:length] = rng.uniform(-1, 1, size=(length - 1, 2))
        teacher_actions[index, :length] = np.tanh(rng.normal(size=(length, 2)))
        visible[index, : min(5, length)] = True
        valid[index, :length] = True
    return PrincipalEpisodeSplit(
        observations=observations,
        previous_actions=previous_actions,
        teacher_actions=teacher_actions,
        puck_visible=visible,
        valid_mask=valid,
        episode_indices=np.arange(count, dtype=np.int64),
        episode_lengths=np.asarray(lengths, dtype=np.int16),
    )


@pytest.mark.parametrize("family_id", PRINCIPAL_FAMILY_IDS)
def test_all_seven_families_share_training_export_and_reload_path(
    family_id, tmp_path
):
    protocol = load_principal_protocol("configs/experiments/principal_sweep_execution_v1.yaml")
    expected = {
        value["id"]: value["expected_total_parameters"]
        for value in protocol["families"]
    }
    split = _split()
    module = PrincipalStudentModule(family_id, seed=14303)
    optimiser = torch.optim.AdamW(module.parameters(), lr=3e-4)

    training_mse = train_principal_epoch(
        module,
        optimiser,
        split,
        batch_size=2,
        sequence_length=8,
        gradient_clip=1.0,
        generator=torch.Generator().manual_seed(14303),
    )
    metrics = evaluate_principal_split(module, split, 2, 8)
    parameters = module.export_numpy_parameters()
    metadata = {
        "student_id": family_id,
        "training_seed": 14303,
        "training_stage": "collector",
    }
    exported = principal_policy_from_parameters(family_id, parameters, metadata)
    checkpoint = tmp_path / f"{family_id}.npz"
    save_principal_checkpoint(family_id, checkpoint, parameters, metadata)
    restored = load_principal_policy(family_id, checkpoint)
    action_error, carry_error, exact = verify_principal_export(
        module,
        exported,
        restored,
        split,
        episode_count=2,
    )

    assert np.isfinite(training_mse)
    assert np.isfinite(metrics["equal_episode_action_mse"])
    assert module.parameter_count == expected[family_id]
    assert exported.parameter_count == expected[family_id]
    assert action_error <= 2e-6
    assert carry_error <= 2e-6
    assert exact


def test_training_objective_weights_complete_episodes_equally():
    class ConstantModule(torch.nn.Module):
        family_id = "feed_forward"

        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward_sequence(self, observations, previous_actions, initial_carry=None):
            actions = self.value.expand(*observations.shape[:2], 2)
            carries = observations.new_zeros((*observations.shape[:2], 0))
            return actions, carries

    split = _split(lengths=(1, 3), seed=82)
    split.teacher_actions[0, 0] = 0.0
    split.teacher_actions[1, :3] = 2.0
    module = ConstantModule()
    optimiser = torch.optim.SGD(module.parameters(), lr=0.0)

    observed = train_principal_epoch(
        module,
        optimiser,
        split,
        batch_size=2,
        sequence_length=2,
        gradient_clip=100.0,
        generator=torch.Generator().manual_seed(82),
    )

    assert observed == pytest.approx(2.0)
    transition_weighted_mse = (0.0 * 1 + 4.0 * 3) / 4
    assert transition_weighted_mse == 3.0
    assert observed != transition_weighted_mse
