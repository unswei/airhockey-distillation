import numpy as np
import pytest

from airhockey_distill.envs.policy_interface import (
    END_EFFECTOR_XY_SLICE,
    PUCK_POSITION_XY_SLICE,
    PUCK_VISIBLE_INDEX,
    PlanarActionAdapter,
    PublicObservationAdapter,
    denormalise_planar_position,
    normalise_planar_position,
)


def upstream_observation() -> np.ndarray:
    return np.linspace(-0.9, 0.9, 20, dtype=np.float32)


def test_visible_observation_keeps_position_and_removes_velocity() -> None:
    adapter = PublicObservationAdapter()
    upstream = upstream_observation()

    public = adapter.adapt(upstream, puck_visible=True)

    assert public.shape == (19,)
    np.testing.assert_array_equal(public[:18], upstream[:18])
    assert public[PUCK_VISIBLE_INDEX] == 1.0
    assert upstream[18] not in public[PUCK_POSITION_XY_SLICE]


def test_blackout_zeroes_only_puck_position_and_visibility() -> None:
    adapter = PublicObservationAdapter()
    upstream = upstream_observation()

    visible = adapter.adapt(upstream, puck_visible=True)
    hidden = adapter.adapt(upstream, puck_visible=False)

    np.testing.assert_array_equal(hidden[:16], visible[:16])
    np.testing.assert_array_equal(hidden[PUCK_POSITION_XY_SLICE], np.zeros(2))
    assert hidden[PUCK_VISIBLE_INDEX] == 0.0


def test_public_observation_is_independent_of_privileged_puck_velocity() -> None:
    adapter = PublicObservationAdapter()
    first = upstream_observation()
    second = first.copy()
    second[18:20] = (-91.0, 37.0)

    np.testing.assert_array_equal(
        adapter.adapt(first, puck_visible=True),
        adapter.adapt(second, puck_visible=True),
    )


def test_public_observation_clips_normalisation_overshoot() -> None:
    adapter = PublicObservationAdapter()
    upstream = upstream_observation()
    upstream[3] = -1.1
    upstream[10] = 1.2

    public = adapter.adapt(upstream, puck_visible=True)

    assert public[3] == -1.0
    assert public[10] == 1.0


def test_action_adapter_fixes_impedance_at_mid_range() -> None:
    full = PlanarActionAdapter().adapt((0.25, -0.5))

    np.testing.assert_array_equal(full, (0.25, -0.5, 0.0, 0.0, 0.0, 0.0))


def test_planar_position_round_trip() -> None:
    workspace = np.asarray(((0.5, 1.0), (-0.4, 0.4)), dtype=np.float32)
    position = np.asarray((0.75, 0.1), dtype=np.float32)

    recovered = denormalise_planar_position(
        normalise_planar_position(position, workspace), workspace
    )

    np.testing.assert_allclose(recovered, position)


def test_adapters_reject_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        PublicObservationAdapter().adapt(np.zeros(19), puck_visible=True)
    with pytest.raises(ValueError):
        PlanarActionAdapter().adapt(np.zeros(3))


def test_end_effector_slice_is_separate_from_puck_position() -> None:
    assert END_EFFECTOR_XY_SLICE.stop == PUCK_POSITION_XY_SLICE.start
