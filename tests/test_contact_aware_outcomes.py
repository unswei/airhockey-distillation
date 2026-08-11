import numpy as np

from airhockey_distill.envs import ContactAwareOutcomeTracker
from airhockey_distill.envs.defend_shot import (
    BackendSnapshot,
    DefendShotTrackingLoss,
    PrivilegedState,
    TableGeometry,
)
from airhockey_distill.envs.shot import DEFAULT_DIRECT_LAUNCH_SHOT, ShotSpec

GEOMETRY = TableGeometry(length=1.948, width=1.038, goal_width=0.25)


def puck_state(
    x: float,
    y: float,
    vx: float,
    vy: float,
    *,
    step: int = 1,
) -> PrivilegedState:
    return PrivilegedState(
        observation_step=step,
        puck_position_table_xy=(x, y),
        puck_velocity_table_xy=(vx, vy),
        puck_position_robot_xy=(x + 1.51, y),
        puck_velocity_robot_xy=(vx, vy),
    )


def observe(
    tracker: ContactAwareOutcomeTracker,
    state: PrivilegedState,
    *,
    contact: bool = False,
    terminal: bool = False,
) -> str | None:
    return tracker.observe(
        state=state,
        geometry=GEOMETRY,
        observation_step=state.observation_step,
        puck_mallet_contact=contact,
        upstream_terminal=terminal,
    )


def test_goal_concession_takes_precedence_over_late_contact() -> None:
    tracker = ContactAwareOutcomeTracker()

    outcome = observe(
        tracker,
        puck_state(-1.0, 0.0, -0.4, 0.0),
        contact=True,
        terminal=True,
    )

    assert outcome == "goal_conceded"


def test_return_requires_contact_and_crossing_the_return_plane() -> None:
    tracker = ContactAwareOutcomeTracker()

    assert (
        observe(
            tracker,
            puck_state(-0.7, 0.05, -0.3, 0.0, step=10),
            contact=True,
        )
        is None
    )
    outcome = observe(
        tracker,
        puck_state(-0.54, 0.05, 0.3, 0.0, step=14),
    )

    assert outcome == "returned"
    assert tracker.first_contact_step == 10


def test_arrest_requires_contact_and_confirmed_low_speed() -> None:
    tracker = ContactAwareOutcomeTracker()

    assert (
        observe(
            tracker,
            puck_state(-0.7, 0.0, 0.04, 0.0, step=20),
            contact=True,
        )
        is None
    )
    assert (
        observe(
            tracker,
            puck_state(-0.69, 0.0, 0.03, 0.0, step=21),
        )
        is None
    )
    outcome = observe(
        tracker,
        puck_state(-0.69, 0.0, 0.02, 0.0, step=22),
    )

    assert outcome == "arrested"


def test_post_contact_non_goal_exit_is_safe_deflection() -> None:
    tracker = ContactAwareOutcomeTracker()
    assert (
        observe(
            tracker,
            puck_state(-0.75, 0.2, -0.5, 0.1),
            contact=True,
        )
        is None
    )

    outcome = observe(
        tracker,
        puck_state(-1.0, 0.2, -0.4, 0.1, step=2),
        terminal=True,
    )

    assert outcome == "safe_deflection"


def test_non_goal_exit_without_contact_is_not_a_save() -> None:
    tracker = ContactAwareOutcomeTracker()

    outcome = observe(
        tracker,
        puck_state(-1.0, 0.2, -0.4, 0.1),
        terminal=True,
    )

    assert outcome == "missed_goal_without_contact"


def test_timeouts_distinguish_contact_from_no_contact() -> None:
    untouched = ContactAwareOutcomeTracker()
    contacted = ContactAwareOutcomeTracker()
    observe(
        contacted,
        puck_state(-0.7, 0.0, -0.3, 0.0),
        contact=True,
    )

    assert untouched.timeout_outcome() == "timeout_without_contact"
    assert contacted.timeout_outcome() == "timeout_after_contact"


class ScriptedBackend:
    ee_workspace_xy = np.asarray(
        ((0.59665, 1.31), (-0.46585, 0.46585)), dtype=np.float32
    )
    table_geometry = GEOMETRY

    def __init__(self, steps: tuple[BackendSnapshot, ...]) -> None:
        self.steps = steps
        self.index = 0

    def reset(self, shot: ShotSpec) -> BackendSnapshot:
        del shot
        self.index = 0
        return BackendSnapshot(
            upstream_policy_observation=np.zeros(20, dtype=np.float32),
            privileged_state=puck_state(0.55, 0.0, -1.6, 0.0, step=0),
        )

    def step(self, upstream_action: np.ndarray) -> BackendSnapshot:
        assert upstream_action.shape == (6,)
        snapshot = self.steps[self.index]
        self.index += 1
        return snapshot

    def close(self) -> None:
        pass


def snapshot(
    state: PrivilegedState,
    *,
    contact: bool = False,
) -> BackendSnapshot:
    return BackendSnapshot(
        upstream_policy_observation=np.zeros(20, dtype=np.float32),
        privileged_state=state,
        puck_mallet_contact=contact,
    )


def test_environment_terminates_early_on_confirmed_return() -> None:
    backend = ScriptedBackend(
        (
            snapshot(
                puck_state(-0.7, 0.0, -0.3, 0.0, step=1),
                contact=True,
            ),
            snapshot(puck_state(-0.54, 0.0, 0.3, 0.0, step=2)),
        )
    )
    environment = DefendShotTrackingLoss(backend, timeout_steps=10)
    environment.reset(shot=DEFAULT_DIRECT_LAUNCH_SHOT)

    _, _, first_terminated, first_truncated, first_info = environment.step((0, 0))
    _, _, terminated, truncated, info = environment.step((0, 0))

    assert not first_terminated
    assert not first_truncated
    assert "outcome" not in first_info
    assert terminated
    assert not truncated
    assert info["outcome"] == "returned"


def test_environment_reports_bare_timeout_separately() -> None:
    backend = ScriptedBackend(
        (
            snapshot(puck_state(-0.2, 0.0, -0.3, 0.0, step=1)),
            snapshot(puck_state(-0.3, 0.0, -0.3, 0.0, step=2)),
        )
    )
    environment = DefendShotTrackingLoss(backend, timeout_steps=2)
    environment.reset(shot=DEFAULT_DIRECT_LAUNCH_SHOT)
    environment.step((0, 0))

    _, _, terminated, truncated, info = environment.step((0, 0))

    assert not terminated
    assert truncated
    assert info["outcome"] == "timeout_without_contact"
