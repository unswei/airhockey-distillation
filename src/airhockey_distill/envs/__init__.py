"""Tracking-loss air-hockey environments and policy interfaces."""

from .defend_shot import DefendShotTrackingLoss, MujocoDirectLaunchBackend
from .outcomes import ContactAwareOutcomeTracker, OutcomeThresholds
from .policy_interface import PlanarActionAdapter, PublicObservationAdapter
from .shot import DEFAULT_DIRECT_LAUNCH_SHOT, ShotSpec
from .shot_distribution import (
    DirectLaunchDistribution,
    GeneratedShot,
    load_direct_launch_distribution,
    manifest_sha256,
    summarise_distribution,
)
from .tracking_loss import BlackoutSchedule
from .training import DirectLaunchTrainingEnv

__all__ = [
    "DEFAULT_DIRECT_LAUNCH_SHOT",
    "BlackoutSchedule",
    "ContactAwareOutcomeTracker",
    "DefendShotTrackingLoss",
    "DirectLaunchDistribution",
    "DirectLaunchTrainingEnv",
    "GeneratedShot",
    "MujocoDirectLaunchBackend",
    "OutcomeThresholds",
    "PlanarActionAdapter",
    "PublicObservationAdapter",
    "ShotSpec",
    "load_direct_launch_distribution",
    "manifest_sha256",
    "summarise_distribution",
]
