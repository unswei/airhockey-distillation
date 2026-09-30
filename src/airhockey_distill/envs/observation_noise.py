"""Paired synthetic measurement noise; never changes simulator state."""

from hashlib import sha256

import numpy as np

from .policy_interface import PublicObservationAdapter, PUCK_POSITION_XY_SLICE


def standard_noise(unit: str, seed: int, observations: int) -> np.ndarray:
    """Key a complete trace by unit, independently of rollout length/order."""
    key = sha256(f"observation_noise_v1:{seed}:{unit}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], "little")))
    return rng.standard_normal((observations, 2))


class NoisyPuckObservationAdapter(PublicObservationAdapter):
    """Add physical-coordinate noise before clipping and visibility masking.

    The evaluator supplies the observation index explicitly: the environment
    may request the same observation more than once during the action lock.
    """

    def __init__(self, standard_normals, std_mm: float, puck_range_metres):
        self.normals = np.asarray(standard_normals, dtype=np.float64)
        ranges = np.asarray(puck_range_metres, dtype=np.float64)
        if self.normals.ndim != 2 or self.normals.shape[1] != 2 or not np.isfinite(self.normals).all():
            raise ValueError("noise trace must be a finite (observations, 2) array")
        if ranges.shape != (2,) or not np.isfinite(ranges).all() or np.any(ranges <= 0):
            raise ValueError("physical coordinate ranges must be finite and positive")
        if not np.isfinite(std_mm) or std_mm < 0:
            raise ValueError("noise standard deviation must be finite and non-negative")
        self.std_mm = std_mm
        self.scale = 2 * (std_mm / 1000) / ranges
        self.observation_step = 0

    def adapt(self, upstream_observation, *, puck_visible):
        # Preserve the exact clean arithmetic and never contaminate masking.
        if self.std_mm == 0 or not puck_visible:
            return super().adapt(upstream_observation, puck_visible=puck_visible)
        if not 0 <= self.observation_step < len(self.normals):
            raise ValueError("observation index outside the frozen noise trace")
        noisy = np.asarray(upstream_observation, dtype=np.float32).copy()
        noisy[PUCK_POSITION_XY_SLICE] += self.normals[self.observation_step] * self.scale
        return super().adapt(noisy, puck_visible=True)
