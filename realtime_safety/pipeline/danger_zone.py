from __future__ import annotations

import numpy as np

from realtime_safety.config import SafetyConfig
from realtime_safety.types import DangerZone, SafetyLevel, Track3DState


class DangerZonePredictor:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def predict(self, tracks: list[Track3DState]) -> list[DangerZone]:
        return [self._predict_one(track) for track in tracks]

    def _predict_one(self, track: Track3DState) -> DangerZone:
        dynamic = track.motion_state == "dynamic"
        if dynamic:
            times = np.arange(
                0.0,
                self.config.prediction_horizon + self.config.prediction_timestep * 0.5,
                self.config.prediction_timestep,
                dtype=np.float32,
            )
        else:
            times = np.array([0.0], dtype=np.float32)
        positions = (
            track.position_xyz[None]
            + times[:, None] * track.velocity_xyz[None]
            + 0.5 * times[:, None] ** 2 * track.acceleration_xyz[None]
        ).astype(np.float32)
        position_covariance = np.asarray(track.covariance[:3, :3], dtype=np.float64)
        base_sigma = float(np.sqrt(max(np.linalg.eigvalsh(position_covariance).max(), 0.0)))
        sigmas = base_sigma + times * np.sqrt(max(float(np.trace(track.covariance[3:, 3:])), 0.0) / 3.0)
        radii = (
            track.radius
            + self.config.robot_radius
            + self.config.safety_margin
            + self.config.uncertainty_gain * sigmas
        ).astype(np.float32)
        planar_distance = np.linalg.norm(positions[:, :2], axis=1)
        clearances = planar_distance - radii
        closest_index = int(np.argmin(clearances))
        closest = float(clearances[closest_index])
        ttc = float(times[closest_index]) if dynamic and closest <= 0.0 and closest_index > 0 else None
        speed = float(np.linalg.norm(track.velocity_xyz))
        direction = track.velocity_xyz / speed if speed > 1e-6 else np.zeros(3, dtype=np.float32)
        risk_score = float(np.clip(1.0 - max(closest, 0.0) / max(self.config.warning_clearance * 2.0, 1e-6), 0.0, 1.0))
        if closest <= self.config.stop_clearance or (ttc is not None and ttc < self.config.stop_ttc):
            level = SafetyLevel.STOP
        elif closest <= self.config.warning_clearance or (ttc is not None and ttc < self.config.warning_ttc):
            level = SafetyLevel.WARNING
        elif dynamic and np.dot(track.position_xyz[:2], track.velocity_xyz[:2]) < 0:
            level = SafetyLevel.CAUTION
        else:
            level = SafetyLevel.SAFE
        return DangerZone(
            track_id=track.track_id,
            predicted_positions=positions,
            radii=radii,
            predicted_direction=direction.astype(np.float32),
            predicted_speed=speed,
            risk_score=risk_score,
            closest_predicted_distance=closest,
            ttc=ttc,
            risk_level=level,
            dynamic=dynamic,
        )
