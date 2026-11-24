from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from realtime_safety.config import TrackingConfig
from realtime_safety.types import BBox3D, ObstacleObservation3D, Track3DState


@dataclass(slots=True)
class _Filter:
    observation: ObstacleObservation3D
    x: np.ndarray
    covariance: np.ndarray
    bbox_minimum: np.ndarray
    bbox_maximum: np.ndarray
    radius: float
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    hits: int = 1
    missing: int = 0
    dynamic_hits: int = 0
    motion_state: str = "static"
    history: list[np.ndarray] = field(default_factory=list)
    filter_timestamp: float = 0.0


class Tracker3D:
    """Constant-velocity Kalman filters keyed primarily by stable 2D Track ID."""

    def __init__(self, config: TrackingConfig, max_history: int = 30) -> None:
        self.config = config
        self.max_history = max_history
        self._filters: dict[int, _Filter] = {}

    def reset(self) -> None:
        self._filters.clear()

    def update(self, observations: list[ObstacleObservation3D], timestamp: float) -> list[Track3DState]:
        observed_ids: set[int] = set()
        for observation in observations:
            track_id = self._associate_id(observation)
            observed_ids.add(track_id)
            if track_id not in self._filters:
                x = np.r_[observation.position_xyz, np.zeros(3)].astype(np.float64)
                self._filters[track_id] = _Filter(
                    observation=observation,
                    x=x,
                    covariance=np.eye(6, dtype=np.float64) * 0.5,
                    bbox_minimum=observation.bbox3d.minimum.copy(),
                    bbox_maximum=observation.bbox3d.maximum.copy(),
                    radius=observation.radius,
                    history=[observation.position_xyz.copy()],
                    filter_timestamp=observation.timestamp,
                )
                continue
            track = self._filters[track_id]
            dt = max(float(observation.timestamp - track.observation.timestamp), 1e-3)
            if observation.class_name == "person" and track.hits >= 2:
                jump = float(np.linalg.norm(observation.position_xyz - track.observation.position_xyz))
                jump_gate = max(
                    float(self.config.obstacle_center_max_step_m),
                    1.25 * max(observation.radius, track.observation.radius),
                )
                if jump > jump_gate:
                    # Limit a one-frame monocular-depth jump. Re-anchoring the
                    # filter made the GUI and downstream obstacle center flash.
                    direction = (
                        observation.position_xyz - track.observation.position_xyz
                    )
                    limited_position = (
                        track.observation.position_xyz
                        + direction * (jump_gate / jump)
                    ).astype(np.float32)
                    correction = limited_position - observation.position_xyz
                    points = (
                        None
                        if observation.points is None
                        else np.asarray(observation.points, dtype=np.float32)
                        + correction
                    )
                    observation = ObstacleObservation3D(
                        track_id=observation.track_id,
                        class_name=observation.class_name,
                        confidence=observation.confidence,
                        position_xyz=limited_position,
                        bbox3d=BBox3D(
                            minimum=observation.bbox3d.minimum + correction,
                            maximum=observation.bbox3d.maximum + correction,
                        ),
                        radius=observation.radius,
                        point_count=observation.point_count,
                        timestamp=observation.timestamp,
                        points=points,
                    )
            previous_velocity = track.x[3:].copy()
            measured_velocity = (observation.position_xyz - track.observation.position_xyz) / dt
            prediction_dt = max(observation.timestamp - track.filter_timestamp, 0.0)
            self._predict(track, prediction_dt)
            track.filter_timestamp = observation.timestamp
            self._correct(track, observation.position_xyz)
            # Velocity is explicitly timestamp-derived, then noise-smoothed with the KF state.
            gain = 0.65 if track.hits < 3 else 0.35
            track.x[3:] = (1.0 - gain) * track.x[3:] + gain * measured_velocity
            measured_acceleration = (track.x[3:] - previous_velocity) / dt
            track.acceleration = 0.8 * track.acceleration + 0.2 * measured_acceleration
            bbox_alpha = float(self.config.bbox_smoothing_alpha)
            track.bbox_minimum = (
                (1.0 - bbox_alpha) * track.bbox_minimum
                + bbox_alpha * observation.bbox3d.minimum
            ).astype(np.float32)
            track.bbox_maximum = (
                (1.0 - bbox_alpha) * track.bbox_maximum
                + bbox_alpha * observation.bbox3d.maximum
            ).astype(np.float32)
            track.radius = (
                (1.0 - bbox_alpha) * track.radius
                + bbox_alpha * observation.radius
            )
            track.observation = observation
            track.hits += 1
            track.missing = 0
            track.history.append(track.x[:3].astype(np.float32).copy())
            del track.history[:-self.max_history]
            self._update_motion_state(track)
        for track_id in list(self._filters):
            if track_id in observed_ids:
                continue
            track = self._filters[track_id]
            dt = max(float(timestamp - track.filter_timestamp), 0.0)
            if dt > 0:
                self._predict(track, min(dt, 0.5))
                track.filter_timestamp = timestamp
            track.missing += 1
            if track.missing > self.config.max_missing:
                del self._filters[track_id]
        return [self._state(track_id, track) for track_id, track in self._filters.items()]

    def predict_to(self, timestamp: float) -> list[Track3DState]:
        states: list[Track3DState] = []
        for track_id, track in self._filters.items():
            dt = max(float(timestamp - track.filter_timestamp), 0.0)
            x = track.x.copy()
            x[:3] += x[3:] * dt
            state = self._state(track_id, track)
            state.position_xyz = x[:3].astype(np.float32)
            states.append(state)
        return states

    def _associate_id(self, observation: ObstacleObservation3D) -> int:
        if observation.track_id >= 0 or observation.track_id in self._filters:
            return observation.track_id
        candidates = [
            (np.linalg.norm(track.x[:3] - observation.position_xyz), track_id)
            for track_id, track in self._filters.items()
            if track.observation.class_name == observation.class_name and track_id < 0
        ]
        if candidates and min(candidates)[0] <= self.config.association_distance:
            return min(candidates)[1]
        return observation.track_id

    @staticmethod
    def _predict(track: _Filter, dt: float) -> None:
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        process = np.eye(6) * max(dt, 1e-3) * 0.08
        track.x = transition @ track.x
        track.covariance = transition @ track.covariance @ transition.T + process

    @staticmethod
    def _correct(track: _Filter, position: np.ndarray) -> None:
        observation_matrix = np.zeros((3, 6))
        observation_matrix[:, :3] = np.eye(3)
        measurement_noise = np.eye(3) * max(0.03, 1.0 - track.observation.confidence) ** 2
        residual = position - observation_matrix @ track.x
        innovation = observation_matrix @ track.covariance @ observation_matrix.T + measurement_noise
        gain = track.covariance @ observation_matrix.T @ np.linalg.inv(innovation)
        track.x += gain @ residual
        track.covariance = (np.eye(6) - gain @ observation_matrix) @ track.covariance

    def _update_motion_state(self, track: _Filter) -> None:
        speed = float(np.linalg.norm(track.x[3:,]))
        if track.motion_state == "static":
            track.dynamic_hits = track.dynamic_hits + 1 if speed >= self.config.dynamic_enter_speed else 0
            if track.dynamic_hits >= self.config.minimum_dynamic_hits:
                track.motion_state = "dynamic"
        elif speed <= self.config.dynamic_exit_speed:
            track.dynamic_hits = max(track.dynamic_hits - 1, 0)
            if track.dynamic_hits == 0:
                track.motion_state = "static"
        else:
            track.dynamic_hits = self.config.minimum_dynamic_hits

    @staticmethod
    def _state(track_id: int, track: _Filter) -> Track3DState:
        predicted_offset = (
            track.x[:3] - track.observation.position_xyz
        ).astype(np.float32)
        return Track3DState(
            track_id=track_id,
            class_name=track.observation.class_name,
            position_xyz=track.x[:3].astype(np.float32).copy(),
            velocity_xyz=track.x[3:].astype(np.float32).copy(),
            acceleration_xyz=track.acceleration.astype(np.float32).copy(),
            covariance=track.covariance.copy(),
            bbox3d=BBox3D(
                minimum=track.bbox_minimum + predicted_offset,
                maximum=track.bbox_maximum + predicted_offset,
            ),
            radius=track.radius,
            hit_count=track.hits,
            missing_count=track.missing,
            last_timestamp=track.observation.timestamp,
            motion_state=track.motion_state,
            confidence=track.observation.confidence,
            history=[point.copy() for point in track.history],
        )
