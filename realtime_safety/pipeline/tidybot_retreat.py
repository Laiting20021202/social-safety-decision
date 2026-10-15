"""Feedback-controlled, one-shot planar retreat for an isolated holonomic simulator.

Input points are in calibrated base coordinates (x forward, y left, z up).
This is a proximity reflex, not an object classifier or collision-free planner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RetreatConfig:
    trigger_distance_m: float = 1.0
    release_distance_m: float = 1.3
    travel_m: float = 2.0
    speed_mps: float = 0.15
    tolerance_m: float = 0.03
    cloud_timeout_s: float = 1.5
    pose_timeout_s: float = 0.5
    max_duration_s: float = 30.0
    stall_timeout_s: float = 3.0
    confirm_frames: int = 3
    min_points: int = 10
    min_valid_points: int = 30
    cell_m: float = 0.15
    min_height_m: float = 0.15
    max_height_m: float = 1.8
    robot_radius_m: float = 0.35
    max_cross_track_m: float = 0.25
    command_frame: str = "body"

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "command_frame":
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.command_frame not in ("body", "world"):
            raise ValueError("command_frame must be body or world")
        for name in ("confirm_frames", "min_points", "min_valid_points"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not self.robot_radius_m < self.trigger_distance_m < self.release_distance_m:
            raise ValueError("require robot radius < trigger < release")
        if self.min_height_m >= self.max_height_m or self.tolerance_m >= self.travel_m:
            raise ValueError("invalid height or travel limits")
        if self.speed_mps > 0.3 or self.travel_m > 2.0:
            raise ValueError("simulation demo limited to 0.3 m/s and 2 m")
        if self.max_duration_s <= self.travel_m / self.speed_mps:
            raise ValueError("max_duration_s must allow the requested travel")


def rotation2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def nearest_obstacle(points: np.ndarray, config: RetreatConfig) -> np.ndarray | None:
    """Require spatial support in an XY cell, rejecting isolated depth noise.

    Floor/ceiling and base-radius points are excluded. An arm or fixture inside
    this ROI can still trigger; this does not replace full robot self-filtering.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < config.min_valid_points:
        raise ValueError("insufficient valid cloud points")
    radius = np.linalg.norm(points[:, :2], axis=1)
    points = points[(points[:, 0] > 0) & (radius > config.robot_radius_m)
                    & (radius <= config.release_distance_m)
                    & (points[:, 2] >= config.min_height_m)
                    & (points[:, 2] <= config.max_height_m)]
    if not len(points):
        return None
    cells = np.floor(points[:, :2] / config.cell_m).astype(np.int64)
    _, ids, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    candidates = [np.median(points[ids == i, :2], axis=0)
                  for i in np.flatnonzero(counts >= config.min_points)]
    return min(candidates, key=np.linalg.norm) if candidates else None


class RetreatController:
    def __init__(self, config: RetreatConfig):
        self.config = config
        self.state = "WAITING"
        self.reason = "waiting for fresh pose and cloud"
        self.pose = None
        self.pose_at = self.cloud_at = None
        self.pose_stamp = self.cloud_stamp = None
        self.near_hits = 0
        self.obstacle = None
        self.origin = self.target = self.direction = None
        self.started_at = self.progress_at = None
        self.best_progress = 0.0

    def fault(self, reason: str):
        self.state = "FAULT"
        self.reason = reason
        self.near_hits = 0

    def update_pose(self, pose, stamp: float, now: float):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(stamp):
            self.pose_at = None
            self.fault("invalid base pose")
            return
        if self.pose_stamp is not None:
            if stamp < self.pose_stamp:
                self.pose_at = None
                self.fault("joint clock reset; restart after checking simulator")
                return
            if stamp == self.pose_stamp:
                return  # Replayed/frozen stamps must not refresh the watchdog.
        if self.pose is not None and self.state == "MOVING":
            dt = max(0.0, now - self.pose_at)
            jump = np.linalg.norm(pose[:2] - self.pose[:2])
            if jump > max(0.1, self.config.speed_mps * dt * 3):
                self.fault("base pose jump or competing motion")
                return
        self.pose, self.pose_stamp, self.pose_at = pose, stamp, now
        # Pose feedback is authoritative for completion. Mark DONE immediately
        # so a subsequently delivered, out-of-order sensor frame cannot replace
        # a successfully reached target with a perception fault.
        if (self.state == "MOVING" and self.target is not None
                and np.linalg.norm(self.target - pose[:2]) <= self.config.tolerance_m):
            self.state = "DONE"
            self.reason = "2 m target reached within tolerance; explicit reset required"

    def update_cloud(self, points, stamp: float, now: float):
        if not math.isfinite(stamp):
            self.cloud_at = None
            self.fault("invalid cloud timestamp")
            return
        if self.cloud_stamp is not None:
            if stamp < self.cloud_stamp:
                if self.state == "DONE":
                    return
                self.cloud_at = None
                self.fault("cloud clock reset")
                return
            if stamp == self.cloud_stamp:
                return
        if self.cloud_at is not None and now - self.cloud_at > self.config.cloud_timeout_s:
            self.near_hits = 0
        try:
            self.obstacle = nearest_obstacle(points, self.config)
        except ValueError as exc:
            self.cloud_at = None
            self.fault(str(exc))
            return
        self.cloud_stamp, self.cloud_at = stamp, now
        near = self.obstacle is not None and np.linalg.norm(self.obstacle) <= self.config.trigger_distance_m
        self.near_hits = self.near_hits + 1 if near else 0

    def reset(self, now: float) -> bool:
        # Explicit reset only, with valid clear space observed in front.
        if not self.fresh(now) or self.obstacle is not None:
            return False
        self.state, self.reason = "IDLE", "reset with fresh clear observations"
        self.origin = self.target = self.direction = None
        self.near_hits = 0
        return True

    def fresh(self, now: float) -> bool:
        return (self.pose_at is not None and self.cloud_at is not None
                and 0 <= now - self.pose_at <= self.config.pose_timeout_s
                and 0 <= now - self.cloud_at <= self.config.cloud_timeout_s)

    def tick(self, now: float) -> tuple[float, float]:
        zero = (0.0, 0.0)
        if self.state in ("FAULT", "DONE"):
            return zero
        if not self.fresh(now):
            if self.state == "MOVING":
                self.fault("stale point cloud or joint feedback")
            else:
                self.state, self.reason = "WAITING", "waiting for fresh pose and cloud"
                self.near_hits = 0
            return zero
        if self.state == "WAITING":
            self.state, self.reason = "IDLE", "waiting for confirmed proximity"
        if self.state == "IDLE":
            if self.near_hits < self.config.confirm_frames or self.obstacle is None:
                return zero
            direction_body = -self.obstacle / np.linalg.norm(self.obstacle)
            self.direction = rotation2(self.pose[2]) @ direction_body
            self.origin = self.pose[:2].copy()
            self.target = self.origin + self.config.travel_m * self.direction
            self.started_at = self.progress_at = now
            self.best_progress = 0.0
            self.state, self.reason = "MOVING", "retreating to latched world target"
        delta = self.pose[:2] - self.origin
        progress = float(delta @ self.direction)
        cross_track = float(np.linalg.norm(delta - progress * self.direction))
        if cross_track > self.config.max_cross_track_m or progress < -0.1:
            self.fault("unexpected movement direction")
            return zero
        if progress > self.config.travel_m + self.config.tolerance_m:
            self.fault("overshot retreat target")
            return zero
        remaining = self.target - self.pose[:2]
        distance = float(np.linalg.norm(remaining))
        if distance <= self.config.tolerance_m:
            self.state, self.reason = "DONE", "2 m target reached within tolerance; explicit reset required"
            return zero
        if progress > self.best_progress + 0.01:
            self.best_progress, self.progress_at = progress, now
        if now - self.started_at > self.config.max_duration_s or now - self.progress_at > self.config.stall_timeout_s:
            self.fault("motion timeout or no progress")
            return zero
        velocity = remaining / distance * min(self.config.speed_mps, distance * 0.8)
        if self.config.command_frame == "body":
            velocity = rotation2(-self.pose[2]) @ velocity
        return tuple(map(float, velocity))
