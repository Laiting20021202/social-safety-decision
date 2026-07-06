from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from social_bev.types import Calibration, Track


def meters_to_bev_pixels(calibration: Calibration, meters: float) -> float:
    cfg = calibration.bev_config
    if not calibration.metric_bev:
        return meters
    x_scale = float(cfg["width_px"]) / max(1e-6, float(cfg["x_max_m"]) - float(cfg["x_min_m"]))
    y_scale = float(cfg["height_px"]) / max(1e-6, float(cfg["y_max_m"]) - float(cfg["y_min_m"]))
    return float(meters) * (x_scale + y_scale) * 0.5


def draw_social_zones(
    image: np.ndarray,
    occupancy_grid: np.ndarray,
    tracks: list[Track],
    calibration: Calibration,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw static or motion-elongated personal space around each person."""

    overlay = image.copy()
    zone_mask = np.zeros(occupancy_grid.shape[:2], dtype=np.uint8)
    for track in tracks:
        if track.bev_position is None:
            continue
        center = (int(round(track.bev_position[0])), int(round(track.bev_position[1])))
        if not _point_inside(center, occupancy_grid.shape):
            continue
        if calibration.metric_bev:
            static_r = max(3, int(round(meters_to_bev_pixels(calibration, float(config.get("static_radius_m", 0.8))))))
            front_r = max(static_r, int(round(meters_to_bev_pixels(calibration, float(config.get("front_radius_m", 1.2))))))
            rear_r = max(2, int(round(meters_to_bev_pixels(calibration, float(config.get("rear_radius_m", 0.6))))))
        else:
            static_r = int(config.get("normalized_static_radius_px", 45))
            front_r = int(config.get("normalized_front_radius_px", 70))
            rear_r = int(config.get("normalized_rear_radius_px", 35))
        direction = _trajectory_direction(track)
        if direction is None:
            cv2.circle(zone_mask, center, static_r, 255, thickness=-1)
            cv2.circle(overlay, center, static_r, (80, 210, 255), thickness=2)
        else:
            dx, dy = direction
            angle = float(np.degrees(np.arctan2(dy, dx)))
            speed = float(np.linalg.norm(direction))
            gain = float(config.get("speed_gain", 0.5))
            major = int(front_r + min(front_r, speed * gain))
            minor = max(3, static_r)
            shifted_center = (
                int(round(center[0] + dx / max(1.0, speed) * (major - rear_r) * 0.25)),
                int(round(center[1] + dy / max(1.0, speed) * (major - rear_r) * 0.25)),
            )
            cv2.ellipse(zone_mask, shifted_center, (major, minor), angle, 0, 360, 255, thickness=-1)
            cv2.ellipse(overlay, shifted_center, (major, minor), angle, 0, 360, (80, 210, 255), thickness=2)
    blended = cv2.addWeighted(image, 0.78, overlay, 0.22, 0.0)
    updated_grid = occupancy_grid.copy()
    updated_grid[(zone_mask > 0) & (updated_grid < 80)] = 50
    return blended, updated_grid


def _trajectory_direction(track: Track) -> tuple[float, float] | None:
    if len(track.trajectory) >= 2:
        x0, y0 = track.trajectory[-2]
        x1, y1 = track.trajectory[-1]
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        if dx * dx + dy * dy >= 9.0:
            return dx, dy
    vx, vy = track.velocity
    if vx * vx + vy * vy >= 9.0:
        return float(vx), float(vy)
    return None


def _point_inside(point: tuple[int, int], shape: tuple[int, ...]) -> bool:
    x, y = point
    h, w = shape[:2]
    return 0 <= x < w and 0 <= y < h

