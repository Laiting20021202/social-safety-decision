from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from social_bev.homography import bev_robot_pixel, image_points_to_bev_pixels, image_to_bev_matrix
from social_bev.social_zone import draw_social_zones
from social_bev.types import BEVResult, Calibration, Detection, ObstacleRegion, Track


class BEVMapBuilder:
    """Build a BEV visualization and occupancy grid from RGB perception outputs."""

    def __init__(self, bev_config: dict[str, Any], social_zone_config: dict[str, Any]) -> None:
        self.bev_config = bev_config
        self.social_zone_config = social_zone_config

    def build(
        self,
        walkable_mask: np.ndarray,
        detections: list[Detection],
        unknown_obstacles: list[ObstacleRegion],
        tracks: list[Track],
        calibration: Calibration,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BEVResult:
        width = int(calibration.bev_config.get("width_px", self.bev_config.get("width_px", 600)))
        height = int(calibration.bev_config.get("height_px", self.bev_config.get("height_px", 800)))
        image = np.full((height, width, 3), (36, 36, 36), dtype=np.uint8)
        grid = np.full((height, width), -1, dtype=np.int16)
        layers: dict[str, np.ndarray] = {
            "free_space": np.zeros((height, width), dtype=np.uint8),
            "known_obstacle": np.zeros((height, width), dtype=np.uint8),
            "unknown_obstacle": np.zeros((height, width), dtype=np.uint8),
            "person": np.zeros((height, width), dtype=np.uint8),
            "social_zone": np.zeros((height, width), dtype=np.uint8),
            "robot": np.zeros((height, width), dtype=np.uint8),
        }

        self._draw_free_space(image, grid, layers, walkable_mask, calibration, frame_shape)
        self._draw_known_obstacles(image, grid, layers, detections, calibration)
        self._draw_unknown_obstacles(image, grid, layers, unknown_obstacles, calibration)
        image, grid = draw_social_zones(image, grid, tracks, calibration, self.social_zone_config)
        layers["social_zone"][grid == 50] = 1
        self._draw_people(image, grid, layers, tracks)
        self._draw_robot(image, grid, layers, calibration)
        self._draw_bev_annotations(image, calibration)
        return BEVResult(
            image=image,
            occupancy_grid=grid,
            metric_bev=calibration.metric_bev,
            label=calibration.label,
            layers=layers,
        )

    def _draw_free_space(
        self,
        image: np.ndarray,
        grid: np.ndarray,
        layers: dict[str, np.ndarray],
        walkable_mask: np.ndarray,
        calibration: Calibration,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        mask = cv2.resize(walkable_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        matrix = image_to_bev_matrix(calibration)
        warped = cv2.warpPerspective(mask, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
        free = warped > 0
        grid[free] = 0
        layers["free_space"][free] = 1
        image[free] = (46, 104, 68)
        self._draw_metric_grid(image, calibration)

    def _draw_known_obstacles(
        self,
        image: np.ndarray,
        grid: np.ndarray,
        layers: dict[str, np.ndarray],
        detections: list[Detection],
        calibration: Calibration,
    ) -> None:
        for detection in detections:
            if detection.category != "known_obstacle":
                continue
            x1, _, x2, y2 = detection.bbox
            points = [(x1, y2), ((x1 + x2) * 0.5, y2), (x2, y2)]
            _, bev = image_points_to_bev_pixels(calibration, points)
            pts = _clip_points(bev, image.shape)
            if len(pts) == 0:
                continue
            if len(pts) >= 2:
                cv2.polylines(image, [pts], isClosed=False, color=(0, 165, 255), thickness=5)
                cv2.polylines(layers["known_obstacle"], [pts], isClosed=False, color=1, thickness=7)
                cv2.polylines(grid, [pts], isClosed=False, color=100, thickness=7)
            for point in pts:
                cv2.circle(image, tuple(point), 5, (0, 120, 255), -1)

    def _draw_unknown_obstacles(
        self,
        image: np.ndarray,
        grid: np.ndarray,
        layers: dict[str, np.ndarray],
        unknown_obstacles: list[ObstacleRegion],
        calibration: Calibration,
    ) -> None:
        for obstacle in unknown_obstacles:
            if not obstacle.ground_points:
                continue
            _, bev = image_points_to_bev_pixels(calibration, obstacle.ground_points)
            obstacle.bev_points = [(float(x), float(y)) for x, y in bev]
            pts = _clip_points(bev, image.shape)
            if len(pts) == 0:
                continue
            if len(pts) >= 2:
                cv2.polylines(image, [pts], isClosed=False, color=(80, 80, 240), thickness=5)
                cv2.polylines(layers["unknown_obstacle"], [pts], isClosed=False, color=1, thickness=9)
                cv2.polylines(grid, [pts], isClosed=False, color=80, thickness=9)
            for point in pts:
                cv2.circle(image, tuple(point), 5, (80, 80, 240), -1)

    def _draw_people(
        self,
        image: np.ndarray,
        grid: np.ndarray,
        layers: dict[str, np.ndarray],
        tracks: list[Track],
    ) -> None:
        for track in tracks:
            if track.bev_position is None:
                continue
            point = (int(round(track.bev_position[0])), int(round(track.bev_position[1])))
            if not _inside(point, image.shape):
                continue
            cv2.circle(image, point, 9, (255, 210, 60), -1)
            cv2.circle(layers["person"], point, 10, 1, -1)
            cv2.circle(grid, point, 10, 100, -1)
            cv2.putText(image, f"ID {track.track_id}", (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if len(track.trajectory) >= 2:
                pts = _clip_points(np.asarray(track.trajectory, dtype=np.float32), image.shape)
                if len(pts) >= 2:
                    cv2.polylines(image, [pts], isClosed=False, color=(255, 235, 120), thickness=2)

    def _draw_robot(
        self,
        image: np.ndarray,
        grid: np.ndarray,
        layers: dict[str, np.ndarray],
        calibration: Calibration,
    ) -> None:
        x, y = bev_robot_pixel(calibration)
        triangle = np.array([[x, y - 18], [x - 14, y + 14], [x + 14, y + 14]], dtype=np.int32)
        cv2.fillPoly(image, [triangle], (230, 230, 230))
        cv2.polylines(image, [triangle], isClosed=True, color=(20, 20, 20), thickness=2)
        cv2.fillPoly(layers["robot"], [triangle], 1)
        cv2.fillPoly(grid, [triangle], 100)

    def _draw_metric_grid(self, image: np.ndarray, calibration: Calibration) -> None:
        step = 100 if calibration.metric_bev else 80
        for x in range(0, image.shape[1], step):
            cv2.line(image, (x, 0), (x, image.shape[0] - 1), (55, 55, 55), 1)
        for y in range(0, image.shape[0], step):
            cv2.line(image, (0, y), (image.shape[1] - 1, y), (55, 55, 55), 1)

    def _draw_bev_annotations(self, image: np.ndarray, calibration: Calibration) -> None:
        label = calibration.label
        if not calibration.metric_bev:
            label = f"{label} / NORMALIZED SOCIAL ZONE"
        cv2.putText(image, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _clip_points(points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    pts = []
    for x, y in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        if np.isfinite(x) and np.isfinite(y):
            pts.append([int(np.clip(round(float(x)), 0, w - 1)), int(np.clip(round(float(y)), 0, h - 1))])
    return np.asarray(pts, dtype=np.int32)


def _inside(point: tuple[int, int], shape: tuple[int, ...]) -> bool:
    h, w = shape[:2]
    x, y = point
    return 0 <= x < w and 0 <= y < h

