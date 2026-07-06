from __future__ import annotations

import cv2
import numpy as np

from social_bev.bev_map import BEVMapBuilder
from social_bev.homography import compute_homography
from social_bev.types import Calibration, Detection, ObstacleRegion, Track


def test_bev_occupancy_grid_values_are_legal() -> None:
    cfg = bev_config()
    image_points = np.array([[45, 45], [115, 45], [150, 119], [10, 119]], dtype=np.float32)
    world_points = np.array([[-1, 1], [1, 1], [1, 0], [-1, 0]], dtype=np.float32)
    calibration = Calibration(
        homography=compute_homography(image_points, world_points),
        image_points=image_points,
        world_points=world_points,
        metric_bev=False,
        bev_config=cfg,
        label="NON-METRIC BEV",
    )
    walkable = np.zeros((120, 160), dtype=np.uint8)
    cv2.fillPoly(walkable, [image_points.astype(np.int32)], 1)
    detections = [
        Detection((70.0, 60.0, 95.0, 105.0), 0.8, 2, "chair", "known_obstacle"),
    ]
    contour = np.array([[[40, 80]], [[55, 80]], [[55, 110]], [[40, 110]]], dtype=np.int32)
    unknown = [
        ObstacleRegion(
            bbox=(40.0, 80.0, 55.0, 110.0),
            contour=contour,
            confidence=0.35,
            label="unknown_obstacle",
            category="unknown_obstacle",
            ground_points=[(40.0, 110.0), (47.0, 110.0), (55.0, 110.0)],
        )
    ]
    tracks = [
        Track(
            track_id=1,
            bbox=(75.0, 35.0, 95.0, 100.0),
            confidence=0.9,
            image_ground_point=(85.0, 100.0),
            bev_position=(80.0, 120.0),
            velocity=(0.0, 0.0),
            age=4,
            missed_frames=0,
            trajectory=[(80.0, 120.0)],
        )
    ]
    result = BEVMapBuilder(cfg, {"normalized_static_radius_px": 8}).build(
        walkable.astype(bool),
        detections,
        unknown,
        tracks,
        calibration,
        frame_shape=(120, 160, 3),
    )
    values = set(np.unique(result.occupancy_grid).tolist())
    assert values.issubset({-1, 0, 50, 80, 100})
    assert result.metric_bev is False


def bev_config() -> dict[str, float | int]:
    return {
        "width_px": 160,
        "height_px": 200,
        "x_min_m": -3.0,
        "x_max_m": 3.0,
        "y_min_m": 0.0,
        "y_max_m": 8.0,
        "resolution_m_per_pixel": 0.05,
        "nonmetric_x_min": -1.0,
        "nonmetric_x_max": 1.0,
        "nonmetric_y_min": 0.0,
        "nonmetric_y_max": 1.0,
    }

