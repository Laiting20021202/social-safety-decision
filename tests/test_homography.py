from __future__ import annotations

import numpy as np

from social_bev.homography import (
    compute_homography,
    estimate_ground_contact,
    is_valid_homography,
    load_calibration,
    project_image_points,
)


def test_homography_maps_four_corners() -> None:
    image_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    world_points = np.array([[-1, 1], [1, 1], [1, 0], [-1, 0]], dtype=np.float32)
    matrix = compute_homography(image_points, world_points)
    calibration = load_calibration(None, bev_config(), frame_shape=(120, 140, 3))
    calibration.homography = matrix
    projected = project_image_points(calibration, image_points)
    assert np.allclose(projected, world_points, atol=1e-4)


def test_invalid_homography_detected() -> None:
    assert not is_valid_homography(np.zeros((3, 3), dtype=np.float64))
    assert not is_valid_homography(np.full((3, 3), np.nan))


def test_person_bottom_center_ground_contact_refines_to_walkable() -> None:
    mask = np.zeros((100, 120), dtype=bool)
    mask[82:90, 54:66] = True
    point, confidence = estimate_ground_contact((40.0, 20.0, 80.0, 84.0), mask, search_radius=10)
    assert confidence == 1.0
    assert 54 <= point[0] <= 66
    assert 82 <= point[1] <= 90


def test_missing_calibration_is_non_metric() -> None:
    calibration = load_calibration(None, bev_config(), frame_shape=(100, 160, 3))
    assert calibration.metric_bev is False
    assert calibration.label == "NON-METRIC BEV"


def bev_config() -> dict[str, float | int]:
    return {
        "width_px": 120,
        "height_px": 160,
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

