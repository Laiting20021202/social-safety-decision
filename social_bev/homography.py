from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from social_bev.config import load_yaml
from social_bev.types import Calibration


class HomographyError(RuntimeError):
    """Raised when an IPM homography cannot be created or used."""


def validate_quad(points: np.ndarray, min_area: float = 100.0) -> bool:
    """Validate a four-point ground quadrilateral in click order."""

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False
    area = abs(float(cv2.contourArea(pts.reshape(-1, 1, 2))))
    if area < min_area:
        return False
    far_left, far_right, near_right, near_left = pts
    if far_left[0] >= far_right[0] or near_left[0] >= near_right[0]:
        return False
    if far_left[1] >= near_left[1] or far_right[1] >= near_right[1]:
        return False
    return True


def compute_homography(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    img = np.asarray(image_points, dtype=np.float32)
    world = np.asarray(world_points, dtype=np.float32)
    if img.shape != (4, 2) or world.shape != (4, 2):
        raise HomographyError("Homography needs exactly four image and four world points")
    if not validate_quad(img):
        raise HomographyError("Image points do not form a reasonable ground quadrilateral")
    matrix = cv2.getPerspectiveTransform(img, world)
    if not is_valid_homography(matrix):
        raise HomographyError("Computed homography is singular or non-finite")
    return matrix.astype(np.float64)


def is_valid_homography(matrix: np.ndarray) -> bool:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3) or not np.isfinite(mat).all():
        return False
    det = float(np.linalg.det(mat))
    return abs(det) > 1e-9


def default_world_points(config: dict[str, float | int], metric: bool = True) -> np.ndarray:
    if metric:
        return np.array(
            [
                [float(config["x_min_m"]), float(config["y_max_m"])],
                [float(config["x_max_m"]), float(config["y_max_m"])],
                [float(config["x_max_m"]), float(config["y_min_m"])],
                [float(config["x_min_m"]), float(config["y_min_m"])],
            ],
            dtype=np.float32,
        )
    return np.array(
        [
            [float(config.get("nonmetric_x_min", -1.0)), float(config.get("nonmetric_y_max", 1.0))],
            [float(config.get("nonmetric_x_max", 1.0)), float(config.get("nonmetric_y_max", 1.0))],
            [float(config.get("nonmetric_x_max", 1.0)), float(config.get("nonmetric_y_min", 0.0))],
            [float(config.get("nonmetric_x_min", -1.0)), float(config.get("nonmetric_y_min", 0.0))],
        ],
        dtype=np.float32,
    )


def default_image_points(frame_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return np.array(
        [
            [0.34 * w, 0.42 * h],
            [0.66 * w, 0.42 * h],
            [0.95 * w, 0.98 * h],
            [0.05 * w, 0.98 * h],
        ],
        dtype=np.float32,
    )


def load_calibration(
    calibration_path: str | Path | None,
    bev_config: dict[str, float | int],
    frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
) -> Calibration:
    """Load metric calibration, or create a clearly marked non-metric fallback."""

    if calibration_path and Path(calibration_path).exists():
        data = load_yaml(calibration_path)
        image_points = np.asarray(data.get("image_points"), dtype=np.float32)
        world_points = np.asarray(data.get("world_points"), dtype=np.float32)
        matrix_data = data.get("homography")
        matrix = np.asarray(matrix_data, dtype=np.float64) if matrix_data is not None else None
        if matrix is None or not is_valid_homography(matrix):
            matrix = compute_homography(image_points, world_points)
        cfg = dict(bev_config)
        cfg.update(data.get("bev", {}) or {})
        return Calibration(
            homography=matrix,
            image_points=image_points,
            world_points=world_points,
            metric_bev=bool(data.get("metric_bev", True)),
            bev_config=cfg,
            label="METRIC BEV" if bool(data.get("metric_bev", True)) else "NON-METRIC BEV",
        )

    if frame_shape is None:
        frame_shape = (360, 640, 3)
    image_points = default_image_points(frame_shape)
    world_points = default_world_points(bev_config, metric=False)
    matrix = compute_homography(image_points, world_points)
    return Calibration(
        homography=matrix,
        image_points=image_points,
        world_points=world_points,
        metric_bev=False,
        bev_config=dict(bev_config),
        label="NON-METRIC BEV",
    )


def project_image_points(calibration: Calibration, points: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, calibration.homography).reshape(-1, 2)
    return projected.astype(np.float32)


def world_to_bev_pixels(calibration: Calibration, world_points: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
    cfg = calibration.bev_config
    pts = np.asarray(world_points, dtype=np.float32).reshape(-1, 2)
    width = float(cfg["width_px"])
    height = float(cfg["height_px"])
    if calibration.metric_bev:
        x_min = float(cfg["x_min_m"])
        x_max = float(cfg["x_max_m"])
        y_min = float(cfg["y_min_m"])
        y_max = float(cfg["y_max_m"])
    else:
        x_min = float(cfg.get("nonmetric_x_min", -1.0))
        x_max = float(cfg.get("nonmetric_x_max", 1.0))
        y_min = float(cfg.get("nonmetric_y_min", 0.0))
        y_max = float(cfg.get("nonmetric_y_max", 1.0))
    denom_x = max(1e-6, x_max - x_min)
    denom_y = max(1e-6, y_max - y_min)
    px = (pts[:, 0] - x_min) / denom_x * (width - 1.0)
    py = (height - 1.0) - (pts[:, 1] - y_min) / denom_y * (height - 1.0)
    return np.stack([px, py], axis=1).astype(np.float32)


def image_points_to_bev_pixels(
    calibration: Calibration,
    image_points: list[tuple[float, float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world = project_image_points(calibration, image_points)
    bev = world_to_bev_pixels(calibration, world)
    return world, bev


def image_to_bev_matrix(calibration: Calibration) -> np.ndarray:
    """Return a 3x3 perspective matrix from source image pixels to BEV pixels."""

    cfg = calibration.bev_config
    width = float(cfg["width_px"])
    height = float(cfg["height_px"])
    if calibration.metric_bev:
        x_min = float(cfg["x_min_m"])
        x_max = float(cfg["x_max_m"])
        y_min = float(cfg["y_min_m"])
        y_max = float(cfg["y_max_m"])
    else:
        x_min = float(cfg.get("nonmetric_x_min", -1.0))
        x_max = float(cfg.get("nonmetric_x_max", 1.0))
        y_min = float(cfg.get("nonmetric_y_min", 0.0))
        y_max = float(cfg.get("nonmetric_y_max", 1.0))
    sx = (width - 1.0) / max(1e-6, x_max - x_min)
    sy = -(height - 1.0) / max(1e-6, y_max - y_min)
    transform = np.array(
        [
            [sx, 0.0, -x_min * sx],
            [0.0, sy, (height - 1.0) + y_min * -sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform @ calibration.homography


def bev_robot_pixel(calibration: Calibration) -> tuple[int, int]:
    cfg = calibration.bev_config
    width = int(cfg["width_px"])
    height = int(cfg["height_px"])
    return width // 2, height - 18


def estimate_ground_contact(
    bbox: tuple[float, float, float, float],
    walkable_mask: np.ndarray,
    search_radius: int = 10,
) -> tuple[tuple[float, float], float]:
    """Refine a box bottom-center point by looking for nearby walkable pixels."""

    h, w = walkable_mask.shape[:2]
    x1, _, x2, y2 = bbox
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round(y2))
    cx = int(np.clip(cx, 0, max(0, w - 1)))
    cy = int(np.clip(cy, 0, max(0, h - 1)))
    y0 = max(0, cy - search_radius)
    y1s = min(h, cy + search_radius + 1)
    x0 = max(0, cx - search_radius)
    x1s = min(w, cx + search_radius + 1)
    window = walkable_mask[y0:y1s, x0:x1s].astype(bool)
    if window.any():
        ys, xs = np.where(window)
        points = np.stack([xs + x0, ys + y0], axis=1)
        distances = np.square(points[:, 0] - cx) + np.square(points[:, 1] - cy)
        best = points[int(np.argmin(distances))]
        return (float(best[0]), float(best[1])), 1.0
    return (float(cx), float(cy)), 0.35

