from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class MetricBevCalibration:
    """Metric work-plane basis used for an orthographic bird's-eye view.

    ``right`` and ``forward`` lie on the fitted work plane. ``normal`` is the
    physical height direction, not a display Euler angle. Raw camera-frame
    points are never modified; :meth:`project` creates a derived BEV copy.
    """

    origin: np.ndarray
    right: np.ndarray
    forward: np.ndarray
    normal: np.ndarray
    bounds_uv: tuple[float, float, float, float]
    inlier_count: int
    inlier_ratio: float
    rms_error_m: float

    def project(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        relative = values - self.origin
        return np.stack(
            (
                relative @ self.right,
                relative @ self.forward,
                relative @ self.normal,
            ),
            axis=1,
        ).astype(np.float32)


def fit_metric_bev(
    pointmap: np.ndarray,
    *,
    distance_threshold_m: float = 0.018,
    iterations: int = 120,
    minimum_inliers: int = 240,
    maximum_samples: int = 12_000,
    seed: int = 7,
) -> MetricBevCalibration:
    """Fit the dominant work surface and return a stable metric BEV basis.

    The lower 78% of the image is used because a behind/above work-cell
    camera often includes a wall along the upper edge. RANSAC finds a physical
    plane and PCA refines it; no user-supplied pitch/roll/yaw is involved.
    """

    dense = np.asarray(pointmap, dtype=np.float32)
    if dense.ndim != 3 or dense.shape[2] != 3:
        raise ValueError("pointmap must have shape HxWx3")
    height, width = dense.shape[:2]
    y0 = int(round(height * 0.22))
    x0 = int(round(width * 0.03))
    x1 = max(x0 + 1, int(round(width * 0.97)))
    candidates = dense[y0:, x0:x1].reshape(-1, 3)
    valid = np.isfinite(candidates).all(axis=1)
    valid &= candidates[:, 1] > 0.05
    candidates = candidates[valid]
    if len(candidates) < minimum_inliers:
        raise ValueError(
            f"not enough finite work-plane points: {len(candidates)} < {minimum_inliers}"
        )

    if len(candidates) > maximum_samples:
        indices = np.linspace(
            0, len(candidates) - 1, maximum_samples, dtype=np.int64
        )
        samples = candidates[indices]
    else:
        samples = candidates

    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_score = -1.0
    threshold = max(float(distance_threshold_m), 1e-4)
    for _ in range(max(int(iterations), 1)):
        chosen = samples[rng.choice(len(samples), size=3, replace=False)]
        normal = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-7:
            continue
        normal /= length
        # A work surface must have a useful vertical component. This rejects
        # the rear wall without assuming the camera itself is level.
        vertical_support = abs(float(normal[2]))
        if vertical_support < 0.18:
            continue
        distances = np.abs((samples - chosen[0]) @ normal)
        mask = distances <= threshold
        count = int(np.count_nonzero(mask))
        score = count * (0.70 + 0.30 * vertical_support)
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None or int(np.count_nonzero(best_mask)) < minimum_inliers:
        raise ValueError("RANSAC could not find a supported work plane")

    inliers = samples[best_mask]
    centroid = np.mean(inliers, axis=0, dtype=np.float64)
    centered = inliers.astype(np.float64) - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal *= -1.0
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    signed_distance = float(np.dot(normal, centroid))
    origin = normal * signed_distance

    # Preserve the intuitive left/right image direction while removing its
    # component normal to the measured work surface.
    camera_right = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    right = camera_right - normal * float(np.dot(camera_right, normal))
    right /= max(float(np.linalg.norm(right)), 1e-12)
    forward = np.cross(normal, right)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    if forward[1] < 0.0:
        # Flip both in-plane axes together so the basis remains right-handed.
        right *= -1.0
        forward *= -1.0

    relative = samples.astype(np.float64) - origin
    u = relative @ right
    v = relative @ forward
    u0, u1 = np.percentile(u, (1.0, 99.0))
    v0, v1 = np.percentile(v, (1.0, 99.0))
    u0, u1 = _padded_metric_extent(float(u0), float(u1))
    v0, v1 = _padded_metric_extent(float(v0), float(v1))
    residuals = centered @ normal

    return MetricBevCalibration(
        origin=origin.astype(np.float32),
        right=right.astype(np.float32),
        forward=forward.astype(np.float32),
        normal=normal.astype(np.float32),
        bounds_uv=(u0, u1, v0, v1),
        inlier_count=len(inliers),
        inlier_ratio=float(len(inliers) / len(samples)),
        rms_error_m=float(np.sqrt(np.mean(np.square(residuals)))),
    )


def rasterize_metric_bev(
    calibration: MetricBevCalibration,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    obstacle_height_m: float = 0.035,
    edge_points: np.ndarray | None = None,
    maximum_height_m: float = 0.75,
    maximum_side_px: int = 520,
) -> np.ndarray:
    """Render an orthographic, metric height/occupancy map as RGB."""

    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    shown_colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(values) != len(shown_colors):
        raise ValueError("points and colors must have equal length")
    bev = calibration.project(values)
    u0, u1, v0, v1 = calibration.bounds_uv
    span_u = max(u1 - u0, 1e-3)
    span_v = max(v1 - v0, 1e-3)
    aspect = span_u / span_v
    if aspect >= 1.0:
        width = maximum_side_px
        height = max(180, int(round(maximum_side_px / aspect)))
    else:
        height = maximum_side_px
        width = max(180, int(round(maximum_side_px * aspect)))
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)

    valid = np.isfinite(bev).all(axis=1)
    valid &= (bev[:, 0] >= u0) & (bev[:, 0] <= u1)
    valid &= (bev[:, 1] >= v0) & (bev[:, 1] <= v1)
    valid &= (bev[:, 2] >= -0.04) & (bev[:, 2] <= maximum_height_m)
    indices = np.flatnonzero(valid)
    if len(indices):
        # Low points are written first; the highest physical sample wins each
        # BEV cell and therefore cannot be hidden by the tabletop below it.
        indices = indices[np.argsort(bev[indices, 2], kind="stable")]
        x = np.clip(
            np.rint((bev[indices, 0] - u0) / span_u * (width - 1)),
            0,
            width - 1,
        ).astype(np.int32)
        y = np.clip(
            np.rint((v1 - bev[indices, 1]) / span_v * (height - 1)),
            0,
            height - 1,
        ).astype(np.int32)
        pixels = (shown_colors[indices].astype(np.float32) * 0.72).astype(np.uint8)
        elevated = bev[indices, 2] >= max(float(obstacle_height_m), 0.0)
        pixels[elevated] = np.clip(
            0.38 * pixels[elevated]
            + 0.62 * np.array((255, 72, 42), dtype=np.float32),
            0,
            255,
        ).astype(np.uint8)
        canvas[y, x] = pixels

    _draw_metric_grid(canvas, calibration.bounds_uv)
    if edge_points is not None:
        _draw_edge_points(canvas, calibration, edge_points)
    cv2.putText(
        canvas,
        f"ORTHOGRAPHIC BEV  elevated >= {obstacle_height_m:.3f} m",
        (9, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (225, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_edge_points(
    canvas: np.ndarray,
    calibration: MetricBevCalibration,
    edge_points: np.ndarray,
) -> None:
    values = np.asarray(edge_points, dtype=np.float32).reshape(-1, 3)
    if not len(values):
        return
    bev = calibration.project(values)
    u0, u1, v0, v1 = calibration.bounds_uv
    valid = np.isfinite(bev).all(axis=1)
    valid &= (bev[:, 0] >= u0) & (bev[:, 0] <= u1)
    valid &= (bev[:, 1] >= v0) & (bev[:, 1] <= v1)
    bev = bev[valid]
    if not len(bev):
        return
    height, width = canvas.shape[:2]
    x = np.rint((bev[:, 0] - u0) / (u1 - u0) * (width - 1)).astype(np.int32)
    y = np.rint((v1 - bev[:, 1]) / (v1 - v0) * (height - 1)).astype(np.int32)
    canvas[
        np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)
    ] = np.array((255, 30, 105), dtype=np.uint8)


def _draw_metric_grid(
    canvas: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> None:
    height, width = canvas.shape[:2]
    u0, u1, v0, v1 = bounds
    step = 0.10
    for u in np.arange(np.ceil(u0 / step) * step, u1, step):
        x = int(round((u - u0) / (u1 - u0) * (width - 1)))
        cv2.line(canvas, (x, 0), (x, height - 1), (34, 48, 52), 1)
    for v in np.arange(np.ceil(v0 / step) * step, v1, step):
        y = int(round((v1 - v) / (v1 - v0) * (height - 1)))
        cv2.line(canvas, (0, y), (width - 1, y), (34, 48, 52), 1)
    cv2.putText(
        canvas,
        "grid 0.10 m",
        (9, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (130, 180, 190),
        1,
        cv2.LINE_AA,
    )


def _padded_metric_extent(low: float, high: float) -> tuple[float, float]:
    center = (low + high) * 0.5
    span = max(high - low, 0.35)
    padding = max(span * 0.04, 0.025)
    return center - span * 0.5 - padding, center + span * 0.5 + padding


__all__ = [
    "MetricBevCalibration",
    "fit_metric_bev",
    "rasterize_metric_bev",
]
