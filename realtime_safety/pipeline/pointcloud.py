from __future__ import annotations

import cv2
import numpy as np


def relative_inverse_depth(prediction: np.ndarray, median_distance: float = 3.0) -> np.ndarray:
    """Convert a relative inverse-depth prediction to a stable positive relative depth."""
    values = np.asarray(prediction, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Depth prediction has no finite values")
    low, high = np.percentile(values[finite], [2.0, 98.0])
    inverse_depth = np.clip(values, low, high) - low
    inverse_depth /= max(float(high - low), 1e-6)
    depth = 1.0 / np.maximum(inverse_depth + 0.08, 0.08)
    scale = median_distance / max(float(np.median(depth[finite])), 1e-6)
    return (depth * scale).astype(np.float32)


def depth_to_pointmap(
    depth: np.ndarray,
    focal_px: float | None = None,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Project depth into x-right, y-forward, z-up robot coordinates."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape HxW")
    height, width = depth.shape
    focal = float(focal_px or 0.85 * max(width, height))
    cx, cy = principal_point or ((width - 1) * 0.5, (height - 1) * 0.5)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - cx) * depth / focal
    y = depth
    z = -(v - cy) * depth / focal
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points, colors, confidence = points[valid], colors[valid], confidence[valid]
    if len(points) == 0:
        return points, colors.astype(np.uint8), confidence
    if voxel_size > 0:
        voxels = np.floor(points / voxel_size).astype(np.int32)
        _, selected = np.unique(voxels, axis=0, return_index=True)
        selected.sort()
        points, colors, confidence = points[selected], colors[selected], confidence[selected]
    if len(points) > max_points:
        # Deterministic evenly-spaced sampling avoids per-frame RNG overhead/flicker.
        selected = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors, confidence = points[selected], colors[selected], confidence[selected]
    if colors.dtype != np.uint8:
        colors = np.clip(colors * 255.0 if colors.max(initial=0) <= 1.0 else colors, 0, 255).astype(np.uint8)
    return points, colors, confidence


def resize_for_pointmap(rgb: np.ndarray, max_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    target = (max(16, int(round(width * scale / 16)) * 16), max(16, int(round(height * scale / 16)) * 16))
    return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
