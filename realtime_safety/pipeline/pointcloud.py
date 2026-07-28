from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np


class ReferenceDepthCalibrator:
    """Recover a stable metric scale from a fixed object at a known distance.

    One measured distance can constrain only a global multiplicative scale.
    The reference is sampled before 3D projection so the same correction is
    applied to forward depth and both lateral axes.
    """

    def __init__(
        self,
        target_depth_m: float,
        roi: Sequence[float],
        percentile: float = 20.0,
        warmup_frames: int = 8,
        ema_alpha: float = 0.08,
    ) -> None:
        self.target_depth_m = float(target_depth_m)
        self.roi = tuple(float(value) for value in roi)
        self.percentile = float(percentile)
        self.warmup_frames = int(warmup_frames)
        self.ema_alpha = float(ema_alpha)
        self._candidates: deque[float] = deque(maxlen=self.warmup_frames)
        self.scale: float | None = None
        self.observed_depth: float | None = None
        self.ready = False

    def update(self, depth: np.ndarray) -> float:
        values = np.asarray(depth, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("depth must have shape HxW")
        height, width = values.shape
        x_min, y_min, x_max, y_max = self.roi
        x0 = max(0, min(width - 1, int(np.floor(x_min * width))))
        x1 = max(x0 + 1, min(width, int(np.ceil(x_max * width))))
        y0 = max(0, min(height - 1, int(np.floor(y_min * height))))
        y1 = max(y0 + 1, min(height, int(np.ceil(y_max * height))))
        crop = values[y0:y1, x0:x1]
        valid = crop[np.isfinite(crop) & (crop > 0.05)]
        if valid.size < 32:
            return self.scale if self.scale is not None else 1.0

        observed = float(np.percentile(valid, self.percentile))
        candidate = self.target_depth_m / observed
        if not np.isfinite(candidate) or candidate <= 0:
            return self.scale if self.scale is not None else 1.0

        # Once initialized, a hand or person passing over the reference ROI
        # must not abruptly rescale the entire world.
        if self.ready and self.scale is not None:
            ratio = candidate / self.scale
            if ratio < 0.65 or ratio > 1.35:
                return self.scale

        self.observed_depth = observed
        self._candidates.append(candidate)
        robust_candidate = float(np.median(self._candidates))
        if self.scale is None or not self.ready:
            self.scale = robust_candidate
        else:
            # Bound each accepted update as a second guard against artificial
            # scene motion, then follow slow model-scale drift with an EMA.
            bounded = float(np.clip(robust_candidate, self.scale * 0.9, self.scale * 1.1))
            self.scale = (1.0 - self.ema_alpha) * self.scale + self.ema_alpha * bounded
        self.ready = len(self._candidates) >= self.warmup_frames
        return self.scale

    def reset(self) -> None:
        self._candidates.clear()
        self.scale = None
        self.observed_depth = None
        self.ready = False


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
    focal_px: float | tuple[float, float] | None = None,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Project depth into x-right, y-forward, z-up robot coordinates."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape HxW")
    height, width = depth.shape
    if isinstance(focal_px, tuple):
        focal_x, focal_y = map(float, focal_px)
    else:
        focal_x = focal_y = float(focal_px or 0.85 * max(width, height))
    if focal_x <= 0 or focal_y <= 0:
        raise ValueError("focal length must be positive")
    cx, cy = principal_point or ((width - 1) * 0.5, (height - 1) * 0.5)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (u - cx) * depth / focal_x
    y = depth
    z = -(v - cy) * depth / focal_y
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
