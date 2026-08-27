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
        adaptation_frames: int = 12,
        spatial_tolerance: float = 0.05,
        min_spatial_support: float = 0.75,
        candidate_stability_tolerance: float = 0.025,
        max_update_fraction: float = 0.025,
    ) -> None:
        self.target_depth_m = float(target_depth_m)
        self.roi = tuple(float(value) for value in roi)
        self.percentile = float(percentile)
        self.warmup_frames = int(warmup_frames)
        self.ema_alpha = float(ema_alpha)
        self._candidates: deque[float] = deque(maxlen=self.warmup_frames)
        self.adaptation_frames = max(1, int(adaptation_frames))
        self.spatial_tolerance = float(spatial_tolerance)
        self.min_spatial_support = float(min_spatial_support)
        self.candidate_stability_tolerance = float(candidate_stability_tolerance)
        self.max_update_fraction = float(max_update_fraction)
        self._warmup_depths: deque[np.ndarray] = deque(maxlen=self.warmup_frames)
        self._adaptation_candidates: deque[float] = deque(maxlen=self.adaptation_frames)
        self._reference_template: np.ndarray | None = None
        self._reference_scale: float | None = None
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

        if not self.ready:
            if self._warmup_depths and self._warmup_depths[0].shape != values.shape:
                self._candidates.clear()
                self._warmup_depths.clear()
            self.observed_depth = observed
            self._candidates.append(candidate)
            self._warmup_depths.append(
                np.where(
                    np.isfinite(values) & (values > 0.05),
                    values,
                    np.nan,
                ).astype(np.float32)
            )
            robust_candidate = float(np.median(self._candidates))
            self.scale = robust_candidate
            self.ready = len(self._candidates) >= self.warmup_frames
            if self.ready:
                depth_stack = np.ma.masked_invalid(np.stack(self._warmup_depths, axis=0))
                self._reference_template = np.ma.median(depth_stack, axis=0).filled(np.nan).astype(np.float32)
                self._reference_scale = self.scale
                self._adaptation_candidates.clear()
            return self.scale

        assert self.scale is not None
        # ``ema_alpha == 0`` is an explicit request to freeze the startup
        # calibration.  Do not even accumulate pending drift while frozen.
        if self.ema_alpha <= 0.0:
            return self.scale

        spatial_candidate = self._spatial_scale_candidate(values)
        if spatial_candidate is None:
            self._adaptation_candidates.clear()
            return self.scale
        candidate, observed = spatial_candidate

        # A large step is much more likely to be an occluder or a changed
        # reference object than model-scale drift.  Keep the historical guard
        # in addition to the spatial and temporal checks below.
        ratio = candidate / self.scale
        if ratio < 0.65 or ratio > 1.35:
            self._adaptation_candidates.clear()
            return self.scale

        self.observed_depth = observed
        if self._adaptation_candidates:
            pending = float(np.median(self._adaptation_candidates))
            if abs(candidate / pending - 1.0) > self.candidate_stability_tolerance:
                self._adaptation_candidates.clear()
        self._adaptation_candidates.append(candidate)
        if len(self._adaptation_candidates) < self.adaptation_frames:
            return self.scale

        robust_candidate = float(np.median(self._adaptation_candidates))
        relative_spread = np.abs(
            np.asarray(self._adaptation_candidates, dtype=np.float64) / robust_candidate - 1.0
        )
        if float(np.max(relative_spread)) > self.candidate_stability_tolerance:
            self._adaptation_candidates.clear()
            return self.scale

        # Even alpha=1 cannot move the global point cloud by more than the
        # configured fraction in one accepted frame.  With the normal small
        # EMA this deliberately takes many corroborated frames to re-anchor.
        bounded = float(
            np.clip(
                robust_candidate,
                self.scale * (1.0 - self.max_update_fraction),
                self.scale * (1.0 + self.max_update_fraction),
            )
        )
        self.scale = (1.0 - self.ema_alpha) * self.scale + self.ema_alpha * bounded
        return self.scale

    def _spatial_scale_candidate(self, depth: np.ndarray) -> tuple[float, float] | None:
        """Return a scale only when the full view supports one depth ratio.

        A Video Depth Anything scale drift multiplies essentially every valid
        scene pixel by the same factor. A hand crossing even the entire metric
        reference ROI changes only a spatial subset of the image, so it cannot
        re-anchor the world unless the surrounding fixed scene corroborates
        the same ratio.
        """
        template = self._reference_template
        reference_scale = self._reference_scale
        if template is None or reference_scale is None or template.shape != depth.shape:
            return None

        matched = (
            np.isfinite(template)
            & (template > 0.05)
            & np.isfinite(depth)
            & (depth > 0.05)
        )
        if int(np.count_nonzero(matched)) < 32:
            return None

        ratios = depth[matched].astype(np.float64) / template[matched].astype(np.float64)
        depth_ratio = float(np.median(ratios))
        if not np.isfinite(depth_ratio) or depth_ratio <= 0.0:
            return None
        inlier = np.abs(ratios / depth_ratio - 1.0) <= self.spatial_tolerance
        if float(np.mean(inlier)) < self.min_spatial_support:
            return None

        # Pixel support alone can be concentrated in one part of the ROI.
        # Requiring agreement across a small grid makes a local foreground
        # patch unable to masquerade as a global monocular-depth scale change.
        height, width = depth.shape
        y_edges = np.linspace(0, height, 5, dtype=np.int32)
        x_edges = np.linspace(0, width, 5, dtype=np.int32)
        cell_agreement: list[bool] = []
        for row in range(4):
            for column in range(4):
                cell_match = matched[
                    y_edges[row] : y_edges[row + 1],
                    x_edges[column] : x_edges[column + 1],
                ]
                if int(np.count_nonzero(cell_match)) < 2:
                    continue
                cell_depth = depth[
                    y_edges[row] : y_edges[row + 1],
                    x_edges[column] : x_edges[column + 1],
                ]
                cell_template = template[
                    y_edges[row] : y_edges[row + 1],
                    x_edges[column] : x_edges[column + 1],
                ]
                cell_ratio = float(
                    np.median(cell_depth[cell_match] / cell_template[cell_match])
                )
                cell_agreement.append(abs(cell_ratio / depth_ratio - 1.0) <= self.spatial_tolerance)
        if len(cell_agreement) < 4 or float(np.mean(cell_agreement)) < self.min_spatial_support:
            return None

        scale_candidate = reference_scale / depth_ratio
        if not np.isfinite(scale_candidate) or scale_candidate <= 0.0:
            return None
        return scale_candidate, self.target_depth_m / scale_candidate

    def reset(self) -> None:
        self._candidates.clear()
        self._warmup_depths.clear()
        self._adaptation_candidates.clear()
        self._reference_template = None
        self._reference_scale = None
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
