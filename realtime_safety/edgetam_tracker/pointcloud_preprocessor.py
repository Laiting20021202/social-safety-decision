from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from realtime_safety.edgetam_tracker.models import CloudFrame


def _negative_infinity() -> np.ndarray:
    return np.full(3, -np.inf, dtype=np.float32)


def _positive_infinity() -> np.ndarray:
    return np.full(3, np.inf, dtype=np.float32)


@dataclass(slots=True)
class PointCloudPreprocessorConfig:
    """Geometry-only preprocessing parameters.

    ``raw_cloud`` in the result is the finite, workspace-cropped cloud before
    lossy voxel/outlier/plane filtering.  It is therefore suitable for later
    high-resolution mask fusion while ``processed_cloud`` is kept small for
    clustering and tracking.
    """

    workspace_min: np.ndarray = field(default_factory=_negative_infinity)
    workspace_max: np.ndarray = field(default_factory=_positive_infinity)
    voxel_size: float = 0.0
    remove_outliers: bool = False
    outlier_method: str = "statistical"
    outlier_mean_k: int = 16
    outlier_stddev: float = 2.0
    outlier_radius: float = 0.08
    outlier_min_neighbors: int = 2
    remove_plane: bool = False
    plane_distance_threshold: float = 0.01
    plane_iterations: int = 100
    plane_min_inliers: int = 30
    plane_min_inlier_ratio: float = 0.1
    plane_normal_axis: np.ndarray | None = None
    plane_max_angle_deg: float = 30.0
    plane_min_distance: float = 0.0
    plane_max_distance: float = float("inf")
    random_seed: int = 31

    def __post_init__(self) -> None:
        self.workspace_min = np.asarray(self.workspace_min, dtype=np.float32).reshape(3)
        self.workspace_max = np.asarray(self.workspace_max, dtype=np.float32).reshape(3)
        if np.any(self.workspace_min > self.workspace_max):
            raise ValueError("workspace_min must not exceed workspace_max")
        if self.voxel_size < 0:
            raise ValueError("voxel_size cannot be negative")
        if self.outlier_method not in {"statistical", "radius"}:
            raise ValueError("outlier_method must be 'statistical' or 'radius'")
        if self.outlier_mean_k < 1:
            raise ValueError("outlier_mean_k must be positive")
        if self.outlier_stddev < 0:
            raise ValueError("outlier_stddev cannot be negative")
        if self.outlier_radius <= 0:
            raise ValueError("outlier_radius must be positive")
        if self.outlier_min_neighbors < 1:
            raise ValueError("outlier_min_neighbors must be positive")
        if self.plane_distance_threshold <= 0:
            raise ValueError("plane_distance_threshold must be positive")
        if self.plane_iterations < 1:
            raise ValueError("plane_iterations must be positive")
        if self.plane_min_inliers < 3:
            raise ValueError("plane_min_inliers must be at least three")
        if not 0.0 <= self.plane_min_inlier_ratio <= 1.0:
            raise ValueError("plane_min_inlier_ratio must be in [0, 1]")
        if not 0.0 <= self.plane_max_angle_deg <= 90.0:
            raise ValueError("plane_max_angle_deg must be in [0, 90]")
        if self.plane_min_distance < 0.0:
            raise ValueError("plane_min_distance cannot be negative")
        if self.plane_max_distance < self.plane_min_distance:
            raise ValueError(
                "plane_max_distance must be at least plane_min_distance"
            )
        if self.plane_normal_axis is not None:
            axis = np.asarray(self.plane_normal_axis, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                raise ValueError("plane_normal_axis must be non-zero")
            self.plane_normal_axis = (axis / norm).astype(np.float32)


# Short name retained for callers constructing the pure geometry pipeline.
PreprocessorConfig = PointCloudPreprocessorConfig


@dataclass(slots=True)
class PlaneModel:
    coefficients: np.ndarray
    inlier_count: int
    inlier_ratio: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(coefficients[:3]))
        if norm <= 1e-12:
            raise ValueError("plane normal must be non-zero")
        self.coefficients = (coefficients / norm).astype(np.float32)
        self.inlier_count = int(self.inlier_count)
        self.inlier_ratio = float(self.inlier_ratio)

    @property
    def normal(self) -> np.ndarray:
        return self.coefficients[:3]

    @property
    def offset(self) -> float:
        return float(self.coefficients[3])

    def distances(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return np.abs(values @ self.normal + self.offset)


@dataclass(slots=True)
class PreprocessResult:
    raw_cloud: CloudFrame
    processed_cloud: CloudFrame
    plane: PlaneModel | None = None
    input_count: int = 0
    finite_count: int = 0
    workspace_count: int = 0
    voxel_count: int = 0
    outlier_count: int = 0
    plane_inlier_count: int = 0

    @property
    def cloud(self) -> CloudFrame:
        """Compatibility alias for the clustering input cloud."""

        return self.processed_cloud

    @property
    def removed_count(self) -> int:
        return max(int(self.input_count) - len(self.processed_cloud.points), 0)


class StaticBackgroundState(str, Enum):
    DISABLED = "disabled"
    WARMING_UP = "warming_up"
    CALIBRATING = "calibrating"
    READY = "ready"


@dataclass(slots=True)
class StaticBackgroundFilterConfig:
    """Startup-calibrated 3D background subtraction for a fixed camera."""

    enabled: bool = False
    warmup_frames: int = 0
    calibration_frames: int = 12
    voxel_size: float = 0.015
    distance_threshold: float = 0.03
    minimum_baseline_points: int = 100
    ray_depth_enabled: bool = False
    depth_axis: int = 1
    horizontal_axis: int = 0
    vertical_axis: int = 2
    ray_distance_threshold: float = 0.025
    alignment_enabled: bool = True
    alignment_min_points: int = 80
    alignment_ratio_tolerance: float = 0.04
    maximum_scale_change: float = 0.35
    alignment_color_distance: float = 55.0
    alignment_require_color: bool = False
    alignment_periphery_fraction: float = 0.18
    alignment_min_support_ratio: float = 0.60
    alignment_min_span_ratio: float = 0.65
    alignment_min_points_per_tile: int = 3
    alignment_min_occupied_tiles: int = 6
    alignment_window_frames: int = 7
    alignment_min_valid_frames: int = 5
    alignment_max_relative_mad: float = 0.0075
    alignment_max_window_range: float = 0.035
    alignment_max_step_fraction: float = 0.03
    alignment_max_upward_rate_per_sec: float = 0.01
    alignment_hold_frames: int = 2

    def __post_init__(self) -> None:
        if self.warmup_frames < 0:
            raise ValueError("background warmup_frames cannot be negative")
        if self.calibration_frames < 1:
            raise ValueError("background calibration_frames must be positive")
        if self.voxel_size <= 0.0:
            raise ValueError("background voxel_size must be positive")
        if self.distance_threshold <= 0.0:
            raise ValueError("background distance_threshold must be positive")
        if self.minimum_baseline_points < 3:
            raise ValueError(
                "background minimum_baseline_points must be at least three"
            )
        axes = (self.depth_axis, self.horizontal_axis, self.vertical_axis)
        if any(axis not in {0, 1, 2} for axis in axes) or len(set(axes)) != 3:
            raise ValueError(
                "background depth/horizontal/vertical axes must be a permutation of 0,1,2"
            )
        if self.ray_distance_threshold <= 0.0:
            raise ValueError("background ray_distance_threshold must be positive")
        if self.alignment_min_points < 3:
            raise ValueError("background alignment_min_points must be at least three")
        if self.alignment_ratio_tolerance <= 0.0:
            raise ValueError(
                "background alignment_ratio_tolerance must be positive"
            )
        if not 0.0 <= self.maximum_scale_change < 1.0:
            raise ValueError(
                "background maximum_scale_change must be in [0, 1)"
            )
        if self.alignment_color_distance < 0.0:
            raise ValueError(
                "background alignment_color_distance cannot be negative"
            )
        if not 0.0 <= self.alignment_periphery_fraction < 0.5:
            raise ValueError(
                "background alignment_periphery_fraction must be in [0, 0.5)"
            )
        if not 0.0 < self.alignment_min_support_ratio <= 1.0:
            raise ValueError(
                "background alignment_min_support_ratio must be in (0, 1]"
            )
        if not 0.0 <= self.alignment_min_span_ratio <= 1.0:
            raise ValueError(
                "background alignment_min_span_ratio must be in [0, 1]"
            )
        if self.alignment_min_points_per_tile < 1:
            raise ValueError(
                "background alignment_min_points_per_tile must be positive"
            )
        if not 1 <= self.alignment_min_occupied_tiles <= 10:
            raise ValueError(
                "background alignment_min_occupied_tiles must be in [1, 10]"
            )
        if self.alignment_window_frames < 1:
            raise ValueError("background alignment_window_frames must be positive")
        if not 1 <= self.alignment_min_valid_frames <= self.alignment_window_frames:
            raise ValueError(
                "background alignment_min_valid_frames must be in [1, window]"
            )
        if self.alignment_max_relative_mad < 0.0:
            raise ValueError(
                "background alignment_max_relative_mad cannot be negative"
            )
        if self.alignment_max_window_range < 0.0:
            raise ValueError(
                "background alignment_max_window_range cannot be negative"
            )
        if not 0.0 < self.alignment_max_step_fraction <= 1.0:
            raise ValueError(
                "background alignment_max_step_fraction must be in (0, 1]"
            )
        if self.alignment_max_upward_rate_per_sec <= 0.0:
            raise ValueError(
                "background alignment_max_upward_rate_per_sec must be positive"
            )
        if self.alignment_hold_frames < 0:
            raise ValueError("background alignment_hold_frames cannot be negative")


@dataclass(slots=True)
class StaticBackgroundFilterResult:
    cloud: CloudFrame
    state: StaticBackgroundState
    removed_points: int
    baseline_points: int
    warmup_progress: int
    calibration_progress: int
    matched_points: int = 0
    alignment_points: int = 0
    depth_scale: float = 1.0
    alignment_valid: bool = False
    alignment_candidate_scale: float = 1.0
    alignment_candidate_points: int = 0
    alignment_candidate_reason: str = "not_evaluated"


class StaticBackgroundFilter:
    """Learn fixed geometry once, then retain only newly introduced points.

    Input clouds may be unordered and voxel sampled differently each frame.
    A nearest-neighbour distance to a multi-frame voxelized baseline is used
    instead of pixel identity, which keeps this filter usable with ordinary
    unorganized PointCloud2 messages.
    """

    def __init__(
        self, config: StaticBackgroundFilterConfig | None = None
    ) -> None:
        self.config = config or StaticBackgroundFilterConfig()
        self._warmup_count = 0
        self._calibration_count = 0
        self._sample_points: list[np.ndarray] = []
        self._sample_colors: list[np.ndarray] = []
        self._samples_have_colors = True
        self._baseline_points = np.empty((0, 3), dtype=np.float32)
        self._baseline_colors: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._ray_tree: cKDTree | None = None
        self._ray_baseline_points = np.empty((0, 3), dtype=np.float32)
        self._ray_baseline_colors: np.ndarray | None = None
        self._ray_bounds: np.ndarray | None = None
        self._alignment_candidates: deque[float | None] = deque(
            maxlen=self.config.alignment_window_frames
        )
        self._alignment_scale = 1.0
        self._alignment_stamp: float | None = None
        self._alignment_has_lock = False
        self._alignment_invalid_frames = 0
        self._latest_alignment_candidate_scale = 1.0
        self._latest_alignment_candidate_points = 0
        self._latest_alignment_candidate_reason = "not_evaluated"

    @property
    def state(self) -> StaticBackgroundState:
        if not self.config.enabled:
            return StaticBackgroundState.DISABLED
        if self._tree is not None:
            return StaticBackgroundState.READY
        if self._warmup_count < self.config.warmup_frames:
            return StaticBackgroundState.WARMING_UP
        return StaticBackgroundState.CALIBRATING

    def reset(self) -> None:
        self._warmup_count = 0
        self._calibration_count = 0
        self._sample_points.clear()
        self._sample_colors.clear()
        self._samples_have_colors = True
        self._baseline_points = np.empty((0, 3), dtype=np.float32)
        self._baseline_colors = None
        self._tree = None
        self._ray_tree = None
        self._ray_baseline_points = np.empty((0, 3), dtype=np.float32)
        self._ray_baseline_colors = None
        self._ray_bounds = None
        self._alignment_candidates.clear()
        self._alignment_scale = 1.0
        self._alignment_stamp = None
        self._alignment_has_lock = False
        self._alignment_invalid_frames = 0
        self._latest_alignment_candidate_scale = 1.0
        self._latest_alignment_candidate_points = 0
        self._latest_alignment_candidate_reason = "not_evaluated"

    def filter(self, cloud: CloudFrame) -> StaticBackgroundFilterResult:
        if not self.config.enabled:
            return self._result(cloud, removed=0)
        if self._warmup_count < self.config.warmup_frames:
            self._warmup_count += 1
            # Preserve the input for generic callers.  The hand-only ROS node
            # checks the state and holds its safety publication until a
            # trustworthy baseline exists.
            return self._result(cloud, removed=0)
        if self._tree is None:
            finite = np.asarray(cloud.points, dtype=np.float32)
            valid = np.isfinite(finite).all(axis=1)
            finite = finite[valid]
            if len(finite):
                self._sample_points.append(finite.copy())
                if cloud.colors is None:
                    self._samples_have_colors = False
                elif self._samples_have_colors:
                    self._sample_colors.append(
                        np.asarray(cloud.colors, dtype=np.uint8)[valid].copy()
                    )
            self._calibration_count += 1
            if self._calibration_count >= self.config.calibration_frames:
                self._build_baseline()
            # Do not report an empty/safe scene from this generic filter.  The
            # hand-only ROS adapter fail-closes by withholding publication
            # while this result remains CALIBRATING.
            return self._result(cloud, removed=0)

        points = np.asarray(cloud.points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            return self._result(cloud, removed=0)
        if self.config.ray_depth_enabled and self._ray_tree is not None:
            keep, matched, alignment_points, depth_scale, alignment_valid = (
                self._foreground_by_ray_depth(cloud)
            )
            return self._result(
                cloud.select(keep),
                removed=int(np.count_nonzero(~keep)),
                matched_points=matched,
                alignment_points=alignment_points,
                depth_scale=depth_scale,
                alignment_valid=alignment_valid,
            )
        distances, _ = self._tree.query(points, k=1, workers=-1)
        keep = np.asarray(distances, dtype=np.float64) > float(
            self.config.distance_threshold
        )
        return self._result(
            cloud.select(keep),
            removed=int(np.count_nonzero(~keep)),
        )

    process = filter

    def _build_baseline(self) -> None:
        if not self._sample_points:
            return
        points = np.concatenate(self._sample_points, axis=0)
        colors = (
            np.concatenate(self._sample_colors, axis=0)
            if self._samples_have_colors
            and len(self._sample_colors) == len(self._sample_points)
            else None
        )
        if self.config.ray_depth_enabled:
            baseline, baseline_colors = self._aggregate_ray_baseline(
                points,
                colors,
            )
        else:
            keys = np.floor(points / self.config.voxel_size).astype(
                np.int64
            )
            _, selected = np.unique(keys, axis=0, return_index=True)
            selected = np.sort(selected)
            baseline = np.asarray(points[selected], dtype=np.float32)
            baseline_colors = (
                None
                if colors is None
                else np.asarray(colors[selected], dtype=np.uint8)
            )
        if len(baseline) < self.config.minimum_baseline_points:
            # Keep collecting instead of declaring a tiny/unrepresentative
            # baseline ready and leaking the whole fixed scene as foreground.
            return
        self._baseline_points = baseline
        self._baseline_colors = baseline_colors
        self._tree = cKDTree(baseline)
        rays, valid = self._ray_coordinates(baseline)
        if int(np.count_nonzero(valid)) >= self.config.minimum_baseline_points:
            valid_rays = rays[valid]
            self._ray_tree = cKDTree(valid_rays)
            self._ray_baseline_points = baseline[valid]
            self._ray_baseline_colors = (
                None
                if self._baseline_colors is None
                else self._baseline_colors[valid]
            )
            # Robust angular bounds define a peripheral reference band.  A
            # foreground object in the centre (for example a large hand) must
            # not become the majority vote used to rescale the whole scene.
            self._ray_bounds = np.quantile(
                valid_rays,
                (0.02, 0.98),
                axis=0,
            ).astype(np.float32)
        self._sample_points.clear()
        self._sample_colors.clear()

    def _aggregate_ray_baseline(
        self,
        points: np.ndarray,
        colors: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build one robust background depth for each angular ray cell.

        A multi-frame 3D voxel union can retain several depths at essentially
        the same camera ray. A nearest-ray query then picks an arbitrary
        calibration frame, which destroys global scale consensus even for a
        fixed scene. Median aggregation keeps a single stable surface while
        still rejecting isolated calibration noise.
        """

        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        rays, valid = self._ray_coordinates(values)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), None
        valid_indices = np.flatnonzero(valid)
        valid_rays = rays[valid]
        # Use cells much finer than the live nearest-ray acceptance radius.
        # This merges repeated samples of the same image neighbourhood
        # without blurring foreground boundaries across unrelated rays.
        cell_size = max(
            float(self.config.ray_distance_threshold) * 0.25,
            1e-4,
        )
        keys = np.rint(valid_rays / cell_size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        ordered_groups = inverse[order]
        starts = np.flatnonzero(
            np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
        )
        stops = np.r_[starts[1:], len(order)]

        aggregated = np.empty((len(starts), 3), dtype=np.float32)
        aggregated_colors = (
            None
            if colors is None
            else np.empty((len(starts), 3), dtype=np.uint8)
        )
        depth_axis = self.config.depth_axis
        horizontal_axis = self.config.horizontal_axis
        vertical_axis = self.config.vertical_axis
        source_colors = (
            None
            if colors is None
            else np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        )
        for output_index, (start, stop) in enumerate(
            zip(starts, stops, strict=True)
        ):
            members = valid_indices[order[start:stop]]
            depth = float(np.median(values[members, depth_axis]))
            ray = np.median(rays[members], axis=0)
            aggregated[output_index, depth_axis] = depth
            aggregated[output_index, horizontal_axis] = ray[0] * depth
            aggregated[output_index, vertical_axis] = ray[1] * depth
            if aggregated_colors is not None and source_colors is not None:
                aggregated_colors[output_index] = np.rint(
                    np.median(source_colors[members], axis=0)
                ).astype(np.uint8)
        return aggregated, aggregated_colors

    def _foreground_by_ray_depth(
        self,
        cloud: CloudFrame,
    ) -> tuple[np.ndarray, int, int, float, bool]:
        """Remove fixed surfaces by angular ray and signed depth residual.

        Monocular metric depth can drift by a common scale after many hours.
        Angular image rays remain stable under that scale, so current points
        are paired with the calibrated background by ``horizontal/depth`` and
        ``vertical/depth``.  A dominant RGB-consistent depth ratio corrects
        the global drift.  Only points measurably *closer* than the background
        survive; farther/missing background surfaces cannot become obstacles.
        """

        points = np.asarray(cloud.points, dtype=np.float32).reshape(-1, 3)
        self._latest_alignment_candidate_scale = 1.0
        self._latest_alignment_candidate_points = 0
        self._latest_alignment_candidate_reason = "not_evaluated"
        rays, valid_depth = self._ray_coordinates(points)
        keep = np.ones(len(points), dtype=bool)
        if self._ray_tree is None or not np.any(valid_depth):
            return keep, 0, 0, 1.0, False
        distances, indices = self._ray_tree.query(
            rays[valid_depth],
            k=1,
            distance_upper_bound=self.config.ray_distance_threshold,
            workers=-1,
        )
        local_matched = np.isfinite(distances) & (
            indices < len(self._ray_baseline_points)
        )
        current_indices = np.flatnonzero(valid_depth)
        matched_current = current_indices[local_matched]
        matched_baseline = np.asarray(indices[local_matched], dtype=np.int64)
        matched_count = len(matched_current)
        if matched_count == 0:
            return keep, 0, 0, 1.0, False

        depth_axis = self.config.depth_axis
        current_depth = points[matched_current, depth_axis].astype(np.float64)
        baseline_depth = self._ray_baseline_points[
            matched_baseline, depth_axis
        ].astype(np.float64)
        ratios = baseline_depth / np.maximum(current_depth, 1e-8)
        alignment_mask = (
            np.isfinite(ratios)
            & (ratios >= 1.0 - self.config.maximum_scale_change)
            & (ratios <= 1.0 + self.config.maximum_scale_change)
        )
        matched_rays = rays[matched_current]
        alignment_mask &= self._peripheral_alignment_mask(matched_rays)
        if (
            cloud.colors is not None
            and self._ray_baseline_colors is not None
            and self.config.alignment_color_distance > 0.0
        ):
            current_colors = np.asarray(cloud.colors, dtype=np.float32)[
                matched_current
            ]
            baseline_colors = self._ray_baseline_colors[
                matched_baseline
            ].astype(np.float32)
            color_distance = np.linalg.norm(
                current_colors - baseline_colors, axis=1
            )
            color_plausible = (
                alignment_mask
                & (color_distance <= self.config.alignment_color_distance)
            )
            if self.config.alignment_require_color or int(
                np.count_nonzero(color_plausible)
            ) >= int(self.config.alignment_min_points):
                alignment_mask = color_plausible
        elif self.config.alignment_require_color:
            # Failing open is deliberate: without the requested RGB evidence,
            # retaining a possible obstacle is safer than explaining it away
            # as monocular scale drift.
            alignment_mask[:] = False

        candidate_scale = 1.0
        candidate_points = 0
        if self.config.alignment_enabled:
            candidate_scale, candidate_points = self._dominant_depth_ratio(
                ratios[alignment_mask],
                matched_rays[alignment_mask],
            )
            self._latest_alignment_candidate_scale = float(candidate_scale)
            self._latest_alignment_candidate_points = int(candidate_points)
            depth_scale, alignment_points, alignment_valid = (
                self._validate_alignment_candidate(
                    candidate_scale,
                    candidate_points,
                    cloud.stamp,
                )
            )
            if not alignment_valid:
                # Applying scale=1 after a metric drift can place a genuinely
                # closer hand behind the old baseline and delete it. Bypass
                # subtraction for this frame instead of claiming empty/safe.
                return keep, matched_count, 0, 1.0, False
        else:
            depth_scale = 1.0
            alignment_points = 0
            alignment_valid = True
        corrected_depth = current_depth * depth_scale
        # Positive residual means a surface newly protrudes toward the camera.
        foreground = (
            baseline_depth - corrected_depth
            > float(self.config.distance_threshold)
        )
        keep[matched_current] = foreground
        return (
            keep,
            matched_count,
            alignment_points,
            depth_scale,
            alignment_valid,
        )

    def _ray_coordinates(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        depth = values[:, self.config.depth_axis]
        valid = np.isfinite(values).all(axis=1) & (depth > 1e-6)
        rays = np.zeros((len(values), 2), dtype=np.float32)
        rays[valid, 0] = (
            values[valid, self.config.horizontal_axis] / depth[valid]
        )
        rays[valid, 1] = (
            values[valid, self.config.vertical_axis] / depth[valid]
        )
        return rays, valid

    def _peripheral_alignment_mask(self, rays: np.ndarray) -> np.ndarray:
        values = np.asarray(rays, dtype=np.float32).reshape(-1, 2)
        fraction = float(self.config.alignment_periphery_fraction)
        if fraction <= 0.0 or self._ray_bounds is None:
            return np.ones(len(values), dtype=bool)
        lower = self._ray_bounds[0]
        upper = self._ray_bounds[1]
        span = np.maximum(upper - lower, 1e-6)
        inner_lower = lower + fraction * span
        inner_upper = upper - fraction * span
        return np.any(
            (values <= inner_lower) | (values >= inner_upper),
            axis=1,
        )

    def _dominant_depth_ratio(
        self,
        ratios: np.ndarray,
        rays: np.ndarray,
    ) -> tuple[float, int]:
        raw_values = np.asarray(ratios, dtype=np.float64).reshape(-1)
        ray_values = np.asarray(rays, dtype=np.float64).reshape(-1, 2)
        if len(raw_values) != len(ray_values):
            raise ValueError("depth ratios and rays must have the same length")
        order = np.argsort(raw_values)
        values = raw_values[order]
        ordered_rays = ray_values[order]
        if len(values) < self.config.alignment_min_points:
            self._latest_alignment_candidate_reason = (
                f"insufficient_points:{len(values)}/"
                f"{self.config.alignment_min_points}"
            )
            return 1.0, 0
        tolerance = float(self.config.alignment_ratio_tolerance)
        best_start = 0
        best_stop = 0
        start = 0
        for stop in range(len(values)):
            while values[stop] - values[start] > tolerance:
                start += 1
            if stop + 1 - start > best_stop - best_start:
                best_start, best_stop = start, stop + 1
        support = best_stop - best_start
        if (
            support < self.config.alignment_min_points
            or support / len(values)
            < float(self.config.alignment_min_support_ratio)
        ):
            self._latest_alignment_candidate_reason = (
                f"insufficient_consensus:{support}/{len(values)}="
                f"{support / len(values):.3f}"
            )
            return 1.0, 0
        scale = float(np.median(values[best_start:best_stop]))
        if self._ray_bounds is not None and self.config.alignment_min_span_ratio > 0:
            inlier_rays = ordered_rays[best_start:best_stop]
            observed_span = np.quantile(inlier_rays, 0.95, axis=0) - np.quantile(
                inlier_rays,
                0.05,
                axis=0,
            )
            reference_span = np.maximum(
                self._ray_bounds[1] - self._ray_bounds[0],
                1e-6,
            )
            if np.any(
                observed_span / reference_span
                < float(self.config.alignment_min_span_ratio)
            ):
                span_ratio = observed_span / reference_span
                self._latest_alignment_candidate_reason = (
                    "insufficient_span:"
                    f"{span_ratio[0]:.3f},{span_ratio[1]:.3f}"
                )
                return 1.0, 0
            normalized = np.clip(
                (inlier_rays - self._ray_bounds[0]) / reference_span,
                0.0,
                1.0 - np.finfo(np.float64).eps,
            )
            columns = np.floor(normalized[:, 0] * 4).astype(np.int64)
            rows = np.floor(normalized[:, 1] * 3).astype(np.int64)
            tile_counts = np.zeros((3, 4), dtype=np.int64)
            np.add.at(tile_counts, (rows, columns), 1)
            occupied = (
                tile_counts >= self.config.alignment_min_points_per_tile
            )
            peripheral = occupied.copy()
            peripheral[1, 1:3] = False
            if (
                int(np.count_nonzero(peripheral))
                < self.config.alignment_min_occupied_tiles
                or not np.any(occupied[:, 0])
                or not np.any(occupied[:, -1])
                or not np.any(occupied[0, :])
                or not np.any(occupied[-1, :])
            ):
                self._latest_alignment_candidate_reason = (
                    "insufficient_tiles:"
                    f"{int(np.count_nonzero(peripheral))}/"
                    f"{self.config.alignment_min_occupied_tiles};"
                    "edges="
                    f"{int(np.any(occupied[:, 0]))}"
                    f"{int(np.any(occupied[:, -1]))}"
                    f"{int(np.any(occupied[0, :]))}"
                    f"{int(np.any(occupied[-1, :]))}"
                )
                return 1.0, 0
        self._latest_alignment_candidate_reason = (
            f"accepted:{support}/{len(values)}="
            f"{support / len(values):.3f}"
        )
        return scale, support

    def _validate_alignment_candidate(
        self,
        candidate: float,
        support: int,
        stamp: float,
    ) -> tuple[float, int, bool]:
        """Temporally validate and slew-limit a global depth correction.

        Invalid or unsettled evidence intentionally bypasses subtraction in
        the caller. A candidate can never silently fall back to scale 1.
        """

        valid_candidate = (
            support >= self.config.alignment_min_points
            and np.isfinite(candidate)
            and candidate > 0.0
        )
        self._alignment_candidates.append(
            float(candidate) if valid_candidate else None
        )
        if not valid_candidate:
            return self._hold_alignment_lock(0)

        values = np.asarray(
            [
                value
                for value in self._alignment_candidates
                if value is not None
            ],
            dtype=np.float64,
        )
        if (
            len(self._alignment_candidates) < self.config.alignment_window_frames
            or len(values) < self.config.alignment_min_valid_frames
        ):
            # Scale 1 is known at baseline creation. Permit only a numerically
            # unchanged candidate while the first temporal window fills.
            if abs(float(candidate) - 1.0) <= 0.005:
                return 1.0, support, True
            if (
                self._alignment_has_lock
                and abs(candidate / self._alignment_scale - 1.0)
                <= self.config.alignment_max_window_range
            ):
                return self._hold_alignment_lock(support)
            return 1.0, 0, False

        target = float(np.median(values))
        relative = values / max(target, 1e-8) - 1.0
        relative_mad = float(np.median(np.abs(relative)))
        relative_range = float(np.max(values) / np.min(values) - 1.0)
        if (
            relative_mad > self.config.alignment_max_relative_mad
            or relative_range > self.config.alignment_max_window_range
        ):
            return self._hold_alignment_lock(0)

        previous = float(self._alignment_scale)
        now = float(stamp) if np.isfinite(stamp) else 0.0
        if self._alignment_stamp is None or now <= self._alignment_stamp:
            elapsed = 1.0 / 12.0
        else:
            elapsed = min(now - self._alignment_stamp, 1.0)
        if target > previous:
            maximum_fraction = min(
                self.config.alignment_max_step_fraction,
                self.config.alignment_max_upward_rate_per_sec * elapsed,
            )
        else:
            maximum_fraction = self.config.alignment_max_step_fraction
        lower = previous * (1.0 - maximum_fraction)
        upper = previous * (1.0 + maximum_fraction)
        self._alignment_scale = float(np.clip(target, lower, upper))
        self._alignment_stamp = now

        # A partial correction is not safe for subtraction: until the ramp is
        # close to its corroborated target, retain all points for this frame.
        if abs(self._alignment_scale / target - 1.0) > 0.01:
            return 1.0, 0, False
        self._alignment_has_lock = True
        self._alignment_invalid_frames = 0
        return self._alignment_scale, support, True

    def _hold_alignment_lock(
        self,
        support: int,
    ) -> tuple[float, int, bool]:
        """Bridge at most a couple of isolated evidence dropouts."""

        self._alignment_invalid_frames += 1
        if (
            self._alignment_has_lock
            and self._alignment_invalid_frames
            <= self.config.alignment_hold_frames
        ):
            return self._alignment_scale, int(support), True
        return 1.0, 0, False

    def _result(
        self,
        cloud: CloudFrame,
        *,
        removed: int,
        matched_points: int = 0,
        alignment_points: int = 0,
        depth_scale: float = 1.0,
        alignment_valid: bool = False,
    ) -> StaticBackgroundFilterResult:
        return StaticBackgroundFilterResult(
            cloud=cloud,
            state=self.state,
            removed_points=int(removed),
            baseline_points=len(self._baseline_points),
            warmup_progress=min(
                self._warmup_count, self.config.warmup_frames
            ),
            calibration_progress=min(
                self._calibration_count,
                self.config.calibration_frames,
            ),
            matched_points=int(matched_points),
            alignment_points=int(alignment_points),
            depth_scale=float(depth_scale),
            alignment_valid=bool(alignment_valid),
            alignment_candidate_scale=float(
                self._latest_alignment_candidate_scale
            ),
            alignment_candidate_points=int(
                self._latest_alignment_candidate_points
            ),
            alignment_candidate_reason=(
                self._latest_alignment_candidate_reason
            ),
        )


class PointCloudPreprocessor:
    def __init__(
        self,
        config: PointCloudPreprocessorConfig | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = PointCloudPreprocessorConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    def process(self, cloud: CloudFrame) -> PreprocessResult:
        input_count = len(cloud.points)
        finite_mask = np.isfinite(cloud.points).all(axis=1)
        finite = cloud.select(finite_mask)
        finite_count = len(finite.points)
        raw_cloud = self.workspace_cloud(finite)
        workspace_count = len(raw_cloud.points)

        processed = self._voxel_downsample(raw_cloud)
        voxel_count = len(processed.points)
        if self.config.remove_outliers and len(processed.points):
            processed = self._remove_outliers(processed)
        outlier_count = len(processed.points)

        plane = None
        plane_inlier_count = 0
        if self.config.remove_plane and len(processed.points) >= 3:
            plane, inliers = self._estimate_plane(processed.points)
            if plane is not None:
                plane_inlier_count = int(np.count_nonzero(inliers))
                processed = processed.select(~inliers)

        return PreprocessResult(
            raw_cloud=raw_cloud,
            processed_cloud=processed,
            plane=plane,
            input_count=input_count,
            finite_count=finite_count,
            workspace_count=workspace_count,
            voxel_count=voxel_count,
            outlier_count=outlier_count,
            plane_inlier_count=plane_inlier_count,
        )

    def workspace_cloud(self, cloud: CloudFrame) -> CloudFrame:
        """Return finite high-resolution points inside the configured workspace."""

        finite_mask = np.isfinite(cloud.points).all(axis=1)
        finite = cloud.select(finite_mask)
        workspace_mask = np.all(
            (finite.points >= self.config.workspace_min)
            & (finite.points <= self.config.workspace_max),
            axis=1,
        )
        return finite.select(workspace_mask)

    # Both spellings are intentionally supported because ROS node code tends to
    # call this stage ``preprocess`` while tests/pure pipelines use ``process``.
    preprocess = process

    def _voxel_downsample(self, cloud: CloudFrame) -> CloudFrame:
        if self.config.voxel_size <= 0 or len(cloud.points) <= 1:
            return cloud
        voxel_keys = np.floor(cloud.points / self.config.voxel_size).astype(np.int64)
        _, selected = np.unique(voxel_keys, axis=0, return_index=True)
        selected.sort()
        return cloud.select(selected)

    def _remove_outliers(self, cloud: CloudFrame) -> CloudFrame:
        points = cloud.points
        if len(points) <= 2:
            return cloud
        tree = cKDTree(points)
        if self.config.outlier_method == "radius":
            neighbor_counts = tree.query_ball_point(
                points,
                self.config.outlier_radius,
                return_length=True,
            )
            # scipy includes the query point itself.
            keep = np.asarray(neighbor_counts, dtype=np.int64) - 1 >= int(
                self.config.outlier_min_neighbors
            )
            return cloud.select(keep)

        neighbor_count = min(int(self.config.outlier_mean_k) + 1, len(points))
        if neighbor_count <= 1:
            return cloud
        distances, _ = tree.query(points, k=neighbor_count)
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim == 1:
            return cloud
        mean_neighbor_distance = np.mean(distances[:, 1:], axis=1)
        finite = np.isfinite(mean_neighbor_distance)
        if not finite.any():
            return cloud.select(finite)
        population = mean_neighbor_distance[finite]
        threshold = float(
            np.mean(population)
            + self.config.outlier_stddev * np.std(population)
        )
        keep = finite & (mean_neighbor_distance <= threshold)
        return cloud.select(keep)

    def _estimate_plane(
        self,
        points: np.ndarray,
    ) -> tuple[PlaneModel | None, np.ndarray]:
        count = len(points)
        empty = np.zeros(count, dtype=bool)
        required = max(
            int(self.config.plane_min_inliers),
            int(np.ceil(self.config.plane_min_inlier_ratio * count)),
        )
        if count < max(required, 3):
            return None, empty

        best_coefficients: np.ndarray | None = None
        best_inliers = empty
        best_error = float("inf")
        for _ in range(int(self.config.plane_iterations)):
            sample_indices = self._rng.choice(count, 3, replace=False)
            sample = np.asarray(points[sample_indices], dtype=np.float64)
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-10:
                continue
            normal /= norm
            if not self._normal_is_allowed(normal):
                continue
            offset = -float(normal @ sample[0])
            if not self._plane_distance_is_allowed(offset):
                continue
            distances = np.abs(points @ normal + offset)
            inliers = distances <= self.config.plane_distance_threshold
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < required:
                continue
            error = float(np.mean(distances[inliers]))
            if inlier_count > int(np.count_nonzero(best_inliers)) or (
                inlier_count == int(np.count_nonzero(best_inliers))
                and error < best_error
            ):
                best_coefficients = np.r_[normal, offset]
                best_inliers = inliers
                best_error = error

        if best_coefficients is None:
            return None, empty

        # Refine the winning model with all inliers, then re-evaluate once.
        inlier_points = np.asarray(points[best_inliers], dtype=np.float64)
        center = np.mean(inlier_points, axis=0)
        _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
        normal = vh[-1]
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        if np.dot(normal, best_coefficients[:3]) < 0:
            normal *= -1.0
        offset = -float(normal @ center)
        if not self._plane_distance_is_allowed(offset):
            return None, empty
        distances = np.abs(points @ normal + offset)
        refined_inliers = distances <= self.config.plane_distance_threshold
        refined_count = int(np.count_nonzero(refined_inliers))
        if refined_count < required or not self._normal_is_allowed(normal):
            return None, empty
        return (
            PlaneModel(
                np.r_[normal, offset],
                inlier_count=refined_count,
                inlier_ratio=refined_count / max(count, 1),
            ),
            refined_inliers,
        )

    def _normal_is_allowed(self, normal: np.ndarray) -> bool:
        axis = self.config.plane_normal_axis
        if axis is None:
            return True
        cosine = abs(float(np.dot(normal, axis)))
        minimum_cosine = float(
            np.cos(np.deg2rad(self.config.plane_max_angle_deg))
        )
        return cosine >= minimum_cosine

    def _plane_distance_is_allowed(self, offset: float) -> bool:
        # PlaneModel normalizes coefficients, so abs(offset) is the shortest
        # distance from the camera/tracking origin to the candidate plane.
        distance = abs(float(offset))
        return bool(
            self.config.plane_min_distance
            <= distance
            <= self.config.plane_max_distance
        )


__all__ = [
    "PlaneModel",
    "PointCloudPreprocessor",
    "PointCloudPreprocessorConfig",
    "PreprocessResult",
    "PreprocessorConfig",
    "StaticBackgroundFilter",
    "StaticBackgroundFilterConfig",
    "StaticBackgroundFilterResult",
    "StaticBackgroundState",
]
