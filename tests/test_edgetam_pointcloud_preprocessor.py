from __future__ import annotations

import numpy as np
import pytest

from realtime_safety.edgetam_tracker.cluster_extractor import (
    ClusterExtractor,
    ClusterExtractorConfig,
)
from realtime_safety.edgetam_tracker.models import CloudFrame, PointCloudQuality
from realtime_safety.edgetam_tracker.pointcloud_preprocessor import (
    PointCloudPreprocessor,
    PointCloudPreprocessorConfig,
    StaticBackgroundFilter,
    StaticBackgroundFilterConfig,
    StaticBackgroundState,
)
from realtime_safety.edgetam_tracker.robot_self_filter import (
    LinkSphere,
    RobotSelfFilter,
    RobotSelfFilterConfig,
    SelfFilterStatus,
)


def _cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> CloudFrame:
    count = len(points)
    if colors is None:
        colors = np.column_stack(
            (
                np.arange(count, dtype=np.uint8),
                np.full(count, 20, dtype=np.uint8),
                np.full(count, 30, dtype=np.uint8),
            )
        )
    return CloudFrame(
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        pixels_uv=np.column_stack((np.arange(count), np.arange(count) + 10)),
        source_indices=np.arange(100, 100 + count),
        stamp=12.5,
        frame_id="tracking",
        image_shape=(240, 320),
    )


def test_preprocessor_keeps_cropped_high_resolution_cloud_and_metadata() -> None:
    cloud = _cloud(
        np.array(
            [
                [np.nan, 0.0, 0.0],
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02],
                [0.25, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [np.inf, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    result = PointCloudPreprocessor(
        PointCloudPreprocessorConfig(
            workspace_min=np.array((0.0, -1.0, -1.0)),
            workspace_max=np.array((1.0, 1.0, 1.0)),
            voxel_size=0.1,
        )
    ).process(cloud)

    assert result.input_count == 6
    assert result.finite_count == 4
    assert result.workspace_count == 3
    assert len(result.raw_cloud.points) == 3
    assert len(result.processed_cloud.points) == 2
    assert result.raw_cloud.source_indices.tolist() == [101, 102, 103]
    assert result.processed_cloud.source_indices.tolist() == [101, 103]
    assert result.processed_cloud.frame_id == "tracking"
    assert result.processed_cloud.image_shape == (240, 320)
    assert np.isfinite(result.processed_cloud.points).all()


@pytest.mark.parametrize("method", ["radius", "statistical"])
def test_preprocessor_removes_isolated_outliers(method: str) -> None:
    rng = np.random.default_rng(4)
    dense = rng.normal(0.0, 0.008, size=(40, 3))
    points = np.vstack((dense, np.array(((2.0, 2.0, 2.0),))))
    config = PointCloudPreprocessorConfig(
        remove_outliers=True,
        outlier_method=method,
        outlier_radius=0.04,
        outlier_min_neighbors=3,
        outlier_mean_k=8,
        outlier_stddev=1.0,
    )

    result = PointCloudPreprocessor(config).process(_cloud(points))

    assert len(result.raw_cloud.points) == 41
    assert len(result.processed_cloud.points) == 40
    assert np.linalg.norm(result.processed_cloud.points, axis=1).max() < 0.1


def test_preprocessor_optionally_removes_a_ransac_plane() -> None:
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(0.2, 1.0, 10))
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    obstacle = np.column_stack(
        (
            np.linspace(0.1, 0.2, 20),
            np.linspace(0.5, 0.6, 20),
            np.full(20, 0.3),
        )
    )
    config = PointCloudPreprocessorConfig(
        remove_plane=True,
        plane_distance_threshold=0.005,
        plane_iterations=80,
        plane_min_inliers=60,
        plane_min_inlier_ratio=0.5,
        plane_normal_axis=np.array((0.0, 0.0, 1.0)),
        plane_max_angle_deg=10.0,
    )

    result = PointCloudPreprocessor(config).process(_cloud(np.vstack((plane, obstacle))))

    assert result.plane is not None
    assert result.plane_inlier_count == 100
    assert len(result.raw_cloud.points) == 120
    assert len(result.processed_cloud.points) == 20
    assert np.all(result.processed_cloud.points[:, 2] > 0.25)
    assert abs(float(result.plane.normal[2])) > 0.99


def test_background_plane_distance_gate_keeps_a_near_planar_obstacle() -> None:
    x, z = np.meshgrid(
        np.linspace(-0.25, 0.25, 12),
        np.linspace(-0.20, 0.20, 10),
    )
    near_obstacle = np.column_stack(
        (x.ravel(), np.full(x.size, 0.20), z.ravel())
    )
    result = PointCloudPreprocessor(
        PointCloudPreprocessorConfig(
            remove_plane=True,
            plane_distance_threshold=0.005,
            plane_iterations=80,
            plane_min_inliers=40,
            plane_min_inlier_ratio=0.30,
            plane_normal_axis=np.array((0.0, 1.0, 0.0)),
            plane_max_angle_deg=10.0,
            plane_min_distance=0.34,
            plane_max_distance=0.48,
        )
    ).process(_cloud(near_obstacle))

    assert result.plane is None
    assert result.plane_inlier_count == 0
    assert len(result.processed_cloud.points) == len(near_obstacle)


def test_static_background_baseline_retains_new_foreground_geometry() -> None:
    x, z = np.meshgrid(
        np.linspace(-0.25, 0.25, 15),
        np.linspace(-0.20, 0.20, 12),
    )
    background = np.column_stack(
        (x.ravel(), np.full(x.size, 0.40), z.ravel())
    ).astype(np.float32)
    hand = np.array(
        [
            [-0.04, 0.26, -0.03],
            [-0.02, 0.25, 0.00],
            [0.00, 0.24, 0.02],
            [0.02, 0.25, 0.01],
            [0.04, 0.26, -0.02],
        ],
        dtype=np.float32,
    )
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            warmup_frames=1,
            calibration_frames=2,
            voxel_size=0.01,
            distance_threshold=0.025,
            minimum_baseline_points=100,
        )
    )

    warmup = background_filter.filter(_cloud(background))
    first = background_filter.filter(_cloud(background))
    second = background_filter.filter(_cloud(background))
    foreground = background_filter.filter(
        _cloud(np.vstack((background, hand)))
    )

    assert warmup.state is StaticBackgroundState.CALIBRATING
    assert len(warmup.cloud.points) == len(background)
    assert first.state is StaticBackgroundState.CALIBRATING
    assert len(first.cloud.points) == len(background)
    assert second.state is StaticBackgroundState.READY
    assert len(second.cloud.points) == len(background)
    assert foreground.state is StaticBackgroundState.READY
    assert foreground.removed_points == len(background)
    assert len(foreground.cloud.points) == len(hand)
    np.testing.assert_allclose(foreground.cloud.points, hand)


def test_ray_depth_background_rejects_global_scale_drift_and_keeps_hand() -> None:
    ray_x, ray_z = np.meshgrid(
        np.linspace(-0.55, 0.55, 20),
        np.linspace(-0.40, 0.40, 16),
    )

    def points_on_rays(depth: float, selector: np.ndarray | None = None) -> np.ndarray:
        x = ray_x.ravel() if selector is None else ray_x.ravel()[selector]
        z = ray_z.ravel() if selector is None else ray_z.ravel()[selector]
        return np.column_stack((x * depth, np.full(len(x), depth), z * depth))

    background = points_on_rays(0.40).astype(np.float32)
    # Learned monocular depth later drifts by a common 15% scale. Angular rays
    # are unchanged, so this must not turn the wooden table into foreground.
    drift = 0.85
    center_rays = np.flatnonzero(
        (np.abs(ray_x.ravel()) < 0.16)
        & (np.abs(ray_z.ravel()) < 0.18)
    )
    hand = points_on_rays(0.25, center_rays).astype(np.float32) * drift
    # Real RGB-D occlusion replaces the table return on these rays; it does
    # not append a second point behind the hand.
    current = (background * drift).copy()
    current[center_rays] = hand
    background_colors = np.tile(
        np.array((105, 92, 75), dtype=np.uint8),
        (len(background), 1),
    )
    current_colors = background_colors.copy()
    current_colors[center_rays] = (205, 155, 125)
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            calibration_frames=2,
            voxel_size=0.005,
            distance_threshold=0.03,
            minimum_baseline_points=200,
            ray_depth_enabled=True,
            ray_distance_threshold=0.005,
            alignment_min_points=100,
            alignment_ratio_tolerance=0.02,
            maximum_scale_change=0.25,
            alignment_require_color=True,
            alignment_periphery_fraction=0.18,
            alignment_min_support_ratio=0.60,
            alignment_min_span_ratio=0.65,
            alignment_window_frames=3,
            alignment_min_valid_frames=3,
            alignment_max_step_fraction=1.0,
            alignment_max_upward_rate_per_sec=10.0,
        )
    )

    background_filter.filter(_cloud(background, background_colors))
    ready = background_filter.filter(_cloud(background, background_colors))
    background_filter.filter(_cloud(current, current_colors))
    background_filter.filter(_cloud(current, current_colors))
    result = background_filter.filter(_cloud(current, current_colors))

    assert ready.state is StaticBackgroundState.READY
    assert result.matched_points == len(current)
    assert result.alignment_points >= 100
    assert result.depth_scale == pytest.approx(1.0 / drift, rel=0.01)
    assert result.removed_points == len(background) - len(hand)
    assert len(result.cloud.points) == len(hand)
    np.testing.assert_allclose(result.cloud.points, hand)


def test_ray_baseline_uses_one_median_surface_per_angular_cell() -> None:
    ray_x, ray_z = np.meshgrid(
        np.linspace(-0.55, 0.55, 24),
        np.linspace(-0.40, 0.40, 18),
    )
    flat_x = ray_x.ravel()
    flat_z = ray_z.ravel()
    spatial_noise = 0.08 * np.sin(9.0 * flat_x) * np.cos(7.0 * flat_z)

    def frame(noise_scale: float) -> np.ndarray:
        depth = 0.40 * (1.0 + noise_scale * spatial_noise)
        return np.column_stack((flat_x * depth, depth, flat_z * depth)).astype(
            np.float32
        )

    colors = np.tile(np.array((100, 90, 75), np.uint8), (flat_x.size, 1))
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            calibration_frames=5,
            voxel_size=0.004,
            distance_threshold=0.018,
            minimum_baseline_points=300,
            ray_depth_enabled=True,
            ray_distance_threshold=0.005,
            alignment_min_points=100,
            alignment_ratio_tolerance=0.02,
            alignment_periphery_fraction=0.18,
            alignment_min_support_ratio=0.60,
            alignment_min_span_ratio=0.65,
            alignment_min_points_per_tile=3,
            alignment_min_occupied_tiles=6,
            alignment_window_frames=3,
            alignment_min_valid_frames=3,
            alignment_max_step_fraction=1.0,
            alignment_max_upward_rate_per_sec=10.0,
        )
    )

    for noise_scale in (1.0, -0.5, 0.0, 0.5, -1.0):
        ready = background_filter.filter(_cloud(frame(noise_scale), colors))
    background_filter.filter(_cloud(frame(0.0), colors))
    background_filter.filter(_cloud(frame(0.0), colors))
    result = background_filter.filter(_cloud(frame(0.0), colors))

    assert ready.state is StaticBackgroundState.READY
    assert ready.baseline_points == flat_x.size
    assert result.alignment_valid
    assert result.alignment_candidate_reason.startswith("accepted:")
    assert result.depth_scale == pytest.approx(1.0, abs=0.005)
    assert result.removed_points == flat_x.size
    assert len(result.cloud.points) == 0


def test_ray_depth_background_recovers_from_day_scale_drift() -> None:
    ray_x, ray_z = np.meshgrid(
        np.linspace(-0.55, 0.55, 24),
        np.linspace(-0.40, 0.40, 18),
    )

    def at_depth(depth: float) -> np.ndarray:
        return np.column_stack(
            (
                ray_x.ravel() * depth,
                np.full(ray_x.size, depth),
                ray_z.ravel() * depth,
            )
        ).astype(np.float32)

    background = at_depth(0.40)
    drift = 0.55
    current = background * drift
    colors = np.tile(
        np.array((100, 90, 75), np.uint8),
        (len(background), 1),
    )
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            calibration_frames=2,
            voxel_size=0.004,
            distance_threshold=0.018,
            minimum_baseline_points=300,
            ray_depth_enabled=True,
            ray_distance_threshold=0.005,
            alignment_min_points=100,
            alignment_ratio_tolerance=0.02,
            maximum_scale_change=0.85,
            alignment_require_color=False,
            alignment_periphery_fraction=0.18,
            alignment_min_support_ratio=0.60,
            alignment_min_span_ratio=0.65,
            alignment_min_points_per_tile=3,
            alignment_min_occupied_tiles=6,
            alignment_window_frames=3,
            alignment_min_valid_frames=3,
            alignment_max_step_fraction=1.0,
            alignment_max_upward_rate_per_sec=10.0,
        )
    )
    background_filter.filter(_cloud(background, colors))
    background_filter.filter(_cloud(background, colors))
    background_filter.filter(_cloud(current, colors))
    background_filter.filter(_cloud(current, colors))

    result = background_filter.filter(_cloud(current, colors))

    assert result.alignment_valid
    assert result.depth_scale == pytest.approx(1.0 / drift, rel=0.01)
    assert result.removed_points == len(current)
    assert len(result.cloud.points) == 0


def test_ray_alignment_does_not_absorb_a_large_central_planar_obstacle() -> None:
    ray_x, ray_z = np.meshgrid(
        np.linspace(-0.55, 0.55, 28),
        np.linspace(-0.40, 0.40, 22),
    )
    flat_x = ray_x.ravel()
    flat_z = ray_z.ravel()

    def points_on_rays(depth: float) -> np.ndarray:
        return np.column_stack(
            (flat_x * depth, np.full(len(flat_x), depth), flat_z * depth)
        ).astype(np.float32)

    background = points_on_rays(0.40)
    drift = 0.85
    current = background * drift
    obstacle_rays = (
        (np.abs(flat_x) < 0.42)
        & (np.abs(flat_z) < 0.30)
    )
    # The sheet-like obstacle covers a majority of all rays and has the same
    # colour as the table. Only the distributed peripheral background vote is
    # therefore allowed to determine the global scale.
    obstacle = points_on_rays(0.37)[obstacle_rays] * drift
    current[obstacle_rays] = obstacle
    colors = np.tile(
        np.array((105, 92, 75), dtype=np.uint8),
        (len(background), 1),
    )
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            calibration_frames=2,
            voxel_size=0.004,
            distance_threshold=0.018,
            minimum_baseline_points=300,
            ray_depth_enabled=True,
            ray_distance_threshold=0.005,
            alignment_min_points=100,
            alignment_ratio_tolerance=0.02,
            maximum_scale_change=0.35,
            alignment_require_color=True,
            alignment_periphery_fraction=0.18,
            alignment_min_support_ratio=0.60,
            alignment_min_span_ratio=0.65,
            alignment_window_frames=3,
            alignment_min_valid_frames=3,
            alignment_max_step_fraction=1.0,
            alignment_max_upward_rate_per_sec=10.0,
        )
    )

    background_filter.filter(_cloud(background, colors))
    background_filter.filter(_cloud(background, colors))
    background_filter.filter(_cloud(current, colors))
    background_filter.filter(_cloud(current, colors))
    result = background_filter.filter(_cloud(current, colors))

    assert np.count_nonzero(obstacle_rays) > len(background) / 2
    assert result.depth_scale == pytest.approx(1.0 / drift, rel=0.01)
    assert len(result.cloud.points) == np.count_nonzero(obstacle_rays)
    np.testing.assert_allclose(result.cloud.points, obstacle)


def test_invalid_ray_alignment_bypasses_subtraction_instead_of_losing_hand() -> None:
    ray_x, ray_z = np.meshgrid(
        np.linspace(-0.4, 0.4, 20),
        np.linspace(-0.3, 0.3, 16),
    )
    flat_x = ray_x.ravel()
    flat_z = ray_z.ravel()

    def at_depth(depth: float) -> np.ndarray:
        return np.column_stack(
            (flat_x * depth, np.full(len(flat_x), depth), flat_z * depth)
        ).astype(np.float32)

    background = at_depth(0.40)
    current = at_depth(0.40) * 1.15
    hand_rays = (np.abs(flat_x) < 0.12) & (np.abs(flat_z) < 0.12)
    current[hand_rays] = at_depth(0.36)[hand_rays] * 1.15
    colors = np.tile(np.array((90, 80, 70), np.uint8), (len(background), 1))
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            calibration_frames=2,
            voxel_size=0.004,
            distance_threshold=0.018,
            minimum_baseline_points=200,
            ray_depth_enabled=True,
            ray_distance_threshold=0.005,
            alignment_min_points=100,
            alignment_require_color=True,
        )
    )
    background_filter.filter(_cloud(background, colors))
    background_filter.filter(_cloud(background, colors))
    missing_rgb = _cloud(current, colors)
    missing_rgb.colors = None

    result = background_filter.filter(missing_rgb)

    assert not result.alignment_valid
    assert result.alignment_points == 0
    assert result.removed_points == 0
    assert len(result.cloud.points) == len(current)


def test_ray_depth_background_does_not_treat_farther_surface_as_obstacle() -> None:
    rays = np.linspace(-0.3, 0.3, 120, dtype=np.float32)
    background = np.column_stack(
        (rays * 0.4, np.full(len(rays), 0.4), np.zeros(len(rays)))
    )
    farther = np.column_stack(
        (rays * 0.5, np.full(len(rays), 0.5), np.zeros(len(rays)))
    )
    background_filter = StaticBackgroundFilter(
        StaticBackgroundFilterConfig(
            enabled=True,
            calibration_frames=1,
            voxel_size=0.002,
            distance_threshold=0.02,
            minimum_baseline_points=100,
            ray_depth_enabled=True,
            ray_distance_threshold=0.004,
            alignment_enabled=False,
        )
    )

    background_filter.filter(_cloud(background))
    result = background_filter.filter(_cloud(farther))

    assert result.matched_points == len(farther)
    assert result.removed_points == len(farther)
    assert len(result.cloud.points) == 0


def test_robot_self_filter_reports_disabled_and_unavailable_without_hiding_cloud() -> None:
    cloud = _cloud(np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))))

    disabled = RobotSelfFilter(RobotSelfFilterConfig(enabled=False)).filter(cloud)
    assert disabled.status is SelfFilterStatus.DISABLED
    assert disabled.cloud is cloud
    assert disabled.safe_to_continue

    unavailable = RobotSelfFilter(
        RobotSelfFilterConfig(enabled=True, fail_closed=True)
    ).filter(cloud, None)
    assert unavailable.status is SelfFilterStatus.UNAVAILABLE
    assert unavailable.cloud is cloud
    assert len(unavailable.cloud.points) == 2
    assert not unavailable.safe_to_continue
    assert "no transformed" in unavailable.reason


def test_robot_self_filter_removes_points_inside_padded_link_spheres() -> None:
    cloud = _cloud(
        np.array(
            (
                (0.0, 0.0, 0.0),
                (0.11, 0.0, 0.0),
                (0.3, 0.0, 0.0),
            )
        )
    )
    self_filter = RobotSelfFilter(
        RobotSelfFilterConfig(enabled=True, padding=0.02)
    )

    result = self_filter.filter(
        cloud,
        {"wrist": LinkSphere(center=np.zeros(3), radius=0.1)},
    )

    assert result.status is SelfFilterStatus.ACTIVE
    assert result.available
    assert result.removed_points == 2
    assert result.sphere_count == 1
    assert result.cloud.source_indices.tolist() == [102]
    np.testing.assert_allclose(result.cloud.points[0], (0.3, 0.0, 0.0))


@pytest.mark.parametrize("method", ["euclidean", "dbscan"])
def test_cluster_extractor_returns_geometry_and_source_correspondence(method: str) -> None:
    rng = np.random.default_rng(8)
    first = rng.normal((0.3, 0.4, 0.2), 0.01, size=(30, 3))
    second = rng.normal((1.0, 0.8, 0.1), 0.012, size=(35, 3))
    noise = np.array(((3.0, 3.0, 3.0),), dtype=np.float32)
    cloud = _cloud(np.vstack((first, second, noise)))
    extractor = ClusterExtractor(
        ClusterExtractorConfig(
            method=method,
            tolerance=0.06,
            min_points=10,
            max_points=100,
            dbscan_min_samples=4,
            min_dimension=0.01,
            max_dimension=0.2,
        )
    )

    clusters = extractor.extract(cloud, robot_origin=np.zeros(3))

    assert len(clusters) == 2
    assert [cluster.cluster_id for cluster in clusters] == [0, 1]
    assert sorted(cluster.point_count for cluster in clusters) == [30, 35]
    for cluster in clusters:
        assert cluster.quality is PointCloudQuality.GOOD
        assert cluster.colors is not None
        assert cluster.pixels_uv is not None
        assert cluster.source_indices is not None
        assert cluster.density > 0
        assert cluster.depth_variance >= 0
        assert cluster.nearest_distance == pytest.approx(
            float(np.linalg.norm(cluster.nearest_point))
        )
        assert np.all(cluster.aabb.size >= 0)
        assert np.all(cluster.obb.size >= 0)
        np.testing.assert_allclose(
            cluster.obb.rotation.T @ cluster.obb.rotation,
            np.eye(3),
            atol=1e-5,
        )
        assert np.linalg.det(cluster.obb.rotation) == pytest.approx(1.0, abs=1e-5)


def test_cluster_extractor_filters_point_count_and_physical_size() -> None:
    rng = np.random.default_rng(14)
    accepted = rng.normal((0.5, 0.5, 0.5), 0.02, size=(20, 3))
    too_small = rng.normal((1.0, 1.0, 1.0), 0.001, size=(20, 3))
    too_few = rng.normal((1.5, 1.5, 1.5), 0.01, size=(4, 3))
    extractor = ClusterExtractor(
        method="euclidean",
        tolerance=0.08,
        min_points=5,
        max_points=30,
        min_dimension=0.02,
        max_dimension=0.2,
    )

    clusters = extractor.extract(_cloud(np.vstack((accepted, too_small, too_few))))

    assert len(clusters) == 1
    assert clusters[0].point_count == len(accepted)
    np.testing.assert_allclose(clusters[0].median_center, (0.5, 0.5, 0.5), atol=0.02)


def test_cluster_missing_depth_ratio_detects_pixel_hole_and_dense_footprint() -> None:
    dense_pixels = np.array(
        [(u, v) for v in range(3, 6) for u in range(4, 7)],
        dtype=np.int32,
    )

    def cloud_from_pixels(pixels: np.ndarray | None) -> CloudFrame:
        effective_pixels = dense_pixels if pixels is None else pixels
        points = np.column_stack(
            (
                effective_pixels[:, 0] * 0.01,
                effective_pixels[:, 1] * 0.01,
                np.ones(len(effective_pixels)),
            )
        ).astype(np.float32)
        return CloudFrame(
            points=points,
            pixels_uv=pixels,
            stamp=1.0,
            frame_id="camera",
            image_shape=(10, 12),
        )

    extractor = ClusterExtractor(
        method="euclidean",
        tolerance=0.015,
        min_points=3,
    )
    dense = extractor.extract(cloud_from_pixels(dense_pixels))
    pixels_with_center_hole = dense_pixels[
        ~np.all(dense_pixels == np.array((5, 4)), axis=1)
    ]
    holed = extractor.extract(cloud_from_pixels(pixels_with_center_hole))
    unknown = extractor.extract(cloud_from_pixels(None))
    sparse_pixels = dense_pixels[::2]
    sparse = extractor.extract(cloud_from_pixels(sparse_pixels))

    assert len(dense) == len(holed) == len(unknown) == len(sparse) == 1
    assert dense[0].missing_depth_ratio == pytest.approx(0.0)
    assert holed[0].missing_depth_ratio == pytest.approx(1.0 / 9.0)
    assert unknown[0].missing_depth_ratio == pytest.approx(0.0)
    assert sparse[0].missing_depth_ratio == pytest.approx(0.0)


def test_close_sparse_returns_survive_normal_cluster_minimum() -> None:
    close = np.array(
        [
            [0.14, 0.00, 0.00],
            [0.15, 0.01, 0.00],
            [0.16, 0.00, 0.01],
        ],
        dtype=np.float32,
    )
    far = close + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cloud = CloudFrame(
        points=np.concatenate((close, far), axis=0),
        stamp=1.0,
        frame_id="tracking",
    )
    extractor = ClusterExtractor(
        ClusterExtractorConfig(
            method="euclidean",
            tolerance=0.04,
            min_points=8,
            emergency_distance=0.25,
            emergency_min_points=3,
        )
    )

    clusters = extractor.extract(
        cloud,
        robot_origin=np.zeros(3, dtype=np.float32),
    )

    assert len(clusters) == 1
    assert clusters[0].point_count == 3
    assert clusters[0].quality is PointCloudQuality.INVALID
    assert clusters[0].nearest_distance < 0.25
