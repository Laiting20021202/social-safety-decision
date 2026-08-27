from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realtime_safety.edgetam_tracker.mask_pointcloud_fusion import (
    FusionConfig,
    clean_mask,
    fuse_mask_with_cloud,
)
from realtime_safety.edgetam_tracker.models import (
    AABB,
    CloudFrame,
    Cluster3D,
    MaskQuality,
    OBB,
    PointCloudQuality,
)
from realtime_safety.edgetam_tracker.projection_utils import (
    ProjectionConfig,
    project_cluster,
    project_points,
)


@dataclass
class _CameraInfo:
    k: list[float]
    width: int
    height: int


def _camera() -> _CameraInfo:
    return _CameraInfo(
        k=[100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0],
        width=100,
        height=80,
    )


def _make_cluster(
    points: np.ndarray,
    *,
    cluster_id: int = 1,
    source_indices: np.ndarray | None = None,
) -> Cluster3D:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    distances = np.linalg.norm(points, axis=1)
    nearest = int(np.argmin(distances))
    return Cluster3D(
        cluster_id=cluster_id,
        points=points,
        centroid=np.mean(points, axis=0),
        median_center=np.median(points, axis=0),
        aabb=AABB(minimum, maximum),
        obb=OBB(
            (minimum + maximum) * 0.5,
            maximum - minimum,
            np.eye(3, dtype=np.float32),
        ),
        nearest_point=points[nearest],
        nearest_distance=float(distances[nearest]),
        point_count=len(points),
        source_indices=source_indices,
        density=1000.0,
        depth_variance=float(np.var(points[:, 1])),
        quality=PointCloudQuality.GOOD,
        quality_score=0.9,
    )


def test_camera_projection_rejects_points_behind_camera_and_applies_tf() -> None:
    internal_points = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.1, 1.0, 0.1],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Internal x-right/y-forward/z-up -> optical x-right/y-down/z-forward.
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    projected = project_points(
        internal_points,
        _camera(),
        tracking_to_camera=transform,
    )
    assert projected.source_indices.tolist() == [0, 1]
    np.testing.assert_allclose(projected.uv[0], [50.0, 40.0])
    np.testing.assert_allclose(projected.uv[1], [60.0, 30.0])


def test_projection_prompt_uses_dense_projected_pixels_not_hollow_box_center() -> None:
    ring_uv = []
    for x in range(35, 66, 5):
        ring_uv.extend([(x, 25), (x, 55)])
    for y in range(30, 51, 5):
        ring_uv.extend([(35, y), (65, y)])
    optical_points = np.array(
        [
            [(u - 50.0) / 100.0, (v - 40.0) / 100.0, 1.0]
            for u, v in ring_uv
        ],
        dtype=np.float32,
    )
    prompt = project_cluster(
        _make_cluster(optical_points),
        _camera(),
        frame_index=7,
        track_id=42,
        config=ProjectionConfig(
            projection_dilation_pixels=2,
            maximum_positive_points=5,
            minimum_positive_spacing_pixels=5.0,
            positive_boundary_margin_pixels=0.0,
        ),
    )
    assert prompt is not None
    assert prompt.track_id == 42
    assert prompt.frame_index == 7
    assert len(prompt.positive_points) >= 3
    for x, y in np.rint(prompt.positive_points).astype(int):
        assert prompt.projection_mask[y, x]
        assert np.linalg.norm(np.array([x, y]) - np.array([50, 40])) > 8.0
    assert prompt.box_xyxy[0] < 35
    assert prompt.box_xyxy[2] > 65


def _rgbd_fixture() -> tuple[CloudFrame, Cluster3D, np.ndarray]:
    height = width = 12
    y_pixels, x_pixels = np.indices((height, width))
    object_mask = (
        (x_pixels >= 3)
        & (x_pixels < 9)
        & (y_pixels >= 3)
        & (y_pixels < 9)
    )
    depth = np.where(object_mask, 1.0, 2.0).astype(np.float32)
    points = np.column_stack(
        (
            (x_pixels.reshape(-1) - 6) * 0.01,
            depth.reshape(-1),
            (y_pixels.reshape(-1) - 6) * 0.01,
        )
    ).astype(np.float32)
    pixels = np.column_stack(
        (x_pixels.reshape(-1), y_pixels.reshape(-1))
    ).astype(np.int32)
    colors = np.tile(
        np.array([[10, 120, 240]], dtype=np.uint8),
        (len(points), 1),
    )
    cloud = CloudFrame(
        points=points,
        colors=colors,
        pixels_uv=pixels,
        image_shape=(height, width),
        stamp=1.0,
        frame_id="tracking",
    )
    object_indices = np.flatnonzero(object_mask.reshape(-1))
    sparse_indices = object_indices[::7]
    cluster = _make_cluster(
        points[sparse_indices],
        source_indices=sparse_indices,
    )
    broad_mask = np.zeros((height, width), dtype=bool)
    broad_mask[2:10, 2:10] = True
    return cloud, cluster, broad_mask


def test_mask_fusion_uses_high_resolution_points_and_rejects_background_depth() -> None:
    cloud, cluster, mask = _rgbd_fixture()
    result = fuse_mask_with_cloud(
        mask,
        cloud,
        cluster,
        mask_quality=MaskQuality.GOOD,
        config=FusionConfig(
            erosion_iterations=0,
            minimum_component_pixels=4,
            aabb_gate_margin=0.03,
            absolute_depth_gate=0.08,
            center_gate_margin=0.04,
            minimum_fused_points=6,
        ),
    )
    assert result.used_mask
    assert not result.used_fallback
    assert result.fused_point_count > cluster.point_count
    assert result.fused_point_count == 36
    np.testing.assert_allclose(result.points[:, 1], 1.0)
    assert result.aabb.maximum[1] < 1.1
    assert np.any(
        np.all(
            np.isclose(result.points, result.nearest_point[None, :]),
            axis=1,
        )
    )


def test_invalid_mask_falls_back_to_pointcloud_instead_of_empty_obstacle() -> None:
    cloud, cluster, mask = _rgbd_fixture()
    result = fuse_mask_with_cloud(
        mask,
        cloud,
        cluster,
        mask_quality=MaskQuality.INVALID,
        config=FusionConfig(erosion_iterations=0),
    )
    assert result.used_fallback
    assert not result.used_mask
    assert result.fused_point_count == cluster.point_count
    assert result.reason == "mask_quality_invalid"
    assert np.isfinite(result.nearest_distance)


def test_fallback_treats_source_indices_as_ids_not_cloud_positions() -> None:
    cloud, cluster, mask = _rgbd_fixture()
    cluster.source_indices = (
        np.arange(cluster.point_count, dtype=np.int64) + 10_000
    )
    expected = cluster.points.copy()

    result = fuse_mask_with_cloud(
        mask,
        cloud,
        cluster,
        mask_quality=MaskQuality.INVALID,
        config=FusionConfig(erosion_iterations=0),
    )

    np.testing.assert_allclose(result.points, expected)
    assert result.source_indices is not None
    assert int(np.min(result.source_indices)) == 10_000


def test_fusion_depth_gate_uses_supplied_optical_z_not_tracking_axis() -> None:
    """A world/base axis is not a valid substitute for optical-camera depth."""

    object_points = np.column_stack(
        (
            np.linspace(0.98, 1.02, 8),
            np.zeros(8),
            np.linspace(-0.02, 0.02, 8),
        )
    ).astype(np.float32)
    background_points = object_points.copy()
    background_points[:, 0] += 0.40
    points = np.concatenate((object_points, background_points), axis=0)
    pixels = np.column_stack(
        (np.arange(len(points), dtype=np.int32), np.zeros(len(points), dtype=np.int32))
    )
    cloud = CloudFrame(
        points=points,
        pixels_uv=pixels,
        image_shape=(1, len(points)),
        stamp=2.0,
        frame_id="tracking",
    )
    cluster = _make_cluster(object_points)
    mask = np.ones(cloud.image_shape, dtype=bool)
    config = FusionConfig(
        erosion_iterations=0,
        minimum_component_pixels=1,
        aabb_gate_margin=0.50,
        absolute_depth_gate=0.08,
        relative_depth_gate=0.0,
        center_gate_margin=1.0,
        robust_distance_mad_scale=1e6,
        minimum_fused_points=4,
    )

    wrong_axis_result = fuse_mask_with_cloud(
        mask,
        cloud,
        cluster,
        mask_quality=MaskQuality.GOOD,
        config=config,
    )
    optical_depth_result = fuse_mask_with_cloud(
        mask,
        cloud,
        cluster,
        mask_quality=MaskQuality.GOOD,
        config=config,
        point_depths=points[:, 0],
        cluster_depths=object_points[:, 0],
    )

    assert wrong_axis_result.used_mask
    assert wrong_axis_result.fused_point_count == 16
    assert optical_depth_result.used_mask
    assert optical_depth_result.fused_point_count == 8
    assert optical_depth_result.aabb.maximum[0] < 1.1


def test_partial_mask_cannot_remove_the_closest_measured_surface() -> None:
    points = np.column_stack(
        (
            np.linspace(0.10, 1.00, 10),
            np.zeros(10),
            np.zeros(10),
        )
    ).astype(np.float32)
    cloud = CloudFrame(
        points=points,
        pixels_uv=np.column_stack(
            (
                np.arange(10, dtype=np.int32),
                np.zeros(10, dtype=np.int32),
            )
        ),
        image_shape=(1, 10),
        stamp=3.0,
        frame_id="tracking",
    )
    cluster = _make_cluster(
        points,
        source_indices=np.arange(10, dtype=np.int64),
    )
    far_half_mask = np.zeros((1, 10), dtype=bool)
    far_half_mask[:, 5:] = True
    result = fuse_mask_with_cloud(
        far_half_mask,
        cloud,
        cluster,
        mask_quality=MaskQuality.DEGRADED,
        config=FusionConfig(
            erosion_iterations=0,
            minimum_component_pixels=1,
            minimum_fused_points=3,
            absolute_depth_gate=0.01,
            relative_depth_gate=0.0,
        ),
    )

    assert result.used_mask
    assert result.nearest_distance <= cluster.nearest_distance + 1e-7
    assert np.any(
        np.all(
            np.isclose(result.points, cluster.nearest_point[None, :]),
            axis=1,
        )
    )


def test_mask_morphology_removes_small_disconnected_speckle() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[1, 1] = True
    cleaned = clean_mask(
        mask,
        config=FusionConfig(
            erosion_iterations=0,
            minimum_component_pixels=10,
        ),
    )
    assert cleaned[10, 10]
    assert not cleaned[1, 1]
