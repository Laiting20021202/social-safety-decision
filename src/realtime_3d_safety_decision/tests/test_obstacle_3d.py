import numpy as np

from realtime_safety.pipeline.obstacle_3d import ObstacleExtractor3D
from realtime_safety.types import Detection2D, PointCloudFrame


def test_mask_maps_to_only_its_3d_points() -> None:
    height, width = 20, 30
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pointmap = np.stack((u * 0.01, np.full_like(u, 3.0), v * 0.01), axis=-1).astype(np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[5:15, 10:20] = True
    detection = Detection2D(
        bbox_xyxy=np.array([10, 5, 20, 15], dtype=np.float32),
        class_id=0,
        class_name="person",
        confidence=0.9,
        centroid_xy=np.array([15, 10], dtype=np.float32),
        timestamp=1.0,
        mask=mask,
        track_id=4,
        image_size=(width, height),
    )
    cloud = PointCloudFrame(
        points=pointmap.reshape(-1, 3),
        colors=np.zeros((height * width, 3), dtype=np.uint8),
        confidence=np.ones(height * width, dtype=np.float32),
        pointmap=pointmap,
        frame_index=1,
        timestamp=1.0,
        anchor_frame_index=1,
        inference_ms=1.0,
        valid=True,
        source="test",
        dense_confidence=np.ones((height, width), dtype=np.float32),
    )
    extractor = ObstacleExtractor3D(minimum_points=10, voxel_size=0.0)
    observations, assigned = extractor.extract([detection], cloud)
    assert len(observations) == 1
    assert observations[0].track_id == 4
    assert observations[0].class_name == "person"
    assert assigned.sum() == mask.sum()
    assert 0.1 < observations[0].position_xyz[0] < 0.2
    diagnostic = extractor.last_diagnostics[0]
    assert diagnostic.mask_pixels == 100
    assert diagnostic.valid_depth_pixels == 64
    assert diagnostic.depth_min_m == 3.0
    assert diagnostic.depth_median_m == 3.0
    assert diagnostic.depth_max_m == 3.0
    assert diagnostic.output_points == observations[0].point_count
    assert diagnostic.reason == "ok"


def test_diagnostics_explain_detection_with_no_valid_depth() -> None:
    height, width = 20, 30
    pointmap = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[5:15, 10:20] = True
    detection = Detection2D(
        bbox_xyxy=np.array([10, 5, 20, 15], dtype=np.float32),
        class_id=0,
        class_name="person",
        confidence=0.9,
        centroid_xy=np.array([15, 10], dtype=np.float32),
        timestamp=1.0,
        mask=mask,
        track_id=4,
        image_size=(width, height),
    )
    cloud = PointCloudFrame(
        points=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.uint8),
        confidence=np.empty((0,), dtype=np.float32),
        pointmap=pointmap,
        frame_index=1,
        timestamp=1.0,
        anchor_frame_index=1,
        inference_ms=1.0,
        valid=False,
        source="test",
        dense_confidence=np.ones((height, width), dtype=np.float32),
    )
    extractor = ObstacleExtractor3D(minimum_points=10, voxel_size=0.0)

    observations, _ = extractor.extract([detection], cloud)

    assert observations == []
    diagnostic = extractor.last_diagnostics[0]
    assert diagnostic.valid_depth_pixels == 0
    assert diagnostic.depth_median_m is None
    assert diagnostic.output_points == 0
    assert diagnostic.reason == "insufficient_points_after_depth_filter"


def test_hand_mask_keeps_near_hand_layer_and_rejects_table_background() -> None:
    height, width = 60, 80
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pointmap = np.stack(
        (
            (u - width / 2) * 0.002,
            0.96 + 0.0005 * v,
            (v - height / 2) * 0.002,
        ),
        axis=-1,
    ).astype(np.float32)
    mask = np.ones((height, width), dtype=bool)
    # The actual hand occupies a minority of the neural mask.  Its surface is
    # closer than the table pixels that otherwise dominate the depth median.
    pointmap[18:42, 12:32, 1] = 0.76 + 0.002 * (u[18:42, 12:32] - 12)
    detection = Detection2D(
        bbox_xyxy=np.array([0, 0, width, height], dtype=np.float32),
        class_id=0,
        class_name="hand",
        confidence=0.95,
        centroid_xy=np.array([22, 30], dtype=np.float32),
        timestamp=1.0,
        mask=mask,
        track_id=7,
        image_size=(width, height),
    )
    cloud = PointCloudFrame(
        points=pointmap.reshape(-1, 3),
        colors=np.zeros((height * width, 3), dtype=np.uint8),
        confidence=np.ones(height * width, dtype=np.float32),
        pointmap=pointmap,
        frame_index=1,
        timestamp=1.0,
        anchor_frame_index=1,
        inference_ms=1.0,
        valid=True,
        source="test",
        dense_confidence=np.ones((height, width), dtype=np.float32),
    )

    observations, _ = ObstacleExtractor3D(
        minimum_points=12, voxel_size=0.0
    ).extract([detection], cloud)

    assert len(observations) == 1
    hand_points = observations[0].points
    assert float(np.median(hand_points[:, 1])) < 0.82
    assert float(np.max(hand_points[:, 1])) < 0.90
    assert observations[0].point_count >= 300
