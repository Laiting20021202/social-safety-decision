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
    observations, assigned = ObstacleExtractor3D(minimum_points=10, voxel_size=0.0).extract([detection], cloud)
    assert len(observations) == 1
    assert observations[0].track_id == 4
    assert observations[0].class_name == "person"
    assert assigned.sum() == mask.sum()
    assert 0.1 < observations[0].position_xyz[0] < 0.2
