from __future__ import annotations

import numpy as np

from realtime_safety.edgetam_tracker.hand_semantic_gate import HandDetection
from realtime_safety.edgetam_tracker.models import CloudFrame
from realtime_safety.edgetam_tracker.tracked_obstacle_node import (
    EdgeTAMPointCloudTrackerNode,
)


def _cloud(source_indices: list[int], offset: float) -> CloudFrame:
    count = len(source_indices)
    return CloudFrame(
        points=np.column_stack(
            (
                np.asarray(source_indices, dtype=np.float32) + offset,
                np.ones(count, dtype=np.float32),
                np.full(count, 0.4, dtype=np.float32),
            )
        ),
        colors=np.tile(np.asarray((10, 20, 30), dtype=np.uint8), (count, 1)),
        pixels_uv=np.column_stack(
            (np.asarray(source_indices, dtype=np.int32), np.full(count, 7))
        ),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        stamp=4.25,
        frame_id="rgbd_color_optical_frame",
        image_shape=(20, 30),
    )


def test_rgb_hand_depth_seed_unions_current_foreground_without_duplicates() -> None:
    foreground = _cloud([1, 2, 3], 0.0)
    rgb_depth_seed = _cloud([3, 4], 100.0)

    merged = EdgeTAMPointCloudTrackerNode._union_clouds(
        foreground,
        rgb_depth_seed,
    )

    assert merged.source_indices.tolist() == [1, 2, 3, 4]
    assert merged.stamp == 4.25
    assert merged.frame_id == "rgbd_color_optical_frame"
    assert merged.image_shape == (20, 30)
    # Duplicate ray 3 keeps the current background-motion sample; the RGB
    # hand seed contributes only the missing ray 4.
    assert merged.points[:, 0].tolist() == [1.0, 2.0, 3.0, 104.0]


def test_rgb_hand_seed_rejects_background_depth_between_fingers() -> None:
    cloud = CloudFrame(
        points=np.asarray(
            [
                (-0.03, 0.01, 0.55),
                (0.00, 0.01, 0.60),
                (0.03, 0.01, 0.82),
            ],
            dtype=np.float32,
        ),
        pixels_uv=np.asarray(((2, 2), (3, 2), (4, 2)), dtype=np.int32),
        source_indices=np.arange(3, dtype=np.int64),
        stamp=8.0,
        frame_id="rgbd_color_optical_frame",
        image_shape=(6, 8),
    )
    mask = np.zeros((6, 8), dtype=bool)
    mask[2, 2:5] = True
    detection = HandDetection(
        bbox_xyxy=np.asarray((2, 2, 5, 3), dtype=np.float32),
        confidence=0.9,
        mask=mask,
        image_size=(8, 6),
    )

    selected = EdgeTAMPointCloudTrackerNode._cloud_inside_hand_detections(
        cloud,
        [detection],
        (6, 8),
        tracking_to_camera=np.eye(4),
        foreground_depth_quantile=0.0,
        foreground_depth_span_m=0.10,
    )

    np.testing.assert_array_equal(selected.source_indices, np.asarray((0, 1)))
    assert float(selected.points[:, 2].max()) < 0.7
