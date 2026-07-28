from __future__ import annotations

import cv2
import numpy as np

from realtime_safety.config import SegmentationConfig
from realtime_safety.pipeline.robot_self_filter import RobotSelfFilter
from realtime_safety.types import Detection2D, FramePacket, PointCloudFrame


def _frame(bgr: np.ndarray, index: int = 0) -> FramePacket:
    height, width = bgr.shape[:2]
    return FramePacket(
        frame_index=index,
        source_timestamp=index / 10.0,
        capture_timestamp=index / 10.0,
        bgr=bgr,
        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        original_fps=10.0,
        original_width=width,
        original_height=height,
    )


def _detection(
    mask: np.ndarray,
    track_id: int = 1,
    class_name: str = "person",
) -> Detection2D:
    ys, xs = np.nonzero(mask)
    return Detection2D(
        bbox_xyxy=np.array(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=np.float32,
        ),
        class_id=0,
        class_name=class_name,
        confidence=0.8,
        centroid_xy=np.array([xs.mean(), ys.mean()], dtype=np.float32),
        timestamp=0.0,
        mask=mask,
        track_id=track_id,
        image_size=(mask.shape[1], mask.shape[0]),
    )


def _robot_image() -> np.ndarray:
    image = np.full((240, 320, 3), 90, dtype=np.uint8)
    # The configured HSV interval contains this saturated green robot casing.
    green_bgr = cv2.cvtColor(
        np.uint8([[[75, 180, 150]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    cv2.rectangle(image, (150, 0), (185, 125), green_bgr.tolist(), -1)
    cv2.rectangle(image, (156, 126), (181, 165), (230, 230, 230), -1)
    return image


def _config() -> SegmentationConfig:
    return SegmentationConfig(
        robot_self_filter=True,
        robot_mask_dilation_px=14,
        robot_tip_extension_px=34,
        robot_reject_overlap=0.65,
        robot_min_residual_pixels=80,
    )


def test_anchored_robot_false_person_is_rejected() -> None:
    image = _robot_image()
    robot_false_mask = np.zeros(image.shape[:2], dtype=bool)
    robot_false_mask[0:132, 145:190] = True
    robot_false_mask[118:168, 154:184] = True
    self_filter = RobotSelfFilter(_config())

    filtered = self_filter.filter_people(
        [_detection(robot_false_mask)],
        _frame(image),
    )

    assert filtered == []
    assert self_filter.latest_mask is not None
    assert self_filter.latest_mask[40, 165]
    assert self_filter.latest_mask[150, 168]


def test_robot_is_rejected_even_when_yolo_calls_it_a_chair() -> None:
    image = _robot_image()
    robot_mask = np.zeros(image.shape[:2], dtype=bool)
    robot_mask[0:168, 142:194] = True
    self_filter = RobotSelfFilter(_config())

    filtered = self_filter.filter_obstacles(
        [_detection(robot_mask, class_name="chair")],
        _frame(image),
    )

    assert filtered == []


def test_real_hand_overlapping_robot_is_kept_and_robot_pixels_are_trimmed() -> None:
    image = _robot_image()
    robot_part = np.zeros(image.shape[:2], dtype=bool)
    robot_part[30:155, 145:194] = True
    hand_part = np.zeros_like(robot_part)
    hand_part[75:145, 70:155] = True
    detection = _detection(robot_part | hand_part)
    self_filter = RobotSelfFilter(_config())

    filtered = self_filter.filter_people([detection], _frame(image))

    assert len(filtered) == 1
    result = filtered[0]
    assert result.mask is not None
    assert result.mask[100, 90]
    assert not result.mask[100, 170]
    assert result.bbox_xyxy[0] == 70
    assert result.bbox_xyxy[2] <= 155


def test_small_but_meaningful_hand_survives_high_robot_overlap() -> None:
    image = _robot_image()
    robot_part = np.zeros(image.shape[:2], dtype=bool)
    robot_part[0:170, 138:199] = True
    hand_part = np.zeros_like(robot_part)
    hand_part[95:125, 105:145] = True
    self_filter = RobotSelfFilter(_config())

    filtered = self_filter.filter_people(
        [_detection(robot_part | hand_part)],
        _frame(image),
    )

    assert len(filtered) == 1
    result = filtered[0]
    assert result.mask is not None
    assert np.count_nonzero(result.mask) >= 900
    assert result.mask[110, 115]
    assert not result.mask[110, 170]


def test_green_clothing_away_from_fixed_robot_base_is_not_filtered() -> None:
    image = np.full((240, 320, 3), 90, dtype=np.uint8)
    green_bgr = cv2.cvtColor(
        np.uint8([[[75, 180, 150]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    cv2.rectangle(image, (20, 70), (90, 210), green_bgr.tolist(), -1)
    person_mask = np.zeros(image.shape[:2], dtype=bool)
    person_mask[60:220, 10:100] = True
    detection = _detection(person_mask)
    self_filter = RobotSelfFilter(_config())

    filtered = self_filter.filter_people([detection], _frame(image))

    assert filtered == [detection]
    assert self_filter.latest_mask is None


def test_short_motion_blur_uses_bounded_previous_robot_mask() -> None:
    self_filter = RobotSelfFilter(_config())
    robot = _robot_image()
    assert self_filter._robot_mask(robot) is not None
    blurred = np.full_like(robot, 90)

    assert self_filter._robot_mask(blurred) is not None
    assert self_filter._robot_mask(blurred) is not None
    assert self_filter._robot_mask(blurred) is not None
    assert self_filter._robot_mask(blurred) is None


def test_robot_center_is_projected_and_temporally_held() -> None:
    image = _robot_image()
    frame = _frame(image)
    height, width = image.shape[:2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pointmap = np.stack(
        (
            (u - width / 2) * 0.001,
            np.full_like(u, 0.40, dtype=np.float32),
            (height / 2 - v) * 0.001,
        ),
        axis=-1,
    ).astype(np.float32)
    cloud = PointCloudFrame(
        points=pointmap.reshape(-1, 3),
        colors=np.zeros((height * width, 3), dtype=np.uint8),
        confidence=np.ones(height * width, dtype=np.float32),
        pointmap=pointmap,
        frame_index=0,
        timestamp=0.0,
        anchor_frame_index=0,
        inference_ms=0.0,
        valid=True,
        source="test",
    )
    self_filter = RobotSelfFilter(_config())
    self_filter.filter_obstacles([], frame)

    state = self_filter.estimate_arm_state(frame, cloud)
    assert state is not None
    assert abs(float(state.center_xyz[1]) - 0.40) < 1e-4
    assert state.mask_pixels > 100
    assert state.point_count > 20

    self_filter._last_core_mask = None
    held = self_filter.estimate_arm_state(_frame(image, 1), cloud)
    assert held is not None
    assert held.held_frames == 1
    np.testing.assert_allclose(held.center_xyz, state.center_xyz)
