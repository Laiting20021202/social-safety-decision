import numpy as np
import pytest

from openarm_safety_bridge.geometry import (
    clustered_swept_boxes,
    estimate_bounded_velocity,
    limit_cloud_center_motion,
    minimum_cloud_to_capsules_distance,
    swept_axis_aligned_box,
)


def test_swept_box_covers_current_and_predicted_hand() -> None:
    center, size = swept_axis_aligned_box(
        np.array([0.0, 0.0, 0.3]),
        np.array([0.1, 0.08, 0.03]),
        np.array([0.0, 0.3, 0.0]),
        0.5,
    )
    assert center == pytest.approx([0.0, 0.075, 0.3])
    assert size == pytest.approx([0.1, 0.23, 0.03])


def test_two_second_prediction_covers_slow_intrusion_during_planning() -> None:
    center, size = swept_axis_aligned_box(
        np.array([-0.30, -0.10, 0.43]),
        np.array([0.16, 0.08, 0.05]),
        np.array([0.0, 0.02, 0.0]),
        2.0,
    )
    assert center == pytest.approx([-0.30, -0.08, 0.43])
    assert size == pytest.approx([0.16, 0.12, 0.05])


def test_velocity_uses_sensor_time_and_clamps_false_jumps() -> None:
    velocity = estimate_bounded_velocity(
        np.zeros(3),
        1.0,
        np.array([0.0, 1.0, 0.0]),
        1.1,
        np.zeros(3),
        smoothing=1.0,
        maximum_speed_mps=0.6,
    )
    assert velocity == pytest.approx([0.0, 0.6, 0.0])


def test_cloud_center_cannot_teleport_from_hand_to_robot() -> None:
    cloud = np.array(
        [[0.40, -0.02, 0.30], [0.42, 0.02, 0.30], [0.41, 0.00, 0.32]]
    )
    limited, center, was_limited = limit_cloud_center_motion(
        cloud,
        np.array([0.0, 0.0, 0.30]),
        10.0,
        10.1,
        maximum_speed_mps=0.20,
        slack_m=0.01,
    )
    assert was_limited
    assert np.linalg.norm(center - [0.0, 0.0, 0.30]) == pytest.approx(0.03)
    assert np.median(limited, axis=0) == pytest.approx(center)


def test_cloud_center_accepts_continuous_hand_motion() -> None:
    cloud = np.array([[0.01, 0.0, 0.30], [0.02, 0.0, 0.30]])
    limited, center, was_limited = limit_cloud_center_motion(
        cloud,
        np.array([0.0, 0.0, 0.30]),
        10.0,
        10.1,
        maximum_speed_mps=0.20,
        slack_m=0.01,
    )
    assert not was_limited
    assert limited == pytest.approx(cloud)
    assert center == pytest.approx([0.015, 0.0, 0.30])


def test_capsule_detects_hand_near_middle_of_long_link() -> None:
    chain = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    cloud = np.array([[0.10, 0.25, 0.0]])
    distance = minimum_cloud_to_capsules_distance(
        cloud, [chain], capsule_radius_m=0.06
    )
    assert distance == pytest.approx(0.04)


def test_robust_capsule_distance_ignores_one_bad_depth_point() -> None:
    chain = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    cloud = np.repeat([[0.30, 0.25, 0.0]], 200, axis=0)
    cloud[0] = [0.0, 0.25, 0.0]
    distance = minimum_cloud_to_capsules_distance(
        cloud, [chain], capsule_radius_m=0.06, distance_quantile=0.01
    )
    assert distance == pytest.approx(0.24)


def test_clustered_boxes_do_not_fill_gap_between_hand_regions() -> None:
    left = np.repeat([[-0.25, 0.0, 0.3]], 100, axis=0)
    right = np.repeat([[0.25, 0.0, 0.3]], 100, axis=0)
    boxes = clustered_swept_boxes(
        np.concatenate((left, right)),
        np.zeros(3),
        0.2,
        padding_m=0.02,
        maximum_boxes=2,
    )
    assert len(boxes) == 2
    assert all(size[0] < 0.10 for _, size in boxes)
