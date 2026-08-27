import cv2
import numpy as np

from openarm_perception_adapter.realtime_obstacle_resampler import (
    filter_robot_self_points,
    mask_metric_support,
    metric_spatial_gate,
    nearest_timestamp,
    robot_dominated_candidate,
    seed_mask_from_model,
    track_mask,
)


def test_nearest_timestamp_accepts_adjacent_rgbd_frame() -> None:
    assert nearest_timestamp((1_000_000_000, 1_067_000_000), 1_033_000_000, 80_000_000) == 1_000_000_000


def test_nearest_timestamp_rejects_expired_frame() -> None:
    assert nearest_timestamp((1_000_000_000,), 1_100_000_001, 80_000_000) is None


def test_robot_self_filter_removes_openarm_points_but_keeps_human_hand() -> None:
    points = np.asarray(
        [
            [0.01, 0.00, 0.50],
            [0.08, 0.00, 0.50],
            [0.30, 0.20, 0.55],
        ],
        dtype=np.float32,
    )
    filtered, removed = filter_robot_self_points(
        points,
        np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([0.06], dtype=np.float32),
        padding_m=0.005,
    )

    assert removed == 1
    np.testing.assert_allclose(filtered, points[1:])


def test_robot_self_filter_covers_moving_link_between_tf_origins() -> None:
    points = np.asarray(
        [
            [0.50, 0.00, 0.50],  # middle of a long articulated link
            [0.50, 0.12, 0.50],  # nearby human-hand surface remains visible
        ],
        dtype=np.float32,
    )
    filtered, removed = filter_robot_self_points(
        points,
        np.asarray([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([0.06, 0.06], dtype=np.float32),
        padding_m=0.005,
        segment_pairs=[(0, 1)],
    )

    assert removed == 1
    np.testing.assert_allclose(filtered, points[1:])


def test_robot_self_filter_rejects_invalid_segment_topology() -> None:
    with np.testing.assert_raises(ValueError):
        filter_robot_self_points(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([0.05], dtype=np.float32),
            segment_pairs=[(0, 2)],
        )


def test_robot_dominated_candidate_rejects_thin_openarm_residue() -> None:
    assert robot_dominated_candidate(
        1000,
        80,
        minimum_retained_fraction=0.15,
    )


def test_robot_dominated_candidate_keeps_real_hand_near_robot() -> None:
    assert not robot_dominated_candidate(
        1000,
        720,
        minimum_retained_fraction=0.15,
    )


def test_model_cloud_seeds_matching_organized_depth_pixels() -> None:
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width]
    xyz = np.column_stack(
        ((xx.ravel() - 16) * 0.01, (yy.ravel() - 12) * 0.01, np.ones(height * width))
    ).astype(np.float32)
    selected = (xx >= 11) & (xx <= 17) & (yy >= 7) & (yy <= 15)
    mask = seed_mask_from_model(xyz, xyz[selected.ravel()], (height, width))
    assert np.count_nonzero(mask[selected]) == np.count_nonzero(selected)
    assert np.count_nonzero(mask) < height * width


def test_model_projection_does_not_fill_unrelated_aabb_pixels() -> None:
    height, width = 80, 100
    yy, xx = np.mgrid[:height, :width]
    depth = np.ones((height, width), dtype=np.float32)
    xyz = np.stack(
        (
            (xx - 50) * depth / 120.0,
            (yy - 40) * depth / 120.0,
            depth,
        ),
        axis=-1,
    ).astype(np.float32)
    hand = (xx >= 18) & (xx <= 35) & (yy >= 25) & (yy <= 48)
    distractor = (xx >= 55) & (xx <= 72) & (yy >= 25) & (yy <= 48)
    model_points = xyz[hand][::5]
    mask = seed_mask_from_model(xyz, model_points, (height, width))

    assert np.count_nonzero(mask[hand]) > 0
    assert np.count_nonzero(mask[distractor]) == 0


def test_sparse_rgb_flow_moves_seed_mask() -> None:
    previous = np.zeros((100, 120), dtype=np.uint8)
    mask = np.zeros_like(previous)
    rng = np.random.default_rng(7)
    previous[30:70, 35:75] = rng.integers(
        0, 256, size=(40, 40), dtype=np.uint8
    )
    mask[30:70, 35:75] = 255
    transform = np.float32(((1, 0, 9), (0, 1, 5)))
    current = cv2.warpAffine(previous, transform, (120, 100))
    moved = track_mask(previous, current, mask)
    old_center = np.mean(np.argwhere(mask > 0), axis=0)
    new_center = np.mean(np.argwhere(moved > 0), axis=0)
    np.testing.assert_allclose(new_center - old_center, (5.0, 9.0), atol=1.5)


def test_metric_spatial_gate_rejects_same_depth_background() -> None:
    points = np.asarray(
        [
            [0.01, 0.02, 0.60],
            [0.08, 0.02, 0.60],
            [0.01, 0.20, 0.60],
            [np.nan, 0.02, 0.60],
        ],
        dtype=np.float32,
    )
    selected = metric_spatial_gate(
        points,
        np.asarray([-0.04, -0.04, 0.55]),
        np.asarray([0.04, 0.08, 0.65]),
    )
    assert selected.tolist() == [True, False, False, False]


def test_mask_metric_support_rejects_wrong_depth_and_position() -> None:
    mask = np.asarray([[255, 255, 255]], dtype=np.uint8)
    points = np.asarray(
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8], [0.4, 0.0, 0.5]],
        dtype=np.float32,
    )
    assert mask_metric_support(
        mask,
        points,
        model_depth=0.5,
        depth_half_width=0.05,
        lower=np.asarray([-0.1, -0.1, 0.4]),
        upper=np.asarray([0.1, 0.1, 0.6]),
    ) == 1
