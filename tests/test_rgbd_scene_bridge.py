from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realtime_safety.ros2_bridge.rgbd_scene_bridge import (
    RgbdFrameProjector,
    RgbdImageSceneBridge,
    RgbdProjectionConfig,
)


def _stamp(seconds: float) -> SimpleNamespace:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _image(array: np.ndarray, encoding: str, stamp: float = 1.0) -> SimpleNamespace:
    values = np.ascontiguousarray(array)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=_stamp(stamp), frame_id="rgbd_color_optical_frame"
        ),
        height=values.shape[0],
        width=values.shape[1],
        step=values.strides[0],
        encoding=encoding,
        is_bigendian=False,
        data=values.tobytes(),
    )


def _info(width: int, height: int, stamp: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=_stamp(stamp), frame_id="rgbd_color_optical_frame"
        ),
        width=width,
        height=height,
        k=[2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
    )


def _rgb(height: int, width: int) -> np.ndarray:
    values = np.zeros((height, width, 3), dtype=np.uint8)
    values[..., 0] = np.arange(width, dtype=np.uint8)
    values[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    values[..., 2] = 200
    return values


def test_image_projector_backprojects_metric_depth_and_rgb_into_both_frames() -> None:
    depth = np.zeros((3, 4), dtype=np.float32)
    depth[1, 2] = 1.0
    projector = RgbdFrameProjector(
        RgbdProjectionConfig(max_points=100),
        camera_position=np.array([1.0, 2.0, 3.0]),
        world_from_optical=np.eye(3),
    )

    result = projector.project(
        _image(_rgb(3, 4), "rgb8"),
        _image(depth, "32FC1"),
        _info(4, 3),
        frame_index=7,
    )

    np.testing.assert_allclose(result.optical_points, [[0.5, 0.0, 1.0]])
    np.testing.assert_array_equal(result.colors, [[2, 1, 200]])
    np.testing.assert_allclose(result.pipeline_cloud.points, [[0.5, 1.0, 0.0]])
    np.testing.assert_allclose(result.world_points, [[1.5, 2.0, 4.0]])
    assert result.pipeline_cloud.pointmap.shape == (3, 4, 3)
    np.testing.assert_allclose(result.pipeline_cloud.pointmap[1, 2], [0.5, 1.0, 0.0])
    assert np.isnan(result.pipeline_cloud.pointmap[0, 0]).all()
    assert result.pipeline_cloud.frame_index == 7
    assert result.pipeline_cloud.timestamp == pytest.approx(1.0)
    assert result.pipeline_cloud.source == "rgbd_depth_backprojection"


def test_projector_uses_bounded_reproducible_depth_noise() -> None:
    depth = np.full((20, 20), 1.0, dtype=np.float32)
    config = RgbdProjectionConfig(
        max_points=100,
        depth_noise_stddev_m=0.001,
        depth_noise_clip_sigma=2.0,
        noise_seed=19,
    )
    first = RgbdFrameProjector(config)
    second = RgbdFrameProjector(config)
    messages = (
        _image(_rgb(20, 20), "rgb8"),
        _image(depth, "32FC1"),
        _info(20, 20),
    )

    first_a = first.project(*messages, frame_index=0).optical_points
    first_b = first.project(*messages, frame_index=1).optical_points
    second_a = second.project(*messages, frame_index=0).optical_points

    np.testing.assert_allclose(first_a, second_a)
    assert not np.array_equal(first_a, first_b)
    assert np.max(np.abs(first_a[:, 2] - 1.0)) <= 0.002001
    assert len(first_a) <= 100


def test_projector_rejects_unsynchronized_images() -> None:
    projector = RgbdFrameProjector(
        RgbdProjectionConfig(max_points=100, sync_slop_sec=0.01)
    )
    with pytest.raises(ValueError, match="timestamps rejected"):
        projector.project(
            _image(_rgb(3, 4), "rgb8", stamp=1.0),
            _image(np.ones((3, 4), np.float32), "32FC1", stamp=1.1),
            _info(4, 3, stamp=1.0),
            frame_index=0,
        )


def test_image_bridge_emits_current_empty_frame_instead_of_holding_old_points() -> None:
    raw = []
    debug = []
    world = []
    bridge = RgbdImageSceneBridge(
        "/color",
        "/depth",
        "/info",
        raw.append,
        lambda points, colors, is_world, rate: debug.append(
            (points.copy(), colors.copy(), is_world, rate)
        ),
        on_world_cloud=world.append,
        projection_config=RgbdProjectionConfig(max_points=100),
    )
    rgb = _image(_rgb(3, 4), "rgb8")
    info = _info(4, 3)

    bridge._receive_synchronized(
        rgb, _image(np.ones((3, 4), np.float32), "32FC1"), info
    )
    bridge._receive_synchronized(
        rgb, _image(np.zeros((3, 4), np.float32), "32FC1"), info
    )

    assert [frame.frame_index for frame in raw] == [0, 1]
    assert raw[0].valid and len(raw[0].points) == 12
    assert not raw[1].valid and len(raw[1].points) == 0
    assert len(world) == 2 and not world[-1].valid
    assert len(debug) == 4
    assert debug[-2][0].shape == (0, 3) and not debug[-2][2]
    assert debug[-1][0].shape == (0, 3) and debug[-1][2]


def test_live_link_pose_converts_optical_forward_to_world_link_x() -> None:
    world = []
    bridge = RgbdImageSceneBridge(
        "/color",
        "/depth",
        "/info",
        lambda _: None,
        None,
        on_world_cloud=world.append,
        projection_config=RgbdProjectionConfig(max_points=100),
    )
    bridge.update_camera_pose(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        stamp=1.0,
        pose_is_optical=False,
    )
    depth = np.zeros((3, 4), dtype=np.float32)
    # At (cx, cy), the optical ray is exactly [0, 0, 1].
    depth[1, 1] = 1.0

    bridge._receive_synchronized(
        _image(_rgb(3, 4), "rgb8"),
        _image(depth, "32FC1"),
        _info(4, 3),
    )

    np.testing.assert_allclose(world[0].points, [[2.0, 2.0, 3.0]], atol=1e-6)

