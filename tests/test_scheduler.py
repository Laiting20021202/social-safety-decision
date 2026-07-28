from __future__ import annotations

import time
import threading
from pathlib import Path

import cv2
import numpy as np

from realtime_safety.config import load_config
from realtime_safety.pipeline.pointcloud import depth_to_pointmap
from realtime_safety.scheduler import (
    RealtimePipeline,
    _observation_pointcloud,
    _observations_with_short_hold,
)
from realtime_safety.types import (
    BBox3D,
    Detection2D,
    FramePacket,
    ObstacleObservation3D,
    PointCloudFrame,
    Track3DState,
)


class FastSegmentation:
    def load(self) -> None: pass
    def warmup(self) -> None: pass
    def close(self) -> None: pass

    def infer(self, frame: FramePacket) -> list[Detection2D]:
        mask = np.zeros(frame.bgr.shape[:2], dtype=bool)
        x = 10 + frame.frame_index % 8
        mask[10:35, x : x + 16] = True
        return [
            Detection2D(
                np.array([x, 10, x + 16, 35], np.float32),
                0,
                "person",
                0.9,
                np.array([x + 8, 22.5], np.float32),
                frame.source_timestamp,
                mask=mask,
                image_size=(frame.original_width, frame.original_height),
            )
        ]


class SlowLoadingSegmentation(FastSegmentation):
    def load(self) -> None:
        __import__("threading").Event().wait(0.4)


class ForbiddenSegmentation(FastSegmentation):
    def load(self) -> None:
        raise AssertionError("YOLO must not load in reconstruction-only mode")


class SlowDepth:
    def __init__(self, delay: float = 0.08) -> None:
        self.delay = delay

    def load(self) -> None: pass
    def warmup(self) -> None: pass
    def close(self) -> None: pass

    def infer(self, frame: FramePacket) -> PointCloudFrame:
        time.sleep(self.delay)
        depth = np.full((48, 64), 3.0, dtype=np.float32)
        pointmap = depth_to_pointmap(depth)
        colors = cv2.resize(frame.rgb, (64, 48)).reshape(-1, 3)
        confidence = np.ones(48 * 64, dtype=np.float32)
        return PointCloudFrame(
            pointmap.reshape(-1, 3),
            colors,
            confidence,
            pointmap,
            frame.frame_index,
            frame.source_timestamp,
            frame.frame_index,
            self.delay * 1000,
            True,
            "test_depth",
            dense_confidence=np.ones((48, 64), dtype=np.float32),
        )


class RecordingReconstructionScene:
    def __init__(self) -> None:
        self.frames: list[tuple[int, int]] = []

    def update_aligned_frame(
        self,
        cloud: PointCloudFrame,
        people,
        yolo_count: int = 0,
        robot_arm=None,
    ) -> None:
        self.frames.append((cloud.frame_index, len(people)))

    def update_people(self, *_args, **_kwargs) -> None:
        pass

    def update_pointcloud(self, cloud: PointCloudFrame) -> None:
        self.frames.append((cloud.frame_index, -1))


class RecordingVideoDashboard:
    def __init__(self) -> None:
        self.frames: list[int] = []

    def update_video(self, image: np.ndarray) -> None:
        self.frames.append(int(image[0, 0, 0]))


def make_video(path: Path, frames: int = 36, fps: float = 30.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64))
    assert writer.isOpened()
    for index in range(frames):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(image, (10 + index % 20, 10), (30 + index % 20, 40), (220, 220, 220), -1)
        writer.write(image)
    writer.release()


def test_workers_are_bounded_and_gui_render_is_not_blocked_by_depth(tmp_path: Path) -> None:
    video = tmp_path / "short.mp4"
    make_video(video)
    config = load_config("realtime_fast")
    config.device = "cpu"
    config.reconstruction.depth_mode = "fast_depth"
    config.reconstruction.fast_depth_frequency_hz = 30.0
    pipeline = RealtimePipeline(config, segmentation_backend=FastSegmentation(), depth_backend=SlowDepth())
    pipeline.start_workers()
    pipeline.start_source(str(video), max_frames=30)
    assert pipeline.wait_until_source_done(timeout=6.0)
    snapshot = pipeline.gui_state.read()
    pipeline.close()
    assert snapshot.frame is not None
    assert snapshot.safety is not None
    assert snapshot.performance.display_fps > 10.0
    assert snapshot.performance.safety_fps > 5.0
    assert snapshot.performance.queue_size <= config.video.queue_size
    assert pipeline.capture_queue.qsize() <= config.video.queue_size
    assert pipeline.reconstruction_queue.qsize() <= config.video.queue_size


def test_raw_video_is_rendered_while_segmentation_is_loading(tmp_path: Path) -> None:
    video = tmp_path / "startup.mp4"
    make_video(video, frames=20)
    config = load_config("realtime_fast")
    config.device = "cpu"
    config.reconstruction.depth_mode = "fast_depth"
    pipeline = RealtimePipeline(config, segmentation_backend=SlowLoadingSegmentation(), depth_backend=SlowDepth(0.01))
    pipeline.start_workers()
    pipeline.start_source(str(video))
    __import__("threading").Event().wait(0.15)
    snapshot = pipeline.gui_state.read()
    pipeline.close()
    assert snapshot.frame is not None
    assert snapshot.annotated_bgr is not None
    assert not snapshot.status["models"]["segmentation"]


def test_reconstruction_mode_skips_yolo_and_safety(tmp_path: Path) -> None:
    video = tmp_path / "viewer.mp4"
    make_video(video, frames=16)
    config = load_config("st4rtrack_viewer")
    config.mode = "reconstruction"
    config.people_overlay = False
    config.device = "cpu"
    config.reconstruction.depth_mode = "fast_depth"
    config.reconstruction.fast_depth_frequency_hz = 30.0
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=ForbiddenSegmentation(),
        depth_backend=SlowDepth(0.01),
    )
    pipeline.start_workers()
    pipeline.start_source(str(video), max_frames=12)
    assert pipeline.wait_until_source_done(timeout=5.0)
    snapshot = pipeline.gui_state.read()
    pipeline.close()
    assert snapshot.frame is not None
    assert snapshot.pointcloud is not None
    assert snapshot.safety is None
    assert snapshot.detections == []
    assert np.array_equal(snapshot.annotated_bgr, snapshot.frame.bgr)
    assert not snapshot.status["models"]["segmentation"]


def test_camera_fov_command_updates_projection_without_depth_scale_change() -> None:
    config = load_config("koch_lan")
    config.device = "cpu"
    target_depth = config.reconstruction.metric_reference_depth_m
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=ForbiddenSegmentation(),
        depth_backend=SlowDepth(0.0),
    )

    pipeline.handle_command("camera_fov", (60.931, 47.609, 320, 240))

    assert abs(config.reconstruction.focal_length_x - 272.0) < 0.02
    assert abs(config.reconstruction.focal_length_y - 272.0) < 0.02
    assert config.reconstruction.principal_point_x == 159.5
    assert config.reconstruction.principal_point_y == 119.5
    assert config.reconstruction.metric_reference_depth_m == target_depth


def test_yolo_observations_become_a_bounded_color_coded_cloud() -> None:
    source = SlowDepth(0.0).infer(
        FramePacket(
            5,
            0.2,
            time.perf_counter(),
            np.zeros((64, 96, 3), dtype=np.uint8),
            np.zeros((64, 96, 3), dtype=np.uint8),
            30.0,
            96,
            64,
        )
    )
    person_points = np.array(((0.0, 0.4, 0.0), (0.1, 0.4, 0.0)), dtype=np.float32)
    chair_points = np.array(((0.2, 0.8, 0.0), (0.3, 0.8, 0.0)), dtype=np.float32)
    observations = [
        ObstacleObservation3D(
            track_id=1,
            class_name="person",
            confidence=0.9,
            position_xyz=person_points.mean(axis=0),
            bbox3d=BBox3D(person_points.min(axis=0), person_points.max(axis=0)),
            radius=0.1,
            point_count=2,
            timestamp=0.2,
            points=person_points,
        ),
        ObstacleObservation3D(
            track_id=2,
            class_name="chair",
            confidence=0.8,
            position_xyz=chair_points.mean(axis=0),
            bbox3d=BBox3D(chair_points.min(axis=0), chair_points.max(axis=0)),
            radius=0.1,
            point_count=2,
            timestamp=0.2,
            points=chair_points,
        ),
    ]

    result = _observation_pointcloud(observations, source, max_points=3)

    assert result.source == "yolo_obstacles"
    assert result.points.shape == (3, 3)
    assert result.colors.shape == (3, 3)
    assert tuple(result.colors[0]) == (255, 64, 64)
    assert tuple(result.colors[-1]) == (192, 96, 255)
    assert result.metric_scale == source.metric_scale


def test_empty_yolo_observations_produce_an_empty_frame() -> None:
    source = SlowDepth(0.0).infer(
        FramePacket(
            5,
            0.2,
            time.perf_counter(),
            np.zeros((64, 96, 3), dtype=np.uint8),
            np.zeros((64, 96, 3), dtype=np.uint8),
            30.0,
            96,
            64,
        )
    )

    result = _observation_pointcloud([], source, max_points=100)

    assert result.points.shape == (0, 3)
    assert result.colors.shape == (0, 3)
    assert not result.valid


def test_person_obstacle_cloud_is_held_briefly_during_a_tracking_miss() -> None:
    points = np.array(((0.0, 0.4, 0.0), (0.1, 0.4, 0.0)), dtype=np.float32)
    observation = ObstacleObservation3D(
        track_id=7,
        class_name="person",
        confidence=0.9,
        position_xyz=points.mean(axis=0),
        bbox3d=BBox3D(points.min(axis=0), points.max(axis=0)),
        radius=0.1,
        point_count=2,
        timestamp=1.0,
        points=points,
    )
    cache: dict[int, ObstacleObservation3D] = {}
    measured_track = Track3DState(
        track_id=7,
        class_name="person",
        position_xyz=observation.position_xyz.copy(),
        velocity_xyz=np.zeros(3, dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=observation.bbox3d,
        radius=0.1,
        hit_count=2,
        missing_count=0,
        last_timestamp=1.0,
        motion_state="static",
        confidence=0.9,
    )
    _observations_with_short_hold([observation], [measured_track], cache, 2, 1.0)
    predicted_track = Track3DState(
        track_id=7,
        class_name="person",
        position_xyz=observation.position_xyz + np.array((0.05, 0.0, 0.0), np.float32),
        velocity_xyz=np.array((0.5, 0.0, 0.0), dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=observation.bbox3d,
        radius=0.1,
        hit_count=2,
        missing_count=1,
        last_timestamp=1.0,
        motion_state="dynamic",
        confidence=0.9,
    )

    held = _observations_with_short_hold([], [predicted_track], cache, 2, 1.1)

    assert len(held) == 1
    assert held[0].track_id == 7
    assert np.allclose(held[0].points, points + np.array((0.05, 0.0, 0.0)))
    assert held[0].confidence < observation.confidence


def test_person_obstacle_cloud_fills_transient_measurement_holes() -> None:
    first_points = np.array(
        ((0.00, 0.40, 0.00), (0.02, 0.40, 0.00), (0.04, 0.40, 0.00)),
        dtype=np.float32,
    )
    second_points = np.array(
        ((0.01, 0.40, 0.00), (0.05, 0.40, 0.00)),
        dtype=np.float32,
    )

    def observation(points: np.ndarray, timestamp: float) -> ObstacleObservation3D:
        return ObstacleObservation3D(
            track_id=9,
            class_name="person",
            confidence=0.9,
            position_xyz=np.median(points, axis=0),
            bbox3d=BBox3D(points.min(axis=0), points.max(axis=0)),
            radius=0.1,
            point_count=len(points),
            timestamp=timestamp,
            points=points,
        )

    first = observation(first_points, 1.0)
    second = observation(second_points, 1.1)
    cache: dict[int, ObstacleObservation3D] = {9: first}
    track = Track3DState(
        track_id=9,
        class_name="person",
        position_xyz=second.position_xyz,
        velocity_xyz=np.zeros(3, dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=second.bbox3d,
        radius=0.1,
        hit_count=3,
        missing_count=0,
        last_timestamp=1.1,
        motion_state="static",
        confidence=0.9,
    )

    fused = _observations_with_short_hold(
        [second],
        [track],
        cache,
        max_missing=12,
        timestamp=1.1,
        voxel_size=0.005,
    )

    assert len(fused) == 1
    assert fused[0].point_count > len(second_points)
    assert len(cache[9].points) == fused[0].point_count


def test_people_overlay_runs_yolo_and_3d_tracking_without_safety(tmp_path: Path) -> None:
    video = tmp_path / "people-viewer.mp4"
    make_video(video, frames=24)
    config = load_config("st4rtrack_viewer")
    config.mode = "reconstruction"
    config.people_overlay = True
    config.device = "cpu"
    config.reconstruction.depth_mode = "fast_depth"
    config.reconstruction.fast_depth_frequency_hz = 30.0
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=FastSegmentation(),
        depth_backend=SlowDepth(0.01),
    )
    pipeline.start_workers()
    pipeline.start_source(str(video), max_frames=20)
    assert pipeline.wait_until_source_done(timeout=5.0)
    snapshot = pipeline.gui_state.read()
    pipeline.close()
    assert snapshot.safety is None
    assert snapshot.people
    assert all(track.class_name == "person" for track in snapshot.people)
    assert snapshot.status["models"]["segmentation"]


def test_renderer_does_not_freeze_cloud_while_people_alignment_lags() -> None:
    config = load_config("st4rtrack_viewer")
    config.device = "cpu"
    scene = RecordingReconstructionScene()
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=FastSegmentation(),
        depth_backend=SlowDepth(0.0),
        scene=scene,
    )
    rgb = np.full((64, 96, 3), 100, dtype=np.uint8)
    frame = FramePacket(9, 0.3, time.perf_counter(), rgb[..., ::-1], rgb, 30.0, 96, 64)
    cloud = SlowDepth(0.0).infer(frame)
    with pipeline._state_lock:
        pipeline._latest_capture = frame
        pipeline._cloud = cloud
        pipeline._cloud_version = 4
        pipeline._people_cloud_version = 3
        pipeline._people_frame_index = 8
    renderer = threading.Thread(target=pipeline._renderer_worker)
    renderer.start()
    deadline = time.perf_counter() + 1.0
    while not scene.frames and time.perf_counter() < deadline:
        time.sleep(0.01)
    pipeline._stop_event.set()
    renderer.join(timeout=1.0)

    # The newest cloud is rendered immediately, but the previous confirmed
    # obstacle handles are not cleared while this frame's YOLO result lags.
    assert scene.frames == [(9, -1)]


def test_reconstruction_video_renderer_is_independent_of_scene_renderer() -> None:
    config = load_config("st4rtrack_viewer")
    config.device = "cpu"
    config.gui.video_fps = 100.0
    dashboard = RecordingVideoDashboard()
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=FastSegmentation(),
        depth_backend=SlowDepth(0.0),
        dashboard=dashboard,
    )
    renderer = threading.Thread(target=pipeline._video_renderer_worker)
    renderer.start()
    for index in range(3):
        bgr = np.full((8, 8, 3), index + 1, dtype=np.uint8)
        with pipeline._state_lock:
            pipeline._latest_capture = FramePacket(
                index,
                index / 30.0,
                time.perf_counter(),
                bgr,
                bgr[..., ::-1],
                30.0,
                8,
                8,
            )
        deadline = time.perf_counter() + 0.5
        while len(dashboard.frames) <= index and time.perf_counter() < deadline:
            time.sleep(0.005)
    pipeline._stop_event.set()
    renderer.join(timeout=1.0)

    assert dashboard.frames == [1, 2, 3]
