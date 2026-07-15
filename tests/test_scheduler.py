from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from realtime_safety.config import load_config
from realtime_safety.pipeline.pointcloud import depth_to_pointmap
from realtime_safety.scheduler import RealtimePipeline
from realtime_safety.types import Detection2D, FramePacket, PointCloudFrame


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
