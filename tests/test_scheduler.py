from __future__ import annotations

import time
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from realtime_safety.config import load_config
from realtime_safety.pipeline.pointcloud import depth_to_pointmap
from realtime_safety.scheduler import (
    RealtimePipeline,
    _model_switch_output_gate,
    _needs_yolo_pipeline,
    _observation_pointcloud,
    _current_observations_with_fusion,
)
from realtime_safety.types import (
    BBox3D,
    Detection2D,
    FramePacket,
    ObstacleObservation3D,
    PointCloudFrame,
    Track3DState,
)
from realtime_safety.utils.validation import validate_config


def test_yolo_worker_runs_for_both_sides_of_pending_handoff() -> None:
    assert not _needs_yolo_pipeline("edgetam", "edgetam")
    assert _needs_yolo_pipeline("yolo", "edgetam")
    assert _needs_yolo_pipeline("edgetam", "yolo")
    assert _needs_yolo_pipeline("yolo", "yolo")


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


class RecordingSegmentation(FastSegmentation):
    def __init__(
        self,
        model: str,
        events: list[tuple[str, str]],
        *,
        load_gate: threading.Event | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.model = model
        self.events = events
        self.load_gate = load_gate
        self.load_error = load_error
        self.closed = False

    def load(self) -> None:
        self.events.append(("load", self.model))
        if self.load_gate is not None:
            self.load_gate.wait(timeout=2.0)
        if self.load_error is not None:
            raise self.load_error

    def warmup(self) -> None:
        self.events.append(("warmup", self.model))

    def close(self) -> None:
        self.closed = True
        self.events.append(("close", self.model))


class RecordingModelDashboard:
    def __init__(self) -> None:
        self.statuses: list[dict[str, object]] = []

    def update_obstacle_model_status(
        self,
        active_model: str,
        *,
        requested_model: str | None = None,
        error: str | None = None,
        ready: bool = True,
        generation: int | None = None,
    ) -> None:
        self.statuses.append(
            {
                "active": active_model,
                "requested": requested_model,
                "error": error,
                "ready": ready,
            }
        )


class RecordingReconstructionDashboard:
    def __init__(self) -> None:
        self.statuses: list[dict[str, object]] = []

    def update_reconstruction_method_status(
        self,
        active_method: str,
        *,
        requested_method: str | None = None,
        error: str | None = None,
        ready: bool = True,
    ) -> None:
        self.statuses.append(
            {
                "active": active_method,
                "requested": requested_method,
                "error": error,
                "ready": ready,
            }
        )


class AvailableMast3rSlam:
    def __init__(self) -> None:
        self.preflight_calls = 0

    def preflight(self) -> None:
        self.preflight_calls += 1

    def close(self) -> None:
        pass


def _switchable_pipeline(
    factory,
) -> tuple[RealtimePipeline, RecordingSegmentation, RecordingModelDashboard]:
    config = load_config("realtime_fast")
    config.device = "cpu"
    config.segmentation.model = "model-a.pt"
    config.segmentation.model_options = ["model-a.pt", "model-b.pt", "model-c.pt"]
    events: list[tuple[str, str]] = []
    current = RecordingSegmentation("model-a.pt", events)
    dashboard = RecordingModelDashboard()
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=current,
        depth_backend=SlowDepth(0.0),
        dashboard=dashboard,
        segmentation_factory=factory,
    )
    pipeline._models_ready["segmentation"] = True
    return pipeline, current, dashboard


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


def test_reconstruction_method_switch_is_allowlisted_and_nonblocking() -> None:
    config = load_config("st4rtrack_viewer")
    config.device = "cpu"
    config.reconstruction.depth_mode = "fast_depth"
    config.reconstruction.depth_mode_options = ["fast_depth", "mast3r_slam"]
    dashboard = RecordingReconstructionDashboard()
    mast3r_slam = AvailableMast3rSlam()
    pipeline = RealtimePipeline(
        config,
        depth_backend=SlowDepth(0.0),
        dashboard=dashboard,
        mast3r_slam_backend=mast3r_slam,
    )
    pipeline._models_ready["depth"] = True

    started = time.perf_counter()
    accepted = pipeline.request_reconstruction_method("mast3r_slam")

    assert accepted
    assert time.perf_counter() - started < 0.05
    assert mast3r_slam.preflight_calls == 1
    assert pipeline.reconstruction_mode == "mast3r_slam"
    assert pipeline.active_reconstruction_mode == "fast_depth"
    assert dashboard.statuses[-1] == {
        "active": "fast_depth",
        "requested": "mast3r_slam",
        "error": None,
        "ready": True,
    }


def test_missing_mast3r_slam_does_not_change_active_method(tmp_path: Path) -> None:
    config = load_config("st4rtrack_viewer")
    config.device = "cpu"
    config.reconstruction.depth_mode = "fast_depth"
    config.reconstruction.depth_mode_options = ["fast_depth", "mast3r_slam"]
    config.reconstruction.mast3r_slam_path = str(tmp_path / "missing")
    config.reconstruction.mast3r_slam_python = str(tmp_path / "missing-python")
    dashboard = RecordingReconstructionDashboard()
    pipeline = RealtimePipeline(
        config,
        depth_backend=SlowDepth(0.0),
        dashboard=dashboard,
    )
    pipeline._models_ready["depth"] = True

    accepted = pipeline.request_reconstruction_method("mast3r_slam")

    assert not accepted
    assert pipeline.reconstruction_mode == "fast_depth"
    assert pipeline.active_reconstruction_mode == "fast_depth"
    assert "setup_mast3r_slam.sh" in str(dashboard.statuses[-1]["error"])


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


def test_metric_bev_commands_do_not_change_projection_intrinsics() -> None:
    config = load_config("koch_lan")
    config.device = "cpu"
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=ForbiddenSegmentation(),
        depth_backend=SlowDepth(0.0),
    )
    calls: list[tuple[object, ...]] = []
    pipeline.scene = SimpleNamespace(
        set_metric_bev_enabled=lambda enabled: calls.append(("enabled", enabled)),
        set_metric_bev_height_threshold=lambda value: calls.append(("height", value)),
        recalibrate_metric_bev=lambda: calls.append(("recalibrate",)),
    )
    focal_x = config.reconstruction.focal_length_x
    focal_y = config.reconstruction.focal_length_y

    pipeline.handle_command("camera_bev_enabled", True)
    pipeline.handle_command("camera_bev_height", 0.055)
    pipeline.handle_command("camera_bev_recalibrate")

    assert calls == [("enabled", True), ("height", 0.055), ("recalibrate",)]
    assert config.gui.metric_bev_enabled
    assert config.gui.metric_bev_obstacle_height_m == 0.055
    assert config.reconstruction.focal_length_x == focal_x
    assert config.reconstruction.focal_length_y == focal_y


def test_koch_profile_preserves_yolo_choices_and_uses_private_mux_topics() -> None:
    config = load_config("koch_lan")

    validate_config(config)

    assert config.people_overlay
    assert config.segmentation.model == "yolo26m-seg.pt"
    assert config.segmentation.model_options == [
        "yolo26m-seg.pt",
        "yolo26s-seg.pt",
        "yolo11m-seg.pt",
        "yolo11s-seg.pt",
        "yolo11n-seg.pt",
    ]
    assert all(
        not model.startswith(("http://", "https://"))
        and Path(model).name == model
        for model in config.segmentation.model_options
    )
    assert config.obstacle_perception.enabled
    assert config.obstacle_perception.backend == "edgetam"
    assert config.obstacle_perception.backend_options == [
        "edgetam",
        "yolo",
    ]
    candidate_and_output_topics = {
        config.obstacle_perception.edgetam_obstacle_cloud_topic,
        config.obstacle_perception.yolo_obstacle_cloud_topic,
        config.obstacle_perception.obstacle_cloud_topic,
    }
    assert candidate_and_output_topics == {
        "/edgetam_tracker/obstacle_cloud",
        "/realtime_safety/yolo_obstacles/candidate_cloud",
        "/realtime_safety/yolo_obstacles/pointcloud",
    }


def test_obstacle_backend_command_switches_mux_and_edge_controller_without_loading_model() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def request_mode(self, mode: str) -> None:
            self.requests.append(mode)

    class NeverLoadedSegmentation(FastSegmentation):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def load(self) -> None:
            self.calls.append("load")

        def warmup(self) -> None:
            self.calls.append("warmup")

        def infer(self, frame: FramePacket) -> list[Detection2D]:
            self.calls.append("infer")
            return super().infer(frame)

    config = load_config("koch_lan")
    config.device = "cpu"
    controller = Recorder()
    mux = Recorder()
    segmentation = NeverLoadedSegmentation()
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=segmentation,
        depth_backend=SlowDepth(0.0),
        obstacle_backend_controller=controller,
        obstacle_cloud_mux=mux,
    )

    pipeline.handle_command("obstacle_backend", "yolo")

    assert pipeline.obstacle_backend_mode == "yolo"
    assert pipeline.active_obstacle_backend_mode == "edgetam"
    assert mux.requests == ["yolo"]
    # EdgeTAM refinement is disabled, but its geometric cloud remains the
    # live source until the mux receives YOLO's first candidate cloud.
    assert controller.requests == ["pointcloud"]

    pipeline.update_obstacle_mux_status(
        SimpleNamespace(active_mode="yolo", state="active", message="ready")
    )
    assert pipeline.active_obstacle_backend_mode == "yolo"

    pipeline.handle_command("obstacle_backend", "edgetam")

    assert pipeline.obstacle_backend_mode == "edgetam"
    assert pipeline.active_obstacle_backend_mode == "yolo"
    assert mux.requests == ["yolo", "edgetam"]
    assert controller.requests == ["pointcloud", "edgetam"]

    pipeline.update_obstacle_mux_status(
        SimpleNamespace(active_mode="edgetam", state="active", message="ready")
    )

    assert pipeline.active_obstacle_backend_mode == "edgetam"
    assert pipeline.segmentation is segmentation
    assert segmentation.calls == []


def test_allowlisted_but_missing_local_model_is_rejected(tmp_path: Path) -> None:
    current_model = tmp_path / "model-a.pt"
    current_model.write_bytes(b"test checkpoint")
    missing_model = tmp_path / "missing.pt"
    config = load_config("realtime_fast")
    config.device = "cpu"
    config.segmentation.model = str(current_model)
    config.segmentation.model_options = [str(current_model), str(missing_model)]
    pipeline = RealtimePipeline(
        config,
        segmentation_backend=FastSegmentation(),
        depth_backend=SlowDepth(0.0),
    )
    pipeline._models_ready["segmentation"] = True

    assert not pipeline.request_segmentation_model(str(missing_model))
    assert pipeline.active_segmentation_model == str(current_model)


def test_segmentation_model_switch_is_queued_then_atomically_applied() -> None:
    created: dict[str, RecordingSegmentation] = {}

    def factory(config, _device: str) -> RecordingSegmentation:
        backend = RecordingSegmentation(config.model, [])
        created[config.model] = backend
        return backend

    pipeline, previous, dashboard = _switchable_pipeline(factory)

    started = time.perf_counter()
    pipeline.handle_command("segmentation_model", "model-b.pt")
    callback_elapsed = time.perf_counter() - started

    assert callback_elapsed < 0.05
    assert created == {}
    assert pipeline.segmentation is previous
    generation, switched = pipeline._apply_pending_segmentation_switch(0)

    assert generation == 1
    assert switched
    assert pipeline.active_segmentation_model == "model-b.pt"
    assert pipeline.config.segmentation.model == "model-b.pt"
    assert pipeline.segmentation is created["model-b.pt"]
    assert created["model-b.pt"].events == [
        ("load", "model-b.pt"),
        ("warmup", "model-b.pt"),
    ]
    assert previous.closed
    assert dashboard.statuses[0]["requested"] == "model-b.pt"
    assert dashboard.statuses[-1] == {
        "active": "model-b.pt",
        "requested": None,
        "error": None,
        "ready": True,
    }


def test_failed_segmentation_model_switch_keeps_previous_backend() -> None:
    created: dict[str, RecordingSegmentation] = {}

    def factory(config, _device: str) -> RecordingSegmentation:
        backend = RecordingSegmentation(
            config.model,
            [],
            load_error=RuntimeError("checkpoint is incompatible"),
        )
        created[config.model] = backend
        return backend

    pipeline, previous, dashboard = _switchable_pipeline(factory)
    pipeline.handle_command("segmentation_model", "model-b.pt")

    generation, switched = pipeline._apply_pending_segmentation_switch(0)

    assert generation == 1
    assert not switched
    assert pipeline.segmentation is previous
    assert pipeline.active_segmentation_model == "model-a.pt"
    assert pipeline.config.segmentation.model == "model-a.pt"
    assert not previous.closed
    assert created["model-b.pt"].closed
    assert pipeline.errors["segmentation_switch"] == "checkpoint is incompatible"
    assert dashboard.statuses[-1]["active"] == "model-a.pt"
    assert dashboard.statuses[-1]["error"] == "checkpoint is incompatible"


def test_unapproved_segmentation_model_requests_never_reach_factory() -> None:
    factory_calls: list[str] = []

    def factory(config, _device: str) -> RecordingSegmentation:
        factory_calls.append(config.model)
        return RecordingSegmentation(config.model, [])

    pipeline, previous, dashboard = _switchable_pipeline(factory)

    assert not pipeline.request_segmentation_model("../outside.pt")
    assert not pipeline.request_segmentation_model("https://example.com/model.pt")
    generation, switched = pipeline._apply_pending_segmentation_switch(0)

    assert (generation, switched) == (0, False)
    assert factory_calls == []
    assert pipeline.segmentation is previous
    assert dashboard.statuses[-1]["error"] is not None


def test_latest_segmentation_model_request_wins_during_slow_load() -> None:
    load_gate = threading.Event()
    created: dict[str, RecordingSegmentation] = {}

    def factory(config, _device: str) -> RecordingSegmentation:
        backend = RecordingSegmentation(
            config.model,
            [],
            load_gate=load_gate if config.model == "model-b.pt" else None,
        )
        created[config.model] = backend
        return backend

    pipeline, previous, _dashboard = _switchable_pipeline(factory)
    pipeline.handle_command("segmentation_model", "model-b.pt")
    results: list[tuple[int, bool]] = []
    switch_thread = threading.Thread(
        target=lambda: results.append(pipeline._apply_pending_segmentation_switch(0))
    )
    switch_thread.start()
    deadline = time.perf_counter() + 1.0
    while "model-b.pt" not in created and time.perf_counter() < deadline:
        time.sleep(0.005)

    pipeline.handle_command("segmentation_model", "model-c.pt")
    load_gate.set()
    switch_thread.join(timeout=2.0)

    assert results == [(1, False)]
    assert created["model-b.pt"].closed
    assert pipeline.segmentation is previous

    generation, switched = pipeline._apply_pending_segmentation_switch(1)

    assert (generation, switched) == (2, True)
    assert pipeline.segmentation is created["model-c.pt"]
    assert pipeline.active_segmentation_model == "model-c.pt"
    assert previous.closed


def test_tracking_preflight_failure_rolls_back_candidate_model() -> None:
    created: dict[str, RecordingSegmentation] = {}

    class BrokenTrackingSegmentation(RecordingSegmentation):
        def track_obstacles(self, _frame: FramePacket) -> list[Detection2D]:
            self.events.append(("track", self.model))
            raise RuntimeError("tracker configuration is invalid")

    def factory(config, _device: str) -> RecordingSegmentation:
        backend = BrokenTrackingSegmentation(config.model, [])
        created[config.model] = backend
        return backend

    pipeline, previous, dashboard = _switchable_pipeline(factory)
    pipeline.handle_command("segmentation_model", "model-b.pt")
    bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    frame = FramePacket(1, 0.1, time.perf_counter(), bgr, bgr[..., ::-1], 30.0, 16, 16)

    generation, switched = pipeline._apply_pending_segmentation_switch(
        0,
        preflight_frame=frame,
        require_tracking_preflight=True,
    )

    assert (generation, switched) == (1, False)
    assert pipeline.segmentation is previous
    assert pipeline.active_segmentation_model == "model-a.pt"
    assert not previous.closed
    assert created["model-b.pt"].closed
    assert ("warmup", "model-b.pt") in created["model-b.pt"].events
    assert ("track", "model-b.pt") in created["model-b.pt"].events
    assert dashboard.statuses[-1]["error"] == "tracker configuration is invalid"


def test_model_switch_output_gate_retains_then_honestly_clears_previous_output() -> None:
    publish, previous_valid, remaining = _model_switch_output_gate(
        current_valid=False,
        previous_valid=True,
        hold_updates=2,
    )
    assert not publish
    assert previous_valid
    assert remaining == 1

    publish, previous_valid, remaining = _model_switch_output_gate(
        current_valid=False,
        previous_valid=previous_valid,
        hold_updates=remaining,
    )
    assert not publish
    assert previous_valid
    assert remaining == 0

    publish, previous_valid, remaining = _model_switch_output_gate(
        current_valid=False,
        previous_valid=previous_valid,
        hold_updates=remaining,
    )
    assert publish
    assert not previous_valid
    assert remaining == 0


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


def test_person_obstacle_cloud_clears_immediately_during_a_tracking_miss() -> None:
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
    measured = _current_observations_with_fusion(
        [observation], cache
    )
    assert len(measured) == 1
    assert 7 in cache

    held = _current_observations_with_fusion([], cache)

    assert held == []
    assert cache == {}


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
    fused = _current_observations_with_fusion(
        [second],
        cache,
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
