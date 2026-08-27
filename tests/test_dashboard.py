from __future__ import annotations

import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from realtime_safety.config import GuiConfig
from realtime_safety.gui.dashboard import (
    Dashboard,
    _fov_degrees,
    _projection_status,
    _resizable_sidebar_bootstrap,
    _view_correction_status,
    _yolo_object_checkpoint_status,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _presentation_dashboard() -> Dashboard:
    dashboard = Dashboard.__new__(Dashboard)
    dashboard._lock = threading.Lock()
    dashboard._input_connected = None
    dashboard._last_video_bgr = np.zeros((120, 200, 3), dtype=np.uint8)
    dashboard._last_video_at = 0.0
    dashboard.presentation_mode = True
    dashboard.reconstruction_only = True
    dashboard.people_overlay = True
    dashboard.camera_status = SimpleNamespace(content="")
    dashboard.file_path = SimpleNamespace(value="")
    dashboard.status = SimpleNamespace(content="")
    dashboard.live_metrics = SimpleNamespace(content="")
    dashboard.yolo_people_status = SimpleNamespace(content="")
    dashboard.video = SimpleNamespace(image=None)
    return dashboard


def test_disconnected_stream_is_visible_and_ready_status_cannot_overwrite_it() -> None:
    dashboard = _presentation_dashboard()

    dashboard.update_camera_status(
        "ROS camera: **reconnecting**", "ros2:///rgbd/color/image_raw", connected=False
    )
    dashboard.update_reconstruction_status(
        "video_depth_anything_metric", "CUDA", True, {}, yolo_ready=True
    )

    assert "CAMERA INPUT LOST" in dashboard.status.content
    assert "INPUT LOST" in dashboard.live_metrics.content
    assert "INFERENCE PAUSED" in dashboard.yolo_people_status.content
    assert "/rgbd/color/image_raw" in dashboard.yolo_people_status.content
    assert dashboard.file_path.value == "ros2:///rgbd/color/image_raw"
    assert dashboard.video.image is not None


def test_ready_status_returns_after_a_decoded_frame_reconnects() -> None:
    dashboard = _presentation_dashboard()
    dashboard.update_camera_status("reconnecting", connected=False)

    dashboard.update_camera_status("connected", connected=True)
    dashboard.update_reconstruction_status(
        "video_depth_anything_metric", "CUDA", True, {}, yolo_ready=True
    )

    assert "SYSTEM READY" in dashboard.status.content


def test_reference_calibration_is_visible_in_presentation_status() -> None:
    dashboard = _presentation_dashboard()
    dashboard.update_reconstruction_status(
        "video_depth_anything_metric_reference_calibrated",
        "CUDA",
        True,
        {},
        metric_scale=0.32,
        reference_depth_m=0.4,
        observed_reference_depth=1.25,
    )

    assert "CALIBRATED 0.40 m" in dashboard.status.content
    assert "0.320×" in dashboard.status.content


def test_projection_geometry_uses_pixel_focal_lengths() -> None:
    assert abs(_fov_degrees(320, 272.0) - 60.93) < 0.01
    status = _projection_status(60.931, 47.609, 320, 240)
    assert "fx=272.0px" in status
    assert "fy=272.0px" in status
    assert "8 cm AprilTag metric lock" in status


def test_view_correction_status_makes_display_only_scope_explicit() -> None:
    status = _view_correction_status(35.0, -2.0, 7.5)

    assert "Legacy angle values are ignored" in status
    assert "orthographic metre coordinates" in status
    assert "controller coordinates are unchanged" in status


def test_reconstruction_dashboard_builds_metric_bev_controls() -> None:
    commands: list[tuple[str, object | None]] = []
    projection = SimpleNamespace(
        focal_length_x=272.0,
        focal_length_y=272.0,
        principal_point_y=119.5,
    )
    dashboard = Dashboard(
        GuiConfig(
            host="127.0.0.1",
            port=_free_port(),
            max_video_width=320,
            metric_bev_enabled=True,
            metric_bev_obstacle_height_m=0.065,
        ),
        lambda command, value: commands.append((command, value)),
        reconstruction_only=True,
        projection_config=projection,
    )
    try:
        assert dashboard.metric_bev_enabled.value is True
        assert dashboard.metric_bev_height.value == 0.065
        assert dashboard.recalibrate_metric_bev is not None

        dashboard.metric_bev_height.value = 0.08

        assert commands[-1] == (
            "camera_bev_height",
            0.08,
        )
    finally:
        dashboard.close()


def test_dashboard_refuses_to_silently_move_off_requested_port() -> None:
    port = _free_port()
    first = Dashboard(
        GuiConfig(host="127.0.0.1", port=port),
        lambda command, value: None,
    )
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            Dashboard(
                GuiConfig(host="127.0.0.1", port=port),
                lambda command, value: None,
            )
    finally:
        first.close()


def test_presentation_bootstrap_pins_webgl_dpr_and_dark_backing_surface() -> None:
    bootstrap = _resizable_sidebar_bootstrap()

    assert "fixedDpr: stableDpr" in bootstrap
    assert "url.searchParams.set" in bootstrap
    assert "window.location.replace" in bootstrap
    assert 'canvas[data-engine^="three.js"]' in bootstrap
    assert "background-color: #111 !important" in bootstrap


def test_obstacle_model_dropdown_emits_runtime_model_command() -> None:
    commands: list[tuple[str, object | None]] = []
    dashboard = Dashboard(
        GuiConfig(host="127.0.0.1", port=_free_port()),
        lambda command, value: commands.append((command, value)),
        reconstruction_only=True,
        people_overlay=True,
        obstacle_model="model-accurate.pt",
        obstacle_model_options=("model-accurate.pt", "model-fast.pt"),
    )
    try:
        assert dashboard.obstacle_model.value == "model-accurate.pt"
        assert tuple(dashboard.obstacle_model.options) == (
            "model-accurate.pt",
            "model-fast.pt",
        )

        dashboard.obstacle_model.value = "model-fast.pt"

        assert commands == [("segmentation_model", "model-fast.pt")]
    finally:
        dashboard.close()


def test_pointcloud_method_dropdown_emits_runtime_switch_command() -> None:
    commands: list[tuple[str, object | None]] = []
    dashboard = Dashboard(
        GuiConfig(host="127.0.0.1", port=_free_port()),
        lambda command, value: commands.append((command, value)),
        reconstruction_only=True,
        reconstruction_method="video_depth",
        reconstruction_method_options=(
            "video_depth",
            "mast3r_slam",
            "st4rtrack",
        ),
    )
    try:
        assert dashboard.reconstruction_method.value == (
            "Metric Video Depth Anything (Temporal)"
        )
        dashboard.reconstruction_method.value = "MASt3R-SLAM (Global Dense Map)"

        assert commands == [("reconstruction_method", "mast3r_slam")]

        dashboard.update_reconstruction_method_status(
            "video_depth",
            requested_method="mast3r_slam",
            error="checkpoint missing",
            ready=True,
        )
        assert "Switch failed" in dashboard.reconstruction_method_status.content
        assert dashboard.reconstruction_method.value == (
            "Metric Video Depth Anything (Temporal)"
        )
        # Programmatic rollback must not emit a second switch command.
        assert commands == [("reconstruction_method", "mast3r_slam")]
    finally:
        dashboard.close()


def test_obstacle_controls_expose_edgetam_yolo_and_yolo_checkpoints() -> None:
    commands: list[tuple[str, object | None]] = []
    dashboard = Dashboard(
        GuiConfig(host="127.0.0.1", port=_free_port()),
        lambda command, value: commands.append((command, value)),
        reconstruction_only=True,
        obstacle_backend="edgetam",
        obstacle_backend_options=("edgetam", "yolo"),
        obstacle_model="yolo26m-seg.pt",
        obstacle_model_options=("yolo26m-seg.pt", "yolo11n-seg.pt"),
    )
    try:
        assert (
            dashboard.obstacle_backend.value
            == "EdgeTAM + RGB Hand Gate + 3D PointCloud"
        )
        assert tuple(dashboard.obstacle_backend.options) == (
            "EdgeTAM + RGB Hand Gate + 3D PointCloud",
            "Legacy COCO YOLO / MediaPipe Hand Gate + RGB-D",
        )
        assert tuple(dashboard.obstacle_model.options) == (
            "yolo26m-seg.pt",
            "yolo11n-seg.pt",
        )

        dashboard.obstacle_backend.value = (
            "Legacy COCO YOLO / MediaPipe Hand Gate + RGB-D"
        )
        dashboard.obstacle_model.value = "yolo11n-seg.pt"

        assert commands == [
            ("obstacle_backend", "yolo"),
            ("segmentation_model", "yolo11n-seg.pt"),
        ]
    finally:
        dashboard.close()


def test_yolo_checkpoint_status_never_calls_coco_weights_a_hand_model() -> None:
    available = _yolo_object_checkpoint_status(
        "yolo26m-seg.pt", available=True
    )
    missing = _yolo_object_checkpoint_status(
        "missing.pt", available=False
    )

    assert "Local COCO YOLO object weights available" in available
    assert "not a human-hand model" in available
    assert "MediaPipe" in available
    assert "object weights are missing" in missing
    assert "not a hand-semantic checkpoint" in missing


def test_legacy_status_does_not_repeat_yolo_hand_mislabel() -> None:
    dashboard = Dashboard.__new__(Dashboard)
    dashboard._obstacle_backend_labels = {
        "yolo": "Legacy COCO YOLO / MediaPipe Hand Gate + RGB-D",
    }
    dashboard._edge_metric = "WAITING"
    dashboard.obstacle_backend_status = SimpleNamespace(content="")

    dashboard.update_obstacle_backend_status(
        "yolo",
        state="active",
        detail="YOLO RGB hand masks active",
        metrics={
            "hand_only": "true",
            "yolo_model": "yolo26m-seg.pt",
            "yolo_fps": "9.0",
        },
    )

    content = dashboard.obstacle_backend_status.content
    assert "YOLO RGB hand" not in content
    assert "MediaPipe supplies the human-hand semantic gate" in content
    assert "COCO checkpoint has no human-hand class" in content


def test_edgetam_status_shows_runtime_evidence() -> None:
    dashboard = Dashboard.__new__(Dashboard)
    dashboard._obstacle_backend_labels = {
        "edgetam": "EdgeTAM + RGB Hand Gate + 3D PointCloud",
        "pointcloud": "3D PointCloud Only (Safe Fallback)",
    }
    dashboard._updating_obstacle_backend = False
    dashboard._edge_metric = "WAITING"
    dashboard.obstacle_backend = SimpleNamespace(
        value="EdgeTAM + RGB Hand Gate + 3D PointCloud"
    )
    dashboard.obstacle_backend_status = SimpleNamespace(content="")

    dashboard.update_obstacle_backend_status(
        "edgetam",
        state="active",
        detail="EdgeTAM ready",
        metrics={
            "fps": "9.8",
            "edge_latency_ms": "41.2",
            "track_count": "2",
            "edge_refined_corrections": "7",
        },
    )

    assert (
        "EdgeTAM + RGB Hand Gate + 3D PointCloud"
        in dashboard.obstacle_backend_status.content
    )
    assert "41.2 ms" in dashboard.obstacle_backend_status.content
    assert "refined: **7**" in dashboard.obstacle_backend_status.content
    assert dashboard._edge_metric == "41.2 ms"

    dashboard.update_obstacle_backend_status(
        "pointcloud",
        state="active",
        detail="EdgeTAM refinement disabled",
        metrics={"fps": "9.5"},
    )
    assert "Safe fallback active" in dashboard.obstacle_backend_status.content
    # Runtime state may report a fallback/error, but it must not rewrite the
    # operator's sticky model selection.
    assert dashboard.obstacle_backend.value == (
        "EdgeTAM + RGB Hand Gate + 3D PointCloud"
    )

    dashboard.update_obstacle_backend_status(
        "edgetam",
        state="active",
        detail="EdgeTAM ready",
        metrics={
            "safety_output_state": "held_untrusted_hand_semantics",
            "pipeline_message": "RGB/3D context unavailable",
            "geometry_fallback_track_count": "0",
        },
    )
    content = dashboard.obstacle_backend_status.content
    assert "Untrusted hand output held" in content
    assert "unverified scene fallback: **BLOCKED**" in content
    assert "controller must STOP on perception timeout" in content


def test_obstacle_model_status_reports_effective_model_and_reverts_on_failure() -> None:
    dashboard = Dashboard.__new__(Dashboard)
    dashboard._lock = threading.Lock()
    dashboard._updating_obstacle_model = False
    dashboard._active_obstacle_model = "model-a.pt"
    dashboard.obstacle_model = SimpleNamespace(value="model-b.pt")
    dashboard.obstacle_model_status = SimpleNamespace(content="")

    dashboard.update_obstacle_model_status(
        "model-a.pt",
        requested_model="model-b.pt",
        ready=True,
    )
    assert "Switching" in dashboard.obstacle_model_status.content
    assert "model-b.pt" in dashboard.obstacle_model_status.content
    assert "model-a.pt" in dashboard.obstacle_model_status.content

    dashboard.update_obstacle_model_status(
        "model-a.pt",
        error="out of memory",
        ready=True,
    )
    assert "Switch failed" in dashboard.obstacle_model_status.content
    assert "out of memory" in dashboard.obstacle_model_status.content
    assert dashboard.obstacle_model.value == "model-a.pt"


def test_obstacle_model_status_ignores_superseded_switch_generation() -> None:
    dashboard = Dashboard.__new__(Dashboard)
    dashboard._lock = threading.Lock()
    dashboard._updating_obstacle_model = False
    dashboard._active_obstacle_model = "model-a.pt"
    dashboard._obstacle_model_status_generation = 0
    dashboard.obstacle_model = SimpleNamespace(value="model-c.pt")
    dashboard.obstacle_model_status = SimpleNamespace(content="")

    dashboard.update_obstacle_model_status(
        "model-b.pt",
        requested_model="model-c.pt",
        ready=True,
        generation=2,
    )
    expected = dashboard.obstacle_model_status.content

    dashboard.update_obstacle_model_status(
        "model-b.pt",
        ready=True,
        generation=1,
    )

    assert dashboard.obstacle_model_status.content == expected
    assert "Switching" in expected
    assert dashboard.obstacle_model.value == "model-c.pt"
