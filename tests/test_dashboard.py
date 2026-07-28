from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from realtime_safety.gui.dashboard import (
    Dashboard,
    _fov_degrees,
    _projection_status,
    _resizable_sidebar_bootstrap,
)


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
        "Network stream: **reconnecting**", "http://camera/stream", connected=False
    )
    dashboard.update_reconstruction_status(
        "video_depth_anything_metric", "CUDA", True, {}, yolo_ready=True
    )

    assert "CAMERA INPUT LOST" in dashboard.status.content
    assert "INPUT LOST" in dashboard.live_metrics.content
    assert "INFERENCE PAUSED" in dashboard.yolo_people_status.content
    assert dashboard.file_path.value == "http://camera/stream"
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
    assert "0.40 m depth is preserved" in status


def test_presentation_bootstrap_pins_webgl_dpr_and_dark_backing_surface() -> None:
    bootstrap = _resizable_sidebar_bootstrap()

    assert "fixedDpr: stableDpr" in bootstrap
    assert "url.searchParams.set" in bootstrap
    assert "window.location.replace" in bootstrap
    assert 'canvas[data-engine^="three.js"]' in bootstrap
    assert "background-color: #111 !important" in bootstrap
