from __future__ import annotations

import base64
import html
import logging
import math
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from realtime_safety.config import GuiConfig
from realtime_safety.types import (
    Detection2D,
    FramePacket,
    PerformanceSnapshot,
    RobotArmState,
    SafetyLevel,
    Track3DState,
)

LOGGER = logging.getLogger(__name__)

_OBSTACLE_BACKEND_LABELS = {
    "edgetam": "EdgeTAM + RGB Hand Gate + 3D PointCloud",
    "pointcloud": "3D PointCloud Only (Safe Fallback)",
    "yolo": "Legacy COCO YOLO / MediaPipe Hand Gate + RGB-D",
}

_RECONSTRUCTION_METHOD_LABELS = {
    "video_depth": "Metric Video Depth Anything (Temporal)",
    "mast3r_slam": "MASt3R-SLAM (Global Dense Map)",
    "st4rtrack": "St4RTrack (Two-frame Pointmap)",
    "hybrid": "St4RTrack + Fast Depth (Hybrid)",
    "fast_depth": "Depth Anything V2 (Per-frame)",
    "rgbd": "RGB-D Camera",
}


def _fov_degrees(image_size: int, focal_length: float) -> float:
    return math.degrees(2.0 * math.atan(float(image_size) / (2.0 * float(focal_length))))


def _projection_status(
    horizontal_fov: float,
    vertical_fov: float,
    width: int,
    height: int,
) -> str:
    focal_x = width / (2.0 * math.tan(math.radians(horizontal_fov) / 2.0))
    focal_y = height / (2.0 * math.tan(math.radians(vertical_fov) / 2.0))
    return (
        f"Projection: **{width}×{height}** · "
        f"`fx={focal_x:.1f}px` · `fy={focal_y:.1f}px`  \n"
        "This changes X/Z ray geometry; reacquire the 8 cm AprilTag metric lock."
    )


def _view_correction_status(
    pitch_down_deg: float,
    roll_deg: float,
    yaw_deg: float,
) -> str:
    return (
        "Legacy angle values are ignored by Metric BEV: "
        f"{pitch_down_deg:.1f}° / {roll_deg:.1f}° / {yaw_deg:.1f}°.  \n"
        "The live work plane is estimated from 3D geometry and projected into "
        "orthographic metre coordinates; controller coordinates are unchanged."
    )


def _yolo_object_checkpoint_status(model: str, *, available: bool) -> str:
    """Describe the legacy object checkpoint without calling it a hand model."""

    escaped = html.escape(str(model))
    if not available:
        return (
            "🔴 **Configured YOLO object weights are missing:** "
            f"`{escaped}`  \n"
            "This is not a hand-semantic checkpoint."
        )
    return (
        "⚪ **Local COCO YOLO object weights available (standby):** "
        f"`{escaped}`  \n"
        "These 80-class COCO weights are **not a human-hand model**. "
        "Hand semantics currently come from the independent MediaPipe gate."
    )


class Dashboard:
    """Viser video/control dashboard. 3D scene rendering is added by Scene3D."""

    def __init__(
        self,
        config: GuiConfig,
        on_command: Callable[[str, object | None], None],
        reconstruction_only: bool = False,
        people_overlay: bool = False,
        projection_config: object | None = None,
        obstacle_model: str | None = None,
        obstacle_model_options: Sequence[str] = (),
        obstacle_backend: str | None = None,
        obstacle_backend_options: Sequence[str] = (),
        reconstruction_method: str | None = None,
        reconstruction_method_options: Sequence[str] = (),
        openarm_control: bool = False,
        openarm_hand_config: dict[str, object] | None = None,
    ) -> None:
        import viser

        self.server = viser.ViserServer(host=config.host, port=config.port)
        actual_port = int(self.server.get_port())
        if actual_port != int(config.port):
            self.server.stop()
            raise RuntimeError(
                f"GUI port {config.port} is already in use; refusing to "
                f"silently start the new dashboard on port {actual_port}. "
                "Stop the existing run_koch_stream.sh process first."
            )
        self.config = config
        self.reconstruction_only = reconstruction_only
        self.people_overlay = people_overlay
        self.openarm_control = bool(openarm_control)
        self.openarm_hand_config = openarm_hand_config or {}
        self.projection_config = projection_config
        self.presentation_mode = bool(config.presentation_mode and reconstruction_only)
        self.max_video_width = config.max_video_width
        self.server.gui.configure_theme(
            titlebar_content=_presentation_titlebar() if self.presentation_mode else None,
            control_layout="collapsible",
            control_width="large" if self.presentation_mode else "medium",
            dark_mode=self.presentation_mode or not reconstruction_only,
            show_logo=not reconstruction_only and not self.presentation_mode,
            show_share_button=False,
            brand_color=(20, 155, 190) if self.presentation_mode else None,
        )
        self._on_command = on_command
        self._lock = threading.Lock()
        model_options = tuple(dict.fromkeys(str(option) for option in obstacle_model_options))
        if obstacle_model and obstacle_model not in model_options:
            model_options = (str(obstacle_model), *model_options)
        self._obstacle_model_options = model_options
        self._active_obstacle_model = str(obstacle_model) if obstacle_model else None
        self._updating_obstacle_model = False
        self._obstacle_model_status_generation = 0
        backend_options = tuple(
            dict.fromkeys(str(option) for option in obstacle_backend_options)
        )
        if obstacle_backend and obstacle_backend not in backend_options:
            backend_options = (str(obstacle_backend), *backend_options)
        self._obstacle_backend_options = backend_options
        self._active_obstacle_backend = (
            str(obstacle_backend) if obstacle_backend else None
        )
        self._obstacle_backend_labels = {
            backend: _OBSTACLE_BACKEND_LABELS.get(backend, backend)
            for backend in backend_options
        }
        self._obstacle_backend_by_label = {
            label: backend
            for backend, label in self._obstacle_backend_labels.items()
        }
        self._updating_obstacle_backend = False
        self._selected_obstacle_backend = self._active_obstacle_backend
        method_options = tuple(
            dict.fromkeys(str(option) for option in reconstruction_method_options)
        )
        if reconstruction_method and reconstruction_method not in method_options:
            method_options = (str(reconstruction_method), *method_options)
        self._reconstruction_method_options = method_options
        self._active_reconstruction_method = (
            str(reconstruction_method) if reconstruction_method else None
        )
        self._reconstruction_method_labels = {
            method: _RECONSTRUCTION_METHOD_LABELS.get(method, method)
            for method in method_options
        }
        self._reconstruction_method_by_label = {
            label: method
            for method, label in self._reconstruction_method_labels.items()
        }
        self._updating_reconstruction_method = False
        self._edge_metric = "WAITING"
        self._latest_people_detections: list[Detection2D] = []
        self._latest_robot_arm: RobotArmState | None = None
        self._latest_obstacle_distances: dict[int, float] = {}
        self._input_connected: bool | None = None
        self._last_video_bgr: np.ndarray | None = None
        self._last_video_at = 0.0
        self._upload_dir = Path(tempfile.mkdtemp(prefix="realtime_safety_upload_"))
        placeholder = _video_placeholder(self.presentation_mode, reconstruction_only)
        self._build_overview()
        if self.people_overlay:
            with self.server.gui.add_folder(
                "Live Perception" if self.presentation_mode else "YOLO obstacle detections",
                order=10,
                expand_by_default=True,
            ):
                self.yolo_people_status = self.server.gui.add_markdown(
                    "Waiting for the first aligned perception frame…"
                )
                self.yolo_people_image = self.server.gui.add_image(
                    placeholder[..., ::-1],
                    label="RGB + obstacle tracking" if self.presentation_mode else "Aligned YOLO frame",
                    format="jpeg",
                    jpeg_quality=76 if self.presentation_mode else 80,
                )
                if self.presentation_mode:
                    # The primary preview is refreshed by the independent RGB
                    # renderer. YOLO annotations are overlaid from the latest
                    # available result instead of limiting video to the slower
                    # 3D-alignment rate.
                    self.video = self.yolo_people_image
        if not (self.presentation_mode and self.people_overlay):
            with self.server.gui.add_folder(
                "Camera Input" if self.presentation_mode else "Live Video",
                order=20,
                expand_by_default=not self.presentation_mode,
            ):
                label = "RGB input" if reconstruction_only else "Annotated RGB"
                self.video = self.server.gui.add_image(
                    placeholder[..., ::-1], label=label, format="jpeg", jpeg_quality=76
                )
        self._build_controls()
        if self.presentation_mode:
            # Viser 1.0 exposes only three fixed sidebar widths. This tiny,
            # client-side enhancement adds the missing desktop drag handle
            # without changing Viser's bundled files or the mobile layout.
            self._sidebar_resizer = self.server.gui.add_html(
                _resizable_sidebar_bootstrap(), order=-1_000
            )

    def _build_overview(self) -> None:
        with self.server.gui.add_folder(
            "System Overview" if self.presentation_mode else "Status",
            order=0,
            expand_by_default=True,
        ):
            if self.presentation_mode:
                self.server.gui.add_markdown(
                    (
                        "**RGB HAND GATE + 3D INTRUSION → EDGETAM TRACKING**"
                        " / **COCO OBJECT WEIGHTS + MEDIAPIPE HAND GATE + "
                        "RGB-D → HAND CLOUD**"
                        if self._obstacle_backend_options
                        else "**MONOCULAR RGB → DEPTH → 3D HUMAN MOTION**"
                    )
                )
            initial_status = (
                "### 🟠 INITIALIZING\nWaiting for the first video frame…"
                if self.presentation_mode
                else "**4D reconstruction:** waiting for a video"
                if self.reconstruction_only
                else "**System:** IDLE"
            )
            self.status = self.server.gui.add_markdown(initial_status)
            perception_label = (
                "Obstacle pipeline" if self._obstacle_backend_options else "YOLO"
            )
            self.live_metrics = self.server.gui.add_markdown(
                f"| Preview | {perception_label} | 4D |\n"
                "|---:|---:|---:|\n| -- | -- | -- |"
            )

    def _build_controls(self) -> None:
        gui = self.server.gui
        if self.openarm_control:
            self._build_openarm_controls(gui)
        if self._reconstruction_method_options:
            initial_method = (
                self._active_reconstruction_method
                or self._reconstruction_method_options[0]
            )
            with gui.add_folder(
                "點雲生成方法 / Point Cloud Method",
                # Keep this primary runtime control above the large live-
                # perception image so it is visible without scrolling.
                order=5,
                expand_by_default=True,
            ):
                self.reconstruction_method = gui.add_dropdown(
                    "Generation method",
                    options=tuple(
                        self._reconstruction_method_labels[method]
                        for method in self._reconstruction_method_options
                    ),
                    initial_value=self._reconstruction_method_labels[initial_method],
                    hint=(
                        "Switches the live point-cloud generator. MASt3R-SLAM "
                        "builds a globally aligned dense map in an isolated "
                        "process; RGB preview remains responsive while a new "
                        "method loads."
                    ),
                )
                self.reconstruction_method_status = gui.add_markdown(
                    "🟠 **正在載入 / Loading:** "
                    f"`{html.escape(self._reconstruction_method_labels[initial_method])}`"
                )

            def request_reconstruction_method(_) -> None:
                if self._updating_reconstruction_method:
                    return
                selected = self._reconstruction_method_by_label.get(
                    str(self.reconstruction_method.value)
                )
                if selected is not None:
                    self._on_command("reconstruction_method", selected)

            self.reconstruction_method.on_update(request_reconstruction_method)

        if self._obstacle_backend_options:
            initial_backend = (
                self._active_obstacle_backend
                or self._obstacle_backend_options[0]
            )
            initial_label = self._obstacle_backend_labels[initial_backend]
            with gui.add_folder(
                "障礙物辨識模型 / Obstacle Model",
                order=15,
                expand_by_default=True,
            ):
                self.obstacle_backend = gui.add_dropdown(
                    "Pipeline / Model",
                    options=tuple(
                        self._obstacle_backend_labels[backend]
                        for backend in self._obstacle_backend_options
                    ),
                    initial_value=initial_label,
                    hint=(
                        "Both paths require an RGB hand confirmation before "
                        "publishing a normal hand obstacle. EdgeTAM is also "
                        "seeded by a sudden RGB-aligned 3D foreground cluster. "
                            "The legacy path has local COCO object weights, but "
                            "they are not a hand-semantic model. Human-hand "
                            "semantics come from the independent MediaPipe gate. "
                            "The controller topic "
                        "is switched atomically after the selected source "
                        "produces its first cloud."
                    ),
                )
                self.obstacle_backend_status = gui.add_markdown(
                    "🟠 **等待 EdgeTAM ROS 2 節點 / Waiting for EdgeTAM node…**"
                )
                self.obstacle_debug_image = gui.add_image(
                    np.zeros((240, 320, 3), dtype=np.uint8),
                    label=(
                        "EdgeTAM live tracking · neural mask + predicted "
                        "intermediate frames"
                    ),
                    format="jpeg",
                    jpeg_quality=82,
                )
                if self._obstacle_model_options:
                    initial_model = (
                        self._active_obstacle_model
                        or self._obstacle_model_options[0]
                    )
                    self.obstacle_model = gui.add_dropdown(
                        "COCO YOLO object checkpoint (not hand model)",
                        options=self._obstacle_model_options,
                        initial_value=initial_model,
                        hint=(
                            "Local 80-class COCO object-segmentation weights for "
                            "the legacy pipeline. They do not contain a human-hand "
                            "class and are never accepted as a hand detector."
                        ),
                    )
                    self.obstacle_model_status = gui.add_markdown(
                        _yolo_object_checkpoint_status(
                            initial_model,
                            available=Path(initial_model).is_file(),
                        )
                    )

            def request_obstacle_backend(_) -> None:
                if self._updating_obstacle_backend:
                    return
                selected = self._obstacle_backend_by_label.get(
                    str(self.obstacle_backend.value)
                )
                if selected is not None:
                    # The dropdown is the operator's sticky selection. Runtime
                    # diagnostics may report loading/error state, but must never
                    # rewrite this choice and trigger visible model jumping.
                    self._selected_obstacle_backend = selected
                    self._on_command("obstacle_backend", selected)

            self.obstacle_backend.on_update(request_obstacle_backend)
            if hasattr(self, "obstacle_model"):
                self.obstacle_model.on_update(
                    lambda _: (
                        None
                        if self._updating_obstacle_model
                        else self._on_command(
                            "segmentation_model",
                            str(self.obstacle_model.value),
                        )
                    )
                )
        elif self._obstacle_model_options:
            initial_model = self._active_obstacle_model or self._obstacle_model_options[0]
            with gui.add_folder(
                "障礙物辨識模型 / Obstacle Model",
                order=15,
                expand_by_default=True,
            ):
                self.obstacle_model = gui.add_dropdown(
                    "模型 / Model",
                    options=self._obstacle_model_options,
                    initial_value=initial_model,
                    hint=(
                        "Runtime-only. The current model stays active until the "
                        "selected local checkpoint has loaded and warmed up."
                    ),
                )
                self.obstacle_model_status = gui.add_markdown(
                    f"🟠 **正在載入 / Loading:** `{html.escape(initial_model)}`"
                )

            def request_obstacle_model(_) -> None:
                if self._updating_obstacle_model:
                    return
                self._on_command("segmentation_model", str(self.obstacle_model.value))

            self.obstacle_model.on_update(request_obstacle_model)

        with gui.add_folder(
            "Source & Playback" if self.presentation_mode else "Input & Playback",
            order=60,
            expand_by_default=not self.presentation_mode,
        ):
            self.file_path = gui.add_text("File / RTSP / webcam", initial_value="auto")
            self.upload = gui.add_upload_button("Upload Video", mime_type="video/*")
            self.start = gui.add_button("Start", color="green")
            self.detect_camera = gui.add_button("Auto-detect USB webcam", color="blue")
            self.camera_status = gui.add_markdown("Webcam: **searching automatically on startup**")
            self.pause = gui.add_button("Pause / Resume")
            self.stop = gui.add_button("Stop", color="red")
            self.restart = gui.add_button("Restart")
            self.loop = gui.add_checkbox("Loop", initial_value=False)
            self.speed = gui.add_slider("Playback speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
            self.seek = gui.add_number("Seek (seconds)", initial_value=0.0, min=0.0)
        self.start.on_click(lambda _: self._on_command("start", self.file_path.value or "auto"))
        self.detect_camera.on_click(lambda _: self._on_command("detect_camera", None))
        self.pause.on_click(lambda _: self._on_command("pause_resume", None))
        self.stop.on_click(lambda _: self._on_command("stop", None))
        self.restart.on_click(lambda _: self._on_command("restart", None))
        self.loop.on_update(lambda _: self._on_command("loop", self.loop.value))
        self.speed.on_update(lambda _: self._on_command("speed", self.speed.value))
        self.seek.on_update(lambda _: self._on_command("seek", self.seek.value))

        if self.reconstruction_only and self.projection_config is not None:
            projection = self.projection_config
            width = max(1, int(self.max_video_width))
            principal_y = getattr(projection, "principal_point_y", None)
            height = (
                max(1, int(round(2.0 * float(principal_y) + 1.0)))
                if principal_y is not None
                else max(1, int(round(width * 0.75)))
            )
            focal_x = getattr(projection, "focal_length_x", None)
            focal_y = getattr(projection, "focal_length_y", None)
            profile_hfov = _fov_degrees(width, focal_x or 0.85 * max(width, height))
            profile_vfov = _fov_degrees(height, focal_y or 0.85 * max(width, height))
            with gui.add_folder(
                "Camera Geometry / BEV 數學視角修正",
                order=65,
                expand_by_default=True,
            ):
                self.horizontal_fov = gui.add_slider(
                    "Horizontal FOV (deg)",
                    min=35.0,
                    max=120.0,
                    step=0.5,
                    initial_value=profile_hfov,
                )
                self.vertical_fov = gui.add_slider(
                    "Vertical FOV (deg)",
                    min=25.0,
                    max=100.0,
                    step=0.5,
                    initial_value=profile_vfov,
                )
                self.projection_status = gui.add_markdown(
                    _projection_status(profile_hfov, profile_vfov, width, height)
                )
                self.metric_bev_enabled = gui.add_checkbox(
                    "啟用正射 BEV / Orthographic BEV",
                    initial_value=bool(
                        getattr(self.config, "metric_bev_enabled", False)
                    ),
                )
                self.metric_bev_height = gui.add_slider(
                    "障礙高度門檻 / Height above plane (m)",
                    min=0.005,
                    max=0.25,
                    step=0.005,
                    initial_value=float(
                        getattr(
                            self.config,
                            "metric_bev_obstacle_height_m",
                            0.035,
                        )
                    ),
                )
                self.view_correction_status = gui.add_markdown(
                    "以 RANSAC 從即時 3D 點估計桌面，投影成正射公尺座標；"
                    "不是旋轉原始點雲。0.10 m 網格與高度圖可直接用於避障。"
                )
                self.recalibrate_metric_bev = gui.add_button(
                    "重新估計工作平面 / Recalibrate BEV",
                    color="blue",
                )
                self.reset_projection = gui.add_button("Reset camera geometry")

            def update_projection(_) -> None:
                horizontal = float(self.horizontal_fov.value)
                vertical = float(self.vertical_fov.value)
                self.projection_status.content = _projection_status(
                    horizontal, vertical, width, height
                )
                self._on_command(
                    "camera_fov",
                    (horizontal, vertical, width, height),
                )

            self.horizontal_fov.on_update(update_projection)
            self.vertical_fov.on_update(update_projection)

            self.metric_bev_enabled.on_update(
                lambda _: self._on_command(
                    "camera_bev_enabled",
                    bool(self.metric_bev_enabled.value),
                )
            )
            self.metric_bev_height.on_update(
                lambda _: self._on_command(
                    "camera_bev_height",
                    float(self.metric_bev_height.value),
                )
            )
            self.recalibrate_metric_bev.on_click(
                lambda _: self._on_command("camera_bev_recalibrate", None)
            )

            def reset_projection(_) -> None:
                self.horizontal_fov.value = profile_hfov
                self.vertical_fov.value = profile_vfov
                self.metric_bev_enabled.value = True
                self.metric_bev_height.value = 0.035
                update_projection(None)
                self._on_command("camera_bev_recalibrate", None)

            self.reset_projection.on_click(reset_projection)

        @self.upload.on_upload
        def _(event) -> None:
            uploaded = event.target.value
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                LOGGER.warning("Rejected uploaded extension: %s", suffix)
                return
            destination = self._upload_dir / Path(uploaded.name).name
            destination.write_bytes(uploaded.content)
            self.file_path.value = str(destination)
            self._on_command("start", str(destination))

        self.visibility = {}
        if not self.reconstruction_only:
            with gui.add_folder("Visibility", order=61):
                self.visibility = {
                    label: gui.add_checkbox(label, initial_value=True)
                    for label in (
                        "Point Cloud", "Tracking Points", "Segmentation", "Static Obstacles",
                        "Dynamic Obstacles", "Bounding Boxes", "Velocity Arrows", "Trajectories",
                        "Danger Zones", "Ground Plane", "Traversable Region", "Candidate Paths", "Selected Path",
                    )
                }
                for label, handle in self.visibility.items():
                    handle.on_update(lambda _, key=label: self._on_command("visibility", (key, self.visibility[key].value)))

        with gui.add_folder(
            "Diagnostics" if self.presentation_mode else "Performance",
            order=70,
            expand_by_default=not self.presentation_mode,
        ):
            self.performance = gui.add_markdown("Waiting for frames…")

    def _build_openarm_controls(self, gui: Any) -> None:
        with gui.add_folder(
            "OpenArm Control",
            order=4,
            expand_by_default=True,
        ):
            self.openarm_targets_note = gui.add_markdown(
                "Move **left_target_cube** and **right_target_cube** directly "
                "in Gazebo. Their gravity is disabled; each cube is the exact "
                "touch target for that arm's gripper TCP (the planner uses "
                "the marker surface facing the arm)."
            )
            self.openarm_move_both_targets = gui.add_button(
                "MOVE BOTH TCPs TO GAZEBO CUBES", color="green"
            )
            self.openarm_move_left_target = gui.add_button(
                "MOVE LEFT TCP", color="blue"
            )
            self.openarm_move_right_target = gui.add_button(
                "MOVE RIGHT TCP", color="blue"
            )
            self.openarm_home = gui.add_button("HOME", color="blue")
            self.openarm_pause = gui.add_button("PAUSE ROBOT", color="orange")
            self.openarm_resume = gui.add_button("RESUME", color="green")
            self.openarm_reset = gui.add_button("RESET")
            self.openarm_estop = gui.add_button("EMERGENCY STOP", color="red")
            self.openarm_planner = gui.add_dropdown(
                "Planner",
                options=("MoveIt Baseline", "MoveIt + Dynamic Safety"),
                initial_value="MoveIt + Dynamic Safety",
            )
            self.openarm_obstacle_source = gui.add_dropdown(
                "Obstacle Source",
                options=("Ground Truth", "Perception"),
                initial_value="Perception",
            )
            self.openarm_hand_note = gui.add_markdown(
                "Human hand pose: edit **human_hand** directly in Gazebo. "
                "The textured LibHand mesh, collision proxy and perception remain live."
            )
            self.manual_hand_speed = gui.add_dropdown(
                "Hand max speed (select then apply)",
                options=("0.01 m/s", "0.02 m/s", "0.05 m/s", "0.1 m/s", "0.3 m/s", "0.6 m/s"),
                initial_value="0.01 m/s",
            )
            self.manual_hand_apply_speed = gui.add_button(
                "APPLY HAND SPEED", color="blue"
            )
            self.manual_hand_auto_sweep = gui.add_checkbox(
                "Auto left/right sweep (fixed height)", initial_value=False
            )
            self.manual_hand_preview = gui.add_button(
                "SHOW HAND + POINT CLOUD", color="green"
            )
            self.manual_hand_withdraw = gui.add_button("Withdraw")
            self.manual_hand_reset = gui.add_button("Reset Hand")
            self.manual_hand_intrusion = gui.add_button(
                "Sudden Intrusion", color="orange"
            )
            self.openarm_control_status = gui.add_markdown(
                "Task: **WAITING** · Safety: **WAITING** · Dynamic: **WAITING**"
            )
        self._openarm_status = {
            "task": "WAITING",
            "safety": "WAITING",
            "dynamic": "WAITING",
            "hand": "WAITING",
            "command": "none",
        }
        self.openarm_move_both_targets.on_click(
            lambda _: self._on_command("openarm_move_both_targets", None)
        )
        self.openarm_move_left_target.on_click(
            lambda _: self._on_command("openarm_move_left_target", None)
        )
        self.openarm_move_right_target.on_click(
            lambda _: self._on_command("openarm_move_right_target", None)
        )
        self.openarm_home.on_click(
            lambda _: self._on_command("openarm_home", None)
        )
        self.openarm_pause.on_click(
            lambda _: self._on_command("openarm_pause", None)
        )
        self.openarm_resume.on_click(
            lambda _: self._on_command("openarm_resume", None)
        )
        self.openarm_reset.on_click(
            lambda _: self._on_command("openarm_reset", None)
        )
        self.openarm_estop.on_click(
            lambda _: self._on_command("openarm_estop", None)
        )

        def planner_changed(_) -> None:
            mode = (
                "dynamic"
                if self.openarm_planner.value == "MoveIt + Dynamic Safety"
                else "moveit"
            )
            self._on_command("openarm_planner", mode)

        def obstacle_changed(_) -> None:
            source = (
                "perception"
                if self.openarm_obstacle_source.value == "Perception"
                else "ground_truth"
            )
            self._on_command("openarm_obstacle_source", source)

        self.openarm_planner.on_update(planner_changed)
        self.openarm_obstacle_source.on_update(obstacle_changed)

        def hand_speed_changed(_) -> None:
            speed = float(str(self.manual_hand_speed.value).split()[0])
            self._on_command("openarm_hand_speed", speed)

        self.manual_hand_speed.on_update(hand_speed_changed)
        self.manual_hand_apply_speed.on_click(hand_speed_changed)

        def hand_auto_sweep_changed(_) -> None:
            self._on_command(
                "openarm_hand_auto_sweep",
                bool(self.manual_hand_auto_sweep.value),
            )

        self.manual_hand_auto_sweep.on_update(hand_auto_sweep_changed)
        self.manual_hand_preview.on_click(
            lambda _: self._on_command("openarm_hand_preview", None)
        )
        self.manual_hand_withdraw.on_click(
            lambda _: self._on_command("openarm_hand_withdraw", None)
        )
        self.manual_hand_reset.on_click(
            lambda _: self._on_command("openarm_hand_reset", None)
        )
        self.manual_hand_intrusion.on_click(
            lambda _: self._on_command("openarm_hand_intrusion", None)
        )

    def update_openarm_control_status(self, key: str, value: str) -> None:
        if not self.openarm_control:
            return
        self._openarm_status[str(key)] = str(value)
        status = self._openarm_status
        self.openarm_control_status.content = (
            f"Task: **{status['task']}** · Safety: **{status['safety']}**  \n"
            f"Dynamic: **{status['dynamic']}**  \n"
            f"Hand: **{status['hand']}**  \n"
            f"Last command: `{status['command']}`"
        )

    def update_reconstruction_method_status(
        self,
        active_method: str,
        *,
        requested_method: str | None = None,
        error: str | None = None,
        ready: bool = True,
    ) -> None:
        if not hasattr(self, "reconstruction_method_status"):
            return
        active_method = str(active_method)
        active_label = self._reconstruction_method_labels.get(
            active_method, active_method
        )
        if error is not None:
            requested = self._reconstruction_method_labels.get(
                str(requested_method), str(requested_method or active_method)
            )
            self.reconstruction_method_status.content = (
                "🔴 **切換失敗 / Switch failed:** "
                f"{html.escape(str(error))}  \n"
                f"仍使用 / Still active: **{html.escape(active_label)}**"
            )
            self._set_reconstruction_method_value(active_method)
            return
        if requested_method is not None and str(requested_method) != active_method:
            requested_label = self._reconstruction_method_labels.get(
                str(requested_method), str(requested_method)
            )
            self.reconstruction_method_status.content = (
                "🟠 **切換中 / Switching:** "
                f"**{html.escape(requested_label)}**  \n"
                f"目前 / Current: **{html.escape(active_label)}**"
            )
            return
        state = "🟢 **已生效 / Active:**" if ready else "🟠 **正在載入 / Loading:**"
        self.reconstruction_method_status.content = (
            f"{state} **{html.escape(active_label)}**"
        )
        self._active_reconstruction_method = active_method
        self._set_reconstruction_method_value(active_method)

    def _set_reconstruction_method_value(self, method: str) -> None:
        if not hasattr(self, "reconstruction_method"):
            return
        label = self._reconstruction_method_labels.get(method, method)
        if self.reconstruction_method.value == label:
            return
        self._updating_reconstruction_method = True
        try:
            self.reconstruction_method.value = label
        finally:
            self._updating_reconstruction_method = False

    def update_obstacle_backend_status(
        self,
        backend: str,
        *,
        state: str,
        detail: str = "",
        metrics: dict[str, str] | None = None,
    ) -> None:
        """Show the effective ROS pipeline and evidence that EdgeTAM ran."""

        if not hasattr(self, "obstacle_backend_status"):
            return
        backend = str(backend)
        state = str(state)
        metrics = dict(metrics or {})
        label = html.escape(
            self._obstacle_backend_labels.get(backend, backend)
        )
        escaped_detail = html.escape(str(detail))
        # Older scheduler diagnostics call this a "YOLO RGB hand" pipeline.
        # In hand-only mode the semantic decision is actually MediaPipe; the
        # selected 80-class COCO weights do not contain a human-hand class.
        # Do not repeat a stale/misleading producer description in the GUI.
        if backend == "yolo" and metrics.get("hand_only", "false") == "true":
            escaped_detail = (
                "COCO YOLO supplies generic object masks; "
                "MediaPipe supplies the human-hand semantic gate."
            )
        background_state = metrics.get("background_state", "disabled")
        safety_output_state = metrics.get(
            "safety_output_state", "publishing_verified"
        )
        edge_model_ready = bool(
            backend == "edgetam"
            and metrics.get("edge_status") == "ready"
            and not metrics.get("edge_error")
            and state in {"ready", "active"}
        )
        if backend == "yolo" and state in {"ready", "active"}:
            icon = "🟢"
            headline = "Legacy hand pipeline active / 舊版手部流程已生效"
        elif edge_model_ready:
            icon = "🟢"
            headline = (
                "EdgeTAM 模型已就緒 / EdgeTAM model ready"
            )
        elif safety_output_state.startswith("held_"):
            icon = "🔴"
            headline = (
                "手部量測未確認，安全輸出暫停 / "
                "Untrusted hand output held"
            )
        elif background_state in {"warming_up", "calibrating"}:
            icon = "🟠"
            headline = "背景校正中，請保持工作區淨空 / Calibrating background"
        elif backend == "pointcloud" and state == "active":
            icon = "🟢"
            headline = "安全降級已生效 / Safe fallback active"
        elif state in {"ready", "active"}:
            icon = "🟢"
            headline = "已生效 / Active"
        elif state in {"loading", "switching", "waiting"}:
            icon = "🟠"
            headline = "啟動中 / Starting"
        elif state in {"disabled", "pointcloud"}:
            icon = "🟢"
            headline = "安全降級已生效 / Safe fallback active"
        else:
            icon = "🔴"
            headline = "無法使用 / Unavailable"

        lines = [f"{icon} **{headline}:** `{label}`"]
        if escaped_detail:
            lines.append(escaped_detail)
        if metrics and backend == "yolo":
            yolo_model = html.escape(metrics.get("yolo_model", "--"))
            yolo_fps = html.escape(metrics.get("yolo_fps", "--"))
            yolo_latency = html.escape(metrics.get("yolo_latency_ms", "--"))
            tracks = html.escape(metrics.get("track_count", "0"))
            output_points = html.escape(metrics.get("output_points", "0"))
            lines.append(
                f"Model: **{yolo_model}** · YOLO: **{yolo_fps} FPS / "
                f"{yolo_latency} ms** · 3D tracks: **{tracks}** · "
                f"obstacle points: **{output_points}**"
            )
            if metrics.get("hand_only", "false") == "true":
                lines.append(
                    "Hand semantic source: **MediaPipe (not YOLO)** · "
                    "the listed COCO checkpoint has no human-hand class · "
                    "output policy: **HAND ONLY**"
                )
            self._edge_metric = f"YOLO {yolo_fps} FPS"
        elif metrics:
            fps = html.escape(metrics.get("fps", "--"))
            latency = html.escape(metrics.get("edge_latency_ms", "--"))
            tracks = html.escape(metrics.get("track_count", "--"))
            refined = html.escape(
                metrics.get("edge_refined_corrections", "0")
            )
            prompts = html.escape(metrics.get("prompt_count", "0"))
            good_masks = html.escape(metrics.get("mask_good_count", "0"))
            degraded_masks = html.escape(
                metrics.get("mask_degraded_count", "0")
            )
            invalid_masks = html.escape(
                metrics.get("mask_invalid_count", "0")
            )
            lines.append(
                "Tracker: **"
                f"{fps} Hz** · Edge latency: **{latency} ms** · "
                f"tracks/prompts: **{tracks}/{prompts}** · "
                f"masks G/D/I: **{good_masks}/{degraded_masks}/{invalid_masks}** · "
                f"refined: **{refined}**"
            )
            hand_candidates = html.escape(
                metrics.get("hand_candidate_count", "0")
            )
            geometry_fallbacks = html.escape(
                metrics.get("geometry_fallback_track_count", "0")
            )
            semantic_status = html.escape(
                metrics.get("hand_semantic_status", "disabled")
            )
            rgb_hands = html.escape(
                metrics.get("hand_rgb_detection_count", "0")
            )
            if semantic_status == "disabled":
                lines.append(
                    "RGB hand semantic gate: **disabled** · "
                    "unverified scene fallback: **BLOCKED** · "
                    f"geometric candidates held: **{tracks}**"
                )
            else:
                lines.append(
                    "RGB hand semantic gate: "
                    f"**{semantic_status}** · RGB hands: **{rgb_hands}** · "
                    f"RGB+3D confirmed hands: **{hand_candidates}** · "
                    "unverified scene fallback: **BLOCKED** · "
                    f"fallback tracks published: **{geometry_fallbacks}**"
                )
            if safety_output_state.startswith("held_"):
                pipeline_message = html.escape(
                    metrics.get("pipeline_message", "")
                )
                lines.append(
                    "🔴 Safety output: **HELD / controller must STOP on "
                    "perception timeout** · Untrusted hand output held"
                    + (f" · {pipeline_message}" if pipeline_message else "")
                )
            semantic_rejects = html.escape(
                metrics.get("hand_semantic_reject_reasons", "")
            )
            if semantic_rejects:
                lines.append(f"Hand semantic gate: `{semantic_rejects}`")
            reject_reasons = html.escape(
                metrics.get("mask_reject_reasons", "")
            )
            if reject_reasons:
                lines.append(f"Mask gate: `{reject_reasons}`")
            if background_state in {"warming_up", "calibrating"}:
                warmup = html.escape(
                    metrics.get("background_warmup", "0/0")
                )
                calibration = html.escape(
                    metrics.get("background_calibration", "0/0")
                )
                lines.append(
                    "Background warmup/calibration: "
                    f"**{warmup} / {calibration}**"
                )
            elif background_state == "ready":
                removed = html.escape(
                    metrics.get("background_removed", "0")
                )
                baseline = html.escape(
                    metrics.get("background_baseline_points", "0")
                )
                matched = html.escape(
                    metrics.get("background_matched_points", "0")
                )
                alignment = html.escape(
                    metrics.get("background_alignment_points", "0")
                )
                depth_scale = html.escape(
                    metrics.get("background_depth_scale", "1.00000")
                )
                candidate_scale = html.escape(
                    metrics.get(
                        "background_candidate_depth_scale", "1.00000"
                    )
                )
                candidate_points = html.escape(
                    metrics.get(
                        "background_candidate_alignment_points", "0"
                    )
                )
                candidate_reason = html.escape(
                    metrics.get(
                        "background_candidate_alignment_reason",
                        "not_evaluated",
                    )
                )
                alignment_valid = (
                    metrics.get("background_alignment_valid", "false")
                    == "true"
                )
                lines.append(
                    "Foreground filter: **ready** · static points removed: "
                    f"**{removed}** · ray matches: **{matched}** · "
                    f"RGB alignment: **{alignment}** · depth scale: "
                    f"**{depth_scale}×** ({'valid' if alignment_valid else 'bypass'})"
                    f" · baseline: **{baseline}**"
                )
                if not alignment_valid:
                    lines.append(
                        "Alignment recovery candidate: "
                        f"**{candidate_scale}× / {candidate_points} points**"
                        f" · reason: `{candidate_reason}`"
                    )
            self._edge_metric = (
                f"{latency} ms" if backend == "edgetam" else f"{fps} Hz"
            )
        elif state in {"loading", "switching", "waiting"}:
            self._edge_metric = "LOADING"
        elif state in {"error", "unavailable"}:
            self._edge_metric = "ERROR"
        else:
            self._edge_metric = "PC ONLY"
        self.obstacle_backend_status.content = "  \n".join(lines)
        self._set_obstacle_backend_value(backend)

    def update_edgetam_debug_image(self, bgr: np.ndarray) -> None:
        """Display the live point-cloud prompts and returned EdgeTAM masks."""

        if not hasattr(self, "obstacle_debug_image"):
            return
        image = np.asarray(bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            return
        display = self._resize_for_gui(image[..., :3])
        with self._lock:
            self.obstacle_debug_image.image = cv2.cvtColor(
                display, cv2.COLOR_BGR2RGB
            )

    def _set_obstacle_backend_value(self, backend: str) -> None:
        self._active_obstacle_backend = str(backend)
        # Runtime state is intentionally separate from the operator's dropdown
        # selection. A stale/error diagnostic may change the effective output,
        # but only a human interaction may change the selected model.

    def update_obstacle_model_status(
        self,
        active_model: str,
        *,
        requested_model: str | None = None,
        error: str | None = None,
        ready: bool = True,
        generation: int | None = None,
    ) -> None:
        """Reflect the effective backend, not merely the dropdown selection."""

        if not hasattr(self, "obstacle_model_status"):
            return
        active = html.escape(str(active_model))
        with self._lock:
            current_generation = getattr(
                self,
                "_obstacle_model_status_generation",
                0,
            )
            if generation is not None and generation < current_generation:
                return
            if generation is not None:
                self._obstacle_model_status_generation = generation
            self._active_obstacle_model = str(active_model)
            if error is not None:
                detail = html.escape(str(error))
                self.obstacle_model_status.content = (
                    f"🔴 **切換失敗 / Switch failed:** {detail}  \n"
                    f"仍使用 / Still active: `{active}`"
                    if ready
                    else (
                        f"🔴 **模型無法使用 / Model unavailable:** {detail}  \n"
                        f"Configured: `{active}`"
                    )
                )
                self._set_obstacle_model_value(str(active_model))
                return
            if requested_model is not None and str(requested_model) != str(active_model):
                requested = html.escape(str(requested_model))
                self.obstacle_model_status.content = (
                    f"🟠 **切換中 / Switching:** `{requested}`  \n"
                    f"目前仍使用 / Current output: `{active}`"
                )
                return
            if ready:
                self.obstacle_model_status.content = (
                    f"🟢 **COCO object weights active:** `{active}`  \n"
                    "This checkpoint is **not the hand-semantic detector**; "
                    "the hand gate remains independent."
                )
                self._set_obstacle_model_value(str(active_model))
            else:
                self.obstacle_model_status.content = (
                    f"🟠 **正在載入 / Loading:** `{active}`"
                )

    def _set_obstacle_model_value(self, model: str) -> None:
        if not hasattr(self, "obstacle_model") or self.obstacle_model.value == model:
            return
        self._updating_obstacle_model = True
        try:
            self.obstacle_model.value = model
        finally:
            self._updating_obstacle_model = False

    def update_camera_status(
        self,
        message: str,
        source: str | int | None = None,
        *,
        connected: bool | None = None,
    ) -> None:
        self.camera_status.content = message
        if source is not None:
            self.file_path.value = str(source)
        if connected is None:
            return
        with self._lock:
            state_changed = connected != self._input_connected
            self._input_connected = connected
            if not connected and state_changed:
                self._show_input_lost_frame()
        if not connected:
            self._show_input_lost_status()

    def update_video(self, annotated_bgr: np.ndarray) -> None:
        with self._lock:
            display = self._resize_for_gui(annotated_bgr)
            if (
                self.presentation_mode
                and self.people_overlay
                and getattr(self, "_active_obstacle_backend", None) == "yolo"
            ):
                display = self._draw_people_overlay(
                    display,
                    self._latest_people_detections,
                    getattr(self, "_latest_robot_arm", None),
                    getattr(self, "_latest_obstacle_distances", {}),
                )
            self._last_video_bgr = display.copy()
            self._last_video_at = time.monotonic()
            self.video.image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    def update_performance(self, perf: PerformanceSnapshot) -> None:
        perception_label = (
            self._obstacle_backend_labels.get(
                getattr(self, "_active_obstacle_backend", ""),
                "Obstacle pipeline",
            )
            if getattr(self, "_obstacle_backend_options", ())
            else "YOLO"
        )
        if self._input_connected is False:
            self.live_metrics.content = (
                f"| Preview | {perception_label} | 4D |\n"
                "|---:|---:|---:|\n"
                "| **INPUT LOST** | **PAUSED** | **PAUSED** |"
            )
        else:
            perception = (
                self._edge_metric
                if getattr(self, "_obstacle_backend_options", ())
                else f"{perf.segmentation_fps:.1f} FPS"
                if self.people_overlay
                else "OFF"
            )
            self.live_metrics.content = (
                f"| Preview | {perception_label} | 4D |\n"
                "|---:|---:|---:|\n"
                f"| **{perf.display_fps:.1f} FPS** | **{perception}** | **{perf.reconstruction_fps:.1f} FPS** |"
            )
        if self.reconstruction_only:
            headers = "| Input | Display | YOLO person | 4D reconstruction |" if self.people_overlay else "| Input | Display | 4D reconstruction |"
            divider = "|---:|---:|---:|---:|" if self.people_overlay else "|---:|---:|---:|"
            values = (
                f"| {perf.input_fps:.1f} FPS | {perf.display_fps:.1f} FPS | {perf.segmentation_fps:.1f} FPS | "
                f"{perf.reconstruction_fps:.1f} FPS |"
                if self.people_overlay
                else f"| {perf.input_fps:.1f} FPS | {perf.display_fps:.1f} FPS | {perf.reconstruction_fps:.1f} FPS |"
            )
            self.performance.content = (
                f"{headers}\n{divider}\n{values}\n\n"
                f"Dropped: **{perf.dropped_frames}** · Queue: **{perf.queue_size}/{perf.queue_capacity}**  \n"
                f"RAM: **{perf.ram_mb:.0f} MB** · VRAM: **{perf.vram_used_mb:.0f} MB**"
            )
            return
        self.performance.content = (
            f"| Input | Display | Segmentation | 3D | Safety |\n"
            f"|---:|---:|---:|---:|---:|\n"
            f"| {perf.input_fps:.1f} FPS | {perf.display_fps:.1f} FPS | {perf.segmentation_fps:.1f} FPS | "
            f"{perf.reconstruction_fps:.1f} FPS | {perf.safety_fps:.1f} FPS |\n\n"
            f"Latency avg / p95: **{perf.average_latency_ms:.1f} / {perf.p95_latency_ms:.1f} ms**  \n"
            f"Dropped: **{perf.dropped_frames}** · Queue: **{perf.queue_size}/{perf.queue_capacity}**  \n"
            f"RAM: **{perf.ram_mb:.0f} MB** · VRAM: **{perf.vram_used_mb:.0f} MB**"
        )

    def update_status(
        self,
        level: SafetyLevel,
        profile: str,
        depth_mode: str,
        scale_mode: str,
        cuda_status: str,
        action: str,
    ) -> None:
        self.status.content = (
            f"# {level.value}\n"
            f"Recommended action: **{action}**  \n"
            f"Profile: **{profile}** · Depth: **{depth_mode}**  \n"
            f"Scale: **{scale_mode.upper()} SCALE** · CUDA: **{cuda_status}**"
        )

    def update_reconstruction_status(
        self,
        depth_mode: str,
        cuda_status: str,
        model_ready: bool,
        errors: dict[str, str],
        yolo_count: int = 0,
        people_3d_count: int = 0,
        yolo_ready: bool = False,
        held_3d_count: int = 0,
        metric_scale: float | None = None,
        reference_depth_m: float | None = None,
        observed_reference_depth: float | None = None,
        apriltag_locked: bool = False,
        apriltag_id: int | None = None,
        apriltag_size_m: float | None = None,
        apriltag_age_frames: int | None = None,
    ) -> None:
        if self.presentation_mode:
            if self._input_connected is False:
                self._show_input_lost_status()
                return
            if model_ready:
                state = (
                    "### 🟢 SYSTEM READY\n"
                    "**Selectable RGB + PointCloud obstacle tracking**"
                    if getattr(self, "_obstacle_backend_options", ())
                    else "### 🟢 SYSTEM READY\n**Real-time 4D human perception**"
                )
            elif errors:
                detail = next(iter(errors.values()))
                state = f"### 🔴 MODEL ERROR\n{detail}"
            else:
                state = "### 🟠 LOADING MODELS\nCamera preview remains available."
            if self.people_overlay:
                stable_3d_count = max(0, people_3d_count - held_3d_count)
                state += (
                    "\n\n| 2D obstacles | 3D tracks | stable |\n"
                    "|---:|---:|---:|\n"
                    f"| **{yolo_count}** | **{people_3d_count}** | **{stable_3d_count}** |"
                )
            backend = depth_mode.replace("_", " ").title()
            acceleration = "GPU accelerated" if "CPU" not in cuda_status.upper() else "CPU"
            if apriltag_locked and metric_scale is not None:
                state += (
                    f"\n\nDepth: **APRILTAG #{apriltag_id} METRIC LOCK**"
                    f" · side **{(apriltag_size_m or 0.0) * 100.0:.1f} cm**"
                    f" · scale **{metric_scale:.3f}×**"
                    f" · age **{apriltag_age_frames or 0}f**"
                )
            elif metric_scale is not None and reference_depth_m is not None:
                raw = (
                    f" · raw **{observed_reference_depth:.2f}**"
                    if observed_reference_depth is not None
                    else ""
                )
                state += (
                    f"\n\nDepth: **CALIBRATED {reference_depth_m:.2f} m**"
                    f" · scale **{metric_scale:.3f}×**{raw}"
                )
            self.status.content = f"{state}\n\n`{backend}` · `{acceleration}`"
            return
        if model_ready:
            state = "### 4D reconstruction\nModel: **ready**"
        elif errors:
            detail = next(iter(errors.values()))
            state = f"### Model error\n{detail}"
        else:
            state = "### Loading model\nThe RGB video remains available while the 3D model loads."
        people = (
            f"  \nYOLO obstacles: **{yolo_count}** · tracked 3D boxes: **{people_3d_count}** "
            f"(short hold: **{held_3d_count}**)  \n"
            f"YOLO model: **{'ready' if yolo_ready else 'loading'}**"
            if self.people_overlay
            else ""
        )
        calibration = (
            f"  \nAprilTag #{apriltag_id}: **{(apriltag_size_m or 0.0) * 100.0:.1f} cm SCALE LOCK**"
            f" · scale: **{metric_scale:.3f}×**"
            if apriltag_locked and metric_scale is not None
            else
            f"  \nMetric reference: **{reference_depth_m:.2f} m** · scale: **{metric_scale:.3f}×**"
            if metric_scale is not None and reference_depth_m is not None
            else ""
        )
        self.status.content = (
            f"{state}  \nBackend: **{depth_mode}** · CUDA: **{cuda_status}**"
            f"{calibration}{people}"
        )

    def _show_input_lost_status(self) -> None:
        age = max(time.monotonic() - self._last_video_at, 0.0) if self._last_video_at else 0.0
        age_text = f"Last frame: **{age:.0f} s ago**" if age else "No camera frame received"
        if self.presentation_mode:
            perception_label = (
                "EdgeTAM"
                if getattr(self, "_obstacle_backend_options", ())
                else "YOLO"
            )
            self.live_metrics.content = (
                f"| Preview | {perception_label} | 4D |\n"
                "|---:|---:|---:|\n"
                "| **INPUT LOST** | **PAUSED** | **PAUSED** |"
            )
            self.status.content = (
                "### 🔴 CAMERA INPUT LOST\n"
                "**The GUI is online; automatic reconnect is active.**\n\n"
                f"{age_text}"
            )
            if self.people_overlay:
                source = str(getattr(self.file_path, "value", "camera input"))
                if source.startswith("ros2://"):
                    source = source.removeprefix("ros2://")
                self.yolo_people_status.content = (
                    "**INPUT LOST · INFERENCE PAUSED**  \n"
                    f"Waiting for `{source}` to resume…"
                )

    def _show_input_lost_frame(self) -> None:
        if self._last_video_bgr is None:
            display = _video_placeholder(self.presentation_mode, self.reconstruction_only)
        else:
            display = self._last_video_bgr.copy()
        height, width = display.shape[:2]
        banner_height = max(54, round(height * 0.18))
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (18, 18, 132), -1)
        display = cv2.addWeighted(overlay, 0.86, display, 0.14, 0.0)
        scale = max(0.48, min(width / 760.0, 0.78))
        cv2.putText(
            display,
            "CAMERA INPUT LOST - RECONNECTING",
            (max(12, round(width * 0.025)), max(34, round(banner_height * 0.62))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 255),
            2,
            cv2.LINE_AA,
        )
        self.video.image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    def update_people_detections(
        self,
        frame: FramePacket,
        detections: list[Detection2D],
        *,
        robot_arm: RobotArmState | None = None,
        tracks: list[Track3DState] | None = None,
    ) -> None:
        if not self.people_overlay:
            return
        distances = (
            {
                int(track.track_id): float(
                    np.linalg.norm(track.position_xyz - robot_arm.center_xyz)
                )
                for track in (tracks or [])
            }
            if robot_arm is not None
            else {}
        )
        if self.presentation_mode:
            with self._lock:
                self._latest_people_detections = list(detections)
                self._latest_robot_arm = robot_arm
                self._latest_obstacle_distances = distances
            edge_mode = getattr(
                self,
                "_active_obstacle_backend",
                None,
            ) != "yolo"
            if edge_mode:
                obstacle_count = len(tracks or [])
                confirmed = sum(
                    track.missing_count == 0 for track in (tracks or [])
                )
                held = sum(
                    track.missing_count > 0 for track in (tracks or [])
                )
                source_label = "EdgeTAM hand candidates"
            else:
                obstacle_count = len(detections)
                confirmed = sum(
                    detection.track_hits >= 3
                    and not detection.is_prediction
                    for detection in detections
                )
                held = sum(detection.is_prediction for detection in detections)
                source_label = "YOLO obstacles"
            closest = min(distances.values(), default=None)
            relation = (
                f" · Nearest arm distance: **{closest:.2f} m**"
                if closest is not None
                else ""
            )
            self.yolo_people_status.content = (
                f"**LIVE · FRAME {frame.frame_index:,}**  \n"
                f"{source_label}: **{obstacle_count}** · "
                f"Fresh: **{confirmed}**"
                + (f" · Short hold: **{held}**" if held else "")
                + relation
            )
            return
        canvas = self._draw_people_overlay(
            self._resize_for_gui(frame.bgr),
            detections,
            robot_arm,
            distances,
        )
        self.yolo_people_image.image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if detections:
            rows = "\n".join(
                f"| {detection.track_id or 0} | {detection.confidence:.2f} | "
                f"{'held' if detection.is_prediction else 'confirmed' if detection.track_hits >= 2 else 'pending'} |"
                for detection in sorted(detections, key=lambda item: item.confidence, reverse=True)
            )
            self.yolo_people_status.content = (
                f"Frame **{frame.frame_index}** · YOLO obstacles: **{len(detections)}**  \n"
                f"| ID | confidence | 3D gate |\n|---:|---:|:---|\n{rows}"
            )
        else:
            self.yolo_people_status.content = f"Frame **{frame.frame_index}** · **no YOLO obstacles detected**"

    @staticmethod
    def _draw_people_overlay(
        canvas: np.ndarray,
        detections: list[Detection2D],
        robot_arm: RobotArmState | None = None,
        obstacle_distances: dict[int, float] | None = None,
    ) -> np.ndarray:
        canvas = canvas.copy()
        obstacle_distances = obstacle_distances or {}
        overlay = canvas.copy()
        for detection in detections:
            if detection.mask is not None and not detection.is_prediction:
                mask = cv2.resize(
                    detection.mask.astype(np.uint8),
                    (canvas.shape[1], canvas.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                overlay[mask] = (
                    0.35 * overlay[mask] + 0.65 * np.array((220, 155, 35), dtype=np.float32)
                )
        canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0.0)
        for detection in detections:
            source_width, source_height = detection.image_size or (canvas.shape[1], canvas.shape[0])
            scale_x = canvas.shape[1] / max(source_width, 1)
            scale_y = canvas.shape[0] / max(source_height, 1)
            x1, y1, x2, y2 = np.rint(
                detection.bbox_xyxy * np.array((scale_x, scale_y, scale_x, scale_y), dtype=np.float32)
            ).astype(int)
            x1 = int(np.clip(x1, 0, canvas.shape[1] - 1))
            y1 = int(np.clip(y1, 0, canvas.shape[0] - 1))
            x2 = int(np.clip(x2, x1 + 1, canvas.shape[1]))
            y2 = int(np.clip(y2, y1 + 1, canvas.shape[0]))
            confirmed = detection.track_hits >= 3
            color = (
                (170, 170, 170)
                if detection.is_prediction
                else (255, 175, 35)
                if confirmed
                else (0, 190, 255)
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            state = " HOLD" if detection.is_prediction else ""
            text = (
                f"ID {detection.track_id or 0} {detection.class_name}"
                f"{state}  {detection.confidence:.0%}"
            )
            font_scale = 0.48
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_top = max(0, y1 - text_height - baseline - 8)
            label_bottom = min(canvas.shape[0], label_top + text_height + baseline + 8)
            label_right = min(canvas.shape[1], x1 + text_width + 12)
            cv2.rectangle(canvas, (x1, label_top), (label_right, label_bottom), (12, 20, 28), -1)
            cv2.putText(
                canvas,
                text,
                (x1 + 6, label_bottom - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        if robot_arm is not None:
            source_width, source_height = robot_arm.image_size
            arm_xy = np.rint(
                robot_arm.center_xy
                * np.array(
                    (
                        canvas.shape[1] / max(source_width, 1),
                        canvas.shape[0] / max(source_height, 1),
                    ),
                    dtype=np.float32,
                )
            ).astype(int)
            arm_point = (
                int(np.clip(arm_xy[0], 0, canvas.shape[1] - 1)),
                int(np.clip(arm_xy[1], 0, canvas.shape[0] - 1)),
            )
            for detection in detections:
                if detection.track_id is None:
                    continue
                source_size = detection.image_size or (
                    canvas.shape[1],
                    canvas.shape[0],
                )
                target = np.rint(
                    detection.centroid_xy
                    * np.array(
                        (
                            canvas.shape[1] / max(source_size[0], 1),
                            canvas.shape[0] / max(source_size[1], 1),
                        ),
                        dtype=np.float32,
                    )
                ).astype(int)
                target_point = (int(target[0]), int(target[1]))
                cv2.line(canvas, arm_point, target_point, (255, 190, 40), 1, cv2.LINE_AA)
                distance = obstacle_distances.get(int(detection.track_id))
                if distance is not None:
                    midpoint = (
                        (arm_point[0] + target_point[0]) // 2,
                        (arm_point[1] + target_point[1]) // 2,
                    )
                    cv2.putText(
                        canvas,
                        f"{distance:.2f}m",
                        midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 235, 120),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.circle(canvas, arm_point, 7, (45, 225, 80), -1, cv2.LINE_AA)
            cv2.circle(canvas, arm_point, 11, (45, 225, 80), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "ARM CENTER",
                (arm_point[0] + 12, max(16, arm_point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (70, 255, 110),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _resize_for_gui(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] <= self.max_video_width:
            return image.copy()
        scale = self.max_video_width / image.shape[1]
        return cv2.resize(
            image,
            (self.max_video_width, max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def close(self) -> None:
        self.server.stop()
        shutil.rmtree(self._upload_dir, ignore_errors=True)


def _video_placeholder(presentation_mode: bool, reconstruction_only: bool) -> np.ndarray:
    if not presentation_mode:
        background = 245 if reconstruction_only else 0
        foreground = (55, 55, 55) if reconstruction_only else (220, 220, 220)
        image = np.full((360, 640, 3), background, dtype=np.uint8)
        message = "Connect a webcam or upload a video" if reconstruction_only else "Connect a webcam or select a source"
        cv2.putText(image, message, (55, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.72, foreground, 2)
        return image

    image = np.full((360, 640, 3), (22, 28, 34), dtype=np.uint8)
    cv2.rectangle(image, (205, 88), (435, 250), (54, 73, 86), 2, cv2.LINE_AA)
    cv2.line(image, (205, 88), (185, 68), (54, 73, 86), 2, cv2.LINE_AA)
    cv2.line(image, (435, 88), (455, 68), (54, 73, 86), 2, cv2.LINE_AA)
    cv2.putText(image, "WAITING FOR VIDEO", (197, 177), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (225, 235, 240), 2)
    cv2.putText(image, "Automatic reconnect is active", (205, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (115, 174, 193), 1)
    return image


def _presentation_titlebar() -> dict[str, object]:
    def svg_data(foreground: str, accent: str) -> str:
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="590" height="40" viewBox="0 0 590 40">
<rect x="0" y="4" width="5" height="32" rx="2.5" fill="{accent}"/>
<text x="18" y="22" font-family="Arial,Helvetica,sans-serif" font-size="18" font-weight="700" letter-spacing="1.2" fill="{foreground}">REAL-TIME 4D SAFETY PERCEPTION</text>
<text x="19" y="36" font-family="Arial,Helvetica,sans-serif" font-size="9" font-weight="600" letter-spacing="2.2" fill="{accent}">HUMAN · MOTION · DEPTH</text>
</svg>'''
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return {
        "buttons": None,
        "image": {
            "image_url_light": svg_data("#17242c", "#148fab"),
            "image_url_dark": svg_data("#ecf5f7", "#33c1d9"),
            "image_alt": "Real-Time 4D Safety Perception",
            "href": None,
        },
    }


def _resizable_sidebar_bootstrap() -> str:
    script = r"""
(() => {
  // Viser adapts the WebGL device-pixel ratio whenever its measured FPS
  // changes. On iPadOS Safari each drawing-buffer resize can expose WebKit's
  // white backing surface for a frame. Pinning DPR keeps the canvas allocation
  // stable while leaving the HTML sidebar at native Retina resolution.
  const appleTouchDevice =
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const stableDpr = appleTouchDevice
    ? 1.0
    : Math.min(1.25, Math.max(1.0, window.devicePixelRatio || 1.0));
  // Viser reads fixedDpr only while its React store is created. Mutating the
  // store after Canvas mount can already be too late on iPad Safari, because
  // AdaptiveDpr may have resized the drawing buffer and exposed a large white
  // backing rectangle. Re-enter once with the supported URL option so the
  // very first WebGL allocation is fixed.
  if (appleTouchDevice) {
    const url = new URL(window.location.href);
    if (!url.searchParams.has('fixedDpr')) {
      url.searchParams.set('fixedDpr', String(stableDpr));
      window.location.replace(url.toString());
      return;
    }
  }
  let dprAttempts = 0;
  const pinWebglDpr = () => {
    const store = window.__viserTestpoints?.devSettings;
    if (!store?.set) return false;
    store.set({fixedDpr: stableDpr});
    document.documentElement.dataset.realtimeSafetyFixedDpr = String(stableDpr);
    return true;
  };
  if (!pinWebglDpr()) {
    const timer = window.setInterval(() => {
      dprAttempts += 1;
      if (pinWebglDpr() || dprAttempts >= 100) window.clearInterval(timer);
    }, 100);
  }

  if (window.matchMedia('(max-width: 575px)').matches) return;
  if (document.getElementById('realtime-safety-sidebar-resizer')) return;

  let panel = document.currentScript?.parentElement || this;
  panel = this;
  while (panel) {
    const style = getComputedStyle(panel);
    if (style.position === 'absolute' && style.right === '0px' && Number(style.zIndex) >= 20) break;
    panel = panel.parentElement;
  }
  if (!panel) return;

  const shadow = panel.previousElementSibling;
  const contents = panel.firstElementChild;
  const handle = document.createElement('div');
  handle.id = 'realtime-safety-sidebar-resizer';
  handle.title = 'Drag to resize the status panel · Double-click to reset';
  handle.setAttribute('role', 'separator');
  handle.setAttribute('aria-orientation', 'vertical');
  handle.setAttribute('aria-label', 'Resize status panel');
  panel.appendChild(handle);

  const storageKey = 'realtime-safety-sidebar-width-v1';
  const defaultWidth = 384;
  let currentWidth = defaultWidth;
  let dragging = false;
  const clamp = (value) => Math.round(Math.max(280, Math.min(value, window.innerWidth * 0.55, 640)));
  const setWidth = (value, persist = false) => {
    const width = clamp(value);
    currentWidth = width;
    for (const element of [panel, shadow, contents]) {
      if (element) element.style.width = `${width}px`;
    }
    handle.setAttribute('aria-valuenow', String(width));
    if (persist) localStorage.setItem(storageKey, String(width));
  };

  const savedWidth = Number(localStorage.getItem(storageKey));
  setWidth(Number.isFinite(savedWidth) && savedWidth > 0 ? savedWidth : defaultWidth);

  const observer = new MutationObserver(() => {
    if (!handle.isConnected) panel.appendChild(handle);
    const collapsed = panel.style.width === '0px';
    handle.style.display = collapsed ? 'none' : 'block';
    if (!collapsed && !dragging && panel.style.width !== `${currentWidth}px`) {
      requestAnimationFrame(() => setWidth(currentWidth));
    }
  });
  observer.observe(panel, {attributes: true, attributeFilter: ['style'], childList: true});

  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    dragging = true;
    handle.setPointerCapture(event.pointerId);
    const startX = event.clientX;
    const startWidth = panel.getBoundingClientRect().width;
    const panelTransition = panel.style.transition;
    const shadowTransition = shadow?.style.transition || '';
    panel.style.transition = 'none';
    if (shadow) shadow.style.transition = 'none';

    const move = (moveEvent) => setWidth(startWidth + startX - moveEvent.clientX);
    const stop = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', stop);
      handle.removeEventListener('pointercancel', stop);
      panel.style.transition = panelTransition;
      if (shadow) shadow.style.transition = shadowTransition;
      dragging = false;
      setWidth(panel.getBoundingClientRect().width, true);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', stop);
    handle.addEventListener('pointercancel', stop);
  });

  handle.addEventListener('dblclick', () => setWidth(defaultWidth, true));
  window.addEventListener('resize', () => setWidth(panel.getBoundingClientRect().width));
})();
""".strip()
    escaped_script = html.escape(script, quote=True)
    return f"""
<style>
html,
body,
#root {{
  background-color: #111 !important;
  color-scheme: dark;
}}
canvas[data-engine^="three.js"] {{
  background-color: #111 !important;
  opacity: 1 !important;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
  transform: translateZ(0);
  -webkit-transform: translateZ(0);
}}
#realtime-safety-sidebar-resizer {{
  position: absolute;
  inset: 0 auto 0 -7px;
  width: 14px;
  z-index: 1000;
  cursor: ew-resize;
  touch-action: none;
  background: linear-gradient(90deg, transparent 5px, rgba(51,193,217,.28) 5px, rgba(51,193,217,.28) 8px, transparent 8px);
  transition: background-color .15s ease, opacity .15s ease;
}}
#realtime-safety-sidebar-resizer:hover,
#realtime-safety-sidebar-resizer:active {{
  background: linear-gradient(90deg, transparent 4px, rgba(51,193,217,.95) 4px, rgba(51,193,217,.95) 9px, transparent 9px);
}}
</style>
<img alt="" aria-hidden="true" style="display:none" onload="{escaped_script}"
  src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" />
""".strip()
