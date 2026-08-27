from __future__ import annotations

import threading
import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from realtime_safety.config import GuiConfig, OpenArmConfig
from realtime_safety.gui.metric_bev import (
    MetricBevCalibration,
    fit_metric_bev,
    rasterize_metric_bev,
)
from realtime_safety.gui.openarm_scene import OpenArmScene
from realtime_safety.types import PointCloudFrame, RobotArmState, Track3DState


LOGGER = logging.getLogger(__name__)


class ReconstructionScene3D:
    """Clean, bounded St4RTrack-style 4D point-cloud viewer.

    Each reconstruction is stored below ``/frames/t...`` in the common anchor
    coordinate system. Only the selected timestep is visible by default, just
    like the upstream St4RTrack visualizer.
    """

    def __init__(
        self,
        server: Any,
        config: GuiConfig,
        openarm_config: OpenArmConfig | None = None,
    ) -> None:
        self.server = server
        self.config = config
        self._lock = threading.RLock()
        self._frames: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._center: np.ndarray | None = None
        self._camera_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._camera_fitted_clients: set[int] = set()
        self._top_down_view = bool(config.metric_bev_enabled)
        self._closed = False
        self._edge_obstacle_handle: Any | None = None
        self._edge_obstacle_points = np.empty((0, 3), dtype=np.float32)
        self._bev_enabled = bool(config.metric_bev_enabled)
        self._bev_obstacle_height_m = float(
            config.metric_bev_obstacle_height_m
        )
        self._bev_calibration: MetricBevCalibration | None = None
        self._bev_recalibrate_requested = True
        self._bev_cloud_handle: Any | None = None
        self._bev_edge_handle: Any | None = None
        self._bev_obstacle_handles: dict[str, Any] = {}
        self._apriltag_handles: dict[str, Any] = {}
        self._apriltag_center_work: np.ndarray | None = None
        self._apriltag_corners_work: np.ndarray | None = None
        self._last_apriltag_scale: float | None = None
        self._sim_debug_enabled = False
        self._sim_debug_handles: dict[str, Any] = {}
        self._sim_debug_local: dict[str, Any] = {}
        self._sim_debug_status: Any | None = None
        self._sim_last_render_at: dict[str, float] = {"world": 0.0, "raw": 0.0}
        # Match the 15 Hz Gazebo depth sensor. This layer is a strict current
        # frame replacement; it never contains reconstruction history.
        self._sim_render_interval_sec = 1.0 / 15.0
        self._sim_debug_frame_index: dict[str, int] = {"world": 0, "raw": 0}
        self._edge_last_input_at = 0.0
        self._edge_last_nonempty_at = 0.0
        # Neural hand masks can miss an isolated RGB-D frame.  Keep the last
        # *visual* cloud briefly so the browser does not blink; sustained
        # empty measurements still clear it and safety timeouts remain owned
        # by the controller-facing ROS path.
        # Simulator mode receives a current-depth resampled cloud at ~10 Hz.
        # Bridge only a few display frames; a longer hold visibly leaves the
        # obstacle behind after the hand exits the RGB image.
        self._edge_visual_hold_sec = 0.35
        self._edge_input_rate_hz = 0.0
        self._edge_last_render_at = 0.0
        self._edge_render_interval_sec = 0.10
        self._requires_apriltag_anchor = bool(
            openarm_config is not None
            and openarm_config.enabled
            and openarm_config.base_anchor == "apriltag"
        )
        self._openarm = (
            OpenArmScene(server, openarm_config)
            if openarm_config is not None and openarm_config.enabled
            else None
        )

        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (3.0, -5.0, 2.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.fov = np.deg2rad(55.0)

        self._frames_root = self.server.scene.add_frame(
            "/frames", show_axes=False, visible=not self._bev_enabled
        )
        self._edge_root = self.server.scene.add_frame(
            "/edgetam", show_axes=False, visible=not self._bev_enabled
        )
        self._bev_root = self.server.scene.add_frame(
            "/metric_bev", show_axes=False, visible=self._bev_enabled
        )
        with self.server.gui.add_folder(
            "Scene Layers" if config.presentation_mode else "4D Reconstruction",
            order=30,
            expand_by_default=not config.presentation_mode,
        ) as folder:
            self._folder = folder
            self._show_reconstruction = self.server.gui.add_checkbox("Reconstruction points", initial_value=True)
            self._show_tracking = self.server.gui.add_checkbox("Tracking points", initial_value=False)
            self._follow_latest = self.server.gui.add_checkbox("Follow latest frame", initial_value=True)
            self._show_all = self.server.gui.add_checkbox("Show frame history", initial_value=False)
            self._history_stride = self.server.gui.add_slider(
                "History stride",
                min=1,
                max=max(1, config.history_frames),
                step=1,
                initial_value=min(config.history_stride, config.history_frames),
                disabled=True,
            )
            self._timestep = self.server.gui.add_slider(
                "Timestep", min=0, max=0, step=1, initial_value=0, disabled=True
            )
            self._frame_status = self.server.gui.add_markdown("Waiting for a reconstructed frame…")

        with self.server.gui.add_folder(
            "3D Safety RGB-D Backprojection",
            order=29,
            expand_by_default=True,
        ) as sim_folder:
            self._sim_folder = sim_folder
            self._show_sim_world = self.server.gui.add_checkbox(
                "World-frame cloud", initial_value=True
            )
            self._show_sim_raw = self.server.gui.add_checkbox(
                "Raw camera-frame cloud", initial_value=False
            )
            self._show_sim_geometry = self.server.gui.add_checkbox(
                "Table / AprilTag / axes", initial_value=True
            )
            self._sim_debug_status = self.server.gui.add_markdown(
                "Waiting for synchronized RGB + aligned depth…"
            )

        with self.server.gui.add_folder(
            "Obstacle Tracking" if config.presentation_mode else "Obstacles · 3D Tracking",
            order=31,
            expand_by_default=not config.presentation_mode,
        ) as people_folder:
            self._people_folder = people_folder
            self._show_person_boxes = self.server.gui.add_checkbox("3D obstacle volumes", initial_value=True)
            self._show_person_centers = self.server.gui.add_checkbox("Obstacle centers", initial_value=True)
            self._show_person_directions = self.server.gui.add_checkbox("Direction arrows", initial_value=True)
            self._show_person_labels = self.server.gui.add_checkbox("Track labels", initial_value=True)
            self._show_edge_foreground = self.server.gui.add_checkbox(
                "Extracted obstacle points", initial_value=True
            )
            self._edge_foreground_status = self.server.gui.add_markdown(
                "Extracted obstacles: **0 points**"
            )
            self._people_status = self.server.gui.add_markdown(
                "Waiting for aligned 3D obstacle tracks…"
            )

        with self.server.gui.add_folder(
            "Robot ↔ Obstacle Geometry",
            order=32,
            expand_by_default=True,
        ) as relationship_folder:
            self._relationship_folder = relationship_folder
            self._show_robot_center = self.server.gui.add_checkbox(
                "Legacy RGB arm center",
                initial_value=self._openarm is None,
            )
            self._show_relationships = self.server.gui.add_checkbox(
                "Center-distance links",
                initial_value=True,
            )
            self._relationship_status = self.server.gui.add_markdown(
                "Waiting for robot localization…"
            )

        with self.server.gui.add_folder(
            "Metric BEV / 數學鳥瞰校正",
            order=66,
            expand_by_default=True,
        ) as bev_folder:
            self._bev_folder = bev_folder
            self._bev_status = self.server.gui.add_markdown(
                "正在估計實體工作平面… / Estimating physical work plane…"
            )
            self._bev_preview = self.server.gui.add_image(
                np.full((270, 360, 3), 12, dtype=np.uint8),
                label="Orthographic metric occupancy / 正射公尺佔用圖",
            )

        self._controls = [
            self._show_reconstruction,
            self._show_tracking,
            self._follow_latest,
            self._show_all,
            self._history_stride,
            self._timestep,
            self._frame_status,
            self._show_sim_world,
            self._show_sim_raw,
            self._show_sim_geometry,
            self._sim_debug_status,
            self._show_person_boxes,
            self._show_person_centers,
            self._show_person_directions,
            self._show_person_labels,
            self._people_status,
            self._show_edge_foreground,
            self._edge_foreground_status,
            self._show_robot_center,
            self._show_relationships,
            self._relationship_status,
            self._bev_status,
            self._bev_preview,
        ]
        self._show_reconstruction.on_update(lambda _: self._refresh_visibility())
        self._show_tracking.on_update(lambda _: self._refresh_visibility())
        self._follow_latest.on_update(lambda _: self._on_follow_latest())
        self._show_all.on_update(lambda _: self._on_show_all())
        self._history_stride.on_update(lambda _: self._refresh_visibility())
        self._timestep.on_update(lambda _: self._on_timestep())
        self._show_sim_world.on_update(lambda _: self._refresh_sim_debug_visibility())
        self._show_sim_raw.on_update(lambda _: self._refresh_sim_debug_visibility())
        self._show_sim_geometry.on_update(lambda _: self._refresh_sim_debug_visibility())
        self._show_person_boxes.on_update(lambda _: self._refresh_visibility())
        self._show_person_centers.on_update(lambda _: self._refresh_visibility())
        self._show_person_directions.on_update(lambda _: self._refresh_visibility())
        self._show_person_labels.on_update(lambda _: self._refresh_visibility())
        self._show_edge_foreground.on_update(
            lambda _: self._refresh_visibility()
        )
        self._show_robot_center.on_update(lambda _: self._refresh_visibility())
        self._show_relationships.on_update(lambda _: self._refresh_visibility())

        @self.server.on_client_connect
        def _(client: Any) -> None:
            self._fit_client_camera(client)

    @property
    def node_count(self) -> int:
        return (
            1
            + int(self._edge_obstacle_handle is not None)
            + int(self._bev_cloud_handle is not None)
            + int(self._bev_edge_handle is not None)
            + len(self._bev_obstacle_handles)
            + len(self._apriltag_handles)
            + sum(
                3 + len(handles["people"])
                for handles in self._frames.values()
            )
        )

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def update_pointcloud(self, frame: PointCloudFrame, defer_visibility: bool = False) -> None:
        # Simulator mode has a dedicated metric world cloud.  Rebuilding the
        # hidden reconstruction tree from the same 307k-point RGB-D message
        # only duplicates WebSocket/GPU traffic and can starve GUI commands.
        if self._sim_debug_enabled:
            return
        points = np.asarray(frame.points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(frame.colors, dtype=np.uint8).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        points, colors = points[valid], colors[valid]
        if len(points) == 0:
            return

        tracking = frame.tracking_points
        tracking_points = (
            np.asarray(tracking, dtype=np.float32).reshape(-1, 3)
            if tracking is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        tracking_points = tracking_points[np.isfinite(tracking_points).all(axis=1)]

        with self._lock:
            if self._closed:
                return
            if frame.apriltag_locked and frame.apriltag_scale_correction is not None:
                scale = float(frame.apriltag_scale_correction)
                # Rebase exactly once when arbitrary model units first become
                # metric. Afterwards the task-plane basis is a latched world
                # frame; slow tag/depth noise must not make it jump while an
                # obstacle enters or leaves. Operators can explicitly request
                # a new fit with the GUI recalibration button.
                needs_metric_rebase = self._last_apriltag_scale is None
                if needs_metric_rebase:
                    # A plane basis and raw-view center fitted before the tag
                    # lock carry the model's old arbitrary units. Refit both
                    # on the now-scaled frame so no mixed-unit geometry remains.
                    self._center = None
                    self._bev_calibration = None
                    self._bev_recalibrate_requested = True
                    self._camera_pose = None
                    self._camera_fitted_clients.clear()
                    self._last_apriltag_scale = scale
            if self._center is None:
                self._center = _robust_center(points)
            centered_points = points - self._center
            centered_tracking = tracking_points - self._center

            existing = self._frames.get(frame.frame_index)
            if existing is None and self.config.history_frames == 1 and self._frames:
                # Live mode reuses one persistent set of Viser/WebGL handles.
                # Replacing scene nodes every update makes the browser briefly
                # render an empty frame while it allocates new GPU buffers.
                _, existing = self._frames.popitem(last=False)
                self._frames[frame.frame_index] = existing
            if existing is not None:
                existing["reconstruction"].points = centered_points
                existing["reconstruction"].colors = colors
                existing["tracking"].points = centered_tracking
                self._frames.move_to_end(frame.frame_index)
            else:
                scene_path = f"/frames/t{frame.frame_index:06d}"
                root = self.server.scene.add_frame(scene_path, show_axes=False, visible=False)
                reconstruction = self.server.scene.add_point_cloud(
                    f"{scene_path}/reconstruction",
                    points=centered_points,
                    colors=colors,
                    point_size=self.config.point_size,
                    point_shape="rounded",
                    precision="float16",
                )
                tracking_handle = self.server.scene.add_point_cloud(
                    f"{scene_path}/tracking",
                    points=centered_tracking,
                    colors=(45, 115, 225),
                    point_size=self.config.point_size * 1.25,
                    point_shape="rounded",
                    precision="float16",
                    visible=False,
                )
                self._frames[frame.frame_index] = {
                    "root": root,
                    "reconstruction": reconstruction,
                    "tracking": tracking_handle,
                    "people": {},
                    "path": scene_path,
                }

            while len(self._frames) > self.config.history_frames:
                _, handles = self._frames.popitem(last=False)
                self._remove_frame_handles(handles)

            last = len(self._frames) - 1
            self._timestep.max = max(last, 0)
            self._timestep.disabled = len(self._frames) <= 1 or self._follow_latest.value or self._show_all.value
            if self._follow_latest.value:
                self._timestep.value = max(last, 0)
            self._frame_status.content = (
                f"**LIVE · FRAME {frame.frame_index:,}** · {len(centered_points):,} points"
                if self.config.presentation_mode
                else f"Frame **{frame.frame_index}** · {len(centered_points):,} points · "
                f"source: **{frame.source}** · history: **{len(self._frames)}/{self.config.history_frames}**"
            )
            if not defer_visibility:
                self._refresh_visibility_locked()
            if self._camera_pose is None:
                self._configure_camera(centered_points)
            self._update_metric_bev(frame, points, colors)
            if self._openarm is not None and not self._sim_debug_enabled:
                self._openarm.set_spatial_context(
                    center=self._center,
                    bev_enabled=self._bev_enabled,
                    calibration=self._bev_calibration,
                    apriltag_center_work=self._apriltag_center_work,
                    apriltag_locked=self._apriltag_center_work is not None,
                )

    def configure_simulator_debug(
        self,
        scene_config: dict[str, Any],
        camera_config: dict[str, Any],
    ) -> None:
        """Create exact-metric simulator overlays in the ROS world frame."""

        explicit_pose = camera_config.get("world_pose")
        if explicit_pose:
            position = np.asarray(explicit_pose["position"], dtype=float)
            world_from_link = Rotation.from_euler(
                "xyz", explicit_pose["rpy_deg"], degrees=True
            ).as_matrix()
        else:
            workspace = np.asarray(
                scene_config["zones"]["workspace"]["center"], dtype=float
            )
            position = workspace + np.array(
                [
                    -float(camera_config["horizontal_offset_to_workspace_center"]),
                    float(camera_config["lateral_offset"]),
                    float(camera_config["height_above_table"]),
                ]
            )
            target = workspace + np.asarray(
                camera_config.get("aim_offset", [0.0, 0.0, 0.0]), dtype=float
            )
            forward = target - position
            forward /= np.linalg.norm(forward)
            left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
            left /= np.linalg.norm(left)
            up = np.cross(forward, left)
            world_from_link = np.column_stack((forward, left, up))
        optical_to_link = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=float,
        )
        optical_xyzw = Rotation.from_matrix(world_from_link @ optical_to_link).as_quat()
        table = scene_config["table"]
        tag = scene_config["apriltag"]
        table_xyzw = Rotation.from_euler(
            "z", float(table.get("yaw_deg", 0.0)), degrees=True
        ).as_quat()
        empty = np.empty((0, 3), dtype=np.float32)
        with self._lock:
            self._sim_debug_enabled = True
            table_center = np.asarray(scene_config["table"]["center"], dtype=float)
            look_at = table_center + np.array([-0.20, 0.0, 0.30])
            view_position = look_at + np.array([1.45, -1.55, 1.20])
            self._camera_pose = (view_position, look_at)
            self.server.initial_camera.position = view_position
            self.server.initial_camera.look_at = look_at
            self.server.initial_camera.up = (0.0, 0.0, 1.0)
            self._camera_fitted_clients.clear()
            self._frames_root.visible = False
            self._show_reconstruction.value = False
            root = self.server.scene.add_frame(
                "/sim_debug/world", show_axes=True, axes_length=0.15, axes_radius=0.004
            )
            camera = self.server.scene.add_frame(
                "/sim_debug/camera_optical",
                show_axes=True,
                axes_length=0.10,
                axes_radius=0.003,
                position=tuple(position),
                wxyz=(
                    float(optical_xyzw[3]),
                    float(optical_xyzw[0]),
                    float(optical_xyzw[1]),
                    float(optical_xyzw[2]),
                ),
            )
            raw = self.server.scene.add_point_cloud(
                "/sim_debug/camera_optical/raw_cloud",
                points=empty,
                colors=np.empty((0, 3), dtype=np.uint8),
                point_size=self.config.point_size,
                point_shape="rounded",
                precision="float16",
                visible=False,
            )
            world = self.server.scene.add_point_cloud(
                "/sim_debug/world/cloud",
                points=empty,
                colors=np.empty((0, 3), dtype=np.uint8),
                point_size=self.config.point_size,
                point_shape="rounded",
                precision="float16",
            )
            table_box = self.server.scene.add_box(
                "/sim_debug/world/table",
                dimensions=np.asarray(table["size"], dtype=float),
                position=tuple(table["center"]),
                wxyz=(
                    float(table_xyzw[3]),
                    float(table_xyzw[0]),
                    float(table_xyzw[1]),
                    float(table_xyzw[2]),
                ),
                color=tuple(int(255 * value) for value in table["color_rgb"]),
                opacity=0.22,
            )
            tag_box = self.server.scene.add_box(
                "/sim_debug/world/apriltag_8cm",
                dimensions=(float(tag["size"]), float(tag["size"]), 0.002),
                position=tuple(tag["center"]),
                color=(255, 190, 35),
                opacity=0.70,
            )
            self._sim_debug_handles = {
                "root": root,
                "camera": camera,
                "raw": raw,
                "world": world,
                "table": table_box,
                "tag": tag_box,
            }
            self._sim_debug_local = {
                "optical_to_link": optical_to_link,
                "camera_position": position.copy(),
                "world_from_optical": world_from_link @ optical_to_link,
                "table_center": np.asarray(table["center"], dtype=float),
                "table_rotation": Rotation.from_quat(table_xyzw).as_matrix(),
                "tag_center": np.asarray(tag["center"], dtype=float),
            }
            if self._openarm is not None:
                self._openarm.set_spatial_context(
                    center=None,
                    bev_enabled=False,
                    calibration=None,
                    apriltag_center_work=None,
                    apriltag_locked=False,
                )
            self._refresh_sim_debug_visibility_locked()

    def update_simulator_entity_poses(self, entities: dict[str, Any]) -> None:
        """Follow Gazebo model edits without feeding ground truth to perception."""

        def pose(name: str) -> tuple[np.ndarray, Rotation] | None:
            document = entities.get(name)
            if not isinstance(document, dict):
                return None
            position = np.asarray(document.get("position", ()), dtype=float)
            quaternion = np.asarray(document.get("orientation_xyzw", ()), dtype=float)
            if position.shape != (3,) or quaternion.shape != (4,):
                return None
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                return None
            if np.linalg.norm(quaternion) <= 1e-9:
                return None
            return position, Rotation.from_quat(quaternion)

        with self._lock:
            if self._closed or not self._sim_debug_enabled:
                return
            camera_pose = pose("rgbd_sensor")
            if camera_pose is not None and "camera" in self._sim_debug_handles:
                position, rotation = camera_pose
                optical_rotation = rotation.as_matrix() @ self._sim_debug_local["optical_to_link"]
                self._sim_debug_local["camera_position"] = position.copy()
                self._sim_debug_local["world_from_optical"] = optical_rotation.copy()
                optical_xyzw = Rotation.from_matrix(optical_rotation).as_quat()
                handle = self._sim_debug_handles["camera"]
                handle.position = tuple(position)
                handle.wxyz = tuple(float(value) for value in (
                    optical_xyzw[3], optical_xyzw[0], optical_xyzw[1], optical_xyzw[2]
                ))
            table_pose = pose("work_table")
            if table_pose is not None and "table" in self._sim_debug_handles:
                origin, rotation = table_pose
                handle = self._sim_debug_handles["table"]
                handle.position = tuple(
                    origin + rotation.apply(self._sim_debug_local["table_center"])
                )
                table_xyzw = Rotation.from_matrix(
                    rotation.as_matrix() @ self._sim_debug_local["table_rotation"]
                ).as_quat()
                handle.wxyz = tuple(float(value) for value in (
                    table_xyzw[3], table_xyzw[0], table_xyzw[1], table_xyzw[2]
                ))
            tag_pose = pose("apriltag_36h11_0")
            if tag_pose is not None and "tag" in self._sim_debug_handles:
                origin, rotation = tag_pose
                handle = self._sim_debug_handles["tag"]
                handle.position = tuple(
                    origin + rotation.apply(self._sim_debug_local["tag_center"])
                )
                tag_xyzw = rotation.as_quat()
                handle.wxyz = tuple(float(value) for value in (
                    tag_xyzw[3], tag_xyzw[0], tag_xyzw[1], tag_xyzw[2]
                ))
            openarm_pose = pose("openarm")
            if openarm_pose is not None and self._openarm is not None:
                self._openarm.update_simulator_model_pose(*openarm_pose)

    def update_simulator_debug_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        world: bool,
        rate_hz: float,
    ) -> None:
        with self._lock:
            if self._closed or not self._sim_debug_enabled:
                return
            key = "world" if world else "raw"
            if key == "raw" and not bool(self._show_sim_raw.value):
                return
            now = time.monotonic()
            if now - self._sim_last_render_at[key] < self._sim_render_interval_sec:
                return
            self._sim_last_render_at[key] = now
            handle = self._sim_debug_handles.get(key)
            if handle is None:
                return
            shown_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            shown_colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
            # Viser/WebGL clients can retain an old GPU allocation when the
            # new cloud loses only a small subset of vertices (the exact case
            # when a hand occludes a cube). Recreate this one bounded 8k-point
            # simulator layer so the browser contains exactly this depth frame.
            visible = bool(
                self._show_sim_world.value if world else self._show_sim_raw.value
            )
            handle.remove()
            handle = self.server.scene.add_point_cloud(
                "/sim_debug/world/cloud"
                if world
                else "/sim_debug/camera_optical/raw_cloud",
                points=shown_points,
                colors=shown_colors,
                point_size=self.config.point_size,
                point_shape="rounded",
                precision="float16",
                visible=visible,
            )
            self._sim_debug_handles[key] = handle
            self._sim_debug_frame_index[key] += 1
            if self._sim_debug_status is not None:
                self._sim_debug_status.content = (
                    f"**3D SAFETY RGB-D BACKPROJECTION · CURRENT FRAME "
                    f"{self._sim_debug_frame_index[key]:,}** · "
                    f"{key}: **{len(shown_points):,} points** · "
                    f"**{rate_hz:.1f} Hz** · no history · metres"
                )

    def _refresh_sim_debug_visibility(self) -> None:
        with self._lock:
            self._refresh_sim_debug_visibility_locked()

    def _refresh_sim_debug_visibility_locked(self) -> None:
        if not self._sim_debug_handles:
            return
        self._sim_debug_handles["world"].visible = bool(self._show_sim_world.value)
        self._sim_debug_handles["raw"].visible = bool(self._show_sim_raw.value)
        geometry = bool(self._show_sim_geometry.value)
        for key in ("root", "camera", "table", "tag"):
            self._sim_debug_handles[key].visible = geometry

    def update_edge_obstacle_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        frame_id: str = "",
    ) -> None:
        """Overlay the controller-facing extracted obstacle cloud in 3D."""

        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        shown_colors = (
            None
            if colors is None
            else np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[valid]
        )
        with self._lock:
            if self._closed:
                return
            now = time.monotonic()
            if self._edge_last_input_at > 0.0:
                interval = now - self._edge_last_input_at
                if interval > 0.0:
                    measured_rate = 1.0 / interval
                    self._edge_input_rate_hz = (
                        measured_rate
                        if self._edge_input_rate_hz <= 0.0
                        else 0.85 * self._edge_input_rate_hz + 0.15 * measured_rate
                    )
            self._edge_last_input_at = now
            if self._sim_debug_enabled:
                # EdgeTAM outputs points in the message's declared tracking
                # frame.  Gazebo uses ROS optical coordinates, so transform
                # those measured RGB-D points into the same world frame as the
                # robot and live environment cloud before drawing them.
                if len(values) and frame_id != "world":
                    rotation = np.asarray(
                        self._sim_debug_local["world_from_optical"], dtype=np.float32
                    )
                    translation = np.asarray(
                        self._sim_debug_local["camera_position"], dtype=np.float32
                    )
                    values = values @ rotation.T + translation
                centered = values
            else:
                # Legacy physical-camera reconstruction coordinates are
                # x-right/y-forward/z-down while this viewer is z-up.
                if len(values):
                    values = values.copy()
                    values[:, 2] *= -1.0
                centered = values if self._center is None else values - self._center
            self._edge_foreground_status.content = (
                f"Extracted obstacles: **{len(values):,} points** · "
                f"**LIVE {self._edge_input_rate_hz:.1f} Hz** · "
                f"`{frame_id or 'unknown'}`"
            )
            if self._center is None and not self._sim_debug_enabled:
                self._edge_obstacle_points = values
                return
            if len(centered) == 0:
                held_for = now - self._edge_last_nonempty_at
                if (
                    self._edge_obstacle_handle is not None
                    and self._edge_last_nonempty_at > 0.0
                    and held_for < self._edge_visual_hold_sec
                ):
                    self._edge_foreground_status.content = (
                        "Extracted obstacles: **temporary detector gap · "
                        f"HOLD {held_for:.2f}/{self._edge_visual_hold_sec:.2f}s**"
                    )
                    return
                self._edge_obstacle_points = values
                if self._edge_obstacle_handle is not None:
                    # Removing and recreating the Viser node is intentional.
                    # Some WebGL clients keep the previous GPU buffer when a
                    # point-cloud update has zero vertices.
                    self._edge_obstacle_handle.remove()
                    self._edge_obstacle_handle = None
                self._update_bev_edge_cloud_locked()
                return
            self._edge_last_nonempty_at = now
            self._edge_obstacle_points = values
            if now - self._edge_last_render_at < self._edge_render_interval_sec:
                return
            self._edge_last_render_at = now
            if shown_colors is None or len(shown_colors) != len(centered):
                shown_colors = np.tile(
                    np.array([[255, 45, 85]], dtype=np.uint8),
                    (len(centered), 1),
                )
            if self._edge_obstacle_handle is None:
                self._edge_obstacle_handle = self.server.scene.add_point_cloud(
                    "/edgetam/foreground_obstacles",
                    points=centered,
                    colors=shown_colors,
                    # Dense current-frame RGB-D points already describe the
                    # hand surface. Oversized splats visually merged them into
                    # a much larger obstacle than the measured geometry.
                    point_size=self.config.point_size * 1.25,
                    point_shape="rounded",
                    precision="float16",
                    visible=(
                        self._show_edge_foreground.value
                        and not self._bev_enabled
                    ),
                )
            else:
                self._edge_obstacle_handle.points = centered
                self._edge_obstacle_handle.colors = shown_colors
                self._edge_obstacle_handle.visible = (
                    self._show_edge_foreground.value and not self._bev_enabled
                )
            self._update_bev_edge_cloud_locked()

    def _update_metric_bev(
        self,
        frame: PointCloudFrame,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Update the separately derived plane-coordinate cloud and BEV map."""

        if not self._bev_enabled:
            return
        if (
            self._requires_apriltag_anchor
            and self._last_apriltag_scale is None
            and frame.apriltag_scale_correction is None
        ):
            self._bev_status.content = (
                "等待 8×8 cm AprilTag 後再鎖定工作平面… / "
                "Waiting for metric tag before fitting work plane…"
            )
            return
        if self._bev_calibration is None or self._bev_recalibrate_requested:
            try:
                self._bev_calibration = fit_metric_bev(frame.pointmap)
                self._bev_recalibrate_requested = False
                self._camera_fitted_clients.clear()
                LOGGER.info(
                    "Metric BEV work plane locked: inliers=%d (%.1f%%) "
                    "rms=%.1fmm normal=(%.3f, %.3f, %.3f)",
                    self._bev_calibration.inlier_count,
                    self._bev_calibration.inlier_ratio * 100.0,
                    self._bev_calibration.rms_error_m * 1000.0,
                    *self._bev_calibration.normal,
                )
            except ValueError as exc:
                self._bev_status.content = (
                    "⚠️ 工作平面尚未鎖定 / Work plane not locked  \n"
                    f"`{exc}`"
                )
                return

        calibration = self._bev_calibration
        assert calibration is not None
        bev_points = calibration.project(points)
        finite = np.isfinite(bev_points).all(axis=1)
        bev_points = bev_points[finite]
        bev_colors = colors[finite]
        u0, u1, v0, v1 = calibration.bounds_uv
        inside = (
            (bev_points[:, 0] >= u0)
            & (bev_points[:, 0] <= u1)
            & (bev_points[:, 1] >= v0)
            & (bev_points[:, 1] <= v1)
            & (bev_points[:, 2] >= -0.05)
            & (bev_points[:, 2] <= 0.75)
        )
        bev_points = bev_points[inside]
        bev_colors = bev_colors[inside]
        if self._bev_cloud_handle is None:
            self._bev_cloud_handle = self.server.scene.add_point_cloud(
                "/metric_bev/orthorectified_pointcloud",
                points=bev_points,
                colors=bev_colors,
                point_size=self.config.point_size,
                point_shape="rounded",
                precision="float16",
                visible=self._bev_enabled,
            )
        else:
            self._bev_cloud_handle.points = bev_points
            self._bev_cloud_handle.colors = bev_colors
            self._bev_cloud_handle.visible = self._bev_enabled

        self._update_bev_edge_cloud_locked()
        self._update_apriltag_locked(frame, calibration)
        preview = rasterize_metric_bev(
            calibration,
            points,
            colors,
            obstacle_height_m=self._bev_obstacle_height_m,
            edge_points=self._edge_obstacle_points,
        )
        self._bev_preview.image = preview
        tilt = float(
            np.degrees(np.arccos(np.clip(calibration.normal[2], -1.0, 1.0)))
        )
        if frame.apriltag_locked and frame.apriltag_scale_correction is not None:
            tag_line = (
                f"  \n🟧 **APRILTAG #{frame.apriltag_id} SCALE LOCK** · "
                f"physical side **{(frame.apriltag_size_m or 0.0) * 100.0:.1f} cm** · "
                f"correction **{frame.apriltag_scale_correction:.4f}×** · "
                f"age **{frame.apriltag_age_frames or 0} frames**"
            )
        elif self._apriltag_center_work is not None:
            tag_line = (
                "  \n🟨 **APRILTAG ANCHOR HOLD** · temporary detection loss; "
                "task-plane pose remains at the last verified lock"
            )
        else:
            tag_line = "  \n🟠 **等待 8×8 cm AprilTag 尺度鎖定 / waiting for tag**"
        self._bev_status.content = (
            "✅ **實體工作平面已鎖定 / Metric plane locked**  \n"
            f"RANSAC: **{calibration.inlier_count:,}** points "
            f"({calibration.inlier_ratio:.0%}) · RMS "
            f"**{calibration.rms_error_m * 1000.0:.1f} mm** · "
            f"mount tilt **{tilt:.1f}°**  \n"
            "紅色＝高於平面的候選障礙；粉紅＝目前幀障礙輸出。  \n"
            "工作平面與 AprilTag 原點保持鎖定，僅手動重新校正才會改變。"
            + tag_line
        )
        if self._bev_enabled and self._camera_fitted_clients == set():
            self._fit_bev_camera_locked()

    def _update_apriltag_locked(
        self,
        frame: PointCloudFrame,
        calibration: MetricBevCalibration,
    ) -> None:
        corners_xyz = frame.apriltag_corners_xyz
        if not frame.apriltag_locked or corners_xyz is None:
            # A temporarily occluded tag is not evidence that the physical
            # table moved. Keep the last spatial anchor and plane instead of
            # clearing them and automatically fitting a different plane when
            # the tag becomes visible again.
            for key, handle in self._apriltag_handles.items():
                handle.visible = self._bev_enabled
                if key == "label" and self._apriltag_center_work is not None:
                    handle.text = "APRILTAG ANCHOR · HOLD (last verified pose)"
            return
        corners = calibration.project(corners_xyz)
        if len(corners) != 4 or not np.isfinite(corners).all():
            return
        if self._apriltag_corners_work is None:
            physical_center = np.mean(corners, axis=0)
            physical_center[2] = 0.0
            self._apriltag_center_work = physical_center.astype(np.float32)
            self._apriltag_corners_work = corners.astype(np.float32).copy()
        # Keep the first verified tag pose as the world origin. Per-frame
        # monocular depth jitter otherwise moves OpenArm and its work grid even
        # though the fixed camera, tag and robot have not physically moved.
        corners = self._apriltag_corners_work.astype(np.float32).copy()
        # Lift the calibration graphic slightly above the fitted tabletop so
        # it remains visible through dense point sprites.
        corners[:, 2] = np.maximum(corners[:, 2], 0.012)
        center = np.mean(corners, axis=0)
        center[2] = 0.016
        closed = np.concatenate((corners, corners[:1]), axis=0)
        segments = np.stack((closed[:-1], closed[1:]), axis=1).astype(np.float32)
        colors = np.tile(
            np.asarray([[(255, 130, 20), (255, 130, 20)]], dtype=np.uint8),
            (4, 1, 1),
        )
        key = "outline"
        if key not in self._apriltag_handles:
            self._apriltag_handles[key] = self.server.scene.add_line_segments(
                "/metric_bev/apriltag/physical_8cm_outline",
                points=segments,
                colors=colors,
                line_width=5.0,
            )
        else:
            self._apriltag_handles[key].points = segments
            self._apriltag_handles[key].visible = self._bev_enabled
        key = "center"
        if key not in self._apriltag_handles:
            self._apriltag_handles[key] = self.server.scene.add_icosphere(
                "/metric_bev/apriltag/center",
                radius=0.012,
                color=(255, 130, 20),
                subdivisions=2,
                position=center,
            )
        else:
            self._apriltag_handles[key].position = center
            self._apriltag_handles[key].visible = self._bev_enabled
        key = "label"
        size_cm = (frame.apriltag_size_m or 0.08) * 100.0
        text = f"APRILTAG #{frame.apriltag_id} · {size_cm:.1f} cm · SCALE LOCK"
        if key not in self._apriltag_handles:
            self._apriltag_handles[key] = self.server.scene.add_label(
                "/metric_bev/apriltag/label",
                text,
                position=center + np.array((0.0, 0.0, 0.03), np.float32),
                anchor="bottom-center",
                font_size_mode="screen",
                font_screen_scale=0.8,
                depth_test=False,
            )
        else:
            self._apriltag_handles[key].text = text
            self._apriltag_handles[key].position = center + np.array(
                (0.0, 0.0, 0.03), np.float32
            )
            self._apriltag_handles[key].visible = self._bev_enabled

    def _update_bev_edge_cloud_locked(self) -> None:
        calibration = self._bev_calibration
        if calibration is None:
            return
        bev_points = calibration.project(self._edge_obstacle_points)
        if not len(bev_points):
            if self._bev_edge_handle is not None:
                self._bev_edge_handle.remove()
                self._bev_edge_handle = None
            return
        colors = np.tile(
            np.array([[255, 30, 105]], dtype=np.uint8),
            (len(bev_points), 1),
        )
        if self._bev_edge_handle is None:
            self._bev_edge_handle = self.server.scene.add_point_cloud(
                "/metric_bev/edgetam_obstacles",
                points=bev_points,
                colors=colors,
                point_size=self.config.point_size * 1.25,
                point_shape="rounded",
                precision="float16",
                visible=(
                    self._bev_enabled and self._show_edge_foreground.value
                ),
            )
        else:
            self._bev_edge_handle.points = bev_points
            self._bev_edge_handle.colors = colors
            self._bev_edge_handle.visible = (
                self._bev_enabled and self._show_edge_foreground.value
            )

    def _fit_bev_camera_locked(self) -> None:
        calibration = self._bev_calibration
        if calibration is None:
            return
        u0, u1, v0, v1 = self._bev_view_bounds_locked()
        look_at = np.array(
            ((u0 + u1) * 0.5, (v0 + v1) * 0.5, 0.0), dtype=np.float64
        )
        span = max(u1 - u0, v1 - v0, 0.4)
        position = look_at + np.array((0.0, 0.0, 1.15 * span))
        self.server.initial_camera.position = position
        self.server.initial_camera.look_at = look_at
        self.server.initial_camera.up = (0.0, 1.0, 0.0)
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = look_at
                client.camera.up_direction = (0.0, 1.0, 0.0)
                client.camera.fov = np.deg2rad(55.0)
            self._camera_fitted_clients.add(client.client_id)

    def update_people(
        self,
        frame_index: int,
        tracks: list[Track3DState],
        yolo_count: int = 0,
        robot_arm: RobotArmState | None = None,
    ) -> None:
        """Attach obstacle geometry and robot-to-obstacle relationships."""
        with self._lock:
            if self._closed or self._center is None or frame_index not in self._frames:
                return
            frame_handles = self._frames[frame_index]
            scene_path = frame_handles["path"]
            people_handles: dict[str, Any] = frame_handles["people"]
            active: set[str] = set()
            arrow_segments: list[np.ndarray] = []
            obstacles = list(tracks)
            for track in obstacles:
                track_id = track.track_id
                center = np.asarray(track.position_xyz, dtype=np.float32) - self._center
                dimensions = np.maximum(np.asarray(track.bbox3d.size, dtype=np.float32), 0.03)

                box_key = f"box:{track_id}"
                if box_key not in people_handles:
                    people_handles[box_key] = self.server.scene.add_box(
                        f"{scene_path}/people/{track_id}/bbox",
                        color=_obstacle_color(track.class_name),
                        dimensions=dimensions,
                        wireframe=True,
                        position=center,
                    )
                else:
                    people_handles[box_key].dimensions = dimensions
                    people_handles[box_key].position = center
                active.add(box_key)

                center_key = f"center:{track_id}"
                if center_key not in people_handles:
                    radius = float(np.clip(np.max(dimensions) * 0.025, 0.012, 0.045))
                    people_handles[center_key] = self.server.scene.add_icosphere(
                        f"{scene_path}/people/{track_id}/center",
                        radius=radius,
                        color=(255, 45, 95),
                        subdivisions=2,
                        position=center,
                    )
                else:
                    people_handles[center_key].position = center
                active.add(center_key)

                label_key = f"label:{track_id}"
                label_position = center + np.array((0.0, 0.0, dimensions[2] * 0.56), dtype=np.float32)
                motion = "MOVING" if track.motion_state == "dynamic" else "STATIONARY"
                hold = f" · HOLD {track.missing_count}" if track.missing_count else ""
                label_text = (
                    f"ID {track_id} · {track.class_name.upper()} · {motion}{hold}"
                )
                if label_key not in people_handles:
                    people_handles[label_key] = self.server.scene.add_label(
                        f"{scene_path}/people/{track_id}/label",
                        label_text,
                        position=label_position,
                        anchor="bottom-center",
                        font_size_mode="screen",
                        font_screen_scale=0.85,
                        depth_test=False,
                    )
                else:
                    people_handles[label_key].text = label_text
                    people_handles[label_key].position = label_position
                active.add(label_key)

                direction = _stable_horizontal_direction(track, dimensions)
                if direction is not None:
                    speed = float(np.linalg.norm(np.asarray(track.velocity_xyz, dtype=np.float32)[:2]))
                    length = float(np.clip(0.22 * np.max(dimensions) + 0.08 * speed, 0.12, 0.38))
                    arrow_segments.append(np.stack((center, center + direction * length), axis=0))

            arrow_key = "directions"
            if arrow_segments:
                segments = np.asarray(arrow_segments, dtype=np.float32)
                colors = np.tile(np.array([[35, 210, 85]], dtype=np.uint8), (len(segments), 1))
                if arrow_key not in people_handles:
                    people_handles[arrow_key] = self.server.scene.add_arrows(
                        f"{scene_path}/people/directions",
                        points=segments,
                        colors=colors,
                        shaft_radius=0.008,
                        head_radius=0.024,
                        head_length=0.05,
                    )
                else:
                    people_handles[arrow_key].points = segments
                    people_handles[arrow_key].colors = colors
                active.add(arrow_key)

            self._update_robot_relationship(
                scene_path,
                people_handles,
                active,
                obstacles,
                robot_arm,
            )
            self._update_bev_obstacles_locked(obstacles)
            if self._openarm is not None:
                display_obstacles: list[tuple[int, str, np.ndarray]] = []
                for track in obstacles:
                    position = np.asarray(track.position_xyz, dtype=np.float32)
                    if self._bev_enabled and self._bev_calibration is not None:
                        position = self._bev_calibration.project(position[None, :])[0]
                    else:
                        position = position - self._center
                    display_obstacles.append(
                        (track.track_id, track.class_name, position)
                    )
                self._openarm.update_obstacles(display_obstacles)

            for key in list(people_handles):
                if key not in active:
                    people_handles.pop(key).remove()
            held = sum(track.missing_count > 0 for track in obstacles)
            self._people_status.content = (
                f"**{len(obstacles)}/{yolo_count} obstacles localized in 3D** · short hold: **{held}**  \n"
                "Wireframe = obstacle volume · magenta = stable center · green = motion"
                if self.config.presentation_mode
                else f"YOLO obstacles: **{yolo_count}** · tracked 3D boxes: **{len(obstacles)}** · "
                f"short hold: **{held}**  \n"
                "Pink dot = stable 3D center · green arrow = confirmed consistent motion"
            )
            self._refresh_visibility_locked()

    def _update_bev_obstacles_locked(
        self, obstacles: list[Track3DState]
    ) -> None:
        active: set[str] = set()
        calibration = self._bev_calibration
        if calibration is not None:
            for track in obstacles:
                minimum = np.asarray(track.bbox3d.minimum, dtype=np.float32)
                maximum = np.asarray(track.bbox3d.maximum, dtype=np.float32)
                corners = np.asarray(
                    [
                        (x, y, z)
                        for x in (minimum[0], maximum[0])
                        for y in (minimum[1], maximum[1])
                        for z in (minimum[2], maximum[2])
                    ],
                    dtype=np.float32,
                )
                projected = calibration.project(corners)
                shown_minimum = projected.min(axis=0)
                shown_maximum = projected.max(axis=0)
                center = (shown_minimum + shown_maximum) * 0.5
                dimensions = np.maximum(shown_maximum - shown_minimum, 0.02)
                box_key = f"box:{track.track_id}"
                if box_key not in self._bev_obstacle_handles:
                    self._bev_obstacle_handles[box_key] = self.server.scene.add_box(
                        f"/metric_bev/tracked_obstacles/{track.track_id}/bbox",
                        color=_obstacle_color(track.class_name),
                        dimensions=dimensions,
                        wireframe=True,
                        position=center,
                    )
                else:
                    handle = self._bev_obstacle_handles[box_key]
                    handle.dimensions = dimensions
                    handle.position = center
                active.add(box_key)

                label_key = f"label:{track.track_id}"
                label_position = center + np.array(
                    (0.0, 0.0, dimensions[2] * 0.55), dtype=np.float32
                )
                text = f"#{track.track_id} · {track.class_name}"
                if label_key not in self._bev_obstacle_handles:
                    self._bev_obstacle_handles[label_key] = self.server.scene.add_label(
                        f"/metric_bev/tracked_obstacles/{track.track_id}/label",
                        text,
                        position=label_position,
                        anchor="bottom-center",
                        font_size_mode="screen",
                        font_screen_scale=0.8,
                        depth_test=False,
                    )
                else:
                    handle = self._bev_obstacle_handles[label_key]
                    handle.text = text
                    handle.position = label_position
                active.add(label_key)
        for key in list(self._bev_obstacle_handles):
            if key not in active:
                self._bev_obstacle_handles.pop(key).remove()

    def _update_robot_relationship(
        self,
        scene_path: str,
        handles: dict[str, Any],
        active: set[str],
        obstacles: list[Track3DState],
        robot_arm: RobotArmState | None,
    ) -> None:
        if robot_arm is None or self._center is None:
            self._relationship_status.content = (
                "Robot arm center: **not localized**  \n"
                "The anchored green component is being reacquired."
            )
            return

        robot_center = (
            np.asarray(robot_arm.center_xyz, dtype=np.float32) - self._center
        )
        robot_key = "robot:center"
        if robot_key not in handles:
            handles[robot_key] = self.server.scene.add_icosphere(
                f"{scene_path}/robot/center",
                radius=0.035,
                color=(35, 230, 90),
                subdivisions=2,
                position=robot_center,
            )
        else:
            handles[robot_key].position = robot_center
        active.add(robot_key)

        robot_label_key = "robot:label"
        robot_label = (
            "KOCH ARM CENTER"
            + (f" · HOLD {robot_arm.held_frames}" if robot_arm.held_frames else "")
        )
        if robot_label_key not in handles:
            handles[robot_label_key] = self.server.scene.add_label(
                f"{scene_path}/robot/label",
                robot_label,
                position=robot_center + np.array((0.0, 0.0, 0.07), np.float32),
                anchor="bottom-center",
                font_size_mode="screen",
                font_screen_scale=0.8,
                depth_test=False,
            )
        else:
            handles[robot_label_key].text = robot_label
            handles[robot_label_key].position = robot_center + np.array(
                (0.0, 0.0, 0.07),
                np.float32,
            )
        active.add(robot_label_key)

        segments: list[np.ndarray] = []
        segment_colors: list[tuple[int, int, int]] = []
        distances: list[tuple[float, Track3DState]] = []
        for track in obstacles:
            obstacle_center = (
                np.asarray(track.position_xyz, dtype=np.float32) - self._center
            )
            distance = float(
                np.linalg.norm(track.position_xyz - robot_arm.center_xyz)
            )
            distances.append((distance, track))
            segments.append(np.stack((robot_center, obstacle_center), axis=0))
            segment_colors.append(_distance_color(distance))
            relation_label_key = f"relation:label:{track.track_id}"
            midpoint = (robot_center + obstacle_center) * 0.5
            text = f"#{track.track_id} · {distance:.2f} m"
            if relation_label_key not in handles:
                handles[relation_label_key] = self.server.scene.add_label(
                    f"{scene_path}/relationships/{track.track_id}/distance",
                    text,
                    position=midpoint,
                    anchor="bottom-center",
                    font_size_mode="screen",
                    font_screen_scale=0.72,
                    depth_test=False,
                )
            else:
                handles[relation_label_key].text = text
                handles[relation_label_key].position = midpoint
            active.add(relation_label_key)

        relation_key = "relation:lines"
        if segments:
            points = np.asarray(segments, dtype=np.float32)
            colors = np.repeat(
                np.asarray(segment_colors, dtype=np.uint8)[:, None, :],
                2,
                axis=1,
            )
            if relation_key not in handles:
                handles[relation_key] = self.server.scene.add_line_segments(
                    f"{scene_path}/relationships/center_distances",
                    points=points,
                    colors=colors,
                    line_width=3.0,
                )
            else:
                handles[relation_key].points = points
                handles[relation_key].colors = colors
            active.add(relation_key)

        if distances:
            closest_distance, closest = min(distances, key=lambda item: item[0])
            self._relationship_status.content = (
                f"Arm center: **({robot_arm.center_xyz[0]:.2f}, "
                f"{robot_arm.center_xyz[1]:.2f}, {robot_arm.center_xyz[2]:.2f}) m**  \n"
                f"Nearest obstacle: **ID {closest.track_id} "
                f"{closest.class_name} · {closest_distance:.2f} m**  \n"
                f"Robot localization: **{robot_arm.confidence:.0%}**"
                + (
                    f" · held **{robot_arm.held_frames}** frames"
                    if robot_arm.held_frames
                    else ""
                )
            )
        else:
            self._relationship_status.content = (
                f"Arm center: **({robot_arm.center_xyz[0]:.2f}, "
                f"{robot_arm.center_xyz[1]:.2f}, {robot_arm.center_xyz[2]:.2f}) m**  \n"
                "No confirmed obstacle center."
            )

    def update_aligned_frame(
        self,
        frame: PointCloudFrame,
        tracks: list[Track3DState],
        yolo_count: int = 0,
        robot_arm: RobotArmState | None = None,
    ) -> None:
        """Apply point cloud, boxes, centers, and arrows in one client transaction."""
        with self.server.atomic():
            self.update_pointcloud(frame, defer_visibility=True)
            self.update_people(
                frame.frame_index,
                tracks,
                yolo_count=yolo_count,
                robot_arm=robot_arm,
            )

    def set_visibility(self, label: str, visible: bool) -> None:
        if label == "Point Cloud":
            self._show_reconstruction.value = visible
        elif label == "Tracking Points":
            self._show_tracking.value = visible

    def update_openarm_joint_state(
        self,
        names: tuple[str, ...],
        positions: tuple[float, ...],
        *,
        received_at: float | None = None,
        header_stamp: float = 0.0,
    ) -> int:
        if self._openarm is None:
            return 0
        return self._openarm.update_joint_state(
            names,
            positions,
            received_at=received_at,
            header_stamp=header_stamp,
        )

    def openarm_robot_state(self, timestamp: float) -> RobotArmState | None:
        if self._openarm is None:
            return None
        return self._openarm.robot_arm_state(timestamp)

    def set_view_correction(
        self,
        pitch_down_deg: float,
        roll_deg: float,
        yaw_deg: float,
    ) -> None:
        """Compatibility no-op: manual Euler rotation is not BEV correction."""

        values = np.asarray(
            (pitch_down_deg, roll_deg, yaw_deg), dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError("View correction angles must be finite")
        self.recalibrate_metric_bev()

    def set_metric_bev_enabled(self, enabled: bool) -> None:
        """Switch between untouched camera coordinates and derived BEV."""

        with self._lock:
            if self._closed:
                return
            self._bev_enabled = bool(enabled)
            self.config.metric_bev_enabled = self._bev_enabled
            self._frames_root.visible = not self._bev_enabled
            self._edge_root.visible = not self._bev_enabled
            self._bev_root.visible = self._bev_enabled
            if self._openarm is not None:
                self._openarm.set_spatial_context(
                    center=self._center,
                    bev_enabled=self._bev_enabled,
                    calibration=self._bev_calibration,
                    apriltag_center_work=self._apriltag_center_work,
                    apriltag_locked=self._apriltag_center_work is not None,
                )
            self._refresh_visibility_locked()
            if self._bev_enabled and self._bev_calibration is not None:
                self._fit_bev_camera_locked()

    def set_metric_bev_height_threshold(self, height_m: float) -> None:
        value = float(height_m)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("BEV obstacle height must be finite and non-negative")
        with self._lock:
            self._bev_obstacle_height_m = value
            self.config.metric_bev_obstacle_height_m = value

    def recalibrate_metric_bev(self) -> None:
        """Request a fresh RANSAC fit on the next reconstructed point map."""

        with self._lock:
            self._bev_calibration = None
            self._bev_recalibrate_requested = True
            self._apriltag_center_work = None
            self._apriltag_corners_work = None
            self._last_apriltag_scale = None
            for handle in self._apriltag_handles.values():
                handle.visible = False
            self._bev_status.content = (
                "等待下一幀重新估計工作平面… / Waiting for next frame…"
            )

    def set_top_down_view(self) -> None:
        """Enable the mathematical BEV and look orthogonally at its plane."""

        with self._lock:
            if self._closed:
                return
            self.set_metric_bev_enabled(True)
            self._top_down_view = True
            self._camera_fitted_clients.clear()
            if self._bev_calibration is not None:
                self._fit_bev_camera_locked()

    def reset(self) -> None:
        with self._lock:
            for handles in self._frames.values():
                self._remove_frame_handles(handles)
            self._frames.clear()
            if self._edge_obstacle_handle is not None:
                self._edge_obstacle_handle.remove()
                self._edge_obstacle_handle = None
            if self._bev_cloud_handle is not None:
                self._bev_cloud_handle.remove()
                self._bev_cloud_handle = None
            if self._bev_edge_handle is not None:
                self._bev_edge_handle.remove()
                self._bev_edge_handle = None
            for handle in self._bev_obstacle_handles.values():
                handle.remove()
            self._bev_obstacle_handles.clear()
            for handle in self._apriltag_handles.values():
                handle.remove()
            self._apriltag_handles.clear()
            self._apriltag_center_work = None
            self._apriltag_corners_work = None
            self._last_apriltag_scale = None
            self._edge_obstacle_points = np.empty((0, 3), dtype=np.float32)
            self._bev_calibration = None
            self._bev_recalibrate_requested = True
            self._center = None
            self._camera_pose = None
            self._top_down_view = False
            self._camera_fitted_clients.clear()
            self._timestep.max = 0
            self._timestep.value = 0
            self._timestep.disabled = True
            self._frame_status.content = "Waiting for a reconstructed frame…"
            self._people_status.content = "Waiting for aligned YOLO obstacle masks…"
            self._edge_foreground_status.content = (
                "Extracted obstacles: **0 points**"
            )
            self._relationship_status.content = (
                "Waiting for the anchored green Koch arm…"
            )
            self._bev_status.content = (
                "正在估計實體工作平面… / Estimating physical work plane…"
            )
            if self._openarm is not None:
                self._openarm.update_obstacles([])
                self._openarm.set_spatial_context(
                    center=None,
                    bev_enabled=self._bev_enabled,
                    calibration=None,
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.reset()
            self._frames_root.remove()
            self._edge_root.remove()
            self._bev_root.remove()
            if self._openarm is not None:
                self._openarm.close()
            for control in self._controls:
                control.remove()
            self._folder.remove()
            self._people_folder.remove()
            self._relationship_folder.remove()
            self._bev_folder.remove()
            self._closed = True

    def _on_follow_latest(self) -> None:
        with self._lock:
            if self._follow_latest.value and self._frames:
                self._timestep.value = len(self._frames) - 1
            self._timestep.disabled = len(self._frames) <= 1 or self._follow_latest.value or self._show_all.value
            self._refresh_visibility_locked()

    def _on_show_all(self) -> None:
        with self._lock:
            self._history_stride.disabled = not self._show_all.value
            self._timestep.disabled = len(self._frames) <= 1 or self._follow_latest.value or self._show_all.value
            self._refresh_visibility_locked()

    def _on_timestep(self) -> None:
        with self._lock:
            if not self._follow_latest.value and not self._show_all.value:
                self._refresh_visibility_locked()

    def _refresh_visibility(self) -> None:
        with self._lock:
            self._refresh_visibility_locked()

    def _refresh_visibility_locked(self) -> None:
        if self._edge_obstacle_handle is not None:
            self._edge_obstacle_handle.visible = (
                self._show_edge_foreground.value and not self._bev_enabled
            )
        if self._bev_edge_handle is not None:
            self._bev_edge_handle.visible = (
                self._show_edge_foreground.value and self._bev_enabled
            )
        for key, handle in self._bev_obstacle_handles.items():
            if key.startswith("box:"):
                handle.visible = self._bev_enabled and self._show_person_boxes.value
            elif key.startswith("label:"):
                handle.visible = self._bev_enabled and self._show_person_labels.value
        self._frames_root.visible = not self._bev_enabled
        self._edge_root.visible = not self._bev_enabled
        self._bev_root.visible = self._bev_enabled
        if not self._frames:
            return
        selected = int(np.clip(self._timestep.value, 0, len(self._frames) - 1))
        stride = max(1, int(self._history_stride.value))
        with self.server.atomic():
            for index, handles in enumerate(self._frames.values()):
                handles["root"].visible = (index % stride == 0) if self._show_all.value else index == selected
                handles["reconstruction"].visible = self._show_reconstruction.value
                handles["tracking"].visible = self._show_tracking.value
                for key, handle in handles["people"].items():
                    if key.startswith("box:"):
                        handle.visible = self._show_person_boxes.value
                    elif key.startswith("center:"):
                        handle.visible = self._show_person_centers.value
                    elif key == "directions":
                        handle.visible = self._show_person_directions.value
                    elif key.startswith("label:"):
                        handle.visible = self._show_person_labels.value
                    elif key == "robot:center":
                        handle.visible = self._show_robot_center.value
                    elif key == "robot:label":
                        handle.visible = self._show_robot_center.value
                    elif key == "relation:lines":
                        handle.visible = self._show_relationships.value
                    elif key.startswith("relation:label:"):
                        handle.visible = self._show_relationships.value

    @staticmethod
    def _remove_frame_handles(handles: dict[str, Any]) -> None:
        for handle in handles["people"].values():
            handle.remove()
        for key in ("tracking", "reconstruction", "root"):
            handles[key].remove()

    def _configure_camera(self, points: np.ndarray) -> None:
        distances = np.linalg.norm(points, axis=1)
        radius = float(np.clip(np.percentile(distances, 90), 0.5, 50.0))
        look_at = np.zeros(3, dtype=np.float64)
        position = np.array((0.85 * radius, -1.35 * radius, 0.65 * radius), dtype=np.float64)
        self._camera_pose = position, look_at
        self.server.initial_camera.position = position
        self.server.initial_camera.look_at = look_at
        for client in self.server.get_clients().values():
            self._fit_client_camera(client)

    def _fit_client_camera(self, client: Any) -> None:
        with self._lock:
            if self._camera_pose is None or client.client_id in self._camera_fitted_clients:
                return
            if self._bev_enabled and self._bev_calibration is not None:
                u0, u1, v0, v1 = self._bev_view_bounds_locked()
                look_at = np.array(
                    ((u0 + u1) * 0.5, (v0 + v1) * 0.5, 0.0),
                    dtype=np.float64,
                )
                span = max(u1 - u0, v1 - v0, 0.4)
                position = look_at + np.array(
                    (0.0, 0.0, 1.15 * span), dtype=np.float64
                )
                with client.atomic():
                    client.camera.position = position
                    client.camera.look_at = look_at
                    client.camera.up_direction = (0.0, 1.0, 0.0)
                    client.camera.fov = np.deg2rad(55.0)
                self._camera_fitted_clients.add(client.client_id)
                return
            position, look_at = self._camera_pose
            if self._top_down_view:
                radius = max(float(np.linalg.norm(position - look_at)), 0.5)
                position = np.array(
                    (0.0, 0.0, 1.45 * radius), dtype=np.float64
                )
                look_at = np.zeros(3, dtype=np.float64)
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = look_at
                client.camera.up_direction = (
                    (0.0, 1.0, 0.0)
                    if self._top_down_view
                    else (0.0, 0.0, 1.0)
                )
                client.camera.fov = np.deg2rad(55.0)
            self._camera_fitted_clients.add(client.client_id)

    def _bev_view_bounds_locked(self) -> tuple[float, float, float, float]:
        """Include the calibrated cloud, OpenArm base, and 1.2 m work grid."""

        assert self._bev_calibration is not None
        u0, u1, v0, v1 = self._bev_calibration.bounds_uv
        if self._openarm is not None:
            base_x, base_y, _ = self._openarm.config.base_position_xyz
            u0 = min(u0, float(base_x) - 0.12, -0.60)
            u1 = max(u1, float(base_x) + 0.12, 0.60)
            v0 = min(v0, float(base_y) - 0.12, -0.60)
            v1 = max(v1, float(base_y) + 0.12, 0.60)
        return float(u0), float(u1), float(v0), float(v1)


def _robust_center(points: np.ndarray) -> np.ndarray:
    low, high = np.percentile(points, (2.0, 98.0), axis=0)
    interior = np.all((points >= low) & (points <= high), axis=1)
    selected = points[interior] if interior.any() else points
    return np.mean(selected, axis=0, dtype=np.float64).astype(np.float32)


def _view_correction_quaternion(
    pitch_down_deg: float,
    roll_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Return a normalized Viser wxyz display quaternion.

    Coordinates are x-right/y-forward/z-up. A positive physical downward
    camera pitch therefore needs a negative rotation around +x to recover a
    level workcell. Roll is around the forward (+y) axis and yaw around +z.
    """

    values = np.asarray(
        (pitch_down_deg, roll_deg, yaw_deg), dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("View correction angles must be finite")

    def axis_angle(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
        half = np.deg2rad(float(degrees)) * 0.5
        xyz = np.asarray(axis, dtype=np.float64) * np.sin(half)
        return np.array((np.cos(half), *xyz), dtype=np.float64)

    def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dtype=np.float64,
        )

    pitch = axis_angle((1.0, 0.0, 0.0), -values[0])
    roll = axis_angle((0.0, 1.0, 0.0), values[1])
    yaw = axis_angle((0.0, 0.0, 1.0), values[2])
    result = multiply(yaw, multiply(roll, pitch))
    return result / max(float(np.linalg.norm(result)), 1e-12)


def _obstacle_color(class_name: str) -> tuple[int, int, int]:
    if class_name == "person":
        return (20, 155, 255)
    if class_name in {"vehicle", "bicycle", "motorcycle"}:
        return (255, 155, 35)
    return (185, 95, 255)


def _distance_color(distance_m: float) -> tuple[int, int, int]:
    if distance_m < 0.30:
        return (255, 45, 55)
    if distance_m < 0.60:
        return (255, 175, 35)
    return (35, 205, 235)


def _stable_horizontal_direction(track: Track3DState, dimensions: np.ndarray) -> np.ndarray | None:
    """Estimate walking direction from recent centers, with smoothed velocity fallback."""
    if track.hit_count < 3:
        return None
    noise_floor = max(0.004, 0.02 * float(np.max(dimensions[:2])))
    if len(track.history) >= 3:
        history = np.asarray(track.history[-6:], dtype=np.float32)[:, :2]
        deltas = np.diff(history, axis=0)
        magnitudes = np.linalg.norm(deltas, axis=1)
        max_step = max(0.20, 2.5 * float(np.max(dimensions[:2])))
        valid = (magnitudes >= noise_floor) & (magnitudes <= max_step)
        if int(valid.sum()) >= 2:
            units = deltas[valid] / magnitudes[valid, None]
            weighted = units * np.minimum(magnitudes[valid, None], 4.0 * noise_floor)
            direction_xy = weighted.sum(axis=0)
            direction_norm = float(np.linalg.norm(direction_xy))
            consistency = direction_norm / max(float(np.minimum(magnitudes[valid], 4.0 * noise_floor).sum()), 1e-6)
            net = deltas[valid].sum(axis=0)
            if consistency >= 0.55 and float(np.linalg.norm(net)) >= 2.0 * noise_floor:
                direction_xy /= direction_norm
                return np.array((direction_xy[0], direction_xy[1], 0.0), dtype=np.float32)

    # Sparse reconstruction updates can leave too few center samples even when
    # the timestamp-aware Kalman velocity is already reliable.
    velocity = np.asarray(track.velocity_xyz, dtype=np.float32)[:2]
    speed = float(np.linalg.norm(velocity))
    if 0.04 <= speed <= 3.0:
        velocity /= speed
        return np.array((velocity[0], velocity[1], 0.0), dtype=np.float32)
    return None
