from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from realtime_safety.config import OpenArmConfig
from realtime_safety.gui.metric_bev import MetricBevCalibration
from realtime_safety.types import RobotArmState


LOGGER = logging.getLogger(__name__)
_MAIN_JOINT_RE = re.compile(r"^openarm_(?:(left|right)_)?joint([1-7])$")


def map_openarm_joint_positions(
    target_names: Sequence[str],
    source_names: Sequence[str],
    source_positions: Sequence[float],
) -> dict[str, float]:
    """Map common OpenArm controller naming variants to official URDF names."""

    if len(source_names) != len(source_positions):
        return {}
    source = {
        str(name): float(position)
        for name, position in zip(source_names, source_positions)
        if np.isfinite(position)
    }
    mapped: dict[str, float] = {}
    for target in target_names:
        core = target.removeprefix("openarm_")
        if core.startswith(("left_", "right_")):
            # Bimanual targets must retain their side; never drive both arms
            # from an ambiguous unprefixed controller joint.
            candidates = (target, core)
        else:
            # Compatibility for the historical single-arm asset and tests.
            candidates = (
                target,
                core,
                f"openarm_right_{core}",
                f"openarm_left_{core}",
                f"right_{core}",
                f"left_{core}",
            )
        for candidate in candidates:
            if candidate in source:
                mapped[target] = source[candidate]
                break
    return mapped


class OpenArmScene:
    """Render and animate an official OpenArm URDF in the Viser work cell."""

    def __init__(self, server: Any, config: OpenArmConfig) -> None:
        self.server = server
        self.config = config
        self._lock = threading.RLock()
        self._closed = False
        self._center: np.ndarray | None = None
        self._bev_enabled = False
        self._bev_calibration: MetricBevCalibration | None = None
        self._apriltag_center_work: np.ndarray | None = None
        self._apriltag_locked = False
        self._display_rotation = np.eye(3, dtype=np.float64)
        self._display_position = np.asarray(
            config.base_position_xyz, dtype=np.float64
        )
        self._last_received_at = 0.0
        self._last_header_stamp = 0.0
        self._last_source_names: tuple[str, ...] = ()
        self._matched_main_joints = 0
        self._required_main_joints = 0
        self._joint_values = np.zeros(0, dtype=np.float64)
        self._joint_names: tuple[str, ...] = ()
        self._joint_limits: dict[str, tuple[float | None, float | None]] = {}
        self._joint_sliders: list[Any] = []
        self._updating_gui = False
        self._obstacles: list[tuple[int, str, np.ndarray]] = []
        self._relationship_handles: dict[str, Any] = {}
        self._last_status = ""
        self._load_error = ""
        self._urdf: Any | None = None
        self._kinematic_model: Any | None = None
        active_base_xyz = (
            config.base_from_apriltag_xyz
            if config.base_anchor == "apriltag"
            else config.base_position_xyz
        )
        self._default_base_xyz = tuple(active_base_xyz)
        self._default_base_rpy = tuple(config.base_rpy_deg)

        self._root = self.server.scene.add_frame(
            "/openarm", show_axes=True, axes_length=0.08, axes_radius=0.003
        )
        self._relationship_root = self.server.scene.add_frame(
            "/openarm_relationships", show_axes=False
        )
        self._workspace_grid = self.server.scene.add_grid(
            "/openarm_workspace/work_plane",
            width=1.2,
            height=1.2,
            cell_size=0.05,
            section_size=0.25,
            plane="xy",
            cell_color=(48, 58, 62),
            cell_thickness=0.35,
            section_color=(85, 115, 125),
            section_thickness=0.8,
            position=(0.0, 0.0, -0.004),
            visible=True,
        )
        self._base_marker = self.server.scene.add_cylinder(
            "/openarm/base_footprint",
            radius=0.065,
            height=0.008,
            color=(25, 220, 235),
            opacity=0.45,
            position=(0.0, 0.0, -0.006),
        )

        with self.server.gui.add_folder(
            "OpenArm · URDF + JointState",
            order=32,
            expand_by_default=True,
        ) as folder:
            self._folder = folder
            self._status = self.server.gui.add_markdown(
                "正在載入 OpenArm URDF… / Loading OpenArm URDF…"
            )
            self._relationship_status = self.server.gui.add_markdown(
                "TCP 已建立；等待障礙物幾何…"
            )
            self._show_model = self.server.gui.add_checkbox(
                "顯示 OpenArm / Show robot", initial_value=config.show_visual
            )
            self._show_workspace = self.server.gui.add_checkbox(
                "顯示 5 cm 工作格 / Work grid", initial_value=True
            )
            self._follow_ros = self.server.gui.add_checkbox(
                "追蹤 ROS JointState", initial_value=True
            )

        with self.server.gui.add_folder(
            "OpenArm 基座外參 / Base calibration",
            order=33,
            expand_by_default=False,
        ) as calibration_folder:
            self._calibration_folder = calibration_folder
            self._base_sliders = [
                self.server.gui.add_slider(
                    label,
                    min=minimum,
                    max=maximum,
                    step=step,
                    initial_value=float(initial),
                )
                for label, minimum, maximum, step, initial in (
                    ("Tag→Base X · right (m)", -1.5, 1.5, 0.005, active_base_xyz[0]),
                    ("Tag→Base Y · forward (m)", -1.5, 1.5, 0.005, active_base_xyz[1]),
                    ("Tag→Base Z · height (m)", -0.5, 1.0, 0.005, active_base_xyz[2]),
                    ("Base roll (deg)", -180.0, 180.0, 1.0, config.base_rpy_deg[0]),
                    ("Base pitch (deg)", -180.0, 180.0, 1.0, config.base_rpy_deg[1]),
                    ("Base yaw (deg)", -180.0, 180.0, 1.0, config.base_rpy_deg[2]),
                )
            ]
            self._reset_base = self.server.gui.add_button("重設照片估算外參 / Reset base")
            self._calibration_note = self.server.gui.add_markdown(
                f"初始參考：相機高度 **{config.camera_height_m:.2f} m**、"
                f"向下 **{config.camera_downward_angle_deg:.1f}°**。  \n"
                "XYZ 是 **AprilTag 中心→雙臂 body 基座**；目前標定距離為 0.500 m。"
            )

        with self.server.gui.add_folder(
            "OpenArm 關節檢查 / Joint inspection",
            order=34,
            expand_by_default=False,
        ) as joint_folder:
            self._joint_folder = joint_folder
            self._joint_status = self.server.gui.add_markdown(
                "尚未建立 joint controls。"
            )
            self._zero_pose = self.server.gui.add_button("零姿態測試 / Zero pose")

        self._controls = [
            self._status,
            self._relationship_status,
            self._show_model,
            self._show_workspace,
            self._follow_ros,
            *self._base_sliders,
            self._reset_base,
            self._calibration_note,
            self._joint_status,
            self._zero_pose,
        ]
        self._show_model.on_update(lambda _: self._refresh_visibility())
        self._show_workspace.on_update(lambda _: self._refresh_visibility())
        self._follow_ros.on_update(lambda _: self._on_follow_ros())
        for slider in self._base_sliders:
            slider.on_update(lambda _, slider=slider: self._on_base_changed(slider))
        self._reset_base.on_click(lambda _: self._reset_base_pose())
        self._zero_pose.on_click(lambda _: self._set_zero_pose())

        self._load_urdf()
        self._update_spatial_transform_locked()
        self._refresh_visibility()
        self.refresh_status()

    @property
    def loaded(self) -> bool:
        return self._urdf is not None

    @property
    def tcp_position(self) -> np.ndarray | None:
        with self._lock:
            positions = self._tcp_positions_locked()
            if not positions:
                return None
            return np.mean(np.stack(tuple(positions.values())), axis=0)

    @property
    def tcp_positions(self) -> dict[str, np.ndarray]:
        with self._lock:
            return self._tcp_positions_locked()

    @property
    def matched_main_joints(self) -> int:
        return self._matched_main_joints

    def _load_urdf(self) -> None:
        urdf_path = self._resolve_project_path(self.config.urdf_path)
        description_root = self._resolve_project_path(self.config.description_path)
        if not urdf_path.is_file():
            self._load_error = (
                f"找不到 {urdf_path}. 請執行 scripts/setup_openarm.sh"
            )
            LOGGER.error(self._load_error)
            return
        try:
            import yourdfpy
            from viser.extras import ViserUrdf

            package_prefix = "package://openarm_description/"

            def resolve_mesh(fname: str) -> str:
                if fname.startswith(package_prefix):
                    return str(description_root / fname[len(package_prefix) :])
                return fname

            model = yourdfpy.URDF.load(
                urdf_path,
                build_scene_graph=True,
                load_meshes=True,
                filename_handler=resolve_mesh,
            )
            self._urdf = ViserUrdf(
                self.server,
                model,
                root_node_name="/openarm",
                load_meshes=True,
                load_collision_meshes=False,
            )
            self._kinematic_model = model
            self._joint_names = self._urdf.get_actuated_joint_names()
            self._required_main_joints = sum(
                _MAIN_JOINT_RE.match(name) is not None for name in self._joint_names
            )
            self._joint_limits = self._urdf.get_actuated_joint_limits()
            self._joint_values = np.zeros(len(self._joint_names), dtype=np.float64)
            self._urdf.update_cfg(self._joint_values)
            self._build_joint_sliders()
            LOGGER.info(
                "Loaded %s URDF with %d actuated joints from %s",
                self.config.model,
                len(self._joint_names),
                urdf_path,
            )
        except Exception as exc:
            self._urdf = None
            self._load_error = f"URDF 載入失敗 / load failed: {exc}"
            LOGGER.exception("Could not load OpenArm URDF")

    def _build_joint_sliders(self) -> None:
        with self._joint_folder:
            for index, name in enumerate(self._joint_names):
                lower, upper = self._joint_limits[name]
                low = float(lower) if lower is not None else -np.pi
                high = float(upper) if upper is not None else np.pi
                slider = self.server.gui.add_slider(
                    name,
                    min=low,
                    max=high,
                    step=0.001,
                    initial_value=0.0,
                    disabled=True,
                )
                slider.on_update(
                    lambda _, index=index, slider=slider: self._on_manual_joint(index, slider)
                )
                self._joint_sliders.append(slider)
                self._controls.append(slider)

    def update_joint_state(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        received_at: float | None = None,
        header_stamp: float = 0.0,
    ) -> int:
        with self._lock:
            if self._closed or self._urdf is None:
                return 0
            mapped = map_openarm_joint_positions(
                self._joint_names, names, positions
            )
            if not mapped:
                self._last_source_names = tuple(str(name) for name in names)
                self._matched_main_joints = 0
                self.refresh_status()
                return 0
            for index, target in enumerate(self._joint_names):
                if target not in mapped:
                    continue
                lower, upper = self._joint_limits[target]
                value = mapped[target]
                if lower is not None:
                    value = max(value, float(lower))
                if upper is not None:
                    value = min(value, float(upper))
                self._joint_values[index] = value
            self._last_source_names = tuple(str(name) for name in names)
            self._last_received_at = (
                time.monotonic() if received_at is None else float(received_at)
            )
            self._last_header_stamp = float(header_stamp)
            self._matched_main_joints = sum(
                name in mapped and _MAIN_JOINT_RE.match(name) is not None
                for name in self._joint_names
            )
            if self._follow_ros.value:
                self._apply_joint_values_locked()
            self.refresh_status()
            return len(mapped)

    def set_spatial_context(
        self,
        *,
        center: np.ndarray | None,
        bev_enabled: bool,
        calibration: MetricBevCalibration | None,
        apriltag_center_work: np.ndarray | None = None,
        apriltag_locked: bool = False,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._center = (
                None if center is None else np.asarray(center, dtype=np.float64)
            )
            self._bev_enabled = bool(bev_enabled)
            self._bev_calibration = calibration
            if apriltag_center_work is not None:
                self._apriltag_center_work = np.asarray(
                    apriltag_center_work, dtype=np.float64
                )
            elif not apriltag_locked:
                self._apriltag_center_work = None
            self._apriltag_locked = bool(apriltag_locked)
            self._update_spatial_transform_locked()
            self._refresh_relationships_locked()
            self.refresh_status()

    def update_obstacles(
        self, obstacles: Iterable[tuple[int, str, np.ndarray]]
    ) -> None:
        with self._lock:
            self._obstacles = [
                (int(track_id), str(label), np.asarray(position, dtype=np.float64))
                for track_id, label, position in obstacles
            ]
            self._refresh_relationships_locked()

    def robot_arm_state(self, timestamp: float) -> RobotArmState | None:
        """Return a controller-facing camera-frame TCP only when FK is fresh."""

        with self._lock:
            age = (
                time.monotonic() - self._last_received_at
                if self._last_received_at
                else float("inf")
            )
            if (
                self._kinematic_model is None
                or self._bev_calibration is None
                or self._required_main_joints == 0
                or self._matched_main_joints < self._required_main_joints
                or age > self.config.stale_after_s
            ):
                return None
            tcp_positions = self._tcp_internal_positions_locked()
            if not tcp_positions:
                return None
            tcp = np.mean(np.stack(tuple(tcp_positions.values())), axis=0)
            localization_source = (
                "urdf_fk_joint_state_bimanual"
                if {"left_tcp", "right_tcp"}.issubset(tcp_positions)
                else "urdf_fk_joint_state"
            )
            return RobotArmState(
                center_xyz=tcp,
                center_xy=np.zeros(2, dtype=np.float32),
                image_size=(0, 0),
                mask_pixels=0,
                point_count=0,
                confidence=1.0,
                timestamp=float(timestamp),
                held_frames=0,
                localization_source=localization_source,
                link_points_xyz=tcp_positions,
            )

    def refresh_status(self) -> None:
        with self._lock:
            if self._load_error:
                status = f"🔴 **OpenArm URDF 未載入**  \n`{self._load_error}`"
            elif self._urdf is None:
                status = "🟠 **OpenArm URDF 載入中**"
            elif not self._last_received_at:
                anchor = (
                    "APRILTAG LOCK"
                    if self._apriltag_locked
                    else "WAITING APRILTAG"
                )
                status = (
                    f"🟡 **URDF READY · 等待 `{self.config.joint_states_topic}`**  \n"
                    f"{self.config.model} · **{anchor}** · 零姿態顯示 · base XYZ "
                    f"{self._base_xyz_text()} m"
                )
            else:
                age = max(time.monotonic() - self._last_received_at, 0.0)
                live = age <= self.config.stale_after_s
                state = "🟢 JOINT LIVE" if live else "🟠 JOINT STALE"
                status = (
                    f"{state} · **{self._matched_main_joints}/{self._required_main_joints} "
                    "left+right arm joints** · "
                    f"age **{age:.2f} s**  \n"
                    f"topic `{self.config.joint_states_topic}` · base XYZ "
                    f"{self._base_xyz_text()} m"
                )
            if status != self._last_status:
                self._status.content = status
                self._last_status = status
            if self._joint_names:
                left = " · ".join(
                    f"L{match.group(2)} `{self._joint_values[index]:+.3f}`"
                    for index, name in enumerate(self._joint_names)
                    if (match := _MAIN_JOINT_RE.match(name)) and match.group(1) == "left"
                )
                right = " · ".join(
                    f"R{match.group(2)} `{self._joint_values[index]:+.3f}`"
                    for index, name in enumerate(self._joint_names)
                    if (match := _MAIN_JOINT_RE.match(name)) and match.group(1) == "right"
                )
                self._joint_status.content = (
                    f"**LEFT** {left}  \n**RIGHT** {right}"
                    if left or right
                    else "等待左右各七軸 JointState…"
                )

    def _apply_joint_values_locked(self) -> None:
        if self._urdf is None:
            return
        self._urdf.update_cfg(self._joint_values)
        self._updating_gui = True
        try:
            for slider, value in zip(self._joint_sliders, self._joint_values):
                slider.value = float(value)
        finally:
            self._updating_gui = False
        self._refresh_relationships_locked()

    def _on_manual_joint(self, index: int, slider: Any) -> None:
        with self._lock:
            if self._updating_gui or self._follow_ros.value or self._urdf is None:
                return
            self._joint_values[index] = float(slider.value)
            self._urdf.update_cfg(self._joint_values)
            self._refresh_relationships_locked()
            self.refresh_status()

    def _on_follow_ros(self) -> None:
        with self._lock:
            for slider in self._joint_sliders:
                slider.disabled = bool(self._follow_ros.value)
            if self._follow_ros.value and self._last_received_at:
                self._apply_joint_values_locked()
            self.refresh_status()

    def _set_zero_pose(self) -> None:
        with self._lock:
            if self._urdf is None:
                return
            self._follow_ros.value = False
            self._joint_values.fill(0.0)
            self._apply_joint_values_locked()
            for slider in self._joint_sliders:
                slider.disabled = False
            self.refresh_status()

    def _on_base_changed(self, _slider: Any) -> None:
        with self._lock:
            if self._updating_gui:
                return
            xyz = tuple(float(slider.value) for slider in self._base_sliders[:3])
            rpy = tuple(float(slider.value) for slider in self._base_sliders[3:])
            if self.config.base_anchor == "apriltag":
                self.config.base_from_apriltag_xyz = xyz
            else:
                self.config.base_position_xyz = xyz
            self.config.base_rpy_deg = rpy
            self._update_spatial_transform_locked()
            self._refresh_relationships_locked()
            self.refresh_status()

    def update_simulator_model_pose(
        self, model_position: np.ndarray, model_rotation: Rotation
    ) -> None:
        """Apply Gazebo's model transform on top of the configured base mount."""

        with self._lock:
            local_position = np.asarray(self._default_base_xyz, dtype=np.float64)
            local_rotation = Rotation.from_euler(
                "xyz", self._default_base_rpy, degrees=True
            )
            position = np.asarray(model_position, dtype=np.float64) + model_rotation.apply(
                local_position
            )
            rotation = model_rotation * local_rotation
            self.config.base_position_xyz = tuple(float(value) for value in position)
            self.config.base_rpy_deg = tuple(float(value) for value in rotation.as_euler("xyz", degrees=True))
            self._update_spatial_transform_locked()

    def _reset_base_pose(self) -> None:
        defaults_xyz = self._default_base_xyz
        defaults_rpy = self._default_base_rpy
        self._updating_gui = True
        try:
            for slider, value in zip(
                self._base_sliders, (*defaults_xyz, *defaults_rpy)
            ):
                slider.value = value
        finally:
            self._updating_gui = False
        if self.config.base_anchor == "apriltag":
            self.config.base_from_apriltag_xyz = defaults_xyz
        else:
            self.config.base_position_xyz = defaults_xyz
        self.config.base_rpy_deg = defaults_rpy
        self._update_spatial_transform_locked()
        self._refresh_relationships_locked()
        self.refresh_status()

    def _update_spatial_transform_locked(self) -> None:
        base_position = self._base_work_position_locked()
        base_rotation = Rotation.from_euler(
            "xyz", self.config.base_rpy_deg, degrees=True
        ).as_matrix()
        if self._bev_enabled:
            position = base_position
            rotation = base_rotation
            workspace_position = np.zeros(3, dtype=np.float64)
            workspace_rotation = np.eye(3, dtype=np.float64)
        elif self._bev_calibration is not None:
            calibration = self._bev_calibration
            basis = np.column_stack(
                (calibration.right, calibration.forward, calibration.normal)
            ).astype(np.float64)
            position = calibration.origin + basis @ base_position
            if self._center is not None:
                position = position - self._center
            rotation = basis @ base_rotation
            workspace_position = calibration.origin.astype(np.float64)
            if self._center is not None:
                workspace_position -= self._center
            workspace_rotation = basis
        else:
            position = base_position.copy()
            if self._center is not None:
                position -= self._center
            rotation = base_rotation
            workspace_position = np.zeros(3, dtype=np.float64)
            if self._center is not None:
                workspace_position -= self._center
            workspace_rotation = np.eye(3, dtype=np.float64)
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        self._display_position = position
        self._display_rotation = rotation
        self._root.position = tuple(position)
        self._root.wxyz = (
            float(quaternion_xyzw[3]),
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
        )
        workspace_xyzw = Rotation.from_matrix(workspace_rotation).as_quat()
        self._workspace_grid.position = tuple(workspace_position)
        self._workspace_grid.wxyz = (
            float(workspace_xyzw[3]),
            float(workspace_xyzw[0]),
            float(workspace_xyzw[1]),
            float(workspace_xyzw[2]),
        )

    def _base_work_position_locked(self) -> np.ndarray:
        if (
            self.config.base_anchor == "apriltag"
            and self._apriltag_center_work is not None
        ):
            return self._apriltag_center_work + np.asarray(
                self.config.base_from_apriltag_xyz, dtype=np.float64
            )
        return np.asarray(self.config.base_position_xyz, dtype=np.float64)

    def _tcp_link_names_locked(self) -> tuple[tuple[str, str], ...]:
        model = self._kinematic_model
        if model is None:
            return ()
        available = model.link_map
        names = (
            ("left_tcp", "openarm_left_hand_tcp"),
            ("right_tcp", "openarm_right_hand_tcp"),
            ("tcp", "openarm_hand_tcp"),
        )
        return tuple(item for item in names if item[1] in available)

    def _link_transform_locked(self, link_name: str) -> np.ndarray | None:
        model = self._kinematic_model
        if model is None:
            return None
        for root_name in ("world", "openarm_body_link0", "openarm_link0"):
            if root_name not in model.link_map:
                continue
            try:
                return np.asarray(
                    model.get_transform(link_name, root_name), dtype=np.float64
                )
            except (KeyError, ValueError):
                continue
        return None

    def _tcp_positions_locked(self) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        for label, link_name in self._tcp_link_names_locked():
            transform = self._link_transform_locked(link_name)
            if transform is None:
                continue
            positions[label] = (
                self._display_position
                + self._display_rotation @ transform[:3, 3]
            ).astype(np.float32)
        return positions

    def _tcp_internal_positions_locked(self) -> dict[str, np.ndarray]:
        local: dict[str, np.ndarray] = {}
        base_rotation = Rotation.from_euler(
            "xyz", self.config.base_rpy_deg, degrees=True
        ).as_matrix()
        base_position = self._base_work_position_locked()
        for label, link_name in self._tcp_link_names_locked():
            transform = self._link_transform_locked(link_name)
            if transform is None:
                continue
            local[label] = base_position + base_rotation @ transform[:3, 3]
        calibration = self._bev_calibration
        if calibration is None:
            return {
                label: value.astype(np.float32) for label, value in local.items()
            }
        basis = np.column_stack(
            (calibration.right, calibration.forward, calibration.normal)
        ).astype(np.float64)
        return {
            label: (calibration.origin + basis @ value).astype(np.float32)
            for label, value in local.items()
        }

    def _refresh_relationships_locked(self) -> None:
        tcp_positions = self._tcp_positions_locked()
        active: set[str] = set()
        for tcp_name, tcp in tcp_positions.items():
            key = f"tcp:{tcp_name}"
            if key not in self._relationship_handles:
                self._relationship_handles[key] = self.server.scene.add_icosphere(
                    f"/openarm_relationships/{tcp_name}",
                    radius=0.022,
                    color=(20, 255, 120)
                    if tcp_name.startswith("left")
                    else (25, 190, 255),
                    subdivisions=2,
                    position=tcp,
                )
            else:
                self._relationship_handles[key].position = tcp
            active.add(key)
        if (
            self.config.base_anchor == "apriltag"
            and self._apriltag_center_work is not None
        ):
            tag = self._work_to_display_locked(self._apriltag_center_work)
            base = self._work_to_display_locked(self._base_work_position_locked())
            key = "apriltag_marker"
            if key not in self._relationship_handles:
                self._relationship_handles[key] = self.server.scene.add_icosphere(
                    "/openarm_relationships/apriltag_center",
                    radius=0.018,
                    color=(255, 132, 25),
                    subdivisions=2,
                    position=tag,
                )
            else:
                self._relationship_handles[key].position = tag
            active.add(key)
            key = "apriltag_to_base"
            points = np.asarray([[tag, base]], dtype=np.float32)
            colors = np.asarray([[(255, 80, 55), (255, 80, 55)]], dtype=np.uint8)
            if key not in self._relationship_handles:
                self._relationship_handles[key] = self.server.scene.add_line_segments(
                    "/openarm_relationships/apriltag_to_body_base",
                    points=points,
                    colors=colors,
                    line_width=5.0,
                )
            else:
                self._relationship_handles[key].points = points
            active.add(key)
            key = "apriltag_distance_label"
            distance = float(np.linalg.norm(base - tag))
            label_text = f"APRILTAG → OPENARM BODY · {distance:.3f} m"
            midpoint = (tag + base) * 0.5
            if key not in self._relationship_handles:
                self._relationship_handles[key] = self.server.scene.add_label(
                    "/openarm_relationships/apriltag_base_distance",
                    label_text,
                    position=midpoint,
                    anchor="bottom-center",
                    font_size_mode="screen",
                    font_screen_scale=0.8,
                    depth_test=False,
                )
            else:
                self._relationship_handles[key].text = label_text
                self._relationship_handles[key].position = midpoint
            active.add(key)

        if tcp_positions and self._obstacles:
            segments = []
            colors = []
            nearest: tuple[float, int, str, str] | None = None
            for track_id, label, center in self._obstacles:
                tcp_name, tcp, distance = min(
                    (
                        (name, position, float(np.linalg.norm(center - position)))
                        for name, position in tcp_positions.items()
                    ),
                    key=lambda item: item[2],
                )
                segments.append(np.stack((tcp, center), axis=0))
                colors.append(
                    (255, 45, 45)
                    if distance < 0.20
                    else (255, 180, 30)
                    if distance < 0.40
                    else (40, 225, 120)
                )
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, track_id, label, tcp_name)
            key = "distance_lines"
            points = np.asarray(segments, dtype=np.float32)
            shown_colors = np.repeat(
                np.asarray(colors, dtype=np.uint8)[:, None, :], 2, axis=1
            )
            if key not in self._relationship_handles:
                self._relationship_handles[key] = self.server.scene.add_line_segments(
                    "/openarm_relationships/tcp_to_obstacles",
                    points=points,
                    colors=shown_colors,
                    line_width=3.0,
                )
            else:
                self._relationship_handles[key].points = points
                self._relationship_handles[key].colors = shown_colors
            active.add(key)
            if nearest is not None:
                distance, track_id, label, tcp_name = nearest
                self._relationship_status.content = (
                    f"左右 TCP 最近障礙 / nearest: **{tcp_name} → "
                    f"#{track_id} {label} · {distance:.3f} m**"
                )
        elif tcp_positions:
            self._relationship_status.content = (
                "LEFT + RIGHT TCP 已建立；目前無確認障礙物 / no confirmed obstacle."
            )
        for key in list(self._relationship_handles):
            if key not in active:
                self._relationship_handles.pop(key).remove()

    def _work_to_display_locked(self, point: np.ndarray) -> np.ndarray:
        value = np.asarray(point, dtype=np.float64)
        if self._bev_enabled:
            return value.astype(np.float32)
        calibration = self._bev_calibration
        if calibration is None:
            shown = value.copy()
        else:
            basis = np.column_stack(
                (calibration.right, calibration.forward, calibration.normal)
            ).astype(np.float64)
            shown = calibration.origin + basis @ value
        if self._center is not None:
            shown -= self._center
        return shown.astype(np.float32)

    def _refresh_visibility(self) -> None:
        with self._lock:
            visible = bool(self._show_model.value)
            self._root.visible = visible
            self._relationship_root.visible = visible
            self._workspace_grid.visible = bool(self._show_workspace.value)

    def _base_xyz_text(self) -> str:
        return "(" + ", ".join(
            f"{value:+.3f}" for value in self._base_work_position_locked()
        ) + ")"

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Removing the parent frame recursively removes all ViserUrdf
            # joint frames and meshes. Calling ViserUrdf.remove() first walks
            # the same hierarchy twice and emits one warning per mesh.
            self._urdf = None
            self._kinematic_model = None
            for handle in self._relationship_handles.values():
                handle.remove()
            self._relationship_handles.clear()
            self._root.remove()
            self._relationship_root.remove()
            self._workspace_grid.remove()
            for control in self._controls:
                control.remove()
            self._folder.remove()
            self._calibration_folder.remove()
            self._joint_folder.remove()


__all__ = ["OpenArmScene", "map_openarm_joint_positions"]
