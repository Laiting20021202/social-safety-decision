from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from realtime_safety.config import GuiConfig
from realtime_safety.types import PointCloudFrame, Track3DState


class ReconstructionScene3D:
    """Clean, bounded St4RTrack-style 4D point-cloud viewer.

    Each reconstruction is stored below ``/frames/t...`` in the common anchor
    coordinate system. Only the selected timestep is visible by default, just
    like the upstream St4RTrack visualizer.
    """

    def __init__(self, server: Any, config: GuiConfig) -> None:
        self.server = server
        self.config = config
        self._lock = threading.RLock()
        self._frames: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._center: np.ndarray | None = None
        self._camera_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._camera_fitted_clients: set[int] = set()
        self._closed = False

        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (3.0, -5.0, 2.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.fov = np.deg2rad(55.0)

        self._frames_root = self.server.scene.add_frame("/frames", show_axes=False)
        with self.server.gui.add_folder("4D Reconstruction", expand_by_default=True) as folder:
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

        with self.server.gui.add_folder("People · YOLO + 3D", expand_by_default=True) as people_folder:
            self._people_folder = people_folder
            self._show_person_boxes = self.server.gui.add_checkbox("3D bounding boxes", initial_value=True)
            self._show_person_centers = self.server.gui.add_checkbox("Person centers", initial_value=True)
            self._show_person_directions = self.server.gui.add_checkbox("Direction arrows", initial_value=True)
            self._people_status = self.server.gui.add_markdown("Waiting for aligned YOLO person masks…")

        self._controls = [
            self._show_reconstruction,
            self._show_tracking,
            self._follow_latest,
            self._show_all,
            self._history_stride,
            self._timestep,
            self._frame_status,
            self._show_person_boxes,
            self._show_person_centers,
            self._show_person_directions,
            self._people_status,
        ]
        self._show_reconstruction.on_update(lambda _: self._refresh_visibility())
        self._show_tracking.on_update(lambda _: self._refresh_visibility())
        self._follow_latest.on_update(lambda _: self._on_follow_latest())
        self._show_all.on_update(lambda _: self._on_show_all())
        self._history_stride.on_update(lambda _: self._refresh_visibility())
        self._timestep.on_update(lambda _: self._on_timestep())
        self._show_person_boxes.on_update(lambda _: self._refresh_visibility())
        self._show_person_centers.on_update(lambda _: self._refresh_visibility())
        self._show_person_directions.on_update(lambda _: self._refresh_visibility())

        @self.server.on_client_connect
        def _(client: Any) -> None:
            self._fit_client_camera(client)

    @property
    def node_count(self) -> int:
        return 1 + sum(3 + len(handles["people"]) for handles in self._frames.values())

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def update_pointcloud(self, frame: PointCloudFrame, defer_visibility: bool = False) -> None:
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
                f"Frame **{frame.frame_index}** · {len(centered_points):,} points · "
                f"source: **{frame.source}** · history: **{len(self._frames)}/{self.config.history_frames}**"
            )
            if not defer_visibility:
                self._refresh_visibility_locked()
            if self._camera_pose is None:
                self._configure_camera(centered_points)

    def update_people(self, frame_index: int, tracks: list[Track3DState], yolo_count: int = 0) -> None:
        """Attach person boxes, robust centers, and velocity arrows to one 4D frame."""
        with self._lock:
            if self._closed or self._center is None or frame_index not in self._frames:
                return
            frame_handles = self._frames[frame_index]
            scene_path = frame_handles["path"]
            people_handles: dict[str, Any] = frame_handles["people"]
            active: set[str] = set()
            arrow_segments: list[np.ndarray] = []
            people = [track for track in tracks if track.class_name == "person"]
            for track in people:
                track_id = track.track_id
                center = np.asarray(track.position_xyz, dtype=np.float32) - self._center
                dimensions = np.maximum(np.asarray(track.bbox3d.size, dtype=np.float32), 0.03)

                box_key = f"box:{track_id}"
                if box_key not in people_handles:
                    people_handles[box_key] = self.server.scene.add_box(
                        f"{scene_path}/people/{track_id}/bbox",
                        color=(20, 155, 255),
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

            for key in list(people_handles):
                if key not in active:
                    people_handles.pop(key).remove()
            self._people_status.content = (
                f"YOLO people: **{yolo_count}** · tracked 3D boxes: **{len(people)}** · "
                f"short hold: **{sum(track.missing_count > 0 for track in people)}**  \n"
                "Pink dot = robust 3D center · green arrow = confirmed consistent motion"
            )
            self._refresh_visibility_locked()

    def update_aligned_frame(
        self,
        frame: PointCloudFrame,
        tracks: list[Track3DState],
        yolo_count: int = 0,
    ) -> None:
        """Apply point cloud, boxes, centers, and arrows in one client transaction."""
        with self.server.atomic():
            self.update_pointcloud(frame, defer_visibility=True)
            self.update_people(frame.frame_index, tracks, yolo_count=yolo_count)

    def set_visibility(self, label: str, visible: bool) -> None:
        if label == "Point Cloud":
            self._show_reconstruction.value = visible
        elif label == "Tracking Points":
            self._show_tracking.value = visible

    def reset(self) -> None:
        with self._lock:
            for handles in self._frames.values():
                self._remove_frame_handles(handles)
            self._frames.clear()
            self._center = None
            self._camera_pose = None
            self._camera_fitted_clients.clear()
            self._timestep.max = 0
            self._timestep.value = 0
            self._timestep.disabled = True
            self._frame_status.content = "Waiting for a reconstructed frame…"
            self._people_status.content = "Waiting for aligned YOLO person masks…"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.reset()
            self._frames_root.remove()
            for control in self._controls:
                control.remove()
            self._folder.remove()
            self._people_folder.remove()
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
            position, look_at = self._camera_pose
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = look_at
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.fov = np.deg2rad(55.0)
            self._camera_fitted_clients.add(client.client_id)


def _robust_center(points: np.ndarray) -> np.ndarray:
    low, high = np.percentile(points, (2.0, 98.0), axis=0)
    interior = np.all((points >= low) & (points <= high), axis=1)
    selected = points[interior] if interior.any() else points
    return np.mean(selected, axis=0, dtype=np.float64).astype(np.float32)


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
