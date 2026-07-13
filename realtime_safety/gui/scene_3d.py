from __future__ import annotations

import threading
from typing import Any

import numpy as np

from realtime_safety.config import GuiConfig
from realtime_safety.pipeline.local_planner import PlannerResult
from realtime_safety.pipeline.traversable_region import TraversableRegion
from realtime_safety.types import DangerZone, PointCloudFrame, SafetyLevel, Track3DState


class Scene3D:
    """Persistent Viser scene. Updating data never writes client camera state."""

    def __init__(self, server: Any, config: GuiConfig) -> None:
        self.server = server
        self.config = config
        self._lock = threading.Lock()
        self._handles: dict[str, Any] = {}
        self.server.scene.set_up_direction("+z")
        self._handles["ground_grid"] = self.server.scene.add_grid(
            "/world/ground_grid",
            width=12.0,
            height=20.0,
            cell_size=0.5,
            section_size=2.0,
            plane="xy",
            cell_color=(65, 80, 70),
            cell_thickness=0.5,
            section_color=(90, 115, 95),
            section_thickness=1.0,
            position=(0.0, 5.0, -0.02),
        )
        self._handles["robot"] = self.server.scene.add_cylinder(
            "/world/robot_footprint",
            radius=0.35,
            height=0.04,
            color=(0, 230, 230),
            opacity=0.6,
            position=(0.0, 0.0, 0.02),
        )
        empty = np.zeros((0, 3), dtype=np.float32)
        self._handles["point_cloud"] = self.server.scene.add_point_cloud(
            "/world/current_point_cloud",
            points=empty,
            colors=np.zeros((0, 3), dtype=np.uint8),
            point_size=config.point_size,
            point_shape="rounded",
            precision="float16",
        )
        self._handles["tracking_points"] = self.server.scene.add_point_cloud(
            "/world/current_tracking_points",
            points=empty,
            colors=(50, 130, 255),
            point_size=config.point_size * 1.6,
            point_shape="circle",
            precision="float16",
        )

    @property
    def node_count(self) -> int:
        return len(self._handles)

    def update_pointcloud(self, frame: PointCloudFrame) -> None:
        with self._lock:
            point_handle = self._handles["point_cloud"]
            point_handle.points = np.asarray(frame.points, dtype=np.float32)
            point_handle.colors = np.asarray(frame.colors, dtype=np.uint8)
            tracking = frame.tracking_points
            tracking_handle = self._handles["tracking_points"]
            tracking_handle.points = (
                np.asarray(tracking, dtype=np.float32) if tracking is not None else np.zeros((0, 3), dtype=np.float32)
            )

    def set_visibility(self, label: str, visible: bool) -> None:
        mapping = {
            "Point Cloud": "point_cloud",
            "Tracking Points": "tracking_points",
            "Ground Plane": "ground_grid",
        }
        key = mapping.get(label)
        if key in self._handles:
            self._handles[key].visible = visible

    def update_obstacles(self, tracks: list[Track3DState], zones: list[DangerZone]) -> None:
        with self._lock:
            active: set[str] = set()
            zone_by_id = {zone.track_id: zone for zone in zones}
            arrow_segments: list[np.ndarray] = []
            arrow_colors: list[tuple[int, int, int]] = []
            for track in tracks:
                dynamic = track.motion_state == "dynamic"
                bbox_key = f"bbox:{track.track_id}"
                label_key = f"label:{track.track_id}"
                color = (55, 145, 255) if dynamic else (145, 145, 145)
                self._upsert_box(bbox_key, track, color)
                self._upsert_label(label_key, track)
                active.update((bbox_key, label_key))
                speed = float(np.linalg.norm(track.velocity_xyz))
                if dynamic and speed > 1e-4:
                    arrow_segments.append(np.stack((track.position_xyz, track.position_xyz + track.velocity_xyz), axis=0))
                    arrow_colors.append((40, 170, 255))
                if len(track.history) >= 2:
                    history_key = f"history:{track.track_id}"
                    self._upsert_spline(history_key, np.asarray(track.history), (80, 150, 255), 2.0)
                    active.add(history_key)
                zone = zone_by_id.get(track.track_id)
                if zone is not None:
                    danger_key = f"danger:{track.track_id}"
                    self._upsert_danger(danger_key, zone)
                    active.add(danger_key)
                    if len(zone.predicted_positions) >= 2:
                        future_key = f"future:{track.track_id}"
                        self._upsert_spline(future_key, zone.predicted_positions, (255, 65, 55), 3.0)
                        active.add(future_key)
            arrows_key = "velocity_arrows"
            if arrow_segments:
                segments = np.asarray(arrow_segments, dtype=np.float32)
                colors = np.asarray(arrow_colors, dtype=np.uint8)
                if arrows_key not in self._handles:
                    self._handles[arrows_key] = self.server.scene.add_arrows(
                        "/world/velocity_arrows", segments, colors, shaft_radius=0.025, head_radius=0.07, head_length=0.15
                    )
                else:
                    self._handles[arrows_key].points = segments
                    self._handles[arrows_key].colors = colors
                active.add(arrows_key)
            managed_prefixes = ("bbox:", "label:", "history:", "future:", "danger:")
            for key in list(self._handles):
                if (key.startswith(managed_prefixes) or key == arrows_key) and key not in active:
                    self._handles.pop(key).remove()

    def update_navigation(self, traversable: TraversableRegion, planner: PlannerResult) -> None:
        with self._lock:
            active: set[str] = set()
            if len(traversable.polygon_xyz) >= 3:
                key = "traversable_mesh"
                polygon = traversable.polygon_xyz.copy()
                polygon[:, 2] += 0.015
                vertices = np.r_[polygon.mean(axis=0, keepdims=True), polygon]
                faces = np.array(
                    [[0, index + 1, (index + 1) % len(polygon) + 1] for index in range(len(polygon))],
                    dtype=np.uint32,
                )
                if key not in self._handles:
                    self._handles[key] = self.server.scene.add_mesh_simple(
                        "/world/traversable_region", vertices, faces, color=(45, 210, 80), opacity=0.18, side="double", cast_shadow=False
                    )
                else:
                    self._handles[key].vertices = vertices
                    self._handles[key].faces = faces
                active.add(key)
            for index, candidate in enumerate(planner.candidates):
                key = f"candidate:{index}"
                color = (45, 225, 90) if candidate.safe else (235, 45, 45)
                self._upsert_named_spline(key, f"/world/paths/candidate_{index}", candidate.points, color, 2.0)
                active.add(key)
                if candidate.collision_point is not None:
                    collision_key = f"collision:{index}"
                    if collision_key not in self._handles:
                        self._handles[collision_key] = self.server.scene.add_icosphere(
                            f"/world/paths/collision_{index}", radius=0.07, color=(255, 0, 0), position=candidate.collision_point
                        )
                    else:
                        self._handles[collision_key].position = candidate.collision_point
                    active.add(collision_key)
            if planner.selected is not None:
                key = "selected_path"
                self._upsert_named_spline(key, "/world/paths/selected", planner.selected.points, (20, 245, 245), 5.0)
                active.add(key)
            managed = ("candidate:", "collision:")
            for key in list(self._handles):
                if (key.startswith(managed) or key in {"traversable_mesh", "selected_path"}) and key not in active:
                    self._handles.pop(key).remove()

    def _upsert_box(self, key: str, track: Track3DState, color: tuple[int, int, int]) -> None:
        dimensions = np.maximum(track.bbox3d.size, 0.03)
        if key not in self._handles:
            self._handles[key] = self.server.scene.add_box(
                f"/world/obstacles/{track.track_id}/bbox",
                color=color,
                dimensions=dimensions,
                wireframe=True,
                position=track.bbox3d.center,
            )
        else:
            handle = self._handles[key]
            handle.dimensions = dimensions
            handle.position = track.bbox3d.center
            handle.color = color

    def _upsert_label(self, key: str, track: Track3DState) -> None:
        text = f"#{track.track_id} {track.class_name} · {track.motion_state} · {np.linalg.norm(track.velocity_xyz):.2f} rel/s"
        position = track.bbox3d.maximum.copy()
        if key not in self._handles:
            self._handles[key] = self.server.scene.add_label(
                f"/world/obstacles/{track.track_id}/label", text, position=position, anchor="bottom-center"
            )
        else:
            self._handles[key].text = text
            self._handles[key].position = position

    def _upsert_spline(self, key: str, points: np.ndarray, color: tuple[int, int, int], width: float) -> None:
        name = key.split(":", 1)[0]
        track_id = key.split(":", 1)[1]
        points = np.asarray(points, dtype=np.float32)
        if key not in self._handles:
            self._handles[key] = self.server.scene.add_spline_catmull_rom(
                f"/world/obstacles/{track_id}/{name}", points=points, color=color, line_width=width
            )
        else:
            self._handles[key].points = points

    def _upsert_named_spline(
        self, key: str, scene_name: str, points: np.ndarray, color: tuple[int, int, int], width: float
    ) -> None:
        points = np.asarray(points, dtype=np.float32)
        if key not in self._handles:
            self._handles[key] = self.server.scene.add_spline_catmull_rom(
                scene_name, points=points, color=color, line_width=width
            )
        else:
            handle = self._handles[key]
            handle.points = points
            handle.color = color

    def _upsert_danger(self, key: str, zone: DangerZone) -> None:
        vertices, faces = _ellipsoid_series_mesh(zone.predicted_positions, zone.radii)
        track_id = key.split(":", 1)[1]
        color = (255, 25, 25) if zone.risk_level in (SafetyLevel.STOP, SafetyLevel.WARNING) else (255, 125, 30)
        if key not in self._handles:
            self._handles[key] = self.server.scene.add_mesh_simple(
                f"/world/obstacles/{track_id}/danger_zone",
                vertices,
                faces,
                color=color,
                opacity=0.23,
                side="double",
                cast_shadow=False,
            )
        else:
            handle = self._handles[key]
            handle.vertices = vertices
            handle.faces = faces
            handle.color = color

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.remove()
            self._handles.clear()


def _ellipsoid_series_mesh(positions: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Low-poly overlapping ellipsoids form a swept trajectory volume."""
    latitude_count, longitude_count = 5, 8
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for center, radius in zip(np.asarray(positions), np.asarray(radii)):
        base = len(vertices)
        for latitude in range(latitude_count + 1):
            phi = np.pi * latitude / latitude_count
            for longitude in range(longitude_count):
                theta = 2.0 * np.pi * longitude / longitude_count
                unit = np.array(
                    [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), 0.55 * np.cos(phi)]
                )
                vertices.append((center + radius * unit).tolist())
        for latitude in range(latitude_count):
            for longitude in range(longitude_count):
                next_longitude = (longitude + 1) % longitude_count
                a = base + latitude * longitude_count + longitude
                b = base + latitude * longitude_count + next_longitude
                c = base + (latitude + 1) * longitude_count + longitude
                d = base + (latitude + 1) * longitude_count + next_longitude
                faces.extend(([a, c, b], [b, c, d]))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)
