import socket

import numpy as np
import viser

from realtime_safety.config import GuiConfig
from realtime_safety.gui.scene_3d import Scene3D
from realtime_safety.pipeline.local_planner import PlannerResult
from realtime_safety.pipeline.traversable_region import TraversableRegion
from realtime_safety.types import BBox3D, DangerZone, PathCandidate, PointCloudFrame, RecommendedAction, SafetyLevel, Track3DState


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_current_nodes_do_not_grow_with_frames() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = Scene3D(server, GuiConfig())
    initial_count = scene.node_count
    for frame_index in range(20):
        count = frame_index + 3
        pointmap = np.zeros((2, 2, 3), dtype=np.float32)
        scene.update_pointcloud(
            PointCloudFrame(
                points=np.zeros((count, 3), dtype=np.float32),
                colors=np.zeros((count, 3), dtype=np.uint8),
                confidence=np.ones(count, dtype=np.float32),
                pointmap=pointmap,
                frame_index=frame_index,
                timestamp=float(frame_index),
                anchor_frame_index=0,
                inference_ms=1.0,
                valid=True,
                source="test",
            )
        )
    assert scene.node_count == initial_count
    scene.close()
    server.stop()


def test_obstacle_nodes_are_reused_and_old_tracks_removed() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = Scene3D(server, GuiConfig())
    position = np.array([0.0, 2.0, 0.5], dtype=np.float32)
    track = Track3DState(
        track_id=3,
        class_name="person",
        position_xyz=position,
        velocity_xyz=np.array([0.1, -0.2, 0.0], dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=BBox3D(position - 0.2, position + 0.2),
        radius=0.3,
        hit_count=5,
        missing_count=0,
        last_timestamp=1.0,
        motion_state="dynamic",
        confidence=0.9,
        history=[position.copy(), position + 0.1],
    )
    positions = np.stack([position, position + track.velocity_xyz], axis=0)
    zone = DangerZone(3, positions, np.array([0.6, 0.7]), track.velocity_xyz, 0.22, 0.8, 0.5, None, SafetyLevel.WARNING, True)
    for _ in range(10):
        scene.update_obstacles([track], [zone])
    active_count = scene.node_count
    scene.update_obstacles([], [])
    assert scene.node_count < active_count
    assert scene.node_count == 4
    scene.close()
    server.stop()


def test_navigation_nodes_are_bounded() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = Scene3D(server, GuiConfig())
    polygon = np.array([[-2, 0, 0], [2, 0, 0], [2, 3, 0], [-2, 3, 0]], dtype=np.float32)
    region = TraversableRegion(polygon, polygon, 0.9)
    points = np.array([[0, 0, 0.03], [0, 1, 0.03], [0, 2, 0.03]], dtype=np.float32)
    candidate = PathCandidate(points, True, 1.0, "center")
    planner = PlannerResult([candidate], candidate, RecommendedAction.CONTINUE)
    for _ in range(20):
        scene.update_navigation(region, planner)
    assert scene.node_count == 7
    scene.update_navigation(TraversableRegion(np.zeros((0, 3)), np.zeros((0, 3)), 0.0), PlannerResult([], None, RecommendedAction.STOP))
    assert scene.node_count == 4
    scene.close()
    server.stop()
