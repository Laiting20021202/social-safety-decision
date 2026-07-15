import socket

import numpy as np
import viser

from realtime_safety.config import GuiConfig
from realtime_safety.gui.reconstruction_scene import ReconstructionScene3D, _stable_horizontal_direction
from realtime_safety.types import BBox3D, PointCloudFrame, Track3DState


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cloud(frame_index: int) -> PointCloudFrame:
    rng = np.random.default_rng(frame_index)
    points = rng.normal(size=(200, 3)).astype(np.float32) + np.array((4.0, 8.0, -2.0), np.float32)
    return PointCloudFrame(
        points=points,
        colors=np.full((len(points), 3), 128, np.uint8),
        confidence=np.ones(len(points), np.float32),
        pointmap=points.reshape(10, 20, 3),
        frame_index=frame_index,
        timestamp=float(frame_index),
        anchor_frame_index=0,
        inference_ms=1.0,
        valid=True,
        source="st4rtrack-test",
        tracking_points=points[:20],
    )


def test_reconstruction_history_is_centered_and_bounded() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = ReconstructionScene3D(server, GuiConfig(history_frames=3))
    for frame_index in range(5):
        scene.update_pointcloud(_cloud(frame_index))
    assert scene.frame_count == 3
    assert scene.node_count == 10  # /frames plus three handles per retained timestep
    assert scene._center is not None
    assert np.linalg.norm(scene._center - np.array((4.0, 8.0, -2.0))) < 0.5
    center = np.array((4.0, 8.0, -2.0), dtype=np.float32)
    scene.update_people(
        4,
        [
            Track3DState(
                track_id=7,
                class_name="person",
                position_xyz=center,
                velocity_xyz=np.array((0.3, 0.1, 0.0), dtype=np.float32),
                acceleration_xyz=np.zeros(3, dtype=np.float32),
                covariance=np.eye(6),
                bbox3d=BBox3D(center - 0.2, center + 0.2),
                radius=0.3,
                hit_count=4,
                missing_count=0,
                last_timestamp=4.0,
                motion_state="dynamic",
                confidence=0.9,
                history=[
                    center + np.array((-0.09, -0.03, 0.0), dtype=np.float32),
                    center + np.array((-0.06, -0.02, 0.0), dtype=np.float32),
                    center + np.array((-0.03, -0.01, 0.0), dtype=np.float32),
                    center,
                ],
            )
        ],
    )
    assert scene.node_count == 13  # box, robust center marker, and direction-arrow handle
    scene.reset()
    assert scene.frame_count == 0
    assert scene.node_count == 1
    scene.close()
    server.stop()


def test_direction_is_hidden_for_oscillating_person_centers() -> None:
    center = np.zeros(3, dtype=np.float32)
    track = Track3DState(
        track_id=3,
        class_name="person",
        position_xyz=center,
        velocity_xyz=np.array((4.0, 0.0, 0.0), dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        bbox3d=BBox3D(center - 0.2, center + 0.2),
        radius=0.3,
        hit_count=6,
        missing_count=0,
        last_timestamp=1.0,
        motion_state="dynamic",
        confidence=0.9,
        history=[
            np.array((0.00, 0.00, 0.0), dtype=np.float32),
            np.array((0.08, 0.00, 0.0), dtype=np.float32),
            np.array((0.00, 0.00, 0.0), dtype=np.float32),
            np.array((0.08, 0.00, 0.0), dtype=np.float32),
            np.array((0.00, 0.00, 0.0), dtype=np.float32),
            np.array((0.08, 0.00, 0.0), dtype=np.float32),
        ],
    )
    assert _stable_horizontal_direction(track, np.full(3, 0.4, dtype=np.float32)) is None


def test_live_mode_reuses_one_persistent_webgl_pointcloud() -> None:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    scene = ReconstructionScene3D(server, GuiConfig(history_frames=1, history_stride=1))
    scene.update_aligned_frame(_cloud(0), [], yolo_count=0)
    first_handles = next(iter(scene._frames.values()))
    reconstruction_id = id(first_handles["reconstruction"])
    root_id = id(first_handles["root"])
    for frame_index in range(1, 8):
        scene.update_aligned_frame(_cloud(frame_index), [], yolo_count=0)
    current_handles = next(iter(scene._frames.values()))
    assert scene.frame_count == 1
    assert id(current_handles["reconstruction"]) == reconstruction_id
    assert id(current_handles["root"]) == root_id
    assert current_handles["root"].visible
    scene.close()
    server.stop()
