from __future__ import annotations

import socket
from types import SimpleNamespace

import numpy as np
import viser

from realtime_safety.config import OpenArmConfig, load_config
from realtime_safety.gui.openarm_scene import (
    OpenArmScene,
    map_openarm_joint_positions,
)
from realtime_safety.gui.metric_bev import MetricBevCalibration
from realtime_safety.ros2_bridge.openarm_joint_state import OpenArmJointStateBridge


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_openarm_joint_mapping_accepts_official_and_controller_aliases() -> None:
    targets = tuple(f"openarm_joint{index}" for index in range(1, 8)) + (
        "openarm_finger_joint1",
    )
    names = tuple(f"right_joint{index}" for index in range(1, 8)) + (
        "finger_joint1",
    )
    positions = tuple(float(index) / 10.0 for index in range(8))

    mapped = map_openarm_joint_positions(targets, names, positions)

    assert mapped["openarm_joint1"] == 0.0
    assert mapped["openarm_joint7"] == 0.6
    assert mapped["openarm_finger_joint1"] == 0.7
    assert map_openarm_joint_positions(targets, ("unknown",), (1.0,)) == {}
    assert map_openarm_joint_positions(targets, ("openarm_joint1",), ()) == {}

    bimanual_targets = (
        "openarm_left_joint1",
        "openarm_right_joint1",
        "openarm_left_finger_joint1",
        "openarm_right_finger_joint1",
    )
    bimanual = map_openarm_joint_positions(
        bimanual_targets,
        ("left_joint1", "right_joint1", "left_finger_joint1", "right_finger_joint1"),
        (0.1, -0.2, 0.3, 0.4),
    )
    assert bimanual == {
        "openarm_left_joint1": 0.1,
        "openarm_right_joint1": -0.2,
        "openarm_left_finger_joint1": 0.3,
        "openarm_right_finger_joint1": 0.4,
    }


def test_koch_profile_enables_official_openarm_v1_bimanual_body() -> None:
    config = load_config("koch_lan")

    assert config.openarm.enabled
    assert config.openarm.model == "openarm_v1.0_bimanual"
    assert config.openarm.joint_states_topic == "/joint_states"
    assert config.openarm.base_anchor == "apriltag"
    assert tuple(config.openarm.base_from_apriltag_xyz) == (0.0, -0.5, 0.0)
    assert config.reconstruction.apriltag_enabled
    assert config.reconstruction.apriltag_size_m == 0.08


def test_joint_state_bridge_forwards_names_positions_and_stamp() -> None:
    calls: list[tuple] = []

    def on_state(names, positions, **kwargs):
        calls.append((names, positions, kwargs))
        return len(names)

    bridge = OpenArmJointStateBridge("/joint_states", on_state)
    message = SimpleNamespace(
        name=["openarm_joint1", "openarm_joint2"],
        position=[0.25, -0.5],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000)),
    )

    bridge._on_joint_state(message)

    assert bridge.message_count == 1
    assert calls[0][0] == ("openarm_joint1", "openarm_joint2")
    assert calls[0][1] == (0.25, -0.5)
    assert calls[0][2]["header_stamp"] == 12.5


def test_openarm_scene_updates_fk_from_joint_state(tmp_path) -> None:
    urdf = tmp_path / "minimal_openarm.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="openarm_test">
  <link name="openarm_link0"/>
  <link name="openarm_link1"/>
  <joint name="openarm_joint1" type="revolute">
    <parent link="openarm_link0"/><child link="openarm_link1"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
  </joint>
  <link name="openarm_hand_tcp"/>
  <joint name="openarm_hand_tcp_joint" type="fixed">
    <parent link="openarm_link1"/><child link="openarm_hand_tcp"/>
    <origin xyz="0 0 0.3"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    config = OpenArmConfig(
        enabled=True,
        urdf_path=str(urdf),
        description_path=str(tmp_path),
        base_position_xyz=(0.1, -0.2, 0.0),
    )
    server = viser.ViserServer(
        host="127.0.0.1", port=_free_port(), verbose=False
    )
    scene = OpenArmScene(server, config)
    before = scene.tcp_position

    matched = scene.update_joint_state(("joint1",), (1.0,))
    after = scene.tcp_position

    assert scene.loaded
    assert matched == 1
    assert scene.matched_main_joints == 1
    assert before is not None and after is not None
    assert not np.allclose(before, after)

    calibration = MetricBevCalibration(
        origin=np.array((0.0, 0.4, -0.5), dtype=np.float32),
        right=np.array((1.0, 0.0, 0.0), dtype=np.float32),
        forward=np.array((0.0, 1.0, 0.0), dtype=np.float32),
        normal=np.array((0.0, 0.0, 1.0), dtype=np.float32),
        bounds_uv=(-0.6, 0.6, -0.6, 0.6),
        inlier_count=500,
        inlier_ratio=0.8,
        rms_error_m=0.005,
    )
    scene.set_spatial_context(
        center=np.zeros(3), bev_enabled=True, calibration=calibration
    )
    # This minimal fixture has one physical joint. Mark the seven-axis
    # completeness gate as satisfied to exercise the controller-facing state
    # conversion independently from the official asset smoke above.
    scene._matched_main_joints = 7
    state = scene.robot_arm_state(timestamp=123.0)
    assert state is not None
    assert state.localization_source == "urdf_fk_joint_state"
    assert state.timestamp == 123.0
    assert np.isfinite(state.center_xyz).all()
    scene.close()
    server.stop()
