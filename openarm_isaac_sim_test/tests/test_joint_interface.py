from pathlib import Path

from openarm_sim.config import load_yaml


def test_bimanual_joint_and_action_contract() -> None:
    robot = load_yaml("config/openarm.yaml")["robot"]
    left = robot["joint_names"]["left"]
    right = robot["joint_names"]["right"]
    assert robot["bimanual"] is True
    assert robot["enabled_self_collisions"] is True
    assert len(left) == len(right) == 7
    assert len(set(left + right)) == 14
    assert robot["controller_actions"]["left"].endswith("/follow_joint_trajectory")
    assert robot["controller_actions"]["right"].endswith("/follow_joint_trajectory")


def test_local_official_urdf_contains_configured_joints() -> None:
    robot = load_yaml("config/openarm.yaml")["robot"]
    root = Path(robot["asset"]["description_root_candidates"][0])
    generated = root / robot["asset"]["generated_urdf"]
    if not generated.is_file():
        return
    text = generated.read_text(encoding="utf-8")
    for name in robot["joint_names"]["left"] + robot["joint_names"]["right"]:
        assert f'name="{name}"' in text


def test_gazebo_home_bootstrap_initializes_both_grippers() -> None:
    from openarm_sim_bringup.gazebo_home_pose import (
        ARM_HOME,
        GRIPPER_JOINTS,
        GRIPPER_OPEN,
        LEFT_ARM_JOINTS,
        RIGHT_ARM_HOME,
        RIGHT_ARM_JOINTS,
    )

    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + GRIPPER_JOINTS
    positions = ARM_HOME + RIGHT_ARM_HOME + [GRIPPER_OPEN, GRIPPER_OPEN]
    assert len(names) == len(positions) == 16
    assert GRIPPER_JOINTS == [
        "openarm_left_finger_joint1",
        "openarm_right_finger_joint1",
    ]
    assert all(0.0 <= value <= 0.044 for value in positions[-2:])
