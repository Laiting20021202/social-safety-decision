from __future__ import annotations

import pytest

from realtime_safety.ros2_bridge.openarm_control import validate_openarm_command


def test_gui_uses_only_validated_high_level_openarm_commands() -> None:
    assert validate_openarm_command("openarm_pick", "green_cube_2") == (
        "task",
        "pick:green_cube_2",
    )
    assert validate_openarm_command("openarm_home", None) == ("task", "home")
    assert validate_openarm_command("openarm_move_both_targets", None) == (
        "task",
        "move_both_targets",
    )
    assert validate_openarm_command("openarm_move_left_target", None) == (
        "task",
        "move_left_target",
    )
    assert validate_openarm_command("openarm_move_right_target", None) == (
        "task",
        "move_right_target",
    )
    assert validate_openarm_command("openarm_estop", None) == (
        "safety",
        "emergency_stop",
    )
    assert validate_openarm_command("openarm_planner", "dynamic") == (
        "planner",
        "dynamic",
    )
    assert validate_openarm_command("openarm_hand_speed", 0.6) == (
        "hand",
        "speed:0.60",
    )
    assert validate_openarm_command("openarm_hand_speed", 0.05) == (
        "hand",
        "speed:0.05",
    )
    assert validate_openarm_command("openarm_hand_speed", 0.02) == (
        "hand",
        "speed:0.02",
    )
    assert validate_openarm_command("openarm_hand_speed", 0.01) == (
        "hand",
        "speed:0.01",
    )
    assert validate_openarm_command("openarm_hand_auto_sweep", True) == (
        "hand",
        "auto_sweep:on",
    )
    assert validate_openarm_command("openarm_hand_auto_sweep", False) == (
        "hand",
        "auto_sweep:off",
    )
    assert validate_openarm_command("openarm_hand_preview", None) == (
        "hand",
        "perception_preview",
    )
    assert validate_openarm_command("openarm_grasp", "auto") == (
        "grasp",
        "auto",
    )
    assert validate_openarm_command("openarm_hand_target", (-0.1, 0.2, 0.9)) == (
        "hand_pose",
        "-0.100000,0.200000,0.900000",
    )
    with pytest.raises(ValueError, match="Unknown cube"):
        validate_openarm_command("openarm_pick", "red_cube_99")
    with pytest.raises(ValueError, match="Unknown OpenArm GUI command"):
        validate_openarm_command("joint_position", "1.0")
