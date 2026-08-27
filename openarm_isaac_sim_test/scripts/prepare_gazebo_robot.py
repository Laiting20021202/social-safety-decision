#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

from openarm_sim.config import PROJECT_ROOT, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the official OpenArm URDF for Gazebo")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "assets/openarm_cache/openarm_v10_bimanual.generated.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "gazebo/generated/openarm_gazebo.urdf",
    )
    args = parser.parse_args()

    robot_config = load_yaml("config/openarm.yaml")["robot"]
    tree = ET.parse(args.source)
    root = tree.getroot()
    description_root = PROJECT_ROOT / "ros2_ws/src/external/openarm_description"
    if not description_root.is_dir():
        description_root = PROJECT_ROOT / "install/openarm_description/share/openarm_description"
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        prefix = "package://openarm_description/"
        if filename.startswith(prefix):
            asset = (description_root / filename[len(prefix) :]).resolve()
            if not asset.is_file():
                raise FileNotFoundError(f"OpenArm mesh does not exist: {asset}")
            mesh.set("filename", asset.as_uri())
    world_joint = root.find("./joint[@name='openarm_body_world_joint']")
    if world_joint is None:
        raise RuntimeError("official OpenArm URDF has no openarm_body_world_joint")
    origin = world_joint.find("origin")
    if origin is None:
        origin = ET.SubElement(world_joint, "origin")
    origin.set("xyz", " ".join(str(value) for value in robot_config["base_position"]))
    origin.set(
        "rpy",
        " ".join(
            str(float(value) * 3.141592653589793 / 180.0)
            for value in robot_config["base_orientation_rpy_deg"]
        ),
    )

    # Gazebo owns the joints while the fixed body remains at the configured
    # table-side mount. Link gravity is disabled to match the verified Isaac
    # high-PD setup until ros2_control is enabled in Phase 2.
    for link in root.findall("link"):
        name = link.get("name")
        if not name or name == "world":
            continue
        gazebo = ET.SubElement(root, "gazebo", {"reference": name})
        ET.SubElement(gazebo, "gravity").text = "false"
        ET.SubElement(gazebo, "self_collide").text = "true"

    # Avoid the uncontrolled zero-pose flailing seen before the Phase-2
    # ros2_control stack is started. The home bootstrap below sets a safe
    # configuration; damping keeps it there without changing joint limits.
    for joint in root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", "2.0")
        dynamics.set("friction", "0.2")

    # Standard ros2_control interface used by MoveIt and the GUI gateway.
    controlled_joint_names = [
        *(f"openarm_left_joint{index}" for index in range(1, 8)),
        *(f"openarm_right_joint{index}" for index in range(1, 8)),
        "openarm_left_finger_joint1",
        "openarm_right_finger_joint1",
    ]
    ros2_control = ET.SubElement(
        root, "ros2_control", {"name": "GazeboSystem", "type": "system"}
    )
    hardware = ET.SubElement(ros2_control, "hardware")
    ET.SubElement(hardware, "plugin").text = "gazebo_ros2_control/GazeboSystem"
    for name in controlled_joint_names:
        source_joint = root.find(f"./joint[@name='{name}']")
        if source_joint is None:
            raise RuntimeError(f"official OpenArm URDF has no joint {name}")
        joint = ET.SubElement(ros2_control, "joint", {"name": name})
        command = ET.SubElement(joint, "command_interface", {"name": "position"})
        limit = source_joint.find("limit")
        if limit is not None:
            if limit.get("lower") is not None:
                ET.SubElement(command, "param", {"name": "min"}).text = limit.get("lower")
            if limit.get("upper") is not None:
                ET.SubElement(command, "param", {"name": "max"}).text = limit.get("upper")
        position_state = ET.SubElement(
            joint, "state_interface", {"name": "position"}
        )
        # Unclaimed finger interfaces otherwise start as NaN and poison the
        # mimic-finger TF tree before their controller is activated.
        ET.SubElement(position_state, "param", {"name": "initial_value"}).text = (
            "0.04" if "finger_joint" in name else "0.0"
        )
        ET.SubElement(joint, "state_interface", {"name": "velocity"})

    control_gazebo = ET.SubElement(root, "gazebo")
    control_plugin = ET.SubElement(
        control_gazebo,
        "plugin",
        {"name": "gazebo_ros2_control", "filename": "libgazebo_ros2_control.so"},
    )
    ET.SubElement(control_plugin, "parameters").text = str(
        PROJECT_ROOT / "config/gazebo_ros2_controllers.yaml"
    )
    ET.SubElement(control_plugin, "robot_param").text = "robot_description"
    ET.SubElement(control_plugin, "robot_param_node").text = (
        "openarm_robot_state_publisher"
    )

    pose_gazebo = ET.SubElement(root, "gazebo")
    pose_plugin = ET.SubElement(
        pose_gazebo,
        "plugin",
        {
            "name": "openarm_phase1_home_pose",
            "filename": "libgazebo_ros_joint_pose_trajectory.so",
        },
    )
    pose_ros = ET.SubElement(pose_plugin, "ros")
    ET.SubElement(pose_ros, "namespace").text = "/openarm"
    ET.SubElement(pose_ros, "remapping").text = (
        "set_joint_trajectory:=phase1_home_trajectory"
    )
    ET.SubElement(pose_plugin, "update_rate").text = "100"

    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
