from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _set_simulated_base_pose(robot_description: str, position: list[float]) -> str:
    """Match MoveIt's world-frame robot pose to the Isaac stage."""

    root = ET.fromstring(robot_description)
    base_joint = root.find("./joint[@name='openarm_body_world_joint']")
    if base_joint is None:
        raise RuntimeError("official OpenArm URDF has no openarm_body_world_joint")
    origin = base_joint.find("origin")
    if origin is None:
        origin = ET.SubElement(base_joint, "origin")
    origin.set("xyz", " ".join(str(float(value)) for value in position))
    origin.set("rpy", "0.0 0.0 0.0")
    return ET.tostring(root, encoding="unicode")


def _yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _text(path: str) -> str:
    with open(path, encoding="utf-8") as stream:
        return stream.read()


def _spawn(context):
    arm_type = context.perform_substitution(LaunchConfiguration("arm_type"))
    use_rviz = context.perform_substitution(LaunchConfiguration("use_rviz")).lower() == "true"
    publish_robot_state = LaunchConfiguration("publish_robot_state")
    description_root = get_package_share_directory("openarm_description")
    moveit_root = get_package_share_directory("openarm_bimanual_moveit_config")
    bringup_root = get_package_share_directory("openarm_sim_bringup")
    with open(os.path.join(bringup_root, "config/openarm.yaml"), encoding="utf-8") as stream:
        base_position = yaml.safe_load(stream)["robot"]["base_position"]
    description_file = os.path.join(
        description_root, "assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
    )
    robot_description = _set_simulated_base_pose(
        xacro.process_file(
            description_file,
            mappings={
                "arm_type": arm_type,
                "bimanual": "true",
                "use_fake_hardware": "true",
                "ros2_control": "false",
            },
        ).toprettyxml(indent="  "),
        base_position,
    )
    official_config = os.path.join(moveit_root, "config/openarm_v1.0")
    params = {
        "robot_description": robot_description,
        "robot_description_semantic": _text(
            os.path.join(official_config, "openarm_bimanual.srdf")
        ),
        "robot_description_kinematics": _yaml(
            os.path.join(official_config, "kinematics.yaml")
        ),
        "robot_description_planning": _yaml(
            os.path.join(official_config, "joint_limits.yaml")
        ),
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": _yaml(os.path.join(bringup_root, "config/ompl_planning.yaml")),
        "allow_trajectory_execution": False,
        "planning_scene_monitor.publish_planning_scene": True,
        "planning_scene_monitor.publish_geometry_updates": True,
        "planning_scene_monitor.publish_state_updates": True,
        "planning_scene_monitor.publish_transforms_updates": True,
        "use_sim_time": True,
    }
    # Keep the official controller parameters present because Humble's
    # MoveGroup constructs its trajectory manager even in plan-only mode.
    # openarm_pose_goal still executes through Gazebo's standard action.
    params.update(_yaml(os.path.join(official_config, "moveit_controllers.yaml")))
    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description, "use_sim_time": True}],
            output="screen",
            condition=IfCondition(publish_robot_state),
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            parameters=[params],
            output="screen",
        ),
    ]
    if use_rviz:
        rviz_config = os.path.join(
            bringup_root, "rviz/openarm_safety.rviz"
        )
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                parameters=[params, {"use_sim_time": True}],
                output="screen",
            )
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("arm_type", default_value="v1.0"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("publish_robot_state", default_value="true"),
            OpaqueFunction(function=_spawn),
        ]
    )
