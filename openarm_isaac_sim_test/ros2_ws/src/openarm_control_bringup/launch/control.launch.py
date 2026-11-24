from __future__ import annotations

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_nodes(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    simulated = value("backend") == "isaac"
    perception = value("perception")
    start_task = value("start_task") == "true" or (simulated and value("start_task") == "auto")
    share = Path(get_package_share_directory("openarm_sim_bringup"))
    config_overrides = {}
    for name in ("safety_config", "robot_config", "task_config"):
        override = value(name)
        config_overrides[name] = "" if override == "default" else override
    if not simulated:
        if perception == "demo_rgbd":
            raise ValueError("demo_rgbd is only available with the Isaac backend")
        if start_task and not all(config_overrides.values()):
            raise ValueError("Hardware motion requires measured safety, robot and task configurations")

    safety_config = config_overrides["safety_config"] or str(share / "config/safety_zones.yaml")
    robot_config = config_overrides["robot_config"] or str(share / "config/openarm.yaml")
    task_config = config_overrides["task_config"] or str(share / "config/sorting_task.yaml")
    with open(safety_config, encoding="utf-8") as stream:
        cloud_input = yaml.safe_load(stream)["topics"]["perception_cloud"]
    if cloud_input != "/perception/obstacles":
        raise ValueError("The shared control pipeline reads /perception/obstacles")
    timing = {"use_sim_time": simulated}
    nodes = []
    if simulated:
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(share / "launch/moveit_sim.launch.py")),
                launch_arguments={"use_rviz": "false"}.items(),
            )
        )
        nodes.append(
            Node(
                package="openarm_sim_bringup",
                executable="isaac_planning_scene",
                output="screen",
                parameters=[timing],
            )
        )

    if perception != "direct":
        cloud_topic = value("obstacle_cloud_topic")
        if perception == "cloud" and (
            not cloud_topic.startswith("/")
            or cloud_topic.startswith("/sim/")
            or cloud_topic == "/perception/obstacles"
        ):
            raise ValueError("Set an absolute measured-cloud input distinct from /perception/obstacles")
        nodes.append(
            Node(
                package="openarm_perception_adapter",
                executable="rgbd_mask_adapter",
                output="screen",
                parameters=[
                    timing,
                    {
                        "input_mode": "social_cloud" if perception == "cloud" else "mask",
                        "obstacle_cloud_topic": cloud_topic,
                        "demo_marker_mask": perception == "demo_rgbd",
                    },
                ],
            )
        )

    nodes.extend(
        [
            Node(
                package="openarm_safety_bridge",
                executable="safety_bridge",
                output="screen",
                parameters=[
                    timing,
                    {
                        "mode": "perception",
                        "allow_obstacle_source_switch": False,
                        "require_fresh_perception": True,
                        "config": safety_config,
                    },
                ],
            ),
            Node(
                package="openarm_dynamic_avoidance",
                executable="dynamic_avoidance",
                output="screen",
                parameters=[
                    timing,
                    {
                        "obstacle_source": "perception",
                        "allow_obstacle_source_switch": False,
                    },
                ],
            ),
        ]
    )
    if start_task:
        nodes.append(
            Node(
                package="openarm_sorting_task",
                executable="pose_goal",
                output="screen",
                parameters=[
                    timing,
                    {
                        "target_source": "pose",
                        "planner_mode": "dynamic",
                        "config": task_config,
                        "robot_config": robot_config,
                    },
                ],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="isaac", choices=["isaac", "hardware"]),
            DeclareLaunchArgument(
                "perception", default_value="cloud", choices=["demo_rgbd", "mask", "cloud", "direct"]
            ),
            DeclareLaunchArgument("start_task", default_value="auto", choices=["auto", "true", "false"]),
            DeclareLaunchArgument(
                "obstacle_cloud_topic", default_value="/edgetam_tracker/obstacle_cloud"
            ),
            DeclareLaunchArgument("safety_config", default_value="default"),
            DeclareLaunchArgument("robot_config", default_value="default"),
            DeclareLaunchArgument("task_config", default_value="default"),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
