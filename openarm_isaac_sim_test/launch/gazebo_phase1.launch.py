from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("openarm_sim_bringup"))
    world = share / "gazebo/worlds/openarm_sorting.world"
    urdf = (share / "gazebo/generated/openarm_gazebo.urdf").read_text()
    if urdf.lstrip().startswith("<?xml"):
        urdf = urdf.split("?>", 1)[1]
    scene = yaml.safe_load((share / "config/scene.yaml").read_text())
    camera = yaml.safe_load((share / "config/camera.yaml").read_text())["camera"]
    workspace = np.asarray(scene["zones"]["workspace"]["center"], dtype=float)
    explicit_pose = camera.get("world_pose")
    if explicit_pose:
        position = np.asarray(explicit_pose["position"], dtype=float)
        roll, pitch, yaw = (
            math.radians(float(value)) for value in explicit_pose["rpy_deg"]
        )
    else:
        target = workspace + np.asarray(
            camera.get("aim_offset", [0.0, 0.0, 0.0]), dtype=float
        )
        position = workspace + np.array(
            [
                -float(camera["horizontal_offset_to_workspace_center"]),
                float(camera["lateral_offset"]),
                float(camera["height_above_table"]),
            ]
        )
        direction = target - position
        direction /= np.linalg.norm(direction)
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        pitch = math.atan2(float(-direction[2]), float(np.linalg.norm(direction[:2])))
        roll = 0.0

    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            ExecuteProcess(
                cmd=[
                    "gazebo", "--verbose",
                    "-s", "libgazebo_ros_init.so",
                    "-s", "libgazebo_ros_factory.so",
                    str(world),
                ],
                output="screen",
                condition=IfCondition(gui),
                additional_env={"GAZEBO_MODEL_DATABASE_URI": ""},
            ),
            ExecuteProcess(
                cmd=[
                    "gzserver", "--verbose",
                    "-s", "libgazebo_ros_init.so",
                    "-s", "libgazebo_ros_factory.so",
                    str(world),
                ],
                output="screen",
                condition=UnlessCondition(gui),
                additional_env={"GAZEBO_MODEL_DATABASE_URI": ""},
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="openarm_robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": urdf, "use_sim_time": True}],
            ),
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package="gazebo_ros",
                        executable="spawn_entity.py",
                        arguments=[
                            "-entity",
                            "openarm",
                            "-topic",
                            "/robot_description",
                        ],
                        output="screen",
                    )
                ],
            ),
            TimerAction(
                period=3.0,
                actions=[
                    Node(
                        package="openarm_sim_bringup",
                        executable="gazebo_home_pose",
                        output="screen",
                        parameters=[{"use_sim_time": True}],
                    )
                ],
            ),
            TimerAction(
                period=6.0,
                actions=[
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "joint_state_broadcaster",
                            "--controller-manager",
                            "/controller_manager",
                            "--controller-manager-timeout",
                            "60",
                        ],
                        output="screen",
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "left_joint_trajectory_controller",
                            "--controller-manager",
                            "/controller_manager",
                            "--controller-manager-timeout",
                            "60",
                        ],
                        output="screen",
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "right_joint_trajectory_controller",
                            "--controller-manager",
                            "/controller_manager",
                            "--controller-manager-timeout",
                            "60",
                        ],
                        output="screen",
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "left_gripper_controller",
                            "--controller-manager",
                            "/controller_manager",
                            "--controller-manager-timeout",
                            "60",
                        ],
                        output="screen",
                    ),
                    Node(
                        package="controller_manager",
                        executable="spawner",
                        arguments=[
                            "right_gripper_controller",
                            "--controller-manager",
                            "/controller_manager",
                            "--controller-manager-timeout",
                            "60",
                        ],
                        output="screen",
                    ),
                ],
            ),
            TimerAction(
                period=9.0,
                actions=[
                    Node(
                        package="openarm_sorting_task",
                        executable="pose_goal",
                        output="screen",
                        parameters=[
                            {"planner_mode": "dynamic", "use_sim_time": True}
                        ],
                    )
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(share / "launch/moveit_sim.launch.py")),
                launch_arguments={
                    "use_rviz": "false",
                    "publish_robot_state": "false",
                }.items(),
            ),
            Node(
                package="openarm_sim_bringup",
                executable="gazebo_scene_pose_sync",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="rgbd_link_to_color_optical",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--qx", "-0.5", "--qy", "0.5", "--qz", "-0.5", "--qw", "0.5",
                    "--frame-id", "rgbd_link", "--child-frame-id", "rgbd_color_optical_frame",
                ],
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="color_to_depth_optical",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                    "--frame-id", "rgbd_color_optical_frame",
                    "--child-frame-id", "rgbd_depth_optical_frame",
                ],
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="openarm_sim_bringup",
                executable="gazebo_rgbd_adapter",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="openarm_sim_bringup",
                executable="gazebo_planning_scene",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="openarm_dynamic_avoidance",
                executable="dynamic_avoidance",
                output="screen",
                parameters=[
                    {
                        "obstacle_source": "perception",
                        "guarded_route_velocity_scale": 0.50,
                        # A 1 cm/s hand otherwise moves for 15 seconds before
                        # the route is refreshed.  Replan on meaningful cloud
                        # motion while retaining a short anti-jitter cooldown.
                        "obstacle_motion_replan_m": 0.04,
                        "replan_cooldown_sec": 0.6,
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="openarm_perception_adapter",
                executable="realtime_obstacle_resampler",
                output="screen",
                parameters=[
                    {
                        # Follow the 8080 GUI-selected EdgeTAM or MediaPipe
                        # RGB-D source through the validated obstacle mux.
                        "model_cloud_topic": "/realtime_safety/yolo_obstacles/pointcloud",
                        "rgb_topic": "/rgbd/color/image_raw",
                        "rgbd_cloud_topic": "/rgbd/points",
                        "output_cloud_topic": "/edgetam_tracker/obstacle_cloud_realtime",
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="openarm_perception_adapter",
                executable="rgbd_mask_adapter",
                output="screen",
                parameters=[
                    {
                        "input_mode": "social_cloud",
                        # The legacy alias is intentionally disabled by the
                        # social-safety runtime.  Consume EdgeTAM's formal
                        # model output instead of waiting on a dead topic.
                        "obstacle_cloud_topic": "/edgetam_tracker/obstacle_cloud_realtime",
                        "allow_hsv_placeholder": False,
                        # Match EdgeTAM's 3.0 s stale-track window.  This is a
                        # liveness timeout, not simulated obstacle data.
                        "timeout_sec": 3.2,
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="openarm_safety_bridge",
                executable="safety_bridge",
                output="screen",
                parameters=[
                    {
                        "mode": "perception",
                        "startup_grace_sec": 30.0,
                        "prediction_horizon_sec": 2.0,
                        "maximum_obstacle_speed_mps": 0.20,
                        "obstacle_center_jump_slack_m": 0.02,
                        "use_sim_time": True,
                    }
                ],
            ),
            TimerAction(
                period=3.0,
                actions=[
                    Node(
                        package="openarm_sim_bringup",
                        executable="gazebo_hand_controller",
                        output="screen",
                        parameters=[{"use_sim_time": True}],
                    )
                ],
            ),
            TimerAction(
                period=4.0,
                actions=[
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        name="openarm_gazebo_rviz",
                        arguments=["-d", str(share / "rviz/openarm_gazebo_phase1.rviz")],
                        parameters=[{"use_sim_time": True}],
                        output="screen",
                        condition=IfCondition(rviz),
                    )
                ],
            ),
        ]
    )
