from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(name: str, arguments: dict[str, object]):
    path = get_package_share_directory("openarm_sim_bringup") + "/launch/" + name
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments={key: value for key, value in arguments.items()}.items(),
    )


def generate_launch_description() -> LaunchDescription:
    scenario = LaunchConfiguration("scenario")
    headless = LaunchConfiguration("headless")
    use_rviz = LaunchConfiguration("use_rviz")
    auto_start = LaunchConfiguration("auto_start")
    output_root = LaunchConfiguration("output_root")
    robot_model = (
        get_package_share_directory("openarm_description")
        + "/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("scenario", default_value="right_side_sweep"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("auto_start", default_value="true"),
            DeclareLaunchArgument("output_root", default_value="results"),
            _include(
                "simulation.launch.py",
                {"mode": "ground_truth", "scenario": scenario, "headless": headless},
            ),
            _include("moveit_sim.launch.py", {"use_rviz": use_rviz}),
            Node(
                package="openarm_dynamic_avoidance",
                executable="dynamic_avoidance",
                parameters=[
                    {
                        "obstacle_source": "ground_truth",
                        "robot_model": robot_model,
                        "use_sim_time": True,
                    }
                ],
                output="screen",
            ),
            # Keep the perception adapter live so the unified GUI can switch
            # obstacle sources without restarting the visible simulator.
            Node(
                package="openarm_perception_adapter",
                executable="rgbd_mask_adapter",
                parameters=[
                    {
                        "input_mode": "social_cloud",
                        "use_sim_time": True,
                    }
                ],
                output="screen",
            ),
            Node(
                package="openarm_safety_bridge",
                executable="safety_bridge",
                parameters=[{"mode": "ground_truth", "use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="openarm_sim_evaluator",
                executable="evaluator",
                parameters=[
                    {
                        "scenario": scenario,
                        "mode": "ground_truth",
                        "output_root": output_root,
                        "use_sim_time": True,
                    }
                ],
                output="screen",
            ),
            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package="openarm_sorting_task",
                        executable="sorting_task",
                        parameters=[{"auto_start": auto_start, "use_sim_time": True}],
                        output="screen",
                    )
                ],
            ),
        ]
    )
