from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_simulator(context: LaunchContext):
    root = Path(
        os.environ.get("OPENARM_SIM_ROOT", get_package_share_directory("openarm_sim_bringup"))
    )
    isaac_python = os.environ.get(
        "ISAAC_SIM_PYTHON",
        str(Path(os.environ.get("ISAAC_SIM_ROOT", "/home/david/isaacsim")) / "python.sh"),
    )
    headless = context.perform_substitution(LaunchConfiguration("headless")).lower() == "true"
    command = [
        isaac_python,
        str(root / "scripts/run_sim.py"),
        "--mode",
        context.perform_substitution(LaunchConfiguration("mode")),
        "--scenario",
        context.perform_substitution(LaunchConfiguration("scenario")),
    ]
    command.append("--headless" if headless else "--no-headless")
    return [ExecuteProcess(cmd=command, output="screen", sigterm_timeout="10")]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="ground_truth"),
            DeclareLaunchArgument("scenario", default_value="no_obstacle"),
            DeclareLaunchArgument("headless", default_value="false"),
            OpaqueFunction(function=_spawn_simulator),
            Node(
                package="hand_obstacle_controller",
                executable="hand_controller",
                parameters=[{"scenario": LaunchConfiguration("scenario"), "use_sim_time": True}],
                output="screen",
            ),
        ]
    )
