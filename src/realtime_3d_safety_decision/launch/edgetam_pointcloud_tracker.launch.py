from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("realtime_3d_safety_decision"),
            "config",
            "edgetam_pointcloud_tracker.yaml",
        ]
    )
    default_rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("realtime_3d_safety_decision"),
            "rviz",
            "edgetam_pointcloud_tracker.rviz",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    use_edgetam = LaunchConfiguration("use_edgetam")
    use_pointcloud_tracking = LaunchConfiguration("use_pointcloud_tracking")
    publish_debug = LaunchConfiguration("publish_debug")
    use_sim_time = LaunchConfiguration("use_sim_time")
    input_mode = LaunchConfiguration("input_mode")
    publish_legacy_alias = LaunchConfiguration("publish_legacy_alias")
    play_bag = LaunchConfiguration("play_bag")
    bag_path = LaunchConfiguration("bag_path")
    bag_rate = LaunchConfiguration("bag_rate")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    tracker = Node(
        package="realtime_3d_safety_decision",
        executable="edgetam_pointcloud_tracker_node",
        name="edgetam_pointcloud_tracker",
        output="screen",
        emulate_tty=True,
        parameters=[
            config_file,
            {
                "edgetam.enabled": ParameterValue(use_edgetam, value_type=bool),
                "tracking.enabled": ParameterValue(
                    use_pointcloud_tracking, value_type=bool
                ),
                "performance.publish_debug_image": ParameterValue(
                    publish_debug, value_type=bool
                ),
                "performance.publish_debug_cloud": ParameterValue(
                    publish_debug, value_type=bool
                ),
                "performance.publish_markers": ParameterValue(
                    publish_debug, value_type=bool
                ),
                "compatibility.publish_legacy_obstacle_alias": ParameterValue(
                    publish_legacy_alias, value_type=bool
                ),
                "input_mode": ParameterValue(input_mode, value_type=str),
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            },
        ],
    )

    replay = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--clock",
            "--rate",
            bag_rate,
        ],
        output="screen",
        condition=IfCondition(play_bag),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        parameters=[
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)}
        ],
        output="screen",
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument(
                "rviz_config", default_value=default_rviz_config
            ),
            DeclareLaunchArgument(
                "use_edgetam",
                default_value="true",
                description="Enable EdgeTAM refinement; point-cloud fallback remains active.",
            ),
            DeclareLaunchArgument(
                "use_pointcloud_tracking",
                default_value="true",
                description="Enable persistent 3D Kalman/Hungarian tracking.",
            ),
            DeclareLaunchArgument(
                "publish_debug",
                default_value="false",
                description="Publish debug image/cloud/markers.",
            ),
            DeclareLaunchArgument(
                "publish_legacy_alias",
                default_value="false",
                description=(
                    "Opt in to publishing the new cloud on the legacy "
                    "controller topic; keep false while the legacy node runs."
                ),
            ),
            DeclareLaunchArgument(
                "input_mode",
                default_value="live",
                description="Diagnostic label: live, bag, or synthetic.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "play_bag",
                default_value="false",
                description="Start ros2 bag play from this launch.",
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value="",
                description="ROS bag path used only when play_bag:=true.",
            ),
            DeclareLaunchArgument("bag_rate", default_value="1.0"),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="false",
                description="Start RViz with the installed tracker visualization.",
            ),
            tracker,
            replay,
            rviz,
        ]
    )
