import xml.etree.ElementTree as ET
import hashlib
import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
import yaml


def make_nodes(context):
    mock = LaunchConfiguration("mock").perform(context) == "true"
    share = Path(get_package_share_directory("openarm_description"))
    model = xacro.process_file(str(share / "urdf/robot/v10.urdf.xacro"), mappings={
        "bimanual": "true", "ros2_control": "false", "hand": "true",
    })
    robot = ET.fromstring(model.toxml())
    gains = yaml.safe_load((share / "config/arm/v10/control_gains.yaml").read_text())
    for side in ("right", "left"):
        control = ET.SubElement(robot, "ros2_control", name=f"openarm_{side}_hardware", type="system")
        hw = ET.SubElement(control, "hardware")
        ET.SubElement(hw, "plugin").text = (
            "mock_components/GenericSystem" if mock else "openarm_right_curl/RightArm")
        ET.SubElement(hw, "param", name="side").text = side
        profile = LaunchConfiguration("tabletop_profile").perform(context)
        if side == "right" and profile:
            raw = Path(profile).read_bytes()
            survey = json.loads(raw)
            angles = survey.get("active_right_pose", [])
            if (survey.get("method") != "physical_encoders" or survey.get("serial") != "319522065801"
                    or survey.get("parent") != "openarm_body_link0" or len(survey.get("registration_sha256", "")) != 64
                    or len(angles) != 7 or not all(math.isfinite(v) for v in angles)
                    or not -.16 <= angles[0] <= .60 or max(map(abs, angles[1:])) > .16 or angles[3] < -.02):
                raise ValueError("expected a surveyed physical tabletop start pose")
            ET.SubElement(hw, "param", name="tabletop_profile_id").text = hashlib.sha256(raw).hexdigest()
            for i, angle in enumerate(angles, 1):
                ET.SubElement(hw, "param", name=f"tabletop_start{i}").text = str(angle)
        for i in range(1, 8):
            for gain in ("kp", "kd"):
                ET.SubElement(hw, "param", name=f"{gain}{i}").text = str(gains[f"joint{i}"][gain])
        for name in [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]:
            joint = ET.SubElement(control, "joint", name=f"openarm_{side}_{name}")
            if side == "right" and name == "joint4":
                ET.SubElement(joint, "command_interface", name="position")
            interfaces = ["position"] if name == "finger_joint1" else ["position", "velocity", "effort"]
            for interface in interfaces:
                state = ET.SubElement(joint, "state_interface", name=interface)
                if mock:
                    ET.SubElement(state, "param", name="initial_value").text = "0.0"

    description = {"robot_description": ET.tostring(robot, encoding="unicode")}
    app = Path(__file__).parent
    nodes = [
        Node(package="controller_manager", executable="ros2_control_node", output="screen",
             parameters=[description, str(app / "controllers.yaml")], on_exit=Shutdown()),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[description, {"publish_frequency": 30.0}]),
        Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"]),
        Node(package="controller_manager", executable="spawner",
             arguments=["right_joint_trajectory_controller", "--inactive"]),
    ]
    if not mock and LaunchConfiguration("embedded_cameras").perform(context) == "true":
        nodes.extend([
            Node(package="realsense2_camera", executable="realsense2_camera_node", name="d415",
                 namespace="camera", output="screen", parameters=[{
                     "camera_name": "d415", "serial_no": "319522065801",
                     "enable_color": True, "enable_depth": True, "enable_sync": True,
                     "align_depth.enable": True, "rgb_camera.color_profile": "640x480x15",
                     "depth_module.depth_profile": "640x480x15", "rgb_camera.color_format": "RGB8",
                     "enable_infra1": True, "enable_infra2": True,
                     "depth_module.infra_profile": "640x480x15",
                     "color_qos": "SENSOR_DATA", "depth_qos": "SENSOR_DATA",
                     "pointcloud.enable": False,
                 }]),
        ])
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mock", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("tabletop_profile", default_value=""),
        DeclareLaunchArgument("embedded_cameras", default_value="false", choices=["true", "false"]),
        OpaqueFunction(function=make_nodes),
    ])
