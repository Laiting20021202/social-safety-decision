#!/usr/bin/env python3
"""Check the ROS 2 contract of the running Docker Compose Isaac stack."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from openarm_sim.config import load_yaml


class IsaacContractProbe(Node):
    def __init__(self) -> None:
        super().__init__("openarm_isaac_contract_probe")
        self.stamps = defaultdict(lambda: deque(maxlen=40))
        self.frames = {}
        self.joint_names = set()
        self.clock_stamps = deque(maxlen=16)
        self.safety_state = ""
        self.nonempty_obstacles = 0
        self.scene_objects = set()
        self.scene_request = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        robot = load_yaml("config/openarm.yaml")["robot"]
        self.expected_joints = {name for side in robot["joint_names"].values() for name in side}
        self.actions = {
            key: ActionClient(
                self, GripperCommand if key.endswith("_gripper") else FollowJointTrajectory, name
            )
            for key, name in robot["controller_actions"].items()
        }
        self.actions["moveit"] = ActionClient(self, MoveGroup, "/move_action")
        self.scene = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.create_timer(0.5, self._get_scene)
        self.create_subscription(Clock, "/clock", self._record_clock, 10)
        self.create_subscription(JointState, "/joint_states", self._joints, 10)
        self.create_subscription(String, "/openarm/safety/state", self._safety, 10)
        self.create_subscription(
            PointCloud2, "/perception/obstacles", self._obstacle, qos_profile_sensor_data
        )
        for key, kind, topic in (
            ("rgb", Image, "/rgbd/color/image_raw"),
            ("depth", Image, "/rgbd/aligned_depth_to_color/image_raw"),
            ("info", CameraInfo, "/rgbd/color/camera_info"),
        ):
            self.create_subscription(
                kind, topic, lambda msg, name=key: self._camera(name, msg), qos_profile_sensor_data
            )

    def _record_clock(self, msg: Clock) -> None:
        self.clock_stamps.append((msg.clock.sec, msg.clock.nanosec))

    def _joints(self, msg: JointState) -> None:
        if len(msg.position) == len(msg.name):
            self.joint_names = set(msg.name)
            self._record("joints", msg)

    def _safety(self, msg: String) -> None:
        self.safety_state = msg.data

    def _get_scene(self) -> None:
        if self.scene_request is not None and not self.scene_request.done():
            return
        if self.scene_request is not None:
            world = self.scene_request.result().scene.world
            self.scene_objects = {obj.id for obj in world.collision_objects}
        if self.scene.service_is_ready():
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
            self.scene_request = self.scene.call_async(request)

    def _obstacle(self, msg: PointCloud2) -> None:
        self._record("obstacles", msg)
        if msg.width * msg.height > 0:
            self.nonempty_obstacles += 1

    def _camera(self, key: str, msg: Image | CameraInfo) -> None:
        if msg.width <= 0 or msg.height <= 0:
            return
        if key == "rgb" and (msg.encoding not in {"rgb8", "bgr8"} or not len(msg.data)):
            return
        if key == "depth" and (msg.encoding != "32FC1" or not len(msg.data)):
            return
        if key == "info" and (msg.k[0] <= 0 or msg.k[4] <= 0):
            return
        self.frames[key] = msg.header.frame_id
        self._record(key, msg)

    def _record(self, key: str, msg: Image | CameraInfo | JointState | PointCloud2) -> None:
        stamp = msg.header.stamp
        self.stamps[key].append((stamp.sec, stamp.nanosec))

    def checks(self, require_obstacle: bool) -> dict[str, bool]:
        synchronized = set(self.stamps["rgb"]) & set(self.stamps["depth"]) & set(self.stamps["info"])
        nodes = {"openarm_safety_bridge", "openarm_dynamic_avoidance"}
        isolated = nodes.issubset(self.get_node_names())
        if isolated:
            for node in nodes:
                subscriptions = self.get_subscriber_names_and_types_by_node(node, "/")
                isolated &= all(
                    not name.startswith("/sim/ground_truth/") and name != "/openarm/obstacle_source"
                    for name, _ in subscriptions
                )
        return {
            "clock_advances": len(set(self.clock_stamps)) >= 2,
            "joint_feedback": self.expected_joints.issubset(self.joint_names)
            and len(set(self.stamps["joints"])) >= 2,
            "synchronized_metric_rgbd": len(synchronized - {(0, 0)}) >= 2
            and len(set(self.frames.values())) == 1,
            "optical_tf": bool(self.tf_buffer.can_transform(
                "world", self.frames.get("rgb", "rgbd_color_optical_frame"), Time()
            )),
            "perception_observations": len(set(self.stamps["obstacles"])) >= 2,
            "safety_state": bool(self.safety_state),
            "workstation_collision_geometry": {
                "isaac_table", "isaac_yellow_table_edge"
            }.issubset(self.scene_objects),
            "perception_only_control": bool(isolated),
            **{f"action_{name}": client.server_is_ready() for name, client in self.actions.items()},
            **({"nonempty_obstacle": self.nonempty_obstacles > 0} if require_obstacle else {}),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--require-obstacle", action="store_true")
    parser.add_argument("--expect-clear", action="store_true")
    args = parser.parse_args()
    rclpy.init()
    probe = IsaacContractProbe()
    try:
        deadline = time.monotonic() + args.timeout
        checks = {}
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.2)
            checks = probe.checks(args.require_obstacle)
            if args.expect_clear:
                checks["clear_workspace"] = (
                    probe.safety_state == "SAFE" and probe.nonempty_obstacles == 0
                )
            if all(checks.values()):
                break
        passed = bool(checks) and all(checks.values())
        print(json.dumps({"passed": passed, "checks": checks, "safety_state": probe.safety_state,
                          "nonempty_obstacle_messages": probe.nonempty_obstacles}, indent=2))
        return 0 if passed else 1
    finally:
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
