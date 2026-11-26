#!/usr/bin/env python3
"""Workstation ROS 2 client. Perform one empty-hand curl and measure the return."""
import argparse
import csv
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

CONTROLLER = "right_joint_trajectory_controller"
ELBOW = "openarm_right_joint4"
JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]


class Curl(Node):
    def __init__(self):
        super().__init__("workstation_right_curl")
        self.rows = []
        self.last_stamp = 0
        self.last_arrival = 0.0
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.create_subscription(JointState, "/joint_states", self.on_state, qos_profile_sensor_data)
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self.switch = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self.action = ActionClient(self, FollowJointTrajectory, f"/{CONTROLLER}/follow_joint_trajectory")

    def on_state(self, message):
        stamp = Time.from_msg(message.header.stamp).nanoseconds
        age = (self.get_clock().now().nanoseconds - stamp) / 1e9
        if stamp <= self.last_stamp or not -0.1 <= age <= 0.5:
            return
        positions = dict(zip(message.name, message.position))
        if not all(math.isfinite(positions.get(joint, math.nan)) for joint in JOINTS):
            return
        try:
            transform = self.tf.lookup_transform("world", "openarm_right_hand_tcp", Time())
        except TransformException:
            return
        tf_stamp = Time.from_msg(transform.header.stamp).nanoseconds
        if abs(tf_stamp - stamp) > 200_000_000:
            return
        xyz = transform.transform.translation
        self.last_stamp = stamp
        self.last_arrival = time.monotonic()
        self.rows.append((stamp, *(positions[joint] for joint in JOINTS), xyz.x, xyz.y, xyz.z))

    def wait(self, predicate, timeout, reason, *, watch_state=False):
        end = time.monotonic() + timeout
        while rclpy.ok() and not predicate():
            if time.monotonic() > end:
                raise RuntimeError(reason)
            rclpy.spin_once(self, timeout_sec=0.05)
            if watch_state and time.monotonic() - self.last_arrival > 0.5:
                raise RuntimeError("Lost live encoder/TF feedback on the workstation")

    def result(self, future, timeout, reason, *, watch_state=False):
        self.wait(future.done, timeout, reason, watch_state=watch_state)
        return future.result()

    def set_active(self, active):
        request = SwitchController.Request()
        request.strictness = SwitchController.Request.STRICT
        request.timeout = Duration(sec=3)
        if active:
            request.activate_controllers = [CONTROLLER]
        else:
            request.deactivate_controllers = [CONTROLLER]
        response = self.result(self.switch.call_async(request), 6, "Controller switch timed out")
        if not response.ok:
            raise RuntimeError("Controller switch rejected by the NUC")

    def run(self):
        self.wait(lambda: len(self.rows) >= 8, 12, "No fresh encoder data and hand TF")
        if not self.controllers.wait_for_service(timeout_sec=4) or not self.switch.wait_for_service(timeout_sec=4):
            raise RuntimeError("NUC controller manager unavailable")
        controllers = self.result(self.controllers.call_async(ListControllers.Request()), 4,
                                  "No controller inventory").controller
        if {c.name: c.state for c in controllers}.get(CONTROLLER) != "inactive":
            raise RuntimeError("Right controller must be inactive before the test")
        base = self.rows[-1]
        q0 = base[4]
        if max(abs(value) for value in base[1:8]) > 0.16 or q0 < -0.02:
            raise RuntimeError("Current right arm posture is outside the starting envelope")
        rest = max(0.0, q0)
        peak = q0 + math.radians(25)
        print(f"Measured start {math.degrees(q0):.2f} deg; target {math.degrees(peak):.2f} deg", flush=True)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [ELBOW]
        for seconds, position in [(1, rest), (8, peak), (10, peak), (17, rest)]:
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=[position], velocities=[0.0], time_from_start=Duration(sec=seconds)))
        handle = None
        completed = False
        first = len(self.rows)
        self.set_active(True)
        try:
            if not self.action.wait_for_server(timeout_sec=4):
                raise RuntimeError("No trajectory action server")
            handle = self.result(self.action.send_goal_async(goal), 4, "No goal acknowledgement")
            if not handle.accepted:
                raise RuntimeError("NUC rejected the trajectory")
            result = self.result(handle.get_result_async(), 23, "Movement timed out", watch_state=True)
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
                raise RuntimeError(f"Action status {result.status}: {result.result.error_string}")
            top = max(self.rows[first:], key=lambda row: row[4])
            end = self.rows[-1]
            if abs(top[4] - peak) > 0.045 or abs(end[4] - rest) > 0.035:
                raise RuntimeError("Encoder measurements did not confirm both lift and return")
            print(f"Encoder start / peak / return: {math.degrees(q0):.2f} / "
                  f"{math.degrees(top[4]):.2f} / {math.degrees(end[4]):.2f} deg")
            print(f"Hand TCP height change: {100 * (max(row[-1] for row in self.rows[first:]) - base[-1]):.2f} cm; "
                  f"{len(self.rows) - first} measured samples")
            completed = True
        finally:
            if handle is not None and handle.accepted and not completed:
                self.result(handle.cancel_goal_async(), 3, "No cancellation acknowledgement")
            if (time.monotonic() - self.last_arrival < 0.5 and
                    all(abs(q - ref) < 0.05 for q, ref in zip(self.rows[-1][1:8], base[1:8]))):
                self.set_active(False)
            else:
                self.get_logger().error("Arm away from its rest pose or feedback lost. Check onsite before releasing support.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Record measured right joints (rad) and TCP position (m)")
    args = parser.parse_args()
    rclpy.init()
    node = Curl()
    try:
        node.run()
    finally:
        if args.csv:
            with open(args.csv, "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["stamp_ns", *JOINTS, "tcp_x_m", "tcp_y_m", "tcp_z_m"])
                writer.writerows(node.rows)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
