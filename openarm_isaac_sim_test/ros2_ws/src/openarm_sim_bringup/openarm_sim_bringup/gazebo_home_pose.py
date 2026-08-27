from __future__ import annotations

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_HOME = [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
RIGHT_ARM_HOME = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
GRIPPER_OPEN = 0.040
LEFT_ARM_JOINTS = [f"openarm_left_joint{index}" for index in range(1, 8)]
RIGHT_ARM_JOINTS = [f"openarm_right_joint{index}" for index in range(1, 8)]
GRIPPER_JOINTS = [
    "openarm_left_finger_joint1",
    "openarm_right_finger_joint1",
]


class GazeboHomePose(Node):
    """Bootstrap the uncontrolled Gazebo articulation into the safe home pose."""

    def __init__(self) -> None:
        super().__init__("openarm_gazebo_home_pose")
        self._publisher = self.create_publisher(
            JointTrajectory, "/openarm/phase1_home_trajectory", 1
        )
        self._attempts = 0
        self._timer = self.create_timer(0.25, self._publish_home)

    def _publish_home(self) -> None:
        if self._publisher.get_subscription_count() == 0:
            return
        # The parallel grippers are part of the Gazebo articulation even when
        # the pick task is disabled.  Leaving their commanded prismatic joint
        # uninitialised can propagate NaN finger transforms through the mimic
        # joint, eventually starving the RGB-D and planning loops.  Seed both
        # actuated fingers at a valid open position together with the arms.
        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.joint_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + GRIPPER_JOINTS
        point = JointTrajectoryPoint()
        point.positions = ARM_HOME + RIGHT_ARM_HOME + [
            GRIPPER_OPEN,
            GRIPPER_OPEN,
        ]
        message.points = [point]
        self._publisher.publish(message)
        self._attempts += 1
        if self._attempts >= 8:
            self.get_logger().info("OpenArm Phase-1 home pose initialized")
            self.destroy_timer(self._timer)


def main() -> None:
    rclpy.init()
    node = GazeboHomePose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
