from __future__ import annotations

import json
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


class GazeboScenePoseSync(Node):
    """Make Gazebo's editable model poses authoritative for ROS and the GUI."""

    TRACKED_MODELS = (
        "rgbd_sensor",
        "work_table",
        "apriltag_36h11_0",
        "human_hand",
        "left_target_cube",
        "right_target_cube",
        "openarm",
    )

    def __init__(self) -> None:
        super().__init__("gazebo_scene_pose_sync")
        self._camera_pub = self.create_publisher(
            PoseStamped, "/sim/camera/pose", 10
        )
        self._scene_pub = self.create_publisher(
            String, "/sim/ground_truth/scene_poses", 10
        )
        self._tf = TransformBroadcaster(self)
        self._camera_pose = None
        self._last_scene_publish = 0.0
        self._received = 0
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_states, 10
        )
        # Gazebo ModelStates is intentionally low-rate.  Re-stamp the latest
        # editable camera pose at sensor-rate so 15 Hz RGB-D frames always
        # have a temporally valid transform without freezing camera edits.
        self.create_timer(1.0 / 30.0, self._publish_camera_pose)

    def _on_states(self, message: ModelStates) -> None:
        by_name = dict(zip(message.name, message.pose, strict=True))
        camera = by_name.get("rgbd_sensor")
        if camera is not None:
            self._camera_pose = camera
        now = time.monotonic()
        if now - self._last_scene_publish < 0.1:
            return
        self._last_scene_publish = now
        entities = {}
        for name in self.TRACKED_MODELS:
            pose = by_name.get(name)
            if pose is None:
                continue
            entities[name] = {
                "position": [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ],
                "orientation_xyzw": [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            }
        self._scene_pub.publish(
            String(data=json.dumps({"frame_id": "world", "entities": entities}))
        )
        self._received += 1
        if self._received == 1:
            self.get_logger().info(
                "Gazebo scene poses are authoritative; direct model edits are live"
            )

    def _publish_camera_pose(self) -> None:
        camera = self._camera_pose
        if camera is None:
            return
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "world"
        pose.pose = camera
        self._camera_pub.publish(pose)
        transform = TransformStamped()
        transform.header = pose.header
        transform.child_frame_id = "rgbd_link"
        transform.transform.translation.x = camera.position.x
        transform.transform.translation.y = camera.position.y
        transform.transform.translation.z = camera.position.z
        transform.transform.rotation = camera.orientation
        self._tf.sendTransform(transform)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboScenePoseSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
