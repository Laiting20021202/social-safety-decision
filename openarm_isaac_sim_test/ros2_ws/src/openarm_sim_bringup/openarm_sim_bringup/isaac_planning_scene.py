from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from openarm_sim.config import load_yaml
from openarm_sim.scene_model import workstation_boxes


class IsaacPlanningScene(Node):
    """Install Isaac's fixed workstation geometry in the MoveIt planning scene."""

    def __init__(self) -> None:
        super().__init__("openarm_isaac_planning_scene")
        config = load_yaml("config/scene.yaml")
        world_frame = config["frames"]["world"]
        self.request = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
        for box in workstation_boxes(config):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = box.position.tolist()
            pose.orientation.z = math.sin(box.yaw / 2.0)
            pose.orientation.w = math.cos(box.yaw / 2.0)
            obstacle = CollisionObject(id=f"isaac_{box.name}", operation=CollisionObject.ADD)
            obstacle.header.frame_id = world_frame
            obstacle.primitives = [
                SolidPrimitive(type=SolidPrimitive.BOX, dimensions=box.size.tolist())
            ]
            obstacle.primitive_poses = [pose]
            self.request.scene.world.collision_objects.append(obstacle)
        self.client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.request_in_flight = None
        self.timer = self.create_timer(0.5, self._apply)

    def _apply(self) -> None:
        if self.request_in_flight is None and self.client.service_is_ready():
            self.request_in_flight = self.client.call_async(self.request)
            self.request_in_flight.add_done_callback(self._applied)

    def _applied(self, future) -> None:
        self.request_in_flight = None
        if future.exception() is None and future.result().success:
            self.get_logger().info("Isaac workstation geometry registered in MoveIt")
            self.timer.cancel()
        else:
            self.get_logger().warning("MoveIt did not apply the workstation geometry")


def main() -> None:
    rclpy.init()
    node = IsaacPlanningScene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
