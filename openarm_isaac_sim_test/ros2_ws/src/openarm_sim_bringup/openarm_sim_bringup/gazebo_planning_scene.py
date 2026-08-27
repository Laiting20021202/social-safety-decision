from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import rclpy
import yaml
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.duration import Duration
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64, String
from tf2_ros import Buffer, TransformListener


def _project_root() -> Path:
    configured = os.environ.get("OPENARM_SIM_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _rotate(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return rotation @ vector


def _pose(position: np.ndarray, quaternion: np.ndarray) -> Pose:
    result = Pose()
    result.position.x, result.position.y, result.position.z = map(float, position)
    result.orientation.x, result.orientation.y, result.orientation.z, result.orientation.w = map(
        float, quaternion
    )
    return result


def _box(name: str, center: np.ndarray, size: list[float], quaternion: np.ndarray) -> CollisionObject:
    result = CollisionObject()
    result.header.frame_id = "world"
    result.id = name
    result.primitives.append(
        SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(value) for value in size])
    )
    result.primitive_poses.append(_pose(center, quaternion))
    result.operation = CollisionObject.ADD
    return result


class GazeboPlanningScene(Node):
    """Mirror direct Gazebo edits into MoveIt without leaking hand GT in perception mode."""

    def __init__(self) -> None:
        super().__init__("gazebo_planning_scene")
        root = _project_root()
        self._scene = yaml.safe_load((root / "config/scene.yaml").read_text())
        self._hand = yaml.safe_load((root / "config/hand_scenarios.yaml").read_text())[
            "defaults"
        ]
        self._poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._source = "perception"
        self._scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self._hand_pub = self.create_publisher(
            CollisionObject, "/sim/ground_truth/hand_collision", 10
        )
        self._distance_pub = self.create_publisher(
            Float64, "/sim/ground_truth/min_distance", 10
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(ModelStates, "/gazebo/model_states", self._states, 10)
        self.create_subscription(String, "/openarm/obstacle_source", self._source_command, 10)
        self.create_timer(0.5, self._publish_scene)
        self.create_timer(0.05, self._publish_ground_truth)
        self.get_logger().info(
            "MoveIt scene follows direct Gazebo edits; hand GT is isolated by obstacle source"
        )

    def _states(self, message: ModelStates) -> None:
        self._poses = {
            name: (
                np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=float),
                np.asarray(
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                    dtype=float,
                ),
            )
            for name, pose in zip(message.name, message.pose, strict=True)
        }

    def _source_command(self, message: String) -> None:
        source = message.data.strip().lower()
        if source not in {"ground_truth", "perception"}:
            return
        previous = self._source
        self._source = source
        if previous == "ground_truth" and source == "perception":
            remove = CollisionObject()
            remove.header.frame_id = "world"
            remove.id = "ground_truth_hand"
            remove.operation = CollisionObject.REMOVE
            self._scene_pub.publish(
                PlanningScene(is_diff=True, world=__import__(
                    "moveit_msgs.msg", fromlist=["PlanningSceneWorld"]
                ).PlanningSceneWorld(collision_objects=[remove]))
            )

    def _world_pose(
        self, model: str, local_position: list[float], local_quaternion: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        model_pose = self._poses.get(model)
        if model_pose is None:
            return None
        origin, orientation = model_pose
        local_rotation = (
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
            if local_quaternion is None
            else local_quaternion
        )
        return (
            origin + _rotate(np.asarray(local_position, dtype=float), orientation),
            _quaternion_multiply(orientation, local_rotation),
        )

    def _publish_scene(self) -> None:
        if not self._poses:
            return
        objects: list[CollisionObject] = []
        table = self._scene["table"]
        table_yaw = math.radians(float(table.get("yaw_deg", 0.0))) / 2.0
        world_pose = self._world_pose(
            "work_table",
            table["center"],
            np.asarray([0.0, 0.0, math.sin(table_yaw), math.cos(table_yaw)]),
        )
        if world_pose is not None:
            objects.append(_box("gazebo_table", *world_pose[:1], table["size"], world_pose[1]))

        bins = self._scene["bins"]
        if bool(bins.get("enabled", True)):
            inner_x, inner_y, height = map(float, bins["inner_size"])
            wall = float(bins["wall_thickness"])
            base = float(bins["base_thickness"])
            for color, center_values in bins["centers"].items():
                center = np.asarray(center_values, dtype=float)
                pieces = (
                    ("base", [center[0], center[1], center[2] - height / 2 + base / 2], [inner_x + 2 * wall, inner_y + 2 * wall, base]),
                    ("left", [center[0] - inner_x / 2 - wall / 2, center[1], center[2]], [wall, inner_y + 2 * wall, height]),
                    ("right", [center[0] + inner_x / 2 + wall / 2, center[1], center[2]], [wall, inner_y + 2 * wall, height]),
                    ("front", [center[0], center[1] - inner_y / 2 - wall / 2, center[2]], [inner_x, wall, height]),
                    ("back", [center[0], center[1] + inner_y / 2 + wall / 2, center[2]], [inner_x, wall, height]),
                )
                for suffix, local, size in pieces:
                    transformed = self._world_pose(f"{color}_bin", local)
                    if transformed is not None:
                        objects.append(_box(f"gazebo_{color}_bin_{suffix}", transformed[0], size, transformed[1]))

        # In dual fingertip-target mode the two gravity-free cubes are visual
        # goal markers, not obstacles.  This lets each TCP occupy the marker's
        # exact center while every real environment/hand obstacle stays active.
        if not bool(self._scene.get("target_cubes", {}).get("enabled", False)):
            size = float(self._scene["cubes"]["size"])
            try:
                from openarm_sim.scene_model import deterministic_cube_layout

                cubes = deterministic_cube_layout(self._scene)
            except Exception as exc:
                self.get_logger().error(f"cannot build cube planning geometry: {exc}")
                cubes = ()
            for cube in cubes:
                transformed = self._world_pose(cube.name, list(cube.position))
                if transformed is not None:
                    objects.append(_box(f"gazebo_{cube.name}", transformed[0], [size] * 3, transformed[1]))

        hand = self._hand_collision()
        if self._source == "ground_truth" and hand is not None:
            objects.append(hand)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = objects
        self._scene_pub.publish(scene)

    def _hand_collision(self) -> CollisionObject | None:
        model_pose = self._poses.get("human_hand")
        if model_pose is None:
            return None
        center, orientation = model_pose
        proxy = self._hand["collision_proxy"]
        result = CollisionObject()
        result.header.frame_id = "world"
        result.id = "ground_truth_hand"
        result.operation = CollisionObject.ADD
        result.primitives.append(
            SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(map(float, proxy["palm_size"])))
        )
        result.primitive_poses.append(_pose(center, orientation))
        if bool(proxy.get("forearm_enabled", True)):
            length = float(proxy["forearm_length"])
            radius = float(proxy["forearm_radius"])
            result.primitives.append(
                SolidPrimitive(
                    type=SolidPrimitive.CYLINDER,
                    dimensions=[length, radius],
                )
            )
            local_rotation = np.asarray(
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
            )
            forearm_center = center + _rotate(
                np.asarray([0.0, -length / 2.0, 0.0]), orientation
            )
            result.primitive_poses.append(
                _pose(
                    forearm_center,
                    _quaternion_multiply(orientation, local_rotation),
                )
            )
        return result

    def _publish_ground_truth(self) -> None:
        hand = self._hand_collision()
        if hand is None:
            return
        self._hand_pub.publish(hand)
        center = np.asarray(
            [
                hand.primitive_poses[0].position.x,
                hand.primitive_poses[0].position.y,
                hand.primitive_poses[0].position.z,
            ]
        )
        distances = []
        for side in ("left", "right"):
            for index in range(8):
                try:
                    transform = self._tf_buffer.lookup_transform(
                        "world",
                        f"openarm_{side}_link{index}",
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.002),
                    )
                except Exception:
                    continue
                point = transform.transform.translation
                distances.append(np.linalg.norm(center - [point.x, point.y, point.z]))
        clearance = float(min(distances) - 0.09) if distances else 0.0
        self._distance_pub.publish(Float64(data=max(clearance, 0.0)))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboPlanningScene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
