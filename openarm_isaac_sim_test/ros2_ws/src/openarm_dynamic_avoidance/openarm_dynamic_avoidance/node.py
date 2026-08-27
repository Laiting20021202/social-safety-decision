from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64, Header, String
from trajectory_msgs.msg import JointTrajectory
from tf2_ros import Buffer, TransformListener

from .policy import (
    BLOCKING_STATES,
    effective_velocity_scale,
    hold_recent_obstacle,
    obstacle_affects_motion_corridor,
    retime_trajectory,
    trajectory_blocked,
)


class DynamicAvoidanceNode(Node):
    """Gate and retime nominal trajectories using the selected live obstacle source.

    This is a deterministic dynamic safety layer, not a learned policy and not
    an alias for OMPL replanning. It only emits a trajectory while the safety
    supervisor reports a clear/warning state; otherwise it holds the latest
    nominal request for recovery from the current measured joints.
    """

    def __init__(self) -> None:
        super().__init__("openarm_dynamic_avoidance")
        self.declare_parameter("obstacle_source", "ground_truth")
        self.declare_parameter("robot_model", "")
        self.declare_parameter("guarded_route_velocity_scale", 0.50)
        self.declare_parameter("obstacle_clear_hold_sec", 0.75)
        # The live RGB-D box moves a few centimetres between neural frames.
        # Replanning every such update starves the controller: MoveIt never
        # gets enough time to execute the new route.  Four centimetres reacts
        # before the slow demo hand crosses a link while avoiding frame churn.
        self.declare_parameter("obstacle_motion_replan_m", 0.04)
        self.declare_parameter("replan_cooldown_sec", 0.60)
        # Planning Scene geometry already carries its own padding.  Six cm is
        # enough to cover the articulated arm corridor without treating every
        # point cloud sharing the same XY projection as an active collision.
        self.declare_parameter("path_influence_margin_m", 0.06)
        self.declare_parameter(
            "environment_cloud_topic",
            "/realtime_safety/environment_cloud_world",
        )
        self.source = ""
        self.safety_state = "SAFE"
        self.velocity_scale = 1.0
        self.current_positions: dict[str, float] = {}
        self.pending: JointTrajectory | None = None
        self.environment_points = 0
        self.dynamic_points = 0
        self.target_pose: PoseStamped | None = None
        self._obstacle_present = False
        self._obstacle_reference: np.ndarray | None = None
        self._last_obstacle_seen = 0.0
        self._obstacle_clear_hold_sec = max(
            float(self.get_parameter("obstacle_clear_hold_sec").value), 0.0
        )
        self._trajectory_active_until = 0.0
        self._last_replan_request = 0.0
        self._spatial_replans = 0
        self._under_routes = 0
        self._guarded_route_active = False
        self._active_side = ""
        self._obstacle_size: np.ndarray | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._source_subscription: Any | None = None
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(
            String, "/openarm/dynamic_avoidance/status", status_qos
        )
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/openarm/dynamic_avoidance/trajectory", 10
        )
        self.obstacle_pub = self.create_publisher(
            PointCloud2, "/openarm/dynamic_obstacles", qos_profile_sensor_data
        )
        self.replan_pub = self.create_publisher(
            String, "/openarm/dynamic_avoidance/replan_request", 10
        )
        self.create_subscription(
            JointTrajectory,
            "/openarm/dynamic_avoidance/input_trajectory",
            self._nominal_trajectory,
            10,
        )
        environment_cloud_topic = str(
            self.get_parameter("environment_cloud_topic").value
        ).strip()
        if not environment_cloud_topic.startswith("/"):
            raise ValueError("environment_cloud_topic must be an absolute ROS topic")
        self.create_subscription(
            PointCloud2,
            environment_cloud_topic,
            self._environment_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "environment geometry source: "
            f"{environment_cloud_topic} (3D Safety RGB-D backprojection)"
        )
        self.create_subscription(JointState, "/joint_states", self._joint_state, 10)
        self.create_subscription(
            PoseStamped,
            "/openarm/dynamic_avoidance/target_pose",
            self._target_pose,
            10,
        )
        self.create_subscription(
            CollisionObject,
            "/openarm/dynamic_avoidance/collision_object",
            self._collision_object,
            10,
        )
        self.create_subscription(String, "/openarm/safety/state", self._safety, 10)
        self.create_subscription(
            Float64, "/openarm/safety/velocity_scaling", self._scaling, 10
        )
        self.create_subscription(
            String, "/openarm/obstacle_source", self._source_command, 10
        )
        self.create_subscription(String, "/openarm/events", self._event, 50)
        robot_model = str(self.get_parameter("robot_model").value)
        if robot_model and not Path(robot_model).is_file():
            self.get_logger().warning(f"robot model path does not exist: {robot_model}")
        self._set_source(str(self.get_parameter("obstacle_source").value))
        self.create_timer(0.5, self._publish_status)

    def _set_source(self, source: str) -> None:
        source = source.strip().lower()
        if source not in {"ground_truth", "perception"}:
            self.get_logger().warning(f"ignored obstacle source: {source}")
            return
        if self._source_subscription is not None:
            self.destroy_subscription(self._source_subscription)
            self._source_subscription = None
        if source == "ground_truth":
            self._source_subscription = self.create_subscription(
                PoseStamped,
                "/sim/ground_truth/hand_pose",
                self._ground_truth_hand,
                10,
            )
        else:
            self._source_subscription = self.create_subscription(
                PointCloud2,
                "/perception/obstacles",
                self._perception_cloud,
                qos_profile_sensor_data,
            )
        self.source = source
        self.dynamic_points = 0
        self._obstacle_present = False
        self._obstacle_reference = None
        self._obstacle_size = None
        self._last_obstacle_seen = 0.0
        self.get_logger().info(f"dynamic obstacle source: {source}")

    def _source_command(self, message: String) -> None:
        self._set_source(message.data)

    def _event(self, message: String) -> None:
        if message.data.startswith("dynamic_under_route_selected,"):
            self._under_routes += 1
            self._guarded_route_active = True
        elif message.data.endswith(",under_recover") and message.data.startswith(
            "dynamic_under_waypoint_reached,"
        ):
            self._guarded_route_active = False
        elif message.data.startswith(("motion_complete,", "planning_failed,")):
            self._guarded_route_active = False

    def _environment_cloud(self, message: PointCloud2) -> None:
        self.environment_points = int(message.width) * int(message.height)

    def _ground_truth_hand(self, message: PoseStamped) -> None:
        center = message.pose.position
        offsets = (-0.04, 0.0, 0.04)
        points = [
            (center.x + dx, center.y + dy, center.z + dz)
            for dx in offsets
            for dy in offsets
            for dz in (-0.0125, 0.0125)
        ]
        header = Header(stamp=message.header.stamp, frame_id=message.header.frame_id or "world")
        cloud = point_cloud2.create_cloud_xyz32(header, points)
        self.dynamic_points = len(points)
        self.obstacle_pub.publish(cloud)

    def _perception_cloud(self, message: PointCloud2) -> None:
        self.dynamic_points = int(message.width) * int(message.height)
        self.obstacle_pub.publish(message)
        rows = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        names = getattr(getattr(rows, "dtype", None), "names", None)
        if names:
            points = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(
                np.float64, copy=False
            )
        else:
            points = np.asarray(list(rows), dtype=np.float64).reshape(-1, 3)
        # The safety bridge transforms this measured cloud to the planning
        # frame and publishes its predicted collision box.  Replan decisions
        # are made from that world-frame box, never from camera coordinates.

    def _collision_object(self, message: CollisionObject) -> None:
        if message.id not in {"ground_truth_hand", "perception_hand_obstacle"}:
            return
        if message.operation == CollisionObject.REMOVE:
            self._update_dynamic_obstacle(None, 0, None)
            return
        if not message.primitives or not message.primitive_poses:
            return
        dimensions = message.primitives[0].dimensions
        if len(dimensions) != 3:
            return
        pose = message.primitive_poses[0].position
        center = np.asarray([pose.x, pose.y, pose.z], dtype=float)
        size = np.asarray(dimensions, dtype=float)
        self._update_dynamic_obstacle(center, max(self.dynamic_points, 1), size)

    def _update_dynamic_obstacle(
        self,
        center: np.ndarray | None,
        point_count: int,
        size: np.ndarray | None,
    ) -> None:
        now = self._now_sec()
        if center is None or point_count <= 0 or not np.isfinite(center).all():
            # Neural masks can legitimately emit a single empty frame between
            # tracked frames. Clearing immediately turns the next valid frame
            # into a false "new obstacle" and repeatedly cancels execution.
            if hold_recent_obstacle(
                self._obstacle_present,
                self._last_obstacle_seen,
                now,
                self._obstacle_clear_hold_sec,
            ):
                return
            self._obstacle_present = False
            self._obstacle_reference = None
            self._obstacle_size = None
            return
        self._last_obstacle_seen = now
        newly_visible = not self._obstacle_present
        moved = (
            self._obstacle_reference is not None
            and float(np.linalg.norm(center - self._obstacle_reference))
            >= max(float(self.get_parameter("obstacle_motion_replan_m").value), 0.005)
        )
        self._obstacle_present = True
        if size is not None and np.isfinite(size).all() and np.all(size > 0.0):
            self._obstacle_size = size.copy()
        if self._obstacle_reference is None:
            self._obstacle_reference = center.copy()
        trajectory_active = now < self._trajectory_active_until
        if (
            trajectory_active
            and (newly_visible or moved)
            and not (newly_visible and self._guarded_route_active)
            and self._obstacle_affects_active_path(center)
        ):
            self._request_spatial_replan(
                "new_obstacle" if newly_visible else "obstacle_moved",
                center,
            )
        elif moved:
            # A far-away moving hand must not accumulate displacement and
            # repeatedly cancel a valid arm trajectory.
            self._obstacle_reference = center.copy()

    def _obstacle_affects_active_path(self, center: np.ndarray) -> bool:
        if self.target_pose is None or self._obstacle_size is None:
            return True
        if self._active_side not in {"left", "right"}:
            return True
        try:
            transform = self._tf_buffer.lookup_transform(
                "world",
                f"openarm_{self._active_side}_hand_tcp",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.02),
            )
        except Exception:
            return True
        translation = transform.transform.translation
        start = np.asarray(
            [translation.x, translation.y, translation.z], dtype=float
        )
        position = self.target_pose.pose.position
        goal = np.asarray([position.x, position.y, position.z], dtype=float)
        return obstacle_affects_motion_corridor(
            start,
            goal,
            center,
            self._obstacle_size,
            margin_m=float(self.get_parameter("path_influence_margin_m").value),
        )

    def _request_spatial_replan(self, reason: str, center: np.ndarray) -> None:
        now = self._now_sec()
        if now - self._last_replan_request < max(
            float(self.get_parameter("replan_cooldown_sec").value), 0.05
        ):
            return
        self._last_replan_request = now
        self._trajectory_active_until = 0.0
        self._obstacle_reference = center.copy()
        self._spatial_replans += 1
        self.replan_pub.publish(String(data=reason))

    def _joint_state(self, message: JointState) -> None:
        if len(message.name) == len(message.position):
            self.current_positions = {
                str(name): float(position)
                for name, position in zip(message.name, message.position, strict=True)
            }

    def _target_pose(self, message: PoseStamped) -> None:
        self.target_pose = message

    def _scaling(self, message: Float64) -> None:
        self.velocity_scale = max(0.0, min(float(message.data), 1.0))

    def _safety(self, message: String) -> None:
        previous = self.safety_state
        self.safety_state = message.data.strip().upper()
        if (
            previous in BLOCKING_STATES
            and self.safety_state in {"SAFE", "WARNING", "REPLAN", "RECOVER"}
            and self.pending is not None
        ):
            self._emit_if_safe()

    def _nominal_trajectory(self, message: JointTrajectory) -> None:
        if any(name.startswith("openarm_left_") for name in message.joint_names):
            self._active_side = "left"
        elif any(name.startswith("openarm_right_") for name in message.joint_names):
            self._active_side = "right"
        self.pending = message
        self._emit_if_safe()

    def _emit_if_safe(self) -> None:
        if self.pending is None or trajectory_blocked(
            self.safety_state, self._guarded_route_active
        ):
            return
        try:
            scale = effective_velocity_scale(
                self.safety_state,
                self.velocity_scale,
                float(
                    self.get_parameter("guarded_route_velocity_scale").value
                ),
            )
        except ValueError as exc:
            self.get_logger().error(f"invalid dynamic velocity configuration: {exc}")
            return
        try:
            trajectory = retime_trajectory(
                self.pending, scale, self.current_positions
            )
        except ValueError as exc:
            self.get_logger().error(f"rejected nominal trajectory: {exc}")
            self.pending = None
            return
        self.pending = None
        self.trajectory_pub.publish(trajectory)
        duration = trajectory.points[-1].time_from_start
        seconds = float(duration.sec) + float(duration.nanosec) * 1e-9
        self._trajectory_active_until = self._now_sec() + max(seconds, 0.1)

    def _now_sec(self) -> float:
        """Use simulator time so slow rendering cannot expire an active path."""

        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_status(self) -> None:
        held = self.pending is not None
        self.status_pub.publish(
            String(
                data=(
                    f"READY source={self.source} state={self.safety_state} "
                    f"environment_points={self.environment_points} "
                    f"dynamic_points={self.dynamic_points} held={str(held).lower()} "
                    "strategy=moveit_spatial_replan+multi_axis_waypoints "
                    "guarded_speed="
                    f"{float(self.get_parameter('guarded_route_velocity_scale').value):.2f} "
                    f"replans={self._spatial_replans} under_routes={self._under_routes} "
                    f"guarded_route={str(self._guarded_route_active).lower()}"
                )
            )
        )


def main() -> None:
    rclpy.init()
    node = DynamicAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
