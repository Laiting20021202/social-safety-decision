from __future__ import annotations

from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from openarm_sim.config import PROJECT_ROOT, load_yaml
from openarm_sim.contracts import RuntimeMode, assert_mode_isolation, subscriptions_for_mode
from openarm_sim.state_machine import SafetyState

from .policy import SafetyPolicy
from .geometry import (
    clustered_swept_boxes,
    estimate_bounded_velocity,
    limit_cloud_center_motion,
    minimum_cloud_to_capsules_distance,
)


_DYNAMIC_OBSTACLE_IDS = ("ground_truth_hand", "perception_hand_obstacle")


def _stale_obstacle_ids(previous_source: str, new_source: str) -> tuple[str, ...]:
    """Return scene objects that must not survive a source/startup transition."""
    if not previous_source or previous_source != new_source:
        return _DYNAMIC_OBSTACLE_IDS
    return ()


class SafetyBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("openarm_safety_bridge")
        self.declare_parameter("mode", "ground_truth")
        self.declare_parameter("config", str(PROJECT_ROOT / "config/safety_zones.yaml"))
        self.declare_parameter("startup_grace_sec", 5.0)
        self.declare_parameter("perception_obstacle_padding_m", 0.03)
        # OMPL planning plus controller hand-off takes noticeably longer than
        # one camera frame.  Predict the slow moving hand far enough ahead that
        # the arm can select a side route before the measured cloud reaches it.
        self.declare_parameter("prediction_horizon_sec", 2.0)
        self.declare_parameter("maximum_obstacle_speed_mps", 0.20)
        self.declare_parameter("obstacle_center_jump_slack_m", 0.02)
        self.declare_parameter("obstacle_velocity_smoothing", 0.25)
        # The cloud contains the measured obstacle surface.  Five centimetres
        # approximates the OpenArm link collision radius without turning the
        # emergency gate into an extra 10+ cm virtual wall.
        self.declare_parameter("robot_capsule_radius_m", 0.05)
        self.declare_parameter("minimum_distance_quantile", 0.01)
        self.declare_parameter("collision_box_count", 3)
        self.mode = RuntimeMode(str(self.get_parameter("mode").value))
        expected_subscriptions = subscriptions_for_mode(self.mode)
        assert_mode_isolation(self.mode, expected_subscriptions)
        self.config = load_yaml(str(self.get_parameter("config").value))
        thresholds = self.config["thresholds_m"]
        self.policy = SafetyPolicy(
            warning_m=float(thresholds["warning"]),
            pause_m=float(thresholds["pause"]),
            emergency_m=float(thresholds["emergency_stop"]),
            resume_m=float(self.config["clearance"]["resume_distance_m"]),
            clear_duration_sec=float(self.config["clearance"]["minimum_clear_duration_sec"]),
            timeout_sec=float(self.config["timeouts"]["obstacle_data_sec"]),
            replan_delay_sec=float(
                self.config.get("behavior", {}).get("replan_delay_sec", 0.30)
            ),
            emergency_confirmation_sec=float(
                self.config.get("behavior", {}).get(
                    "emergency_confirmation_sec", 0.20
                )
            ),
        )
        self.policy.last_observation_sec = self._now()
        self._startup_sec = self._now()
        self._received_observation = False
        self.planning_frame = self.config["frames"]["planning"]
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state_pub = self.create_publisher(String, self.config["topics"]["safety_state"], 10)
        self.scaling_pub = self.create_publisher(
            Float64, self.config["topics"]["velocity_scaling"], 10
        )
        self.distance_pub = self.create_publisher(
            Float64, "/openarm/safety/min_distance", 10
        )
        self.events_pub = self.create_publisher(String, self.config["topics"]["events"], 50)
        self.scene_pub = self.create_publisher(PlanningScene, self.config["topics"]["planning_scene"], 10)
        self.dynamic_obstacle_pub = self.create_publisher(
            CollisionObject, "/openarm/dynamic_avoidance/collision_object", 10
        )
        self.create_subscription(
            String, "/openarm/safety/command", self._safety_command, 10
        )
        self.last_distance = float("inf")
        self.last_state = self.policy.state
        self._perception_obstacle_present = False
        self._perception_cleanup_pending = True
        self._last_obstacle_center: np.ndarray | None = None
        self._last_obstacle_stamp_sec: float | None = None
        self._obstacle_velocity = np.zeros(3, dtype=float)
        self.obstacle_source = ""
        self._obstacle_subscriptions: list[Any] = []
        self.create_subscription(
            String, "/openarm/obstacle_source", self._obstacle_source_command, 10
        )
        self._set_obstacle_source(self.mode.value)
        self.create_service(Trigger, "/openarm/safety/reset_estop", self._reset_estop)
        self.create_timer(0.05, self._timer)

    def _ground_truth_collision(self, obstacle: CollisionObject) -> None:
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obstacle)
        self.scene_pub.publish(scene)
        self.dynamic_obstacle_pub.publish(obstacle)

    def _obstacle_source_command(self, message: String) -> None:
        self._set_obstacle_source(message.data)

    def _set_obstacle_source(self, source: str) -> None:
        source = source.strip().lower()
        if source not in {"ground_truth", "perception"}:
            self.get_logger().warning(f"ignored obstacle source: {source}")
            return
        previous_source = self.obstacle_source
        # MoveIt's PlanningScene outlives this node.  Explicitly remove the
        # old dynamic object before changing subscriptions; merely clearing
        # our local boolean leaves a stale collision body that can overlap an
        # OpenArm finger and make every HOME plan fail at its start state.
        for obstacle_id in _stale_obstacle_ids(previous_source, source):
            self._remove_obstacle(obstacle_id)
        for subscription in self._obstacle_subscriptions:
            self.destroy_subscription(subscription)
        self._obstacle_subscriptions = []
        if source == "ground_truth":
            self._obstacle_subscriptions.extend(
                (
                    self.create_subscription(
                        CollisionObject,
                        self.config["topics"]["ground_truth_collision"],
                        self._ground_truth_collision,
                        10,
                    ),
                    self.create_subscription(
                        Float64,
                        self.config["topics"]["ground_truth_min_distance"],
                        self._distance_callback,
                        10,
                    ),
                )
            )
        else:
            self._obstacle_subscriptions.append(
                self.create_subscription(
                    PointCloud2,
                    self.config["topics"]["perception_cloud"],
                    self._perception_cloud,
                    qos_profile_sensor_data,
                )
            )
        self.obstacle_source = source
        self._perception_obstacle_present = False
        # A REMOVE published during node startup can precede the
        # move_group subscriber connection.  The first empty perception frame
        # retries it once, after discovery has completed.
        self._perception_cleanup_pending = source == "perception"
        self._last_obstacle_center = None
        self._last_obstacle_stamp_sec = None
        self._obstacle_velocity.fill(0.0)
        self.policy.last_observation_sec = self._now()
        self.get_logger().info(f"safety obstacle source: {source}")

    def _distance_callback(self, message: Float64) -> None:
        self._received_observation = True
        self.last_distance = float(message.data)
        self.policy.observe(self.last_distance, self._now())
        self._publish_policy()

    def _perception_cloud(self, message: PointCloud2) -> None:
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
        if not len(points):
            self._received_observation = True
            self.policy.observe(float("inf"), self._now())
            if (
                self._perception_obstacle_present
                or self._perception_cleanup_pending
            ):
                self._remove_obstacle("perception_hand_obstacle")
                self._perception_obstacle_present = False
                self._perception_cleanup_pending = False
            self._last_obstacle_center = None
            self._last_obstacle_stamp_sec = None
            self._obstacle_velocity.fill(0.0)
            self._publish_policy()
            return
        try:
            # The Gazebo camera is an operator-editable kinematic model and
            # ModelStates has no acquisition timestamp.  Its broadcaster can
            # therefore only represent the current extrinsic calibration,
            # not a truthful historical pose for each RGB-D stamp.  Transform
            # every fresh measured cloud with that latest calibration.  An
            # unavailable transform still remains fail-safe via the timeout.
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame,
                message.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception as error:
            self.get_logger().warning(
                f"perception cloud TF unavailable: {error}",
                throttle_duration_sec=2.0,
            )
            return
        world_points = _transform_points(points, transform.transform)
        stamp_sec = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )
        if stamp_sec <= 0.0:
            stamp_sec = self._now()
        world_points, measured_center, center_limited = limit_cloud_center_motion(
            world_points,
            self._last_obstacle_center,
            self._last_obstacle_stamp_sec,
            stamp_sec,
            maximum_speed_mps=float(
                self.get_parameter("maximum_obstacle_speed_mps").value
            ),
            slack_m=float(
                self.get_parameter("obstacle_center_jump_slack_m").value
            ),
        )
        if center_limited:
            self.get_logger().warning(
                "bounded implausible perception-cloud center jump",
                throttle_duration_sec=1.0,
            )
        lower = np.quantile(world_points, 0.02, axis=0)
        upper = np.quantile(world_points, 0.98, axis=0)
        measured_size = np.maximum(upper - lower, 0.025)
        # Use the robust quantile-box center for velocity after the median
        # center continuity gate has translated the fresh cloud.
        measured_center = (lower + upper) / 2.0
        self._obstacle_velocity = estimate_bounded_velocity(
            self._last_obstacle_center,
            self._last_obstacle_stamp_sec,
            measured_center,
            stamp_sec,
            self._obstacle_velocity,
            smoothing=float(
                self.get_parameter("obstacle_velocity_smoothing").value
            ),
            maximum_speed_mps=float(
                self.get_parameter("maximum_obstacle_speed_mps").value
            ),
        )
        self._last_obstacle_center = measured_center.copy()
        self._last_obstacle_stamp_sec = stamp_sec
        horizon = max(
            float(self.get_parameter("prediction_horizon_sec").value), 0.0
        )
        padding = max(
            float(self.get_parameter("perception_obstacle_padding_m").value),
            0.0,
        )
        boxes = clustered_swept_boxes(
            world_points,
            self._obstacle_velocity,
            horizon,
            padding_m=padding,
            maximum_boxes=int(self.get_parameter("collision_box_count").value),
        )
        obstacle = _boxes_collision(
            "perception_hand_obstacle", self.planning_frame, boxes
        )
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obstacle)
        self.scene_pub.publish(scene)
        self.dynamic_obstacle_pub.publish(obstacle)
        self._perception_obstacle_present = True
        self._perception_cleanup_pending = False
        link_chains = self._robot_link_chains()
        if link_chains:
            # The swept boxes above are deliberately predictive so MoveIt can
            # route around where the hand is going.  Emergency-stop distance,
            # however, must describe the obstacle *now*.  Feeding future
            # points into the immediate safety policy caused a safe arm (often
            # 7-10 cm from the measured hand) to latch E-stop merely because a
            # two-second prediction intersected its current pose.
            self.last_distance = minimum_cloud_to_capsules_distance(
                world_points,
                link_chains,
                capsule_radius_m=float(
                    self.get_parameter("robot_capsule_radius_m").value
                ),
                distance_quantile=float(
                    self.get_parameter("minimum_distance_quantile").value
                ),
            )
        else:
            # A freshly started TransformListener needs a few camera frames to
            # fill its cache.  Do not latch a false E-stop during the explicit
            # startup grace period; after it expires, missing robot geometry
            # remains fail-safe.
            if self._now() - self._startup_sec < float(
                self.get_parameter("startup_grace_sec").value
            ):
                return
            self.last_distance = 0.0
        # A raw cloud is not yet a usable safety observation until both its
        # camera transform and robot-link geometry are available.  Marking it
        # received before those checks caused a fresh node to bypass startup
        # grace, time out, and latch a false emergency stop.
        self._received_observation = True
        self.policy.observe(self.last_distance, self._now())
        self._publish_policy()

    def _robot_link_chains(self) -> list[np.ndarray]:
        chains: list[np.ndarray] = []
        for side in ("left", "right"):
            points = []
            frames = [f"openarm_{side}_link{index}" for index in range(8)]
            frames.extend((f"openarm_{side}_hand", f"openarm_{side}_hand_tcp"))
            for frame in frames:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.planning_frame, frame, rclpy.time.Time(), timeout=Duration(seconds=0.005)
                    )
                except Exception:
                    continue
                translation = transform.transform.translation
                points.append([translation.x, translation.y, translation.z])
            if len(points) >= 2:
                chains.append(np.asarray(points, dtype=float).reshape(-1, 3))
        return chains

    def _robot_link_points(self) -> np.ndarray:
        """Compatibility view used by older diagnostics/tests."""

        chains = self._robot_link_chains()
        if not chains:
            return np.empty((0, 3), dtype=float)
        return np.concatenate(chains, axis=0)

    def _timer(self) -> None:
        if (
            not self._received_observation
            and self._now() - self._startup_sec
            < float(self.get_parameter("startup_grace_sec").value)
        ):
            self._publish_policy()
            return
        self.policy.check_timeout(self._now())
        self._publish_policy()

    def _safety_command(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "emergency_stop":
            self.policy.state = SafetyState.EMERGENCY_STOP
        elif command == "pause" and self.policy.state is not SafetyState.EMERGENCY_STOP:
            self.policy.state = SafetyState.PAUSE
        elif command == "resume" and self.policy.state is not SafetyState.EMERGENCY_STOP:
            self.policy.state = SafetyState.RECOVER
        elif command == "reset":
            self.policy.reset_estop(self._now())
        elif command == "escape_started":
            self.policy.grant_escape_grace(
                self._now(),
                float(
                    self.config.get("behavior", {}).get(
                        "escape_execution_grace_sec", 3.0
                    )
                ),
            )
        else:
            self.get_logger().warning(f"ignored safety command: {command}")
            return
        self._publish_policy()

    def _publish_policy(self) -> None:
        state = self.policy.state
        scales = self.config["velocity_scaling"]
        scale = {
            SafetyState.SAFE: scales["safe"],
            SafetyState.WARNING: scales["warning"],
            SafetyState.PAUSE: scales["pause"],
            SafetyState.REPLAN: scales["warning"],
            SafetyState.RECOVER: scales["warning"],
            SafetyState.EMERGENCY_STOP: scales["pause"],
        }[state]
        self.state_pub.publish(String(data=state.value))
        self.scaling_pub.publish(Float64(data=float(scale)))
        self.distance_pub.publish(Float64(data=float(self.last_distance)))
        if state is not self.last_state:
            previous = self.last_state
            self.events_pub.publish(
                String(data=f"safety_transition,{previous.value},{state.value},{self.last_distance}")
            )
            phase_event = {
                SafetyState.REPLAN: "replan_requested",
                SafetyState.RECOVER: "replan_complete",
                SafetyState.SAFE: "motion_resumed",
            }.get(state)
            if phase_event is not None:
                self.events_pub.publish(String(data=f"{phase_event},{previous.value}"))
            self.last_state = state

    def _remove_obstacle(self, obstacle_id: str) -> None:
        collision = CollisionObject()
        collision.header.frame_id = self.planning_frame
        collision.id = obstacle_id
        collision.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(collision)
        self.scene_pub.publish(scene)
        self.dynamic_obstacle_pub.publish(collision)

    def _reset_estop(self, _request: Trigger.Request, response: Trigger.Response):
        self.policy.reset_estop(self._now())
        self._publish_policy()
        response.success = True
        response.message = "E-stop latch reset; entering REPLAN"
        return response

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def _boxes_collision(
    name: str,
    frame: str,
    boxes: list[tuple[np.ndarray, np.ndarray]],
) -> CollisionObject:
    collision = CollisionObject()
    collision.header.frame_id = frame
    collision.id = name
    for center, size in boxes:
        collision.primitives.append(
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[float(value) for value in size],
            )
        )
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, center)
        pose.orientation.w = 1.0
        collision.primitive_poses.append(pose)
    collision.operation = CollisionObject.ADD
    return collision


def _transform_points(points: np.ndarray, transform: Any) -> np.ndarray:
    quaternion = transform.rotation
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    translation = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z]
    )
    return points @ rotation.T + translation


def main() -> None:
    rclpy.init()
    node = SafetyBridgeNode()
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
