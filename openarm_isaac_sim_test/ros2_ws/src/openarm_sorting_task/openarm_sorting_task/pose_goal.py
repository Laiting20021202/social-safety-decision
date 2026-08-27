from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
)
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64, String
from trajectory_msgs.msg import JointTrajectory
from tf2_ros import Buffer, TransformListener

from openarm_sim.config import PROJECT_ROOT, load_yaml

from .route_planner import (
    avoidance_route_waypoints,
    guarded_route_needs_restart,
    preserve_home_after_side_failure,
    progressive_pose_tolerance,
    route_progress,
    target_hold_sequence,
    target_contact_point,
    waiting_path_should_retry,
)
from .idle_avoidance import (
    active_escape_can_continue,
    crisis_escape_point,
    obstacle_has_withdrawn,
    obstacle_is_clear_for_recovery,
    safety_evade_sequence,
    select_evading_sides,
    select_evading_sides_from_links,
    union_axis_aligned_boxes,
)


ROUTE_STEP_KINDS = frozenset(
    {"under_approach", "under_entry", "under_exit", "under_recover"}
)
CRISIS_STEP_KINDS = frozenset({"safety_evade", "idle_evade"})


class OpenArmPoseGoal(Node):
    """Plan collision-aware left/right TCP motions to Gazebo target markers."""

    def __init__(self) -> None:
        super().__init__("openarm_pose_goal")
        self.declare_parameter("config", str(PROJECT_ROOT / "config/sorting_task.yaml"))
        self.declare_parameter("planner_mode", "moveit")
        self._config = yaml.safe_load(
            Path(str(self.get_parameter("config").value)).read_text()
        )
        self._robot = load_yaml("config/openarm.yaml")["robot"]
        self._target: PoseStamped | None = None
        self._targets: dict[str, PoseStamped] = {}
        self._active_side = "left"
        self._current_step: tuple[str, str] | None = None
        self._sequence: list[tuple[str, str]] = []
        self._request_kind = ""
        self._planner_mode = str(self.get_parameter("planner_mode").value).strip().lower()
        if self._planner_mode not in {"moveit", "dynamic"}:
            raise ValueError("planner_mode must be moveit or dynamic")
        self._safety_state = "SAFE"
        self._state = "IDLE"
        self._joint_names: set[str] = set()
        self._joint_positions: dict[str, float] = {}
        self._busy = False
        self._resume_pending = False
        self._move_goal: Any | None = None
        self._trajectory_goal: Any | None = None
        self._request_generation = 0
        self._spatial_replan_count = 0
        self._dynamic_obstacle: tuple[str, np.ndarray, np.ndarray] | None = None
        self._waiting_obstacle_center: np.ndarray | None = None
        self._safety_distance = float("inf")
        self._idle_saved_joints: dict[str, list[float]] | None = None
        self._idle_evading_sides: tuple[str, ...] = ()
        self._idle_evaded = False
        self._idle_clear_since: float | None = None
        self._idle_trigger_obstacle_center: np.ndarray | None = None
        self._idle_retry_after = 0.0
        self._crisis_escape_targets: dict[str, PoseStamped] = {}
        self._held_target_sides: set[str] = set()
        self._estop_trigger_obstacle_center: np.ndarray | None = None
        self._estop_clear_since: float | None = None
        self._estop_reset_requested = False
        self._route_waypoints: dict[str, PoseStamped] = {}
        self._route_obstacle_center: np.ndarray | None = None
        self._route_refreshes: dict[str, int] = {}
        self._force_local_detour_sides: set[str] = set()
        self._under_routed_steps: set[tuple[str, str]] = set()
        self._under_route_active = False
        self._planning_failures: dict[tuple[str, str], int] = {}
        self._waiting_retries: dict[tuple[str, str], int] = {}
        self._failed_sides: dict[str, int] = {}
        self._retry_timer: Any | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._state_pub = self.create_publisher(String, "/openarm/task/state", 10)
        self._events_pub = self.create_publisher(String, "/openarm/events", 50)
        self._safety_command_pub = self.create_publisher(
            String, "/openarm/safety/command", 10
        )
        self._dynamic_input_pub = self.create_publisher(
            JointTrajectory, "/openarm/dynamic_avoidance/input_trajectory", 10
        )
        self._dynamic_target_pub = self.create_publisher(
            PoseStamped, "/openarm/dynamic_avoidance/target_pose", 10
        )
        self.create_subscription(PoseStamped, "/openarm/target_pose", self._target_pose, 10)
        self.create_subscription(ModelStates, "/gazebo/model_states", self._model_states, 10)
        self.create_subscription(String, "/openarm/task/command", self._command, 10)
        self.create_subscription(String, "/openarm/planner/mode", self._planner, 10)
        self.create_subscription(String, "/openarm/safety/state", self._safety, 10)
        self.create_subscription(
            Float64,
            "/openarm/safety/min_distance",
            self._safety_distance_callback,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._joints, 10)
        self.create_subscription(PlanningScene, "/planning_scene", self._planning_scene, 10)
        self.create_subscription(
            CollisionObject,
            "/openarm/dynamic_avoidance/collision_object",
            self._dynamic_collision,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            "/openarm/dynamic_avoidance/trajectory",
            self._dynamic_trajectory,
            10,
        )
        self.create_subscription(
            String,
            "/openarm/dynamic_avoidance/replan_request",
            self._dynamic_replan_request,
            10,
        )
        self._move_client = ActionClient(
            self, MoveGroup, str(self._config["move_group_action"])
        )
        self._trajectory_clients = {
            side: ActionClient(
                self,
                FollowJointTrajectory,
                str(self._robot["controller_actions"][side]),
            )
            for side in ("left", "right")
        }
        self.create_timer(0.2, self._publish_state)
        self.get_logger().info(
            "Dual TCP target planner ready; targets follow direct Gazebo model edits"
        )

    def _model_states(self, message: ModelStates) -> None:
        by_name = dict(zip(message.name, message.pose, strict=True))
        targets: dict[str, PoseStamped] = {}
        for side, model_name in self._config["target_models"].items():
            pose = by_name.get(str(model_name))
            if pose is None:
                continue
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = str(self._config["planning_frame"])
            target.pose = pose
            quaternion = self._config["tool_orientation_xyzw"]
            target.pose.orientation.x = float(quaternion[0])
            target.pose.orientation.y = float(quaternion[1])
            target.pose.orientation.z = float(quaternion[2])
            target.pose.orientation.w = float(quaternion[3])
            targets[str(side)] = target
        self._targets = targets

    def _target_pose(self, message: PoseStamped) -> None:
        if message.header.frame_id not in {"", str(self._config["planning_frame"])}:
            self._set_state(f"TARGET_FRAME_REJECTED:{message.header.frame_id}")
            return
        message.header.frame_id = str(self._config["planning_frame"])
        if (
            message.pose.orientation.x == 0.0
            and message.pose.orientation.y == 0.0
            and message.pose.orientation.z == 0.0
            and message.pose.orientation.w == 0.0
        ):
            quaternion = self._config["tool_orientation_xyzw"]
            message.pose.orientation.x = float(quaternion[0])
            message.pose.orientation.y = float(quaternion[1])
            message.pose.orientation.z = float(quaternion[2])
            message.pose.orientation.w = float(quaternion[3])
        self._target = message
        self._dynamic_target_pub.publish(message)
        if not self._busy:
            self._set_state("TARGET_READY")

    def _command(self, message: String) -> None:
        command = message.data.strip().lower()
        if command in {
            "move_target",
            "move_both_targets",
            "move_left_target",
            "move_right_target",
            "home",
        }:
            # A new operator command becomes the desired posture.  Do not
            # later restore a stale posture captured by an older idle reflex.
            self._clear_idle_avoidance()
        if command == "move_target":
            if self._target is None:
                self._set_state("TARGET_NOT_SET")
            else:
                self._held_target_sides.discard("left")
                self._begin_request(
                    "target",
                    target_hold_sequence(("left",), "explicit_target"),
                )
        elif command in {"move_both_targets", "move_left_target", "move_right_target"}:
            sides = {
                "move_both_targets": ("left", "right"),
                "move_left_target": ("left",),
                "move_right_target": ("right",),
            }[command]
            missing = [side for side in sides if side not in self._targets]
            if missing:
                self._set_state(f"GAZEBO_TARGET_NOT_READY:{','.join(missing)}")
                return
            self._held_target_sides.difference_update(sides)
            self._begin_request(
                command.removeprefix("move_"),
                target_hold_sequence(sides),
            )
        elif command == "home":
            # HOME is a preemptive high-level command.  It must never be lost
            # behind an older MoveGroup/controller callback or a PAUSE state.
            self._held_target_sides.clear()
            self._begin_request(
                "home_both", [("left", "home"), ("right", "home")]
            )
        elif command == "pause":
            self._resume_pending = self._busy or bool(self._request_kind)
            self._cancel_motion()
            self._set_state("PAUSED")
        elif command == "resume":
            if self._safety_state != "EMERGENCY_STOP" and self._resume_pending:
                self._resume_pending = False
                if self._current_step is None:
                    self._start_next_step()
                else:
                    self._replan_last_request()
        elif command == "reset":
            self._cancel_motion()
            self._sequence = []
            self._current_step = None
            self._request_kind = ""
            self._resume_pending = False
            self._waiting_obstacle_center = None
            self._held_target_sides.clear()
            self._clear_idle_avoidance()
            self._set_state(self._holding_state())
        elif command.startswith("pick:") or command == "start":
            self._set_state("PICK_TASK_PAUSED")

    def _begin_request(
        self, request_kind: str, sequence: list[tuple[str, str]]
    ) -> None:
        self._cancel_motion()
        if request_kind != "idle_evade":
            self._crisis_escape_targets = {}
        self._request_kind = request_kind
        self._sequence = list(sequence)
        self._current_step = None
        self._resume_pending = False
        self._route_waypoints = {}
        self._route_obstacle_center = None
        self._route_refreshes = {}
        self._force_local_detour_sides = set()
        self._under_routed_steps = set()
        self._under_route_active = False
        self._planning_failures = {}
        self._waiting_retries = {}
        self._failed_sides = {}
        self._waiting_obstacle_center = None
        self._set_state(f"COMMAND_ACCEPTED:{request_kind.upper()}")
        self._start_next_step()

    def _planner(self, message: String) -> None:
        mode = message.data.strip().lower()
        if mode in {"moveit", "dynamic"}:
            self._planner_mode = mode
            self._emit(f"planner_mode,{mode}")

    def _safety(self, message: String) -> None:
        previous = self._safety_state
        self._safety_state = message.data.strip().upper()
        if self._safety_state == previous:
            return
        if self._safety_state == "EMERGENCY_STOP" and previous != "EMERGENCY_STOP":
            self._estop_trigger_obstacle_center = (
                self._dynamic_obstacle[1].copy()
                if self._dynamic_obstacle is not None
                else None
            )
            self._estop_clear_since = None
            self._estop_reset_requested = False
        elif previous == "EMERGENCY_STOP" and self._safety_state != "EMERGENCY_STOP":
            self._estop_trigger_obstacle_center = None
            self._estop_clear_since = None
            self._estop_reset_requested = False
        if (
            self._safety_state == "REPLAN"
            and previous == "PAUSE"
            and self._request_kind
            and self._planner_mode == "dynamic"
            and self._dynamic_obstacle is not None
            and self._current_step is not None
            and self._current_step[1]
            in {"model_target", "explicit_target", *ROUTE_STEP_KINDS}
        ):
            # Preserve the selected marker as the goal.  The old behavior
            # inserted a Home retreat here, so every close hand erased target
            # progress.  Rebuild a local route from the measured TCP instead.
            self._emit(
                f"dynamic_progress_replan,{self._active_side},"
                f"{self._current_step[1]},{self._safety_distance:.3f}"
            )
            self._force_local_detour_sides.add(self._active_side)
            self._restart_dynamic_route(self._active_side)
            return
        if self._guarded_under_route() and self._safety_state == "REPLAN":
            # PAUSE is intentionally excluded: an incoming moving obstacle
            # invalidates even an earlier collision-checked bypass.
            self._emit(
                f"dynamic_under_route_guarded,{self._active_side},{self._safety_state}"
            )
            if guarded_route_needs_restart(
                self._safety_state,
                self._under_route_active,
                self._busy,
            ):
                # PAUSE cancels both controller and MoveGroup goals.  If that
                # cancellation happened while a guarded waypoint was being
                # planned, REPLAN must submit it again from current joints;
                # returning here used to leave the task stuck forever.
                self._resume_pending = False
                self._set_state("DYNAMIC_GUARDED_ROUTE_RESTART")
                self._replan_last_request()
            return
        if active_escape_can_continue(
            self._safety_state,
            self._current_step[1] if self._current_step is not None else None,
            self._busy,
        ):
            # WARNING may already have started the dedicated, collision-
            # checked vertical lift.  PAUSE and the following REPLAN must not
            # cancel and resubmit that same goal: doing so created the visible
            # shoulder/wrist reversal before the arm finally lifted.
            self._resume_pending = False
            self._set_state("CRISIS_ESCAPE_IN_PROGRESS")
            return
        if self._safety_state == "EMERGENCY_STOP":
            self._resume_pending = bool(self._request_kind)
            self._cancel_motion()
            self._set_state("EMERGENCY_STOP")
        elif self._safety_state == "PAUSE":
            self._resume_pending = bool(self._request_kind)
            self._cancel_motion()
            if not self._begin_active_crisis_evade():
                self._set_state("PAUSE")
        elif self._safety_state == "REPLAN":
            if self._request_kind:
                self._resume_pending = False
                self._set_state("DYNAMIC_REPLANNING")
                if self._current_step is None:
                    self._start_next_step()
                else:
                    self._replan_last_request()
            else:
                self._set_state("REPLAN")
        elif (
            previous in {"PAUSE", "REPLAN", "EMERGENCY_STOP"}
            and self._safety_state in {"SAFE", "WARNING", "RECOVER"}
            and self._resume_pending
        ):
            # WARNING permits a collision-checked trajectory at reduced
            # velocity.  Waiting for SAFE here deadlocked a request whenever
            # the supervisor naturally relaxed PAUSE -> WARNING while the
            # hand was still visible but no longer close.
            self._resume_pending = False
            self._set_state("REPLANNING")
            if self._current_step is None:
                self._start_next_step()
            else:
                self._replan_last_request()
        elif (
            previous in {"PAUSE", "REPLAN"}
            and self._safety_state in {"SAFE", "RECOVER"}
            and not self._busy
            and not self._request_kind
        ):
            # A safety-latch reset with no active motion is complete once the
            # scene is clear.  Leaving the task display stuck on REPLAN made
            # otherwise valid GUI commands look disabled even though the
            # supervisor had already returned to SAFE.
            self._set_state(self._holding_state())
        self._idle_avoidance_tick()

    def _is_crisis_step(self) -> bool:
        return bool(
            self._current_step is not None
            and self._current_step[1] in CRISIS_STEP_KINDS
        )

    def _prepare_crisis_escape_targets(
        self,
        sides: tuple[str, ...],
        center: np.ndarray,
        size: np.ndarray,
        *,
        vertical_only: bool = False,
    ) -> tuple[str, ...]:
        config = self._config.get("crisis_avoidance", {})
        idle_config = self._config.get("always_on_avoidance", {})
        prepared: list[str] = []
        for side in sides:
            tcp = self._tcp_position(side)
            if tcp is None:
                continue
            xyz = crisis_escape_point(
                tcp,
                center,
                size,
                side=side,
                minimum_lift_m=float(
                    idle_config.get("vertical_lift_m", 0.14)
                    if vertical_only
                    else config.get("minimum_lift_m", 0.16)
                ),
                horizontal_escape_m=(
                    0.0
                    if vertical_only
                    else float(config.get("horizontal_escape_m", 0.10))
                ),
                vertical_clearance_m=float(
                    idle_config.get("vertical_clearance_m", 0.10)
                    if vertical_only
                    else config.get("vertical_clearance_m", 0.14)
                ),
                maximum_tcp_z_m=float(config.get("maximum_tcp_z_m", 1.12)),
            )
            target = self._waypoint_message(xyz)
            if vertical_only:
                orientation = self._tcp_orientation(side)
                if orientation is None:
                    orientation = np.asarray(
                        self._config["tool_orientation_xyzw"], dtype=float
                    )
                target.pose.orientation.x = float(orientation[0])
                target.pose.orientation.y = float(orientation[1])
                target.pose.orientation.z = float(orientation[2])
                target.pose.orientation.w = float(orientation[3])
            self._crisis_escape_targets[side] = target
            prepared.append(side)
            self._emit(
                f"{'idle_vertical_lift' if vertical_only else 'crisis_escape_goal'},"
                f"{side},{xyz[0]:.3f},"
                f"{xyz[1]:.3f},{xyz[2]:.3f}"
            )
        return tuple(prepared)

    def _begin_active_crisis_evade(self) -> bool:
        config = self._config.get("crisis_avoidance", {})
        if (
            not bool(config.get("enabled", True))
            or self._planner_mode != "dynamic"
            or not self._request_kind
            or self._current_step is None
            or self._dynamic_obstacle is None
        ):
            return False

        _, center, size = self._dynamic_obstacle
        if self._request_kind in {"idle_evade", "idle_restore"}:
            sides = self._idle_evading_sides
            if not sides:
                sides = self._select_evading_arms(
                    center,
                    size,
                    trigger_distance_m=float(
                        self._config.get("always_on_avoidance", {}).get(
                            "tcp_trigger_distance_m", 0.16
                        )
                    ),
                )
            sides = self._prepare_crisis_escape_targets(
                sides, center, size, vertical_only=True
            )
            if not sides:
                return False
            # A hand returning during descent must restart the same simple
            # vertical reflex.  Preserve the original joint snapshot; never
            # replace it with a marker route or Home request.
            self._idle_evading_sides = sides
            self._idle_trigger_obstacle_center = center.copy()
            self._idle_clear_since = None
            self._idle_evaded = False
            self._begin_request(
                "idle_evade", [(side, "idle_evade") for side in sides]
            )
            return True

        if self._is_crisis_step():
            sides = self._prepare_crisis_escape_targets(
                (self._current_step[0],), center, size
            )
            if not sides:
                return False
            self._resume_pending = False
            self._set_state("CRISIS_ESCAPE_REFRESH")
            self._replan_last_request()
            return True

        sides = self._select_evading_arms(
            center,
            size,
            trigger_distance_m=float(
                config.get("tcp_trigger_distance_m", 0.24)
            ),
        )
        sides = self._prepare_crisis_escape_targets(sides, center, size)
        if not sides:
            return False

        target_kind = (
            "explicit_target" if self._request_kind == "target" else "model_target"
        )
        interrupted = self._current_step
        self._sequence = safety_evade_sequence(
            interrupted,
            self._sequence,
            sides,
            target_kind=target_kind,
            route_step_kinds=ROUTE_STEP_KINDS,
        )
        # When MOVE BOTH has already placed one arm, an intrusion while the
        # second arm is moving may evade that held arm too. Requeue its marker
        # exactly once so it also returns after the obstacle clears.
        for held_side in sorted(self._held_target_sides.intersection(sides)):
            restore = (held_side, "model_target")
            if restore not in self._sequence:
                self._sequence.append(restore)
        self._current_step = None
        self._route_waypoints = {}
        self._route_obstacle_center = None
        self._under_routed_steps = {
            step for step in self._under_routed_steps if step[0] not in sides
        }
        self._under_route_active = False
        self._force_local_detour_sides.update(sides)
        self._resume_pending = False
        self._emit(
            f"crisis_escape_triggered,{','.join(sides)},"
            f"{self._safety_distance:.3f},resume={interrupted[1]}"
        )
        # Grant a short bounded planning window immediately.  Waiting until
        # OMPL returns can let the near-contact confirmation latch while the
        # arm is still computing the very escape requested by PAUSE.
        self._safety_command_pub.publish(String(data="escape_started"))
        self._set_state("CRISIS_ESCAPE_QUEUED")
        self._start_next_step()
        return True

    def _safety_distance_callback(self, message: Float64) -> None:
        self._safety_distance = float(message.data)
        if (
            self._state
            in {"DYNAMIC_WAIT_FOR_OBSTACLE_CLEAR", "DYNAMIC_WAIT_FOR_CLEAR_PATH"}
            and self._dynamic_obstacle is not None
            and self._waiting_obstacle_center is not None
            and not self._busy
            and self._retry_timer is None
            and waiting_path_should_retry(
                self._safety_state,
                self._safety_distance,
                float(
                    self._config.get("always_on_avoidance", {}).get(
                        "release_distance_m", 0.26
                    )
                ),
                coarse_path_blocked=self._dynamic_path_is_blocked(),
            )
        ):
            # The hand can become safely separated without moving another
            # full center-displacement threshold. Ask MoveIt immediately; it
            # remains the authority that collision-checks the resumed route.
            self._retry_waiting_request(self._dynamic_obstacle[1])

    def _joints(self, message: JointState) -> None:
        self._joint_names = set(message.name)
        if len(message.name) == len(message.position):
            self._joint_positions = {
                str(name): float(position)
                for name, position in zip(message.name, message.position, strict=True)
            }

    def _planning_scene(self, message: PlanningScene) -> None:
        for obstacle in message.world.collision_objects:
            self._dynamic_collision(obstacle)

    def _dynamic_collision(self, obstacle: CollisionObject) -> None:
        dynamic_ids = {"ground_truth_hand", "perception_hand_obstacle"}
        if obstacle.id not in dynamic_ids:
            return
        if obstacle.operation == obstacle.REMOVE:
            if self._dynamic_obstacle is not None and self._dynamic_obstacle[0] == obstacle.id:
                self._dynamic_obstacle = None
                step = self._current_step
                if (
                    self._resume_pending
                    and self._request_kind
                    and not self._busy
                    and step is not None
                    and step[1] in {"model_target", "explicit_target"}
                ):
                    self._resume_pending = False
                    self._planning_failures.pop(step, None)
                    self._set_state("DYNAMIC_OBSTACLE_CLEARED_RETRY")
                    self._schedule_route_retry(step, 0.20)
                self._waiting_obstacle_center = None
            return
        if not obstacle.primitives or not obstacle.primitive_poses:
            return
        centers: list[np.ndarray] = []
        sizes: list[np.ndarray] = []
        for primitive, pose in zip(
            obstacle.primitives, obstacle.primitive_poses, strict=False
        ):
            if primitive.type != SolidPrimitive.BOX or len(primitive.dimensions) != 3:
                continue
            centers.append(
                np.asarray(
                    [pose.position.x, pose.position.y, pose.position.z], dtype=float
                )
            )
            sizes.append(np.asarray(primitive.dimensions, dtype=float))
        union = union_axis_aligned_boxes(centers, sizes)
        if union is not None:
            center, size = union
            self._dynamic_obstacle = (obstacle.id, center, size)
            if (
                self._state
                in {"DYNAMIC_WAIT_FOR_OBSTACLE_CLEAR", "DYNAMIC_WAIT_FOR_CLEAR_PATH"}
                and self._waiting_obstacle_center is not None
                and float(np.linalg.norm(center - self._waiting_obstacle_center))
                >= float(self._config.get("waiting_obstacle_retry_motion_m", 0.06))
            ):
                self._retry_waiting_request(center)
            self._idle_avoidance_tick()

    def _start_next_step(self) -> None:
        if self._busy:
            self._set_state("BUSY")
            return
        if self._safety_state == "EMERGENCY_STOP":
            self._resume_pending = bool(self._request_kind)
            self._set_state(
                f"{self._request_kind.upper()}_QUEUED_{self._safety_state}"
            )
            return
        if (
            self._safety_state == "PAUSE"
            and not self._guarded_under_route()
            and not self._can_plan_soft_stop_detour()
            and not self._is_crisis_step()
            and not (
                self._sequence and self._sequence[0][1] in CRISIS_STEP_KINDS
            )
        ):
            self._resume_pending = bool(self._request_kind)
            self._set_state(
                f"{self._request_kind.upper()}_QUEUED_{self._safety_state}"
            )
            return
        if not self._sequence:
            self._current_step = None
            completed_request = self._request_kind
            if completed_request == "idle_evade":
                failed = tuple(sorted(self._failed_sides))
                self._request_kind = ""
                self._resume_pending = False
                self._under_route_active = False
                if failed:
                    self._idle_saved_joints = None
                    self._idle_evading_sides = ()
                    self._idle_evaded = False
                    self._idle_trigger_obstacle_center = None
                    self._idle_retry_after = self._now_sec() + float(
                        self._config.get("always_on_avoidance", {}).get(
                            "retry_backoff_sec", 2.0
                        )
                    )
                    self._set_state(
                        "IDLE_AVOIDANCE_FAILED:" + ",".join(failed)
                    )
                    self._emit("idle_avoidance_failed," + ",".join(failed))
                else:
                    self._idle_evaded = True
                    self._idle_clear_since = None
                    self._set_state("IDLE_AVOIDANCE_HOLD")
                    self._emit(
                        "idle_avoidance_hold," + ",".join(self._idle_evading_sides)
                    )
                return
            if completed_request == "idle_restore":
                failed = tuple(sorted(self._failed_sides))
                self._request_kind = ""
                self._resume_pending = False
                self._under_route_active = False
                if failed:
                    self._idle_evaded = True
                    self._idle_clear_since = None
                    self._idle_retry_after = self._now_sec() + float(
                        self._config.get("always_on_avoidance", {}).get(
                            "retry_backoff_sec", 2.0
                        )
                    )
                    self._set_state(
                        "IDLE_AVOIDANCE_RESTORE_DEFERRED:" + ",".join(failed)
                    )
                    self._emit("idle_avoidance_restore_deferred," + ",".join(failed))
                else:
                    restored = self._idle_evading_sides
                    self._clear_idle_avoidance()
                    self._set_state(self._holding_state(restored=True))
                    self._emit("idle_avoidance_restored," + ",".join(restored))
                return
            if self._failed_sides:
                failed = ",".join(sorted(self._failed_sides))
                complete = f"TARGET_UNREACHABLE:{failed}"
                if completed_request in {
                    "target",
                    "both_targets",
                    "left_target",
                    "right_target",
                }:
                    complete = (
                        f"PARTIAL_TARGET_HOLDING:FAILED={failed}"
                        if self._held_target_sides
                        else f"TARGET_UNREACHABLE:{failed}"
                    )
                elif completed_request == "home_both":
                    complete = f"PARTIAL_MOTION_COMPLETE:FAILED={failed}"
            else:
                complete = (
                    "HOME_REACHED"
                    if completed_request == "home_both"
                    else self._holding_state()
                )
            self._set_state(complete)
            self._emit(f"motion_complete,{completed_request},{self._planner_mode}")
            self._request_kind = ""
            self._resume_pending = False
            self._under_route_active = False
            return
        self._current_step = self._sequence.pop(0)
        self._active_side, kind = self._current_step
        if kind == "model_target":
            self._target = self._model_contact_target(self._active_side)
            self._dynamic_target_pub.publish(self._target)
            self._plan_target_or_under_route(kind)
        elif kind == "explicit_target":
            self._plan_target_or_under_route(kind)
        elif kind in ROUTE_STEP_KINDS:
            self._plan_route_waypoint(kind)
        elif kind in CRISIS_STEP_KINDS:
            self._plan_crisis_evade()
        elif kind == "idle_restore":
            self._plan_idle_restore()
        else:
            self._plan_home()

    def _plan_target_or_under_route(self, kind: str) -> None:
        if self._target is None:
            self._set_state("TARGET_NOT_SET")
            return
        route_key = (self._active_side, kind)
        route_config = self._config.get("dynamic_under_route", {})
        if (
            self._planner_mode == "dynamic"
            and bool(route_config.get("enabled", True))
            and route_key not in self._under_routed_steps
            and self._dynamic_obstacle is not None
        ):
            current = self._current_tcp_position()
            goal = np.asarray(
                [
                    self._target.pose.position.x,
                    self._target.pose.position.y,
                    self._target.pose.position.z,
                ],
                dtype=float,
            )
            if current is not None:
                obstacle_id, center, size = self._dynamic_obstacle
                route = avoidance_route_waypoints(
                    current,
                    goal,
                    center,
                    size,
                    collision_margin_m=float(route_config.get("collision_margin_m", 0.03)),
                    horizontal_clearance_m=float(
                        route_config.get("horizontal_clearance_m", 0.08)
                    ),
                    vertical_clearance_m=float(
                        route_config.get("vertical_clearance_m", 0.04)
                    ),
                    minimum_tcp_z_m=float(route_config.get("minimum_tcp_z_m", 0.16)),
                    maximum_tcp_z_m=float(route_config.get("maximum_tcp_z_m", 1.12)),
                    entry_horizontal_clearance_m=float(
                        route_config.get("entry_horizontal_clearance_m", 0.08)
                    ),
                    exit_horizontal_clearance_m=float(
                        route_config.get("exit_horizontal_clearance_m", 0.03)
                    ),
                    maximum_backtrack_m=float(
                        route_config.get("maximum_backtrack_m", 0.02)
                    ),
                    force_local_detour=(
                        self._active_side in self._force_local_detour_sides
                    ),
                    forced_local_clearance_m=float(
                        route_config.get("forced_local_clearance_m", 0.06)
                    ),
                )
                if route is not None:
                    entry_progress = route_progress(current, goal, route.entry)
                    exit_progress = route_progress(current, goal, route.exit)
                    self._under_routed_steps.add(route_key)
                    self._under_route_active = True
                    self._route_obstacle_center = center.copy()
                    self._route_waypoints = {
                        "under_approach": self._waypoint_message(
                            np.asarray(
                                [
                                    route.entry[0],
                                    route.entry[1],
                                    max(
                                        float(current[2]),
                                        float(
                                            center[2]
                                            + size[2] / 2.0
                                            + float(
                                                route_config.get(
                                                    "approach_clearance_m", 0.08
                                                )
                                            )
                                        ),
                                    ),
                                ],
                                dtype=float,
                            )
                        ),
                        "under_entry": self._waypoint_message(route.entry),
                        "under_exit": self._waypoint_message(route.exit),
                        # Rise only after reaching the far side of the box.
                        # Planning the final boundary target directly from a
                        # low under-exit posture repeatedly exhausted OMPL.
                        "under_recover": self._waypoint_message(
                            np.asarray(
                                [
                                    route.exit[0],
                                    route.exit[1],
                                    max(
                                        float(goal[2]),
                                        float(
                                            center[2]
                                            + size[2] / 2.0
                                            + float(
                                                route_config.get(
                                                    "approach_clearance_m", 0.08
                                                )
                                            )
                                        ),
                                    ),
                                ],
                                dtype=float,
                            )
                        ),
                    }
                    original = self._current_step
                    assert original is not None
                    self._sequence = [
                        (self._active_side, "under_entry"),
                        (self._active_side, "under_exit"),
                        (self._active_side, "under_recover"),
                        original,
                        *self._sequence,
                    ]
                    self._current_step = (self._active_side, "under_approach")
                    self._emit(
                        "dynamic_under_route_selected,"
                        f"{self._active_side},{obstacle_id},{route.strategy},"
                        f"{route.entry[0]:.3f},{route.entry[1]:.3f},{route.entry[2]:.3f},"
                        f"{route.exit[0]:.3f},{route.exit[1]:.3f},{route.exit[2]:.3f},"
                        f"entry_progress={entry_progress:.3f},"
                        f"exit_progress={exit_progress:.3f}"
                    )
                    self._set_state(f"{self._active_side.upper()}_UNDER_ROUTE_APPROACH")
                    self._plan_route_waypoint("under_approach")
                    return
                self._emit(
                    f"dynamic_under_route_skipped,{self._active_side},not_blocking_or_infeasible"
                )
            else:
                self._emit(f"dynamic_under_route_skipped,{self._active_side},tcp_tf_unavailable")
        elif self._planner_mode == "dynamic" and self._dynamic_obstacle is None:
            self._emit(f"dynamic_under_route_skipped,{self._active_side},no_obstacle_box")
        self._plan_target()

    def _current_tcp_position(self) -> np.ndarray | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self._config["planning_frame"]),
                str(self._config["end_effector_links"][self._active_side]),
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception as error:
            self.get_logger().warning(
                f"TCP transform unavailable for under-route: {error}",
                throttle_duration_sec=2.0,
            )
            return None
        translation = transform.transform.translation
        return np.asarray([translation.x, translation.y, translation.z], dtype=float)

    def _waypoint_message(self, xyz: np.ndarray) -> PoseStamped:
        waypoint = PoseStamped()
        waypoint.header.frame_id = str(self._config["planning_frame"])
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.pose = Pose()
        waypoint.pose.position.x, waypoint.pose.position.y, waypoint.pose.position.z = map(
            float, xyz
        )
        waypoint.pose.orientation.w = 1.0
        return waypoint

    def _model_contact_target(self, side: str) -> PoseStamped:
        marker = self._targets[side]
        target = PoseStamped()
        target.header = marker.header
        target.pose = Pose()
        target.pose.orientation = marker.pose.orientation
        center = np.asarray(
            [
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z,
            ],
            dtype=float,
        )
        current = self._current_tcp_position()
        if current is None:
            contact = center
        else:
            contact = target_contact_point(
                center,
                current,
                float(self._config.get("target_contact_offset_m", 0.0)),
            )
        (
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
        ) = map(float, contact)
        self._emit(
            f"target_contact_goal,{side},{contact[0]:.3f},"
            f"{contact[1]:.3f},{contact[2]:.3f}"
        )
        return target

    def _plan_route_waypoint(self, kind: str) -> None:
        waypoint = self._route_waypoints.get(kind)
        if waypoint is None:
            self._set_state("UNDER_ROUTE_WAYPOINT_MISSING")
            return
        # The dynamic monitor must check the segment that is actually being
        # executed.  Leaving the final marker here made every safe bypass look
        # as if it still crossed the hand's original straight-line corridor.
        self._dynamic_target_pub.publish(waypoint)
        self._set_state(f"PLANNING_{kind.upper()}")
        route_config = self._config.get("dynamic_under_route", {})
        tolerance_key = {
            "under_exit": "exit_waypoint_tolerance_m",
            "under_recover": "recover_waypoint_tolerance_m",
        }.get(kind, "waypoint_tolerance_m")
        self._plan_position(
            waypoint,
            float(route_config.get(tolerance_key, 0.025)),
            constrain_orientation=False,
        )

    def _plan_target(self) -> None:
        if self._target is None:
            self._set_state("TARGET_NOT_SET")
            return
        step = self._current_step
        failures = self._planning_failures.get(step, 0) if step is not None else 0
        base_tolerance = float(self._config["pose_tolerance_m"])
        tolerance = progressive_pose_tolerance(
            base_tolerance,
            failures,
            float(self._config.get("maximum_pose_tolerance_m", base_tolerance)),
        )
        if tolerance > base_tolerance:
            self._emit(
                f"target_tolerance_fallback,{self._active_side},"
                f"{failures},{tolerance:.4f}"
            )
        self._plan_position(
            self._target,
            tolerance,
            constrain_orientation=bool(
                self._config.get("pose_goal_constrain_orientation", False)
            ),
        )

    def _plan_position(
        self,
        target: PoseStamped,
        tolerance: float,
        *,
        constrain_orientation: bool,
    ) -> None:
        position = PositionConstraint()
        position.header = target.header
        position.link_name = str(
            self._config["end_effector_links"][self._active_side]
        )
        position.weight = 1.0
        position.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[2.0 * tolerance] * 3)
        )
        position.constraint_region.primitive_poses.append(target.pose)
        constraints = Constraints(position_constraints=[position])
        if constrain_orientation:
            orientation = OrientationConstraint()
            orientation.header = target.header
            orientation.link_name = str(
                self._config["end_effector_links"][self._active_side]
            )
            orientation.orientation = target.pose.orientation
            orientation.absolute_x_axis_tolerance = float(
                self._config["orientation_tolerance_rad"]
            )
            orientation.absolute_y_axis_tolerance = orientation.absolute_x_axis_tolerance
            orientation.absolute_z_axis_tolerance = orientation.absolute_x_axis_tolerance
            orientation.weight = 1.0
            constraints.orientation_constraints.append(orientation)
        self._send_plan(constraints)

    def _plan_home(self) -> None:
        self._plan_joint_values(
            [float(value) for value in self._robot["home"][self._active_side]]
        )

    def _plan_idle_restore(self) -> None:
        values = (
            self._idle_saved_joints.get(self._active_side)
            if self._idle_saved_joints is not None
            else None
        )
        if values is None:
            self._failed_sides[self._active_side] = MoveItErrorCodes.INVALID_ROBOT_STATE
            self._current_step = None
            self._start_next_step()
            return
        self._plan_joint_values(values)

    def _plan_crisis_evade(self) -> None:
        target = self._crisis_escape_targets.get(self._active_side)
        if target is None:
            if self._current_step is not None and self._current_step[1] == "idle_evade":
                self._failed_sides[self._active_side] = (
                    MoveItErrorCodes.INVALID_MOTION_PLAN
                )
                self._current_step = None
                self._start_next_step()
                return
            # Missing TF/obstacle geometry must not generate an unchecked
            # Cartesian guess.  The raised, collision-checked Home posture is
            # the deterministic fallback.
            self._emit(f"crisis_escape_home_fallback,{self._active_side},no_target")
            self._plan_home()
            return
        config = self._config.get("crisis_avoidance", {})
        self._dynamic_target_pub.publish(target)
        self._set_state(f"CRISIS_ESCAPE_PLANNING:{self._active_side.upper()}")
        self._plan_position(
            target,
            float(config.get("pose_tolerance_m", 0.035)),
            # While holding a target, the reflex is deliberately a plain
            # vertical lift. Keeping the tool orientation prevents OMPL from
            # producing the large wrist/shoulder swivel seen in the GUI.
            constrain_orientation=bool(
                self._current_step is not None
                and self._current_step[1] == "idle_evade"
            ),
        )

    def _plan_joint_values(self, values: list[float]) -> None:
        constraints = Constraints()
        for name, value in zip(
            self._robot["joint_names"][self._active_side],
            values,
            strict=True,
        ):
            constraint = JointConstraint()
            constraint.joint_name = str(name)
            constraint.position = float(value)
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        self._send_plan(constraints)

    def _send_plan(self, constraints: Constraints) -> None:
        if self._busy:
            self._set_state("BUSY")
            return
        if not set(self._robot["joint_names"][self._active_side]).issubset(
            self._joint_names
        ):
            self._set_state("JOINT_STATES_NOT_READY")
            return
        if self._safety_state == "EMERGENCY_STOP" or (
            self._safety_state == "PAUSE" and not self._guarded_under_route()
            and not self._is_crisis_step()
        ):
            self._resume_pending = True
            self._set_state(
                f"{self._active_side.upper()}_TARGET_BLOCKED_{self._safety_state}"
            )
            return
        if not self._move_client.wait_for_server(timeout_sec=0.5):
            self._set_state("MOVEIT_NOT_READY")
            return
        goal = MoveGroup.Goal()
        goal.request.group_name = str(self._config["groups"][self._active_side])
        step_kind = self._current_step[1] if self._current_step else "target"
        route_config = self._config.get("dynamic_under_route", {})
        crisis_config = self._config.get("crisis_avoidance", {})
        is_route = step_kind in ROUTE_STEP_KINDS
        is_crisis = step_kind in CRISIS_STEP_KINDS
        goal.request.num_planning_attempts = int(
            crisis_config.get("planning_attempts", self._config["planning_attempts"])
            if is_crisis
            else (
                route_config.get(
                    "planning_attempts", self._config["planning_attempts"]
                )
                if is_route
                else self._config["planning_attempts"]
            )
        )
        goal.request.allowed_planning_time = float(
            crisis_config.get("planning_time_sec", self._config["planning_time_sec"])
            if is_crisis
            else (
                route_config.get(
                    "planning_time_sec", self._config["planning_time_sec"]
                )
                if is_route
                else self._config["planning_time_sec"]
            )
        )
        idle_config = self._config.get("always_on_avoidance", {})
        is_idle_reflex = step_kind in {"idle_evade", "idle_restore"}
        goal.request.max_velocity_scaling_factor = float(
            idle_config.get(
                "restore_velocity_scaling", self._config["velocity_scaling"]
            )
            if step_kind == "idle_restore"
            else idle_config.get("velocity_scaling", 0.12)
            if is_idle_reflex
            else (
                crisis_config.get(
                    "velocity_scaling", self._config["velocity_scaling"]
                )
                if is_crisis
                else self._config["velocity_scaling"]
            )
        )
        goal.request.max_acceleration_scaling_factor = float(
            idle_config.get(
                "restore_acceleration_scaling",
                self._config["acceleration_scaling"],
            )
            if step_kind == "idle_restore"
            else idle_config.get("acceleration_scaling", 0.10)
            if is_idle_reflex
            else self._config["acceleration_scaling"]
        )
        goal.request.pipeline_id = str(self._config["planning_pipeline"])
        goal.request.planner_id = str(self._config["planner_id"])
        goal.request.goal_constraints = [constraints]
        goal.request.start_state.is_diff = True
        goal.planning_options = PlanningOptions(plan_only=True, look_around=False, replan=False)
        self._busy = True
        self._set_state(f"PLANNING:{self._active_side.upper()}:{step_kind.upper()}")
        generation = self._request_generation
        self._move_client.send_goal_async(goal).add_done_callback(
            lambda future, value=generation: self._plan_response(future, value)
        )

    def _plan_response(self, future: Any, generation: int) -> None:
        goal = future.result()
        if generation != self._request_generation:
            if goal is not None and goal.accepted:
                goal.cancel_goal_async()
            return
        if goal is None or not goal.accepted:
            self._busy = False
            self._set_state("PLANNING_REJECTED")
            return
        self._move_goal = goal
        goal.get_result_async().add_done_callback(
            lambda result_future, value=generation: self._plan_result(
                result_future, value
            )
        )

    def _plan_result(self, future: Any, generation: int) -> None:
        if generation != self._request_generation:
            return
        response = future.result()
        self._move_goal = None
        if response.status == GoalStatus.STATUS_CANCELED:
            self._busy = False
            return
        result = response.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._busy = False
            step = self._current_step
            if step is not None and step[1] in CRISIS_STEP_KINDS:
                failures = self._planning_failures.get(step, 0) + 1
                self._planning_failures[step] = failures
                if failures <= int(
                    self._config.get("crisis_avoidance", {}).get(
                        "planning_retry_attempts", 1
                    )
                ):
                    if step[1] == "idle_evade":
                        self._emit(
                            f"idle_vertical_lift_retry,{step[0]},"
                            f"planning_error={result.error_code.val}"
                        )
                        self._set_state("IDLE_VERTICAL_LIFT_RETRY")
                        self._plan_crisis_evade()
                        return
                    self._emit(
                        f"crisis_escape_home_fallback,{step[0]},"
                        f"planning_error={result.error_code.val}"
                    )
                    self._set_state("CRISIS_ESCAPE_HOME_FALLBACK")
                    self._plan_home()
                    return
            route_config = self._config.get("dynamic_under_route", {})
            if (
                self._planner_mode == "dynamic"
                and step is not None
                and step[1] in ROUTE_STEP_KINDS
            ):
                failures = self._planning_failures.get(step, 0) + 1
                self._planning_failures[step] = failures
                maximum = int(route_config.get("planning_retry_attempts", 3))
                moved = bool(
                    self._dynamic_obstacle is not None
                    and self._route_obstacle_center is not None
                    and float(
                        np.linalg.norm(
                            self._dynamic_obstacle[1] - self._route_obstacle_center
                        )
                    )
                    >= float(route_config.get("route_refresh_motion_m", 0.05))
                )
                refreshes = self._route_refreshes.get(step[0], 0)
                maximum_refreshes = int(
                    route_config.get("maximum_route_refresh_attempts", 4)
                )
                if moved and refreshes < maximum_refreshes:
                    self._route_refreshes[step[0]] = refreshes + 1
                    self._emit(
                        f"dynamic_route_refreshed,{step[0]},{step[1]},"
                        f"{refreshes + 1},{result.error_code.val}"
                    )
                    self._restart_dynamic_route(step[0])
                    return
                if failures <= maximum:
                    self._set_state(f"PLANNING_RETRY:{step[1]}:{failures}")
                    self._emit(
                        f"dynamic_under_planning_retry,{step[0]},{step[1]},{failures},"
                        f"{result.error_code.val}"
                    )
                    self._schedule_route_retry(
                        step,
                        float(route_config.get("planning_retry_backoff_sec", 0.8)),
                    )
                    return
                if (
                    self._dynamic_obstacle is not None
                    and self._dynamic_path_is_blocked()
                ):
                    # The target is not unreachable merely because a moving
                    # hand invalidated every short-lived guide. Preserve the
                    # target and resume as soon as the measured hand leaves
                    # the current TCP-to-target corridor.
                    self._wait_for_clear_dynamic_path(step[0])
                    return
                if self._dynamic_obstacle is not None:
                    self._emit(
                        f"dynamic_route_failed_clear_path,{step[0]},"
                        f"{step[1]},{result.error_code.val}"
                    )
                # A short detector gap can remove the collision object at the
                # exact instant a stale guide fails. Rebuild the original
                # target immediately instead of mislabelling it unreachable.
                if self._dynamic_obstacle is None:
                    self._emit(
                        f"dynamic_route_obstacle_vanished,{step[0]},{step[1]}"
                    )
                    self._restart_dynamic_route(step[0])
                    return
            if (
                self._planner_mode == "dynamic"
                and step is not None
                and step[1] in {"model_target", "explicit_target"}
            ):
                failures = self._planning_failures.get(step, 0) + 1
                self._planning_failures[step] = failures
                maximum = int(self._config.get("maximum_replan_attempts", 3))
                if failures <= maximum:
                    # A moving obstacle can enter while OMPL is sampling a
                    # previously clear straight goal.  Re-read its latest box
                    # and reconsider the under-route instead of immediately
                    # declaring a reachable marker impossible.
                    self._under_routed_steps.discard(step)
                    self._set_state(
                        f"DYNAMIC_TARGET_RETRY:{step[0]}:{failures}"
                    )
                    self._emit(
                        f"dynamic_target_planning_retry,{step[0]},"
                        f"{failures},{result.error_code.val}"
                    )
                    self._schedule_route_retry(
                        step,
                        float(self._config.get("replan_backoff_sec", 0.75)),
                    )
                    return
                if (
                    self._dynamic_obstacle is not None
                    and self._dynamic_path_is_blocked()
                ):
                    # Do not convert a temporarily blocked moving-hand scene
                    # into a permanent task failure.  Keep the request and
                    # resume automatically when the perception obstacle is
                    # removed after the hand leaves the camera workspace.
                    self._resume_pending = True
                    self._waiting_obstacle_center = self._dynamic_obstacle[1].copy()
                    self._set_state("DYNAMIC_WAIT_FOR_OBSTACLE_CLEAR")
                    self._emit(
                        f"dynamic_wait_for_clear,{step[0]},"
                        f"{failures},{result.error_code.val}"
                    )
                    return
            failed_side = self._active_side
            self._failed_sides[failed_side] = int(result.error_code.val)
            self._set_state(
                f"{failed_side.upper()}_TARGET_UNREACHABLE:{result.error_code.val}"
            )
            self._emit(
                f"planning_failed,{result.error_code.val},{failed_side},"
                f"{step[1] if step else 'unknown'}"
            )
            self._sequence = preserve_home_after_side_failure(
                self._sequence, failed_side
            )
            self._current_step = None
            # A failed left target must not make MOVE BOTH look dead. Continue
            # with the independent arm, then report a visible partial result.
            self._start_next_step()
            return
        if self._current_step is not None:
            self._planning_failures.pop(self._current_step, None)
        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.points:
            self._busy = False
            self._set_state("EMPTY_TRAJECTORY")
            return
        if (
            self._planner_mode == "dynamic"
            and self._current_step is not None
            and self._current_step[1] in CRISIS_STEP_KINDS
        ):
            # This is the only PAUSE bypass: the goal was generated upward and
            # away from the live hand, then collision-checked by MoveIt.  It is
            # sent directly because the nominal dynamic gate intentionally
            # blocks every ordinary trajectory during PAUSE.  A true latched
            # EMERGENCY_STOP still cancels it through _safety().
            if self._safety_state == "EMERGENCY_STOP":
                self._busy = False
                self._resume_pending = True
                self._set_state("CRISIS_ESCAPE_BLOCKED_ESTOP")
                return
            self._emit(
                f"crisis_escape_execute,{self._active_side},"
                f"{self._current_step[1]},{self._safety_state}"
            )
            self._safety_command_pub.publish(String(data="escape_started"))
            self._set_state("CRISIS_ESCAPE_EXECUTING")
            self._execute(trajectory)
        elif self._planner_mode == "dynamic":
            self._dynamic_input_pub.publish(trajectory)
            self._set_state("DYNAMIC_SAFETY_CHECK")
        else:
            self._execute(trajectory)

    def _dynamic_trajectory(self, message: JointTrajectory) -> None:
        if self._planner_mode == "dynamic" and self._busy:
            self._execute(message)

    def _dynamic_replan_request(self, message: String) -> None:
        if (
            self._planner_mode != "dynamic"
            or not self._request_kind
            or self._current_step is None
            or self._safety_state == "EMERGENCY_STOP"
        ):
            return
        self._spatial_replan_count += 1
        self._emit(
            f"dynamic_spatial_replan,{message.data},{self._spatial_replan_count}"
        )
        self._cancel_motion()
        if self._current_step[1] in {
            "model_target",
            "explicit_target",
            *ROUTE_STEP_KINDS,
        }:
            # The obstacle moved, so old Cartesian waypoints are stale. Build
            # a fresh route from the measured TCP and latest Planning Scene
            # box. Planning is allowed during soft PAUSE; execution remains
            # gated until this new route has been collision checked.
            self._restart_dynamic_route(self._active_side)
            return
        if self._safety_state == "PAUSE":
            self._resume_pending = True
            self._set_state("DYNAMIC_REPLAN_QUEUED")
            return
        self._set_state("DYNAMIC_REPLANNING")
        self._replan_last_request()

    def _restart_dynamic_route(self, side: str) -> None:
        """Discard stale guides and route again from the measured TCP."""

        target_kind = (
            "explicit_target" if self._request_kind == "target" else "model_target"
        )
        self._sequence = preserve_home_after_side_failure(self._sequence, side)
        self._sequence.insert(0, (side, target_kind))
        self._current_step = None
        self._route_waypoints = {}
        self._route_obstacle_center = None
        self._under_routed_steps = {
            routed for routed in self._under_routed_steps if routed[0] != side
        }
        self._under_route_active = False
        self._failed_sides.pop(side, None)
        self._resume_pending = False
        self._set_state("DYNAMIC_ROUTE_REFRESH")
        self._start_next_step()

    def _wait_for_clear_dynamic_path(self, side: str) -> None:
        target_kind = (
            "explicit_target" if self._request_kind == "target" else "model_target"
        )
        self._sequence = preserve_home_after_side_failure(self._sequence, side)
        self._current_step = (side, target_kind)
        self._active_side = side
        self._route_waypoints = {}
        self._route_obstacle_center = None
        self._under_routed_steps = {
            routed for routed in self._under_routed_steps if routed[0] != side
        }
        self._under_route_active = False
        self._busy = False
        self._resume_pending = True
        self._waiting_obstacle_center = self._dynamic_obstacle[1].copy()
        self._set_state("DYNAMIC_WAIT_FOR_CLEAR_PATH")
        self._emit(f"dynamic_wait_for_clear_path,{side}")

    def _execute(self, trajectory: JointTrajectory) -> None:
        trajectory_client = self._trajectory_clients[self._active_side]
        if not trajectory_client.wait_for_server(timeout_sec=0.5):
            self._busy = False
            self._set_state("CONTROLLER_NOT_READY")
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self._set_state("EXECUTING")
        generation = self._request_generation
        trajectory_client.send_goal_async(goal).add_done_callback(
            lambda future, value=generation: self._trajectory_response(
                future, value
            )
        )

    def _trajectory_response(self, future: Any, generation: int) -> None:
        goal = future.result()
        if generation != self._request_generation:
            if goal is not None and goal.accepted:
                goal.cancel_goal_async()
            return
        if goal is None or not goal.accepted:
            self._busy = False
            self._set_state("TRAJECTORY_REJECTED")
            return
        self._trajectory_goal = goal
        goal.get_result_async().add_done_callback(
            lambda result_future, value=generation: self._trajectory_result(
                result_future, value
            )
        )

    def _trajectory_result(self, future: Any, generation: int) -> None:
        if generation != self._request_generation:
            return
        response = future.result()
        self._trajectory_goal = None
        self._busy = False
        if response.status == GoalStatus.STATUS_CANCELED:
            return
        if (
            response.status == GoalStatus.STATUS_SUCCEEDED
            and response.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        ):
            step_kind = self._current_step[1] if self._current_step else "target"
            if step_kind in {"model_target", "explicit_target"}:
                self._held_target_sides.add(self._active_side)
            elif step_kind == "home":
                self._held_target_sides.discard(self._active_side)
            if step_kind in ROUTE_STEP_KINDS:
                self._emit(
                    f"dynamic_under_waypoint_reached,{self._active_side},{step_kind}"
                )
            if step_kind == "under_recover":
                self._under_route_active = False
            suffix = {
                "home": "HOME_REACHED",
                "idle_evade": "IDLE_EVADE_REACHED",
                "safety_evade": "SAFETY_EVADE_REACHED",
                "idle_restore": "IDLE_RESTORE_REACHED",
                "under_approach": "UNDER_APPROACH_REACHED",
                "under_entry": "UNDER_ENTRY_REACHED",
                "under_exit": "UNDER_EXIT_REACHED",
                "under_recover": "UNDER_RECOVER_REACHED",
            }.get(step_kind, "TARGET_REACHED")
            self._set_state(f"{self._active_side.upper()}_{suffix}")
            self._busy = False
            self._start_next_step()
        else:
            self._set_state(f"TRAJECTORY_FAILED:{response.result.error_code}")

    def _replan_last_request(self) -> None:
        self._busy = False
        if self._current_step is None:
            self._start_next_step()
            return
        _, kind = self._current_step
        if kind in {"model_target", "explicit_target"}:
            if kind == "model_target" and self._active_side in self._targets:
                self._target = self._model_contact_target(self._active_side)
            self._plan_target_or_under_route(kind)
        elif kind in ROUTE_STEP_KINDS:
            self._plan_route_waypoint(kind)
        elif kind == "home":
            self._plan_home()
        elif kind in CRISIS_STEP_KINDS:
            self._plan_crisis_evade()
        elif kind == "idle_restore":
            self._plan_idle_restore()

    def _cancel_motion(self) -> None:
        self._request_generation += 1
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self.destroy_timer(self._retry_timer)
            self._retry_timer = None
        if self._move_goal is not None:
            self._move_goal.cancel_goal_async()
        if self._trajectory_goal is not None:
            self._trajectory_goal.cancel_goal_async()
        self._move_goal = None
        self._trajectory_goal = None
        self._busy = False

    def _schedule_route_retry(
        self, step: tuple[str, str], delay_sec: float
    ) -> None:
        generation = self._request_generation
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self.destroy_timer(self._retry_timer)
        timer: Any | None = None

        def retry() -> None:
            assert timer is not None
            timer.cancel()
            self.destroy_timer(timer)
            if self._retry_timer is timer:
                self._retry_timer = None
            if (
                generation == self._request_generation
                and self._current_step == step
                and not self._busy
            ):
                if step[1] in ROUTE_STEP_KINDS:
                    self._plan_route_waypoint(step[1])
                else:
                    if step[1] == "model_target":
                        self._target = self._model_contact_target(step[0])
                        self._dynamic_target_pub.publish(self._target)
                    self._plan_target_or_under_route(step[1])

        timer = self.create_timer(max(float(delay_sec), 0.05), retry)
        self._retry_timer = timer

    def _guarded_under_route(self) -> bool:
        return bool(
            self._planner_mode == "dynamic"
            and self._under_route_active
            and self._current_step is not None
            and self._current_step[1]
            in ROUTE_STEP_KINDS
        )

    def _can_plan_soft_stop_detour(self) -> bool:
        """Allow collision-checked detour planning while motion is paused.

        PAUSE still blocks execution of a straight trajectory.  In Dynamic
        mode, however, refusing to *plan* before a guarded route exists creates
        a deadlock: the route can never become guarded.  Only target requests
        with a current Planning Scene obstacle may pass this planning gate.
        """

        if self._planner_mode != "dynamic" or self._dynamic_obstacle is None:
            return False
        if not self._sequence:
            return False
        return self._sequence[0][1] in {"model_target", "explicit_target"}

    def _retry_waiting_request(self, obstacle_center: np.ndarray) -> None:
        step = self._current_step
        if (
            step is None
            or step[1] not in {"model_target", "explicit_target"}
            or self._busy
            or self._retry_timer is not None
        ):
            return
        if self._state == "DYNAMIC_WAIT_FOR_CLEAR_PATH" and not waiting_path_should_retry(
            self._safety_state,
            self._safety_distance,
            float(
                self._config.get("always_on_avoidance", {}).get(
                    "release_distance_m", 0.26
                )
            ),
            coarse_path_blocked=self._dynamic_path_is_blocked(),
        ):
            # Track the moving hand without consuming a planning attempt.
            # The next measured displacement will check the corridor again.
            self._waiting_obstacle_center = obstacle_center.copy()
            return
        attempts = self._waiting_retries.get(step, 0) + 1
        self._waiting_retries[step] = attempts
        maximum = int(self._config.get("waiting_obstacle_retry_attempts", 2))
        if attempts > maximum:
            failed_side = step[0]
            self._resume_pending = False
            self._failed_sides[failed_side] = MoveItErrorCodes.PLANNING_FAILED
            self._sequence = preserve_home_after_side_failure(
                self._sequence, failed_side
            )
            self._current_step = None
            self._waiting_obstacle_center = obstacle_center.copy()
            self._set_state(
                f"DYNAMIC_RETRY_EXHAUSTED:{failed_side}:{attempts - 1}"
            )
            self._emit(
                f"dynamic_wait_retry_exhausted,{failed_side},{attempts - 1}"
            )
            self._start_next_step()
            return
        self._waiting_obstacle_center = obstacle_center.copy()
        self._resume_pending = False
        self._planning_failures.pop(step, None)
        self._under_routed_steps.discard(step)
        self._set_state("DYNAMIC_MOVED_OBSTACLE_RETRY")
        self._emit(f"dynamic_moved_obstacle_retry,{step[0]}")
        self._schedule_route_retry(step, 0.20)

    def _dynamic_path_is_blocked(self) -> bool:
        if self._dynamic_obstacle is None or self._target is None:
            return False
        current = self._current_tcp_position()
        if current is None:
            return True
        goal = np.asarray(
            [
                self._target.pose.position.x,
                self._target.pose.position.y,
                self._target.pose.position.z,
            ],
            dtype=float,
        )
        _, center, size = self._dynamic_obstacle
        config = self._config.get("dynamic_under_route", {})
        route = avoidance_route_waypoints(
            current,
            goal,
            center,
            size,
            collision_margin_m=float(config.get("collision_margin_m", 0.03)),
            horizontal_clearance_m=float(
                config.get("horizontal_clearance_m", 0.08)
            ),
            vertical_clearance_m=float(config.get("vertical_clearance_m", 0.04)),
            minimum_tcp_z_m=float(config.get("minimum_tcp_z_m", 0.16)),
            maximum_tcp_z_m=float(config.get("maximum_tcp_z_m", 1.12)),
            entry_horizontal_clearance_m=float(
                config.get("entry_horizontal_clearance_m", 0.08)
            ),
            exit_horizontal_clearance_m=float(
                config.get("exit_horizontal_clearance_m", 0.03)
            ),
            maximum_backtrack_m=float(config.get("maximum_backtrack_m", 0.02)),
            force_local_detour=(self._active_side in self._force_local_detour_sides),
            forced_local_clearance_m=float(
                config.get("forced_local_clearance_m", 0.06)
            ),
        )
        return route is not None

    def _idle_avoidance_tick(self) -> None:
        config = self._config.get("always_on_avoidance", {})
        if not bool(config.get("enabled", True)) or self._planner_mode != "dynamic":
            return
        if self._request_kind:
            return
        now = self._now_sec()
        if now < self._idle_retry_after:
            return
        obstacle = self._dynamic_obstacle
        trigger = float(config.get("trigger_distance_m", 0.35))
        release = float(config.get("release_distance_m", 0.40))
        dangerous = bool(
            obstacle is not None
            and np.isfinite(self._safety_distance)
            and self._safety_distance <= trigger
            and self._safety_state in {"WARNING", "PAUSE", "REPLAN"}
        )
        if self._idle_evaded:
            current_center = obstacle[1] if obstacle is not None else None
            if not obstacle_has_withdrawn(
                self._idle_trigger_obstacle_center,
                current_center,
                minimum_motion_m=float(
                    config.get("restore_obstacle_motion_m", 0.18)
                ),
            ):
                self._idle_clear_since = None
                return
            if dangerous:
                self._idle_clear_since = None
                return
            distance_is_clear = obstacle is None or (
                np.isfinite(self._safety_distance)
                and self._safety_distance >= release
            )
            if not distance_is_clear:
                self._idle_clear_since = None
                return
            if self._idle_clear_since is None:
                self._idle_clear_since = now
                return
            if self._safety_state == "EMERGENCY_STOP":
                # The general dynamic E-stop recovery tick owns reset.  Once
                # it advances the supervisor to RECOVER, this saved posture is
                # sent through MoveIt without losing the original target.
                return
            if self._safety_state not in {"SAFE", "RECOVER"}:
                # Keep the accumulated clear interval while the supervisor
                # advances RESET -> REPLAN -> RECOVER.  Clearing it here used
                # to add another delay and could miss short safe windows.
                return
            if now - self._idle_clear_since < float(
                config.get("clear_duration_sec", 1.0)
            ):
                return
            if self._idle_saved_joints and self._idle_evading_sides:
                self._begin_request(
                    "idle_restore",
                    [(side, "idle_restore") for side in self._idle_evading_sides],
                )
            return
        if not dangerous or obstacle is None:
            return
        snapshot: dict[str, list[float]] = {}
        for side in ("left", "right"):
            names = [str(name) for name in self._robot["joint_names"][side]]
            if not all(name in self._joint_positions for name in names):
                return
            snapshot[side] = [self._joint_positions[name] for name in names]
        _, center, size = obstacle
        sides = self._select_evading_arms(
            center,
            size,
            trigger_distance_m=float(config.get("tcp_trigger_distance_m", 0.30)),
        )
        if not sides:
            return
        self._idle_saved_joints = snapshot
        self._idle_evading_sides = sides
        self._idle_trigger_obstacle_center = center.copy()
        self._idle_clear_since = None
        sides = self._prepare_crisis_escape_targets(
            sides, center, size, vertical_only=True
        )
        if not sides:
            return
        self._idle_evading_sides = sides
        self._emit(
            f"idle_avoidance_triggered,{','.join(sides)},{self._safety_distance:.3f}"
        )
        self._safety_command_pub.publish(String(data="escape_started"))
        self._begin_request(
            "idle_evade", [(side, "idle_evade") for side in sides]
        )

    def _tcp_position(self, side: str) -> np.ndarray | None:
        previous = self._active_side
        self._active_side = side
        try:
            return self._current_tcp_position()
        finally:
            self._active_side = previous

    def _tcp_orientation(self, side: str) -> np.ndarray | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self._config["planning_frame"]),
                str(self._config["end_effector_links"][side]),
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None
        rotation = transform.transform.rotation
        value = np.asarray(
            [rotation.x, rotation.y, rotation.z, rotation.w], dtype=float
        )
        norm = float(np.linalg.norm(value))
        return value / norm if np.isfinite(norm) and norm > 1e-9 else None

    def _arm_link_points(self, side: str) -> np.ndarray:
        """Sample every arm link plus segment midpoints in the world frame."""

        frames = [
            *(f"openarm_{side}_link{index}" for index in range(1, 8)),
            f"openarm_{side}_hand",
            str(self._config["end_effector_links"][side]),
        ]
        points: list[np.ndarray] = []
        for frame in frames:
            try:
                transform = self._tf_buffer.lookup_transform(
                    str(self._config["planning_frame"]),
                    frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.02),
                )
            except Exception:
                continue
            translation = transform.transform.translation
            points.append(
                np.asarray(
                    [translation.x, translation.y, translation.z], dtype=float
                )
            )
        if not points:
            return np.empty((0, 3), dtype=float)
        # Link origins alone can miss an obstacle at the middle of a long
        # forearm. Segment midpoints make side selection follow the visible
        # arm instead of whichever TCP happens to be closer.
        midpoints = [
            (first + second) / 2.0
            for first, second in zip(points, points[1:], strict=False)
        ]
        return np.stack([*points, *midpoints])

    def _select_evading_arms(
        self,
        center: np.ndarray,
        size: np.ndarray,
        *,
        trigger_distance_m: float,
    ) -> tuple[str, ...]:
        points = {
            side: self._arm_link_points(side) for side in ("left", "right")
        }
        sides = select_evading_sides_from_links(
            points,
            center,
            size,
            trigger_distance_m=trigger_distance_m,
        )
        if sides:
            return sides
        # Preserve the previous TCP-only fallback during initial TF startup.
        tcp_positions = {
            side: position
            for side in ("left", "right")
            if (position := self._tcp_position(side)) is not None
        }
        return select_evading_sides(
            tcp_positions,
            center,
            size,
            trigger_distance_m=trigger_distance_m,
        )

    def _clear_idle_avoidance(self) -> None:
        self._idle_saved_joints = None
        self._idle_evading_sides = ()
        self._idle_evaded = False
        self._idle_clear_since = None
        self._idle_trigger_obstacle_center = None

    def _holding_state(self, *, restored: bool = False) -> str:
        sides = tuple(
            side for side in ("left", "right") if side in self._held_target_sides
        )
        if not sides:
            return "IDLE_AVOIDANCE_RESTORED" if restored else "IDLE"
        prefix = "TARGETS_RESTORED" if restored else "TARGETS_REACHED"
        if len(sides) == 1:
            prefix = f"{sides[0].upper()}_TARGET_" + (
                "RESTORED" if restored else "REACHED"
            )
        return f"{prefix}_HOLDING"

    def _automatic_estop_recovery_tick(self) -> None:
        config = self._config.get("dynamic_estop_recovery", {})
        if (
            not bool(config.get("enabled", False))
            or self._planner_mode != "dynamic"
            or self._safety_state != "EMERGENCY_STOP"
            or self._estop_reset_requested
        ):
            return
        obstacle = self._dynamic_obstacle
        current_center = obstacle[1] if obstacle is not None else None
        if self._estop_trigger_obstacle_center is None and current_center is not None:
            self._estop_trigger_obstacle_center = current_center.copy()
            self._estop_clear_since = None
            return
        if not obstacle_is_clear_for_recovery(
            self._estop_trigger_obstacle_center,
            current_center,
            minimum_motion_m=float(config.get("obstacle_motion_m", 0.08)),
            current_distance_m=self._safety_distance,
            release_distance_m=float(config.get("release_distance_m", 0.26)),
        ):
            self._estop_clear_since = None
            return
        now = self._now_sec()
        if self._estop_clear_since is None:
            self._estop_clear_since = now
            return
        if now - self._estop_clear_since < float(config.get("clear_duration_sec", 1.5)):
            return
        self._estop_reset_requested = True
        self._safety_command_pub.publish(String(data="reset"))
        self._set_state("DYNAMIC_CLEAR_RESETTING_ESTOP")
        self._emit("dynamic_estop_reset_after_obstacle_clear")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_state(self, value: str) -> None:
        if value != self._state:
            self._state = value
            self._events_pub.publish(String(data=f"pose_goal_state,{value}"))

    def _emit(self, value: str) -> None:
        self._events_pub.publish(String(data=value))

    def _publish_state(self) -> None:
        self._automatic_estop_recovery_tick()
        self._idle_avoidance_tick()
        self._state_pub.publish(String(data=self._state))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OpenArmPoseGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
