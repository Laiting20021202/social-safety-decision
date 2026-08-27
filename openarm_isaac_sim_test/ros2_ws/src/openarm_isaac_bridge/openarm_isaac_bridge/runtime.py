from __future__ import annotations

import os
import threading
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from openarm_sim.camera_math import (
    ISAAC_WORLD_LINK_TO_ROS_OPTICAL_XYZW,
    back_project_depth,
    intrinsics_from_horizontal_fov,
    optical_points_to_world,
)
from openarm_sim.config import PROJECT_ROOT, load_yaml
from openarm_sim.contracts import RuntimeMode, assert_mode_isolation, subscriptions_for_mode

from .control import TrajectoryController
from .messages import camera_info_message, image_message, point_cloud_message, stamp_from_seconds


class IsaacRosBridge:
    def __init__(
        self,
        mode: str,
        camera: Any,
        robot: Any,
        hand: Any,
        camera_config: dict[str, Any],
        robot_config: dict[str, Any],
        hand_config: dict[str, Any],
        world: Any,
        cube_specs: list[Any],
    ) -> None:
        os.environ.setdefault("ROS_DOMAIN_ID", "0")
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.msg import CollisionObject
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
        from std_msgs.msg import Float64, String
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.mode = RuntimeMode(mode)
        assert_mode_isolation(self.mode, subscriptions_for_mode(self.mode))
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("openarm_isaac_bridge", parameter_overrides=[])
        self.camera = camera
        self.robot = robot
        self.hand = hand
        self.camera_config = camera_config
        self.robot_config = robot_config
        self.hand_config = hand_config
        self.world = world
        self.cube_specs = cube_specs
        self.task_state = "HOME"
        self._rng = np.random.default_rng(20260817)
        self._hand_commands: deque[str] = deque()
        self._manual_hand_targets: deque[tuple[float, float, float]] = deque(maxlen=1)
        self._cube_commands: deque[tuple[str, str]] = deque()
        self._attached_cube: Any | None = None
        self._attached_cube_name: str | None = None
        self._grasp_constraint_path: str | None = None
        self._grasp_config = load_yaml("config/sorting_task.yaml")["grasp"]
        self._frame_queue: deque[tuple[float, Any, np.ndarray, np.ndarray]] = deque()
        self._noise_config = dict(camera_config["noise"])
        for environment, key in (
            ("OPENARM_CAMERA_LATENCY_MS", "latency_ms"),
            ("OPENARM_CAMERA_FRAME_DROP_PROBABILITY", "frame_drop_probability"),
        ):
            if environment in os.environ:
                self._noise_config[key] = float(os.environ[environment])
                self._noise_config["enabled"] = True
        self._clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        # MoveIt's CurrentStateMonitor requests reliable durability. JointState
        # is control feedback, not a lossy image stream.
        self._joint_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self._color_pub = self.node.create_publisher(Image, camera_config["topics"]["color"], qos_profile_sensor_data)
        self._depth_pub = self.node.create_publisher(Image, camera_config["topics"]["depth"], qos_profile_sensor_data)
        self._aligned_pub = self.node.create_publisher(Image, camera_config["topics"]["aligned_depth"], qos_profile_sensor_data)
        self._color_info_pub = self.node.create_publisher(CameraInfo, camera_config["topics"]["color_info"], qos_profile_sensor_data)
        self._depth_info_pub = self.node.create_publisher(CameraInfo, camera_config["topics"]["depth_info"], qos_profile_sensor_data)
        self._points_pub = self.node.create_publisher(PointCloud2, camera_config["topics"]["points"], qos_profile_sensor_data)
        self._world_points_pub = self.node.create_publisher(
            PointCloud2,
            camera_config["topics"]["world_points"],
            qos_profile_sensor_data,
        )
        self._pointcloud_counter = 0
        self._tf = TransformBroadcaster(self.node)
        self._tf_static = StaticTransformBroadcaster(self.node)
        # Ground truth remains available for the evaluator in both modes. The
        # perception safety bridge is structurally forbidden from subscribing.
        self._gt_pose_pub = self.node.create_publisher(PoseStamped, "/sim/ground_truth/hand_pose", 10)
        self._gt_collision_pub = self.node.create_publisher(CollisionObject, "/sim/ground_truth/hand_collision", 10)
        self._gt_distance_pub = self.node.create_publisher(Float64, "/sim/ground_truth/min_distance", 10)
        self._cube_states_pub = self.node.create_publisher(String, "/sim/ground_truth/cube_states", 10)
        self._event_pub = self.node.create_publisher(String, "/openarm/events", 50)
        self.node.create_subscription(String, "/openarm/task/state", self._task_state_callback, 10)
        self.node.create_subscription(String, "/openarm/events", self._cube_event_callback, 50)
        self.node.create_subscription(String, "/sim/hand/command", self._hand_command_callback, 10)
        self.node.create_subscription(
            PoseStamped,
            "/sim/hand/manual_target_pose",
            self._manual_hand_target_callback,
            10,
        )
        self.control = TrajectoryController(robot, self.node, robot_config["controller_actions"])
        self.node.create_subscription(String, "/openarm/safety/state", self._safety_callback, 10)
        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        width = int(camera_config["resolution"]["width"])
        height = int(camera_config["resolution"]["height"])
        self.intrinsics = intrinsics_from_horizontal_fov(
            width, height, float(camera_config["horizontal_fov_deg"])
        )
        self._static_sent = False
        self._last_ground_truth_time = float("-inf")
        self._ground_truth_period_sec = 0.05
        self._screenshot_states = set(load_yaml("config/evaluation.yaml")["screenshot_states"])
        self._captured_states: set[str] = set()

    def update(self, sim_time: float, publish_camera: bool) -> None:
        from rosgraph_msgs.msg import Clock

        stamp = stamp_from_seconds(sim_time)
        self._clock_pub.publish(Clock(clock=stamp))
        self.control.update(sim_time)
        self._update_cube_attachment()
        self._publish_joint_state(stamp)
        self._publish_tf(stamp)
        if publish_camera:
            self._queue_rgbd(stamp, sim_time)
        # Ground-truth obstacle updates are simulator state, not camera
        # observations. Keep their watchdog cadence independent of RTX render
        # rate so low-FPS diagnostics cannot cause a false safety timeout.
        if sim_time - self._last_ground_truth_time >= self._ground_truth_period_sec - 1e-9:
            self._publish_ground_truth(stamp)
            self._last_ground_truth_time = sim_time
        self._flush_rgbd(sim_time)
        self._capture_state_screenshot(sim_time)

    def close(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        self._spin_thread.join(timeout=1.0)

    def consume_hand_commands(self) -> list[str]:
        commands = list(self._hand_commands)
        self._hand_commands.clear()
        return commands

    def consume_manual_hand_target(self) -> tuple[float, float, float] | None:
        if not self._manual_hand_targets:
            return None
        target = self._manual_hand_targets[-1]
        self._manual_hand_targets.clear()
        return target

    def publish_event(self, event: str, *values: str) -> None:
        from std_msgs.msg import String

        self._event_pub.publish(String(data=",".join((event, *values))))

    def _capture_state_screenshot(self, sim_time: float) -> None:
        if self.task_state not in self._screenshot_states or self.task_state in self._captured_states:
            return
        rgba = self.camera.get_rgba()
        if rgba is None:
            return
        rgba_array = np.asarray(rgba)
        # RTX annotators can briefly return an empty array while the first
        # render product is warming up.  Keep the state eligible for a later
        # screenshot instead of terminating the whole simulation.
        if rgba_array.ndim != 3 or rgba_array.shape[0] == 0 or rgba_array.shape[1] == 0:
            return
        from PIL import Image

        output_root = Path(os.environ.get("OPENARM_OUTPUT_ROOT", PROJECT_ROOT / "results"))
        run_id = os.environ.get("OPENARM_RUN_ID")
        if not run_id:
            return
        directory = output_root / run_id / "screenshots"
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{sim_time:09.3f}_{self.task_state.lower()}.png"
        Image.fromarray(rgba_array[..., :3].astype(np.uint8)).save(directory / filename)
        self._captured_states.add(self.task_state)

    def _queue_rgbd(self, stamp: Any, sim_time: float) -> None:
        rgba = self.camera.get_rgba()
        depth = self.camera.get_depth()
        if rgba is None or depth is None:
            return
        rgba_array = np.asarray(rgba)
        depth_array = np.asarray(depth)
        if (
            rgba_array.ndim != 3
            or depth_array.ndim != 2
            or rgba_array.shape[0] == 0
            or rgba_array.shape[1] == 0
            or rgba_array.shape[:2] != depth_array.shape
        ):
            return
        rgb = rgba_array[..., :3].astype(np.uint8, copy=False)
        depth_m = np.asarray(depth_array, dtype=np.float32)
        noise = self._noise_config
        if noise["enabled"]:
            if self._rng.random() < float(noise["frame_drop_probability"]):
                return
            if float(noise["rgb_stddev_255"]) > 0.0:
                rgb = np.clip(
                    rgb.astype(np.float32)
                    + self._rng.normal(0.0, float(noise["rgb_stddev_255"]), rgb.shape),
                    0,
                    255,
                ).astype(np.uint8)
            if float(noise["depth_stddev_m"]) > 0.0:
                depth_m = depth_m + self._rng.normal(
                    0.0, float(noise["depth_stddev_m"]), depth_m.shape
                ).astype(np.float32)
        release_time = sim_time + float(noise["latency_ms"]) / 1000.0
        self._frame_queue.append((release_time, stamp, rgb.copy(), depth_m.copy()))

    def _flush_rgbd(self, sim_time: float) -> None:
        while self._frame_queue and self._frame_queue[0][0] <= sim_time + 1e-9:
            _, stamp, rgb, depth_m = self._frame_queue.popleft()
            self._publish_rgbd_frame(stamp, rgb, depth_m)

    def _publish_rgbd_frame(
        self, stamp: Any, rgb: np.ndarray, depth_m: np.ndarray
    ) -> None:
        color_frame = self.camera_config["color_frame"]
        depth_frame = self.camera_config["depth_frame"]
        self._color_pub.publish(image_message(rgb, "rgb8", color_frame, stamp))
        depth_message = image_message(depth_m, "32FC1", depth_frame, stamp)
        self._depth_pub.publish(depth_message)
        self._aligned_pub.publish(image_message(depth_m, "32FC1", color_frame, stamp))
        self._color_info_pub.publish(camera_info_message(self.intrinsics, color_frame, stamp))
        self._depth_info_pub.publish(camera_info_message(self.intrinsics, depth_frame, stamp))
        points, colors = back_project_depth(
            depth_m,
            self.intrinsics,
            rgb,
            float(self.camera_config["near_clip"]),
            float(self.camera_config["far_clip"]),
        )
        assert colors is not None
        self._pointcloud_counter += 1
        camera_fps = float(self.camera_config["fps"])
        cloud_fps = float(self.camera_config["pointcloud"]["fps"])
        publish_every = max(1, round(camera_fps / cloud_fps))
        if self._pointcloud_counter % publish_every != 0:
            return
        stride = max(1, int(self.camera_config["pointcloud"]["pixel_stride"]))
        points = points[::stride]
        colors = colors[::stride]
        self._points_pub.publish(point_cloud_message(points, colors, color_frame, stamp))
        camera_position, camera_orientation = self.camera.get_world_pose()
        world_points = optical_points_to_world(
            points,
            np.asarray(camera_position),
            np.asarray(camera_orientation),
        )
        self._world_points_pub.publish(
            point_cloud_message(world_points, colors, "world", stamp)
        )

    def _publish_joint_state(self, stamp: Any) -> None:
        from sensor_msgs.msg import JointState

        positions = self.robot.get_joint_positions()
        if positions is None:
            return
        velocities = self.robot.get_joint_velocities()
        efforts = self.robot.get_measured_joint_efforts()
        message = JointState()
        message.header.stamp = stamp
        message.name = list(self.robot.dof_names)
        message.position = np.asarray(positions, dtype=float).tolist()
        message.velocity = (
            np.zeros_like(positions) if velocities is None else np.asarray(velocities)
        ).astype(float).tolist()
        message.effort = (
            np.zeros_like(positions) if efforts is None else np.asarray(efforts)
        ).astype(float).tolist()
        self._joint_pub.publish(message)

    def _publish_tf(self, stamp: Any) -> None:
        from geometry_msgs.msg import TransformStamped

        position, orientation = self.camera.get_world_pose()
        link = TransformStamped()
        link.header.stamp = stamp
        link.header.frame_id = "world"
        link.child_frame_id = self.camera_config["link_frame"]
        link.transform.translation.x = float(position[0])
        link.transform.translation.y = float(position[1])
        link.transform.translation.z = float(position[2])
        link.transform.rotation.w = float(orientation[0])
        link.transform.rotation.x = float(orientation[1])
        link.transform.rotation.y = float(orientation[2])
        link.transform.rotation.z = float(orientation[3])
        if not self._static_sent:
            optical_transforms = []
            for frame in (self.camera_config["color_frame"], self.camera_config["depth_frame"]):
                optical = TransformStamped()
                optical.header.stamp = stamp
                optical.header.frame_id = self.camera_config["link_frame"]
                optical.child_frame_id = frame
                # Camera.get_world_pose() defaults to Isaac's world camera
                # convention: link +X forward, +Y left, +Z up.  REP-103
                # optical is +Z forward, +X right, +Y down.  This fixed
                # quaternion implements columns [right, down, forward] =
                # [-link_Y, -link_Z, link_X].
                quaternion = ISAAC_WORLD_LINK_TO_ROS_OPTICAL_XYZW
                optical.transform.rotation.x = quaternion[0]
                optical.transform.rotation.y = quaternion[1]
                optical.transform.rotation.z = quaternion[2]
                optical.transform.rotation.w = quaternion[3]
                optical_transforms.append(optical)
            self._tf_static.sendTransform(optical_transforms)
            self._static_sent = True
        self._tf.sendTransform(link)

    def _publish_ground_truth(self, stamp: Any) -> None:
        from geometry_msgs.msg import Pose, PoseStamped
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from std_msgs.msg import Float64, String

        assert self._gt_pose_pub is not None
        assert self._gt_collision_pub is not None
        assert self._gt_distance_pub is not None
        position, orientation = self.hand.get_world_pose()
        pose_message = PoseStamped()
        pose_message.header.stamp = stamp
        pose_message.header.frame_id = "world"
        pose_message.pose.position.x = float(position[0])
        pose_message.pose.position.y = float(position[1])
        pose_message.pose.position.z = float(position[2])
        pose_message.pose.orientation.w = float(orientation[0])
        pose_message.pose.orientation.x = float(orientation[1])
        pose_message.pose.orientation.y = float(orientation[2])
        pose_message.pose.orientation.z = float(orientation[3])
        self._gt_pose_pub.publish(pose_message)

        collision = CollisionObject()
        collision.header = pose_message.header
        collision.id = "sim_hand_proxy"
        offsets_and_sizes = (
            ([0.0, 0.0, 0.0], self.hand_config["collision_proxy"]["palm_size"]),
            ([-0.10, 0.0, 0.0], [0.16, 0.065, 0.065]),
            ([-0.30, 0.0, 0.0], [0.34, 0.09, 0.09]),
        )
        for offset, size in offsets_and_sizes:
            primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in size])
            proxy_pose = Pose()
            proxy_pose.position.x = float(position[0] + offset[0])
            proxy_pose.position.y = float(position[1] + offset[1])
            proxy_pose.position.z = float(position[2] + offset[2])
            proxy_pose.orientation = pose_message.pose.orientation
            collision.primitives.append(primitive)
            collision.primitive_poses.append(proxy_pose)
        collision.operation = CollisionObject.ADD
        self._gt_collision_pub.publish(collision)
        self._gt_distance_pub.publish(Float64(data=self._minimum_collision_distance()))
        states = []
        for spec in self.cube_specs:
            cube = self.world.scene.get_object(spec.name)
            if cube is None:
                continue
            cube_position, _ = cube.get_world_pose()
            states.append(
                {"name": spec.name, "color": spec.color, "position": np.asarray(cube_position).tolist()}
            )
        self._cube_states_pub.publish(String(data=json.dumps(states, separators=(",", ":"))))

    def _minimum_collision_distance(self) -> float:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = self.robot.prim.GetStage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        robot_bounds = []
        robot_root = stage.GetPrimAtPath(f"{self.robot_config['prim_path']}/root_joint")
        for prim in Usd.PrimRange(robot_root):
            # URDF-imported OpenArm collision meshes can be authored beneath a
            # rigid link without CollisionAPI on the link prim itself.  A link
            # world bound still encloses those collision children and is the
            # correct fallback for the safety-distance watchdog.
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                robot_bounds.append((np.asarray(box.GetMin()), np.asarray(box.GetMax())))
        hand_bounds = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.hand.prim_path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                hand_bounds.append((np.asarray(box.GetMin()), np.asarray(box.GetMax())))
        if not robot_bounds or not hand_bounds:
            return float("inf")
        minimum = float("inf")
        for robot_min, robot_max in robot_bounds:
            for hand_min, hand_max in hand_bounds:
                delta = np.maximum(0.0, np.maximum(robot_min - hand_max, hand_min - robot_max))
                minimum = min(minimum, float(np.linalg.norm(delta)))
        return minimum

    def _task_state_callback(self, message: Any) -> None:
        self.task_state = message.data

    def _safety_callback(self, message: Any) -> None:
        self.control.set_safety_state(message.data)

    def _hand_command_callback(self, message: Any) -> None:
        self._hand_commands.append(message.data.strip())

    def _manual_hand_target_callback(self, message: Any) -> None:
        if message.header.frame_id not in {"", "world"}:
            self.node.get_logger().warning(
                f"ignored manual hand target in frame {message.header.frame_id!r}"
            )
            return
        target = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            float(message.pose.position.z),
        )
        if not np.isfinite(target).all():
            self.node.get_logger().warning("ignored non-finite manual hand target")
            return
        self._manual_hand_targets.append(target)

    def _cube_event_callback(self, message: Any) -> None:
        fields = message.data.split(",")
        if len(fields) < 2:
            return
        if fields[0] == "magnetic_attach":
            self._cube_commands.append(("attach", fields[1]))
        elif fields[0] == "magnetic_detach":
            self._cube_commands.append(("detach", fields[1]))

    def _update_cube_attachment(self) -> None:
        while self._cube_commands:
            command, cube_name = self._cube_commands.popleft()
            if command == "attach":
                cube = self.world.scene.get_object(cube_name)
                if cube is None:
                    self.node.get_logger().error(f"cannot attach unknown cube: {cube_name}")
                    continue
                if not self._create_magnetic_constraint(cube, cube_name):
                    continue
                self._attached_cube = cube
                self._attached_cube_name = cube_name
                self.publish_event("magnetic_attach_applied", cube_name)
            elif command == "detach" and cube_name == self._attached_cube_name:
                self._remove_magnetic_constraint()
                self._attached_cube = None
                self._attached_cube_name = None

    def _create_magnetic_constraint(self, cube: Any, cube_name: str) -> bool:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        hand_path = f"{self.robot_config['prim_path']}/root_joint/openarm_left_hand"
        hand_prim = self.robot.prim.GetStage().GetPrimAtPath(hand_path)
        if not hand_prim.IsValid():
            raise RuntimeError(f"active gripper link is missing from the stage: {hand_path}")
        transform = UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        fingertip = np.asarray(
            transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0835)), dtype=float
        )
        cube_position, cube_orientation = cube.get_world_pose()
        distance = float(np.linalg.norm(np.asarray(cube_position) - fingertip))
        limit = float(self._grasp_config["magnetic_max_distance_m"])
        if distance > limit:
            self.node.get_logger().error(
                f"magnetic attach rejected for {cube_name}: "
                f"distance={distance:.4f}m limit={limit:.4f}m"
            )
            self.publish_event(
                "magnetic_attach_rejected", cube_name, f"{distance:.6f}"
            )
            return False
        self._remove_magnetic_constraint()
        stage = self.robot.prim.GetStage()
        constraint_path = f"/World/GraspConstraints/{cube_name}_fixed"
        joint = UsdPhysics.FixedJoint.Define(stage, constraint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(cube.prim_path)])
        # Lock the cube's current pose relative to the hand. This avoids a
        # one-time snap as well as the former per-frame teleport behavior.
        relative_position = transform.GetInverse().Transform(
            Gf.Vec3d(*np.asarray(cube_position, dtype=float))
        )
        hand_rotation = transform.ExtractRotationQuat()
        cube_rotation = Gf.Quatd(
            float(cube_orientation[0]),
            Gf.Vec3d(*np.asarray(cube_orientation[1:4], dtype=float)),
        )
        relative_rotation = hand_rotation.GetInverse() * cube_rotation
        relative_imaginary = relative_rotation.GetImaginary()
        joint.CreateLocalPos0Attr(
            Gf.Vec3f(
                float(relative_position[0]),
                float(relative_position[1]),
                float(relative_position[2]),
            )
        )
        joint.CreateLocalRot0Attr(
            Gf.Quatf(
                float(relative_rotation.GetReal()),
                Gf.Vec3f(
                    float(relative_imaginary[0]),
                    float(relative_imaginary[1]),
                    float(relative_imaginary[2]),
                ),
            )
        )
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)
        self._grasp_constraint_path = constraint_path
        return True

    def _remove_magnetic_constraint(self) -> None:
        if self._grasp_constraint_path is None:
            return
        self.robot.prim.GetStage().RemovePrim(self._grasp_constraint_path)
        self._grasp_constraint_path = None
