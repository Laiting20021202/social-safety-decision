"""ROS adapter for the bounded tidybot proximity reflex. Dry-run by default."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import yaml

from realtime_safety.pipeline.tidybot_retreat import RetreatConfig, RetreatController


def cloud_xyz(message) -> np.ndarray:
    """Decode XYZ float32/64 with declared endianness and organized row padding."""
    fields = {f.name: f for f in message.fields}
    endian = ">" if message.is_bigendian else "<"
    formats, offsets = [], []
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in (7, 8):
            raise ValueError("cloud requires scalar float32/64 XYZ")
        size = 4 if field.datatype == 7 else 8
        if field.offset < 0 or field.offset + size > message.point_step:
            raise ValueError("invalid XYZ field offset")
        formats.append(endian + ("f4" if size == 4 else "f8"))
        offsets.append(field.offset)
    if (message.height <= 0 or message.width <= 0 or message.point_step <= 0
            or message.row_step < message.width * message.point_step
            or len(message.data) < message.row_step * message.height):
        raise ValueError("empty or truncated point cloud")
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": formats,
                      "offsets": offsets, "itemsize": message.point_step})
    values = np.ndarray((message.height, message.width), dtype=dtype,
                        buffer=memoryview(message.data),
                        strides=(message.row_step, message.point_step))
    return np.column_stack([values[k].ravel() for k in ("x", "y", "z")])


def load_settings(path: str, execute: bool, accept_uncalibrated_simulation: bool = False):
    data = yaml.safe_load(Path(path).read_text())
    config = RetreatConfig(**data["controller"])
    transform = np.asarray(data["cloud_to_base"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("cloud_to_base must be a finite 4x4 rigid transform")
    if (not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
        raise ValueError("cloud_to_base must be a proper rigid transform")
    if not math.isfinite(data["cloud_scale_to_m"]) or data["cloud_scale_to_m"] <= 0:
        raise ValueError("cloud_scale_to_m must be finite and positive")
    for field in ("cloud_topic", "joint_topic", "command_topic"):
        if not isinstance(data[field], str) or not data[field].startswith("/") or any(c.isspace() for c in data[field]):
            raise ValueError(f"invalid {field}")
    if data["command_topic"] != "/tidybot/cmd_vel":
        raise ValueError("this adapter targets only /tidybot/cmd_vel")
    if execute:
        required = ("metric_scale_verified", "camera_extrinsics_verified",
                    "base_pose_world_meters_verified", "holonomic_command_frame_verified",
                    "receiver_watchdog_verified", "isolated_simulation",
                    "retreat_corridor_verified_clear")
        missing = [k for k in required if data.get("verified", {}).get(k) is not True]
        accepted = {"metric_scale_verified", "retreat_corridor_verified_clear"}
        if accept_uncalibrated_simulation:
            if data.get("verified", {}).get("isolated_simulation") is not True:
                raise ValueError("Uncalibrated override is restricted to verified isolated simulation")
            missing = [k for k in missing if k not in accepted]
        if missing:
            raise ValueError("Execute blocked; verify configuration first: " + ", ".join(missing))
    return data, config, transform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tidybot_retreat.yaml")
    parser.add_argument("--execute", action="store_true", help="Publish real simulation Twist commands after verification")
    parser.add_argument("--accept-uncalibrated-simulation", action="store_true",
                        help="In isolated simulation only, accept unverified RGB scale and retreat corridor")
    parser.add_argument("--duration", type=float, help="Optional bounded wall-clock run in seconds")
    args = parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be positive")
    try:
        if args.accept_uncalibrated_simulation and not args.execute:
            parser.error("--accept-uncalibrated-simulation requires --execute")
        settings, config, transform = load_settings(
            args.config, args.execute, args.accept_uncalibrated_simulation)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import PointCloud2, JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    class RetreatNode(Node):
        def __init__(self):
            super().__init__("tidybot_retreat")
            self.controller = RetreatController(config)
            self.command = None
            self.started = False
            self.last_status = 0.0
            self.last_command_at = time.monotonic()
            self.endpoint_discovery_deadline = time.monotonic() + 3.0
            # No /cmd_vel publisher at all in dry-run, including zero commands.
            if args.execute:
                self.command = self.create_publisher(Twist, settings["command_topic"],
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE))
            self.status = self.create_publisher(String, "/realtime_safety/tidybot_retreat/status", 1)
            self.create_subscription(PointCloud2, settings["cloud_topic"], self.on_cloud, qos_profile_sensor_data)
            self.create_subscription(JointState, settings["joint_topic"], self.on_pose, qos_profile_sensor_data)
            self.create_service(Trigger, "~/reset", self.reset)
            self.create_service(Trigger, "~/stop", self.stop)
            self.create_timer(0.05, self.tick)
            self.get_logger().info("EXECUTE: simulation velocity enabled" if args.execute else
                "DRY RUN: only status/proposed velocity; /tidybot/cmd_vel is untouched")
            if args.accept_uncalibrated_simulation:
                self.get_logger().warning(
                    "UNCALIBRATED SIMULATION OVERRIDE: RGB metric scale and retreat corridor accepted by operator")

        def on_pose(self, msg):
            names = list(msg.name)
            required = ("joint_x", "joint_y", "joint_th")
            if not any(k in names for k in required):
                return
            if len(names) != len(msg.position) or len(set(names)) != len(names):
                self.controller.pose_at = None
                self.controller.fault("malformed joint feedback")
                return
            values = dict(zip(names, msg.position))
            if not all(k in values for k in required):
                self.controller.pose_at = None
                self.controller.fault("missing base joint feedback")
                return
            self.controller.update_pose([values[k] for k in required],
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic())

        def on_cloud(self, msg):
            if msg.header.frame_id != settings["cloud_frame"]:
                self.controller.cloud_at = None
                self.controller.fault("unexpected cloud frame: " + msg.header.frame_id)
                return
            try:
                xyz = cloud_xyz(msg) * settings["cloud_scale_to_m"]
                base = xyz @ transform[:3, :3].T + transform[:3, 3]
                self.controller.update_cloud(base,
                    msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic())
            except (ValueError, TypeError, OverflowError) as exc:
                self.controller.cloud_at = None
                self.controller.fault("invalid cloud: " + str(exc))

        def reset(self, request, response):
            response.success = self.controller.reset(time.monotonic())
            response.message = "reset" if response.success else "need fresh pose and clear, valid cloud"
            return response

        def stop(self, request, response):
            self.controller.fault("operator stop")
            self.send_zero()
            response.success, response.message = True, "stopped; explicit reset required"
            return response

        def send_zero(self):
            if self.command is not None and self.started:
                self.command.publish(Twist())

        def tick(self):
            now = time.monotonic()
            allowed = True
            if self.command is not None:
                pubs = self.get_publishers_info_by_topic(settings["command_topic"])
                subscribers = self.get_subscriptions_info_by_topic(settings["command_topic"])
                # Avoid command arbitration with keyboard teleop or other controllers.
                own = [i for i in pubs if i.node_name == self.get_name() and i.node_namespace == self.get_namespace()]
                allowed = len(pubs) == len(own) == 1 and len(subscribers) == 1
                if not allowed:
                    # DDS graph discovery is asynchronous. Never move during the
                    # grace period, but do not latch a permanent fault before our
                    # own publisher and the already-audited Isaac subscriber appear.
                    if not self.started and now < self.endpoint_discovery_deadline:
                        return
                    self.controller.fault("require sole cmd_vel publisher and exactly one simulator subscriber")
                if self.started and now - self.last_command_at > 0.5:
                    self.controller.fault("control-loop watchdog")
            vx, vy = self.controller.tick(now)
            if self.command is not None:
                msg = Twist()
                if allowed:
                    msg.linear.x, msg.linear.y = vx, vy
                    self.command.publish(msg)
                    self.started = True
                elif self.started:
                    self.command.publish(msg)  # Stop our previous motion; conflict remains latched.
                    self.started = False
                self.last_command_at = now
            if now - self.last_status >= 0.5:
                c = self.controller
                mode = ("execute_uncalibrated_sim" if args.accept_uncalibrated_simulation
                        else "execute" if args.execute else "dry_run")
                payload = {"mode": mode,
                           "state": c.state, "reason": c.reason,
                           "proposed_velocity": [vx, vy], "command_frame": config.command_frame,
                           "target_world_xy": None if c.target is None else c.target.tolist(),
                           "pose_world_xy_yaw": None if c.pose is None else c.pose.tolist(),
                           "obstacle_base_xy": None if c.obstacle is None else c.obstacle.tolist(),
                           "cloud_age_s": None if c.cloud_at is None else now-c.cloud_at,
                           "pose_age_s": None if c.pose_at is None else now-c.pose_at,
                           "verification": settings.get("verified", {})}
                message = String(data=json.dumps(payload, allow_nan=False))
                self.status.publish(message)
                self.get_logger().info(message.data)
                self.last_status = now

    # Keep the context alive long enough to publish zero on Ctrl+C/SIGTERM.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    old_term = signal.signal(signal.SIGTERM, interrupt)
    node = RetreatNode()
    started = time.monotonic()
    try:
        while rclpy.ok() and (args.duration is None or time.monotonic()-started < args.duration):
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.send_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    main()
