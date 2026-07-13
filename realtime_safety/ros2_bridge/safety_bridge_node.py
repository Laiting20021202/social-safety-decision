from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    args, ros_args = parser.parse_known_args()

    class SafetyBridge(Node):
        def __init__(self, path: Path) -> None:
            super().__init__("realtime_safety_bridge")
            self.path = path
            self.offset = 0
            self.state_pub = self.create_publisher(String, "/realtime_safety/state", 10)
            self.twist_pub = self.create_publisher(Twist, "/realtime_safety/recommended_cmd_vel", 10)
            self.create_timer(0.1, self.poll)

        def poll(self) -> None:
            if not self.path.exists():
                return
            with self.path.open("r", encoding="utf-8") as stream:
                stream.seek(self.offset)
                records = stream.readlines()
                self.offset = stream.tell()
            if not records:
                return
            raw = records[-1].strip()
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                return
            self.state_pub.publish(String(data=raw))
            command = Twist()
            action = record.get("recommended_action")
            if record.get("metric_valid") and action in {"CONTINUE", "DETOUR_LEFT", "DETOUR_RIGHT"}:
                command.linear.x = 0.25
            elif record.get("metric_valid") and action == "SLOW_DOWN":
                command.linear.x = 0.1
            if action == "DETOUR_LEFT":
                command.angular.z = 0.25
            elif action == "DETOUR_RIGHT":
                command.angular.z = -0.25
            self.twist_pub.publish(command)

    rclpy.init(args=ros_args)
    node = SafetyBridge(Path(args.jsonl))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
