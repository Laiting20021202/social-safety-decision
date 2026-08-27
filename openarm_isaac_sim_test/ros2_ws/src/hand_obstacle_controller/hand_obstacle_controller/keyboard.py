from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


KEYS = {
    "r": "reset",
    "s": "start",
    "p": "pause",
    "o": "resume",
    "t": "trigger_hand",
    "w": "withdraw",
}


class KeyboardController(Node):
    def __init__(self) -> None:
        super().__init__("hand_obstacle_keyboard")
        self.publisher = self.create_publisher(String, "/sim/hand/command", 10)


def main() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("hand_keyboard requires an interactive terminal")
    rclpy.init()
    node = KeyboardController()
    original = termios.tcgetattr(sys.stdin)
    print("r reset | s start | p pause | o resume | t trigger | w withdraw | q quit")
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                break
            if command := KEYS.get(key):
                node.publisher.publish(String(data=command))
                print(command, flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
