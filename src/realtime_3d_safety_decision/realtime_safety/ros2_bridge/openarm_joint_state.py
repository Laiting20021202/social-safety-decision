from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any


LOGGER = logging.getLogger(__name__)


class OpenArmJointStateBridge:
    """Latest-only ROS 2 JointState subscriber for the GUI URDF overlay."""

    def __init__(
        self,
        topic: str,
        on_state: Callable[..., int | None],
        node_name: str = "realtime_safety_openarm_joint_state",
    ) -> None:
        if not topic.startswith("/") or any(char.isspace() for char in topic):
            raise ValueError("JointState topic must be absolute and contain no spaces")
        self.topic = topic
        self.on_state = on_state
        self.node_name = node_name
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._subscription: Any | None = None
        self._lock = threading.Lock()
        self._message_count = 0
        self._last_received_at = 0.0

    @property
    def message_count(self) -> int:
        with self._lock:
            return self._message_count

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        runtime = acquire_ros2_runtime()
        node = Node(self.node_name, context=runtime.context)
        subscription = node.create_subscription(
            JointState, self.topic, self._on_joint_state, qos_profile_sensor_data
        )
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._subscription = subscription
        LOGGER.info("OpenArm GUI subscribing to JointState %s", self.topic)

    def _on_joint_state(self, message: Any) -> None:
        names = tuple(str(name) for name in message.name)
        positions = tuple(float(value) for value in message.position)
        if not names or len(names) != len(positions):
            LOGGER.warning(
                "Ignoring malformed JointState on %s: names=%d positions=%d",
                self.topic,
                len(names),
                len(positions),
            )
            return
        received_at = time.monotonic()
        header_stamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) / 1_000_000_000.0
        )
        matched = self.on_state(
            names,
            positions,
            received_at=received_at,
            header_stamp=header_stamp,
        )
        with self._lock:
            self._message_count += 1
            self._last_received_at = received_at
            count = self._message_count
        if count == 1:
            LOGGER.info(
                "First OpenArm JointState received: names=%s matched=%s",
                names,
                matched,
            )

    def close(self) -> None:
        if self._node is None:
            return
        from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

        runtime = self._runtime
        node = self._node
        self._node = None
        self._subscription = None
        self._runtime = None
        if runtime is not None:
            runtime.remove_node(node)
        node.destroy_node()
        if runtime is not None:
            release_ros2_runtime(runtime)


__all__ = ["OpenArmJointStateBridge"]
