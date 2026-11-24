from __future__ import annotations

import json
import logging
from typing import Any, Callable

from realtime_safety.ros2_bridge.runtime import (
    acquire_ros2_runtime,
    release_ros2_runtime,
)


LOGGER = logging.getLogger(__name__)


class SimulatorPoseBridge:
    """GUI-only bridge for editable Gazebo model poses.

    This ground-truth topic is never forwarded into the perception planner.
    """

    def __init__(
        self,
        topic: str,
        on_poses: Callable[[dict[str, Any]], None],
    ) -> None:
        self.topic = topic
        self.on_poses = on_poses
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._subscription: Any | None = None
        self._received = 0

    def start(self) -> None:
        if self._node is not None:
            return
        from rclpy.node import Node
        from std_msgs.msg import String

        runtime = acquire_ros2_runtime()
        node = Node("realtime_safety_simulator_pose_bridge", context=runtime.context)
        subscription = node.create_subscription(String, self.topic, self._on_message, 10)
        runtime.add_node(node)
        self._runtime = runtime
        self._node = node
        self._subscription = subscription

    def _on_message(self, message: Any) -> None:
        try:
            document = json.loads(message.data)
            if document.get("frame_id") != "world":
                return
            entities = document.get("entities")
            if not isinstance(entities, dict):
                return
            self.on_poses(entities)
            self._received += 1
            if self._received == 1:
                LOGGER.info("GUI is following direct Gazebo model edits")
        except Exception:
            LOGGER.exception("Could not apply Gazebo scene poses to the GUI")

    def close(self) -> None:
        runtime, node = self._runtime, self._node
        self._runtime = None
        self._node = None
        self._subscription = None
        if runtime is None or node is None:
            return
        runtime.remove_node(node)
        node.destroy_node()
        release_ros2_runtime(runtime)


__all__ = ["SimulatorPoseBridge"]
