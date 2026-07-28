from __future__ import annotations

import threading
from typing import Any


class SharedRos2Runtime:
    """One rclpy context/executor shared by every ROS endpoint in the app."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor

        context = Context()
        rclpy.init(args=[], context=context)
        executor = MultiThreadedExecutor(num_threads=2, context=context)
        thread = threading.Thread(
            name="realtime-safety-ros2-executor",
            target=executor.spin,
            daemon=True,
        )
        self.context: Any = context
        self.executor: Any = executor
        self.thread = thread
        self.users = 0
        thread.start()

    def add_node(self, node: Any) -> None:
        self.executor.add_node(node)

    def remove_node(self, node: Any) -> None:
        self.executor.remove_node(node)

    def shutdown(self) -> None:
        self.executor.shutdown(timeout_sec=3.0)
        self.thread.join(timeout=3.0)
        self.context.try_shutdown()


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: SharedRos2Runtime | None = None


def acquire_ros2_runtime() -> SharedRos2Runtime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = SharedRos2Runtime()
        _RUNTIME.users += 1
        return _RUNTIME


def release_ros2_runtime(runtime: SharedRos2Runtime) -> None:
    global _RUNTIME
    shutdown = False
    with _RUNTIME_LOCK:
        if runtime is not _RUNTIME:
            return
        runtime.users = max(runtime.users - 1, 0)
        if runtime.users == 0:
            _RUNTIME = None
            shutdown = True
    if shutdown:
        runtime.shutdown()
