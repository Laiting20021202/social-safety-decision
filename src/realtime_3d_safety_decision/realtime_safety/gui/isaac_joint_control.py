"""Small-step Isaac Sim joint control with fresh, named OpenArm feedback."""
from __future__ import annotations

import math
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def load_arm_limits(urdf: str | Path) -> dict[str, tuple[float, float]]:
    limits = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        name = joint.attrib["name"]
        limit = joint.find("limit")
        if re.fullmatch(r"openarm_(left|right)_joint[1-7]", name) and limit is not None:
            limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    if len(limits) != 14:
        raise ValueError("Expected the 14 OpenArm arm joints in the URDF")
    return limits


class JointJogState:
    def __init__(self, limits: dict[str, tuple[float, float]]) -> None:
        self.limits = limits
        self.positions: dict[str, float] = {}
        self.received_at: float | None = None

    def update(self, names: list[str], positions: list[float], now: float) -> bool:
        if len(names) != len(positions) or len(set(names)) != len(names):
            return False
        sample = dict(zip(names, positions))
        if not all(name in sample and math.isfinite(sample[name]) for name in self.limits):
            return False
        self.positions = {name: float(sample[name]) for name in self.limits}
        self.received_at = now
        return True

    def fresh(self, now: float) -> bool:
        return self.received_at is not None and 0 <= now - self.received_at <= 2.0

    def target(self, name: str, delta: float, now: float) -> float:
        if not self.fresh(now):
            raise ValueError("等待新的 OpenArm 關節回饋")
        if name not in self.limits or not math.isfinite(delta) or not 0 < abs(delta) <= 0.05:
            raise ValueError("關節步進必須在 0–0.05 rad 之間")
        value = self.positions[name] + delta
        lower, upper = self.limits[name]
        if not lower <= value <= upper:
            raise ValueError("目標超出 OpenArm URDF 關節範圍")
        return value


class IsaacJointControlPanel:
    def __init__(self, server: Any, urdf_path: str) -> None:
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from realtime_safety.ros2_bridge.runtime import acquire_ros2_runtime

        self.state = JointJogState(load_arm_limits(urdf_path))
        self._lock = threading.RLock()
        self._pending: tuple[str, float, float] | None = None
        self._message = "選擇關節後，按 − / ＋ 小幅移動模擬手臂。"
        self._runtime = acquire_ros2_runtime()
        self._node = Node("realtime_safety_isaac_joint_control", context=self._runtime.context)
        self._publisher = self._node.create_publisher(JointState, "/joint_command", 10)
        self._subscription = self._node.create_subscription(
            JointState, "/joint_states", self._on_state, qos_profile_sensor_data
        )
        with server.gui.add_folder("Isaac Sim 關節控制", order=4, expand_by_default=True):
            server.gui.add_markdown(
                "**OpenArm 左右臂 · 模擬關節步進**  \n"
                "點雲來源：前方 RGB + 深度估計，尚無 depth topic；尺度未驗證。"
            )
            self._status = server.gui.add_markdown("等待 Isaac Sim 連線…")
            self._joint = server.gui.add_dropdown(
                "關節", options=tuple(sorted(self.state.limits)),
                initial_value="openarm_left_joint1",
            )
            self._step = server.gui.add_slider(
                "步進 (rad)", min=0.005, max=0.05, step=0.005, initial_value=0.01
            )
            self._minus = server.gui.add_button("− 移動選取關節", disabled=True)
            self._plus = server.gui.add_button("＋ 移動選取關節", disabled=True)
            self._result = server.gui.add_markdown(self._message)
        self._minus.on_click(lambda _: self.jog(-1))
        self._plus.on_click(lambda _: self.jog(1))
        self._timer = self._node.create_timer(0.25, self._refresh)
        self._runtime.add_node(self._node)

    def _on_state(self, message: Any) -> None:
        with self._lock:
            self.state.update(list(message.name), list(message.position), time.monotonic())

    def _connected(self) -> bool:
        endpoints = self._node.get_subscriptions_info_by_topic("/joint_command")
        return (
            len(endpoints) == 1
            and "JointControlGraph_ROS2SubscribeJointState" in endpoints[0].node_name
            and self._publisher.get_subscription_count() == 1
        )

    def _refresh(self) -> None:
        with self._lock:
            now = time.monotonic()
            connected = self._connected()
            fresh = self.state.fresh(now)
            if self._pending is not None:
                name, target, sent_at = self._pending
                actual = self.state.positions.get(name)
                if fresh and self.state.received_at > sent_at and actual is not None and abs(actual - target) <= 0.003:
                    self._message = f"已抵達 `{name}`：目標 {target:.4f}，回饋 {actual:.4f} rad。"
                    self._pending = None
                elif now - sent_at > 5.0:
                    self._message = "指令已送出，但 5 秒內未確認抵達；請檢查模擬是否播放。"
                    self._pending = None
            ready = connected and fresh and self._pending is None
            self._minus.disabled = self._plus.disabled = not ready
            actual = self.state.positions.get(self._joint.value)
            state = "🟢 已連線" if connected and fresh else "🟠 等待控制連線／新回饋"
            position = "等待中" if actual is None else f"{actual:.4f} rad"
            self._status.content = f"**{state}** · `{self._joint.value}` = **{position}**"
            self._result.content = self._message

    def jog(self, direction: int) -> None:
        from sensor_msgs.msg import JointState

        with self._lock:
            try:
                if direction not in (-1, 1) or not self._connected():
                    raise ValueError("尚未連線到唯一的 Isaac Sim 關節控制 subscriber")
                if self._pending is not None:
                    raise ValueError("上一筆關節指令尚待回饋")
                now = time.monotonic()
                name = self._joint.value
                target = self.state.target(name, direction * float(self._step.value), now)
                message = JointState()
                message.name = [name]
                message.position = [target]
                self._publisher.publish(message)
                self._pending = (name, target, now)
                self._message = f"已送出 `{name}` → {target:.4f} rad；等待實際回饋。"
            except ValueError as exc:
                self._message = str(exc)
        self._refresh()

    def close(self) -> None:
        from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

        self._runtime.remove_node(self._node)
        self._node.destroy_node()
        release_ros2_runtime(self._runtime)
