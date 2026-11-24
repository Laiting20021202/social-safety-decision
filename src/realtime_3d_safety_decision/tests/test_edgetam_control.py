from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from app import _edge_control_initial_mode
from realtime_safety.ros2_bridge.edgetam_control import (
    EdgeTAMControlBridge,
    EdgeTAMControlStatus,
)


def test_three_way_backend_maps_to_two_way_edge_startup_mode() -> None:
    assert _edge_control_initial_mode("edgetam") == "edgetam"
    assert _edge_control_initial_mode("pointcloud") == "pointcloud"
    assert _edge_control_initial_mode("yolo") == "pointcloud"
    with pytest.raises(ValueError, match="Unsupported obstacle backend"):
        _edge_control_initial_mode("unknown")


class _Request:
    def __init__(self) -> None:
        self.data = False


class _Future:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[Any], None]] = []
        self._result: Any = None
        self._error: Exception | None = None
        self._done = False
        self.cancelled = False

    def add_done_callback(self, callback: Callable[[Any], None]) -> None:
        self._callbacks.append(callback)
        if self._done:
            callback(self)

    def result(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._result

    def complete(self, *, success: bool, message: str = "") -> None:
        self._result = SimpleNamespace(success=success, message=message)
        self._done = True
        for callback in tuple(self._callbacks):
            callback(self)

    def cancel(self) -> None:
        self.cancelled = True


class _Client:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.requests: list[_Request] = []
        self.futures: list[_Future] = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request: _Request) -> _Future:
        self.requests.append(request)
        future = _Future()
        self.futures.append(future)
        return future


def _attach_client(bridge: EdgeTAMControlBridge, client: _Client) -> None:
    bridge._client = client
    bridge._request_type = _Request


def _diagnostics(
    state: str,
    *,
    level: Any = 0,
    message: str = "",
    error: str = "",
    refined: str = "0",
) -> Any:
    values = [
        SimpleNamespace(key="state", value=state),
        SimpleNamespace(key="error", value=error),
        SimpleNamespace(key="refined_corrections", value=refined),
    ]
    edge = SimpleNamespace(
        name="/edgetam_pointcloud_tracker/edgetam",
        hardware_id="official_edgetam",
        level=level,
        message=message,
        values=values,
    )
    pipeline = SimpleNamespace(
        name="/edgetam_pointcloud_tracker/pipeline",
        hardware_id="rgbd_pointcloud_tracker",
        level=0,
        message="point-cloud tracking active",
        values=[SimpleNamespace(key="fps", value="8.2")],
    )
    return SimpleNamespace(status=[pipeline, edge])


def test_only_edgetam_and_pointcloud_modes_are_accepted() -> None:
    bridge = EdgeTAMControlBridge(lambda _: None)

    assert bridge.request_mode("edgetam")
    assert bridge.request_mode(" pointcloud ")
    with pytest.raises(ValueError, match="exactly 'edgetam' or 'pointcloud'"):
        bridge.request_mode("yolo")


def test_obstacle_cloud_callback_decodes_points_and_preserves_empty_clear() -> None:
    received: list[tuple[np.ndarray, np.ndarray | None, str]] = []
    bridge = EdgeTAMControlBridge(
        lambda _: None,
        on_obstacle_cloud=lambda points, colors, frame_id: received.append(
            (points.copy(), None if colors is None else colors.copy(), frame_id)
        ),
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
    )
    payload = np.zeros(2, dtype=dtype)
    payload["x"] = (0.1, 0.2)
    payload["y"] = (0.4, 0.5)
    payload["z"] = (-0.2, 0.1)
    payload["rgb"] = (0xFF002D, 0x22CC66)
    fields = [
        SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)
        for name, offset, datatype in (
            ("x", 0, 7),
            ("y", 4, 7),
            ("z", 8, 7),
            ("rgb", 12, 6),
        )
    ]
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0),
        frame_id="realtime_safety_frame",
    )
    message = SimpleNamespace(
        header=header,
        height=1,
        width=2,
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=32,
        data=payload.tobytes(),
    )

    bridge._on_obstacle_cloud_message(message)
    np.testing.assert_allclose(
        received[-1][0], [[0.1, 0.4, -0.2], [0.2, 0.5, 0.1]]
    )
    np.testing.assert_array_equal(
        received[-1][1], [[255, 0, 45], [34, 204, 102]]
    )
    assert received[-1][2] == "realtime_safety_frame"

    message.width = 0
    message.row_step = 0
    message.data = b""
    bridge._on_obstacle_cloud_message(message)
    assert received[-1][0].shape == (0, 3)
    assert received[-1][1] is None
    assert received[-1][2] == "realtime_safety_frame"


def test_request_is_async_and_maps_modes_to_set_bool() -> None:
    updates: list[EdgeTAMControlStatus] = []
    bridge = EdgeTAMControlBridge(updates.append)
    client = _Client()
    _attach_client(bridge, client)

    assert bridge.request_mode("edgetam")

    assert updates[-1].state == "loading"
    assert updates[-1].requested_mode == "edgetam"
    assert updates[-1].active_mode == "pointcloud"
    assert len(client.requests) == 1
    assert client.requests[0].data is True

    client.futures[0].complete(
        success=True,
        message="EdgeTAM enable accepted; model is loading asynchronously",
    )

    assert updates[-1].state == "loading"
    assert updates[-1].active_mode == "pointcloud"

    bridge._on_diagnostics(_diagnostics("ready", message="ready"))
    assert updates[-1].state == "active"
    assert updates[-1].active_mode == "edgetam"

    bridge.request_mode("pointcloud")
    assert client.requests[-1].data is False


def test_latest_request_wins_without_reporting_stale_service_success() -> None:
    updates: list[EdgeTAMControlStatus] = []
    bridge = EdgeTAMControlBridge(updates.append)
    client = _Client()
    _attach_client(bridge, client)

    bridge.request_mode("edgetam")
    first = client.futures[0]
    bridge.request_mode("pointcloud")

    # Calls are intentionally serialized so service side effects stay ordered.
    assert len(client.requests) == 1
    first.complete(success=True, message="stale enable completed")
    assert len(client.requests) == 2
    assert client.requests[1].data is False
    assert not any(
        update.state == "active" and update.active_mode == "edgetam"
        for update in updates[2:]
    )

    client.futures[1].complete(success=True, message="disabled")
    assert updates[-1].state == "active"
    assert updates[-1].requested_mode == "pointcloud"
    assert updates[-1].active_mode == "pointcloud"


def test_unavailable_service_times_out_without_blocking_request() -> None:
    updates: list[EdgeTAMControlStatus] = []
    bridge = EdgeTAMControlBridge(updates.append, request_timeout_sec=0.01)
    client = _Client(ready=False)
    _attach_client(bridge, client)

    bridge.request_mode("edgetam")
    assert updates[-1].state == "loading"
    assert client.requests == []

    assert bridge._pending is not None
    bridge._pending.deadline = 0.0
    bridge._poll_requests()

    assert updates[-1].state == "error"
    assert "/edgetam_tracker/set_enabled" in updates[-1].message


def test_late_response_cannot_turn_a_timed_out_request_active() -> None:
    updates: list[EdgeTAMControlStatus] = []
    bridge = EdgeTAMControlBridge(updates.append, request_timeout_sec=0.01)
    client = _Client()
    _attach_client(bridge, client)

    bridge.request_mode("edgetam")
    assert bridge._inflight is not None
    bridge._inflight[0].deadline = 0.0
    bridge._poll_requests()
    timed_out_status = updates[-1]
    assert timed_out_status.state == "error"

    client.futures[0].complete(success=True, message="too late")

    assert updates[-1] == timed_out_status
    assert bridge.active_mode == "pointcloud"


def test_diagnostics_adopt_actual_mode_and_expose_values() -> None:
    updates: list[EdgeTAMControlStatus] = []
    bridge = EdgeTAMControlBridge(updates.append)

    bridge._on_diagnostics(
        _diagnostics(
            "degraded",
            level=1,
            message="independent-state compatibility fallback",
            refined="3",
        )
    )

    update = updates[-1]
    assert update.state == "active"
    assert update.requested_mode == "edgetam"
    assert update.active_mode == "edgetam"
    assert update.diagnostics["state"] == "degraded"
    assert update.diagnostics["refined_corrections"] == "3"
    assert update.diagnostics["pipeline.fps"] == "8.2"
    assert update.diagnostics["edge.level"] == "1"


def test_diagnostics_report_model_error_while_pointcloud_mode_is_retained() -> None:
    updates: list[EdgeTAMControlStatus] = []
    bridge = EdgeTAMControlBridge(updates.append)
    bridge.request_mode("edgetam")

    bridge._on_diagnostics(
        _diagnostics(
            "error",
            # ROS 2 Humble exposes uint8 constants as one-byte values.
            level=b"\x02",
            message="EdgeTAM unavailable; point-cloud fallback active",
            error="checkpoint checksum mismatch",
        )
    )

    update = updates[-1]
    assert update.state == "error"
    assert update.requested_mode == "edgetam"
    assert update.active_mode == "pointcloud"
    assert update.message == "checkpoint checksum mismatch"


def test_start_and_close_use_shared_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import rclpy.node
    import realtime_safety.ros2_bridge.runtime as runtime_module

    client = _Client()

    class FakeNode:
        def __init__(self, name: str, *, context: Any) -> None:
            self.name = name
            self.context = context
            self.destroyed: list[tuple[str, Any]] = []

        def create_subscription(self, message_type, topic, callback, qos):
            self.subscription = SimpleNamespace(
                message_type=message_type,
                topic=topic,
                callback=callback,
                qos=qos,
            )
            return self.subscription

        def create_client(self, service_type, service_name):
            self.service_type = service_type
            self.service_name = service_name
            return client

        def create_timer(self, period, callback):
            self.timer = SimpleNamespace(period=period, callback=callback)
            return self.timer

        def destroy_subscription(self, value):
            self.destroyed.append(("subscription", value))

        def destroy_client(self, value):
            self.destroyed.append(("client", value))

        def destroy_timer(self, value):
            self.destroyed.append(("timer", value))

        def destroy_node(self):
            self.destroyed.append(("node", self))

    runtime = SimpleNamespace(
        context=object(),
        added=[],
        removed=[],
        add_node=lambda node: runtime.added.append(node),
        remove_node=lambda node: runtime.removed.append(node),
    )
    released: list[Any] = []
    monkeypatch.setattr(rclpy.node, "Node", FakeNode)
    monkeypatch.setattr(runtime_module, "acquire_ros2_runtime", lambda: runtime)
    monkeypatch.setattr(runtime_module, "release_ros2_runtime", released.append)

    bridge = EdgeTAMControlBridge(lambda _: None)
    bridge.start()
    node = runtime.added[0]

    assert bridge.is_started
    assert node.subscription.topic == "/edgetam_tracker/diagnostics"
    assert node.service_name == "/edgetam_tracker/set_enabled"

    bridge.close()

    assert runtime.removed == [node]
    assert released == [runtime]
    assert {kind for kind, _ in node.destroyed} == {
        "subscription",
        "client",
        "timer",
        "node",
    }
    assert not bridge.is_started
