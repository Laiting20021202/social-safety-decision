from __future__ import annotations

"""Non-blocking ROS 2 control bridge for the EdgeTAM tracker.

The dashboard process uses the repository's shared ROS executor, so a GUI
callback must never wait for a service or for the model to load.  This bridge
therefore keeps only the newest requested mode, calls ``std_srvs/SetBool``
asynchronously, and reconciles the service result with the tracker's
diagnostics.

Importing this module does not require ROS 2.  ROS message and client types are
loaded lazily by :meth:`EdgeTAMControlBridge.start`.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Literal, Mapping


LOGGER = logging.getLogger(__name__)

EdgeTAMMode = Literal["edgetam", "pointcloud"]
EdgeTAMUiState = Literal["loading", "active", "error"]

_VALID_MODES = frozenset({"edgetam", "pointcloud"})
_EDGE_ACTIVE_STATES = frozenset({"ready", "degraded"})


@dataclass(frozen=True, slots=True)
class EdgeTAMControlStatus:
    """One immutable UI update from :class:`EdgeTAMControlBridge`.

    ``requested_mode`` is the user's latest selection.  ``active_mode`` is
    changed only after a successful service response or authoritative
    diagnostics, allowing the UI to say that the previous path remains active
    while EdgeTAM is loading.
    """

    state: EdgeTAMUiState
    requested_mode: EdgeTAMMode
    active_mode: EdgeTAMMode
    message: str
    diagnostics: Mapping[str, str]


@dataclass(slots=True)
class _Request:
    generation: int
    mode: EdgeTAMMode
    deadline: float


class EdgeTAMControlBridge:
    """Control ``/edgetam_tracker`` without blocking a dashboard callback."""

    valid_modes = ("edgetam", "pointcloud")

    def __init__(
        self,
        on_status: Callable[[EdgeTAMControlStatus], None],
        *,
        on_debug_image: Callable[[Any], None] | None = None,
        on_obstacle_cloud: Callable[[Any, Any, str], None] | None = None,
        diagnostics_topic: str = "/edgetam_tracker/diagnostics",
        debug_image_topic: str = "/edgetam_tracker/debug_image",
        obstacle_cloud_topic: str = "/edgetam_tracker/obstacle_cloud",
        service_name: str = "/edgetam_tracker/set_enabled",
        node_name: str = "realtime_safety_edgetam_control",
        request_timeout_sec: float = 8.0,
        initial_mode: EdgeTAMMode = "pointcloud",
    ) -> None:
        self._validate_ros_name(diagnostics_topic, "Diagnostics topic")
        self._validate_ros_name(debug_image_topic, "Debug image topic")
        self._validate_ros_name(obstacle_cloud_topic, "Obstacle cloud topic")
        self._validate_ros_name(service_name, "Service")
        if not node_name or any(character.isspace() for character in node_name):
            raise ValueError("ROS node name must not be empty or contain whitespace")
        initial = self._validate_mode(initial_mode)
        if request_timeout_sec <= 0.0:
            raise ValueError("EdgeTAM request timeout must be positive")

        self.diagnostics_topic = diagnostics_topic
        self.debug_image_topic = debug_image_topic
        self.obstacle_cloud_topic = obstacle_cloud_topic
        self.service_name = service_name
        self.node_name = node_name
        self.request_timeout_sec = float(request_timeout_sec)
        self._on_status = on_status
        self._on_debug_image = on_debug_image
        self._on_obstacle_cloud = on_obstacle_cloud

        self._lock = threading.RLock()
        self._runtime: Any | None = None
        self._node: Any | None = None
        self._subscription: Any | None = None
        self._debug_subscription: Any | None = None
        self._obstacle_cloud_subscription: Any | None = None
        self._client: Any | None = None
        self._timer: Any | None = None
        self._request_type: Any | None = None
        self._pending: _Request | None = None
        self._inflight: tuple[_Request, Any] | None = None
        self._generation = 0
        self._requested_mode: EdgeTAMMode = initial
        self._active_mode: EdgeTAMMode = initial
        self._state: EdgeTAMUiState = "loading"
        self._message = "Waiting for EdgeTAM diagnostics"
        self._diagnostics: dict[str, str] = {}
        self._mode_confirmed = False
        self._closed = False

    @staticmethod
    def _validate_ros_name(value: str, label: str) -> None:
        if not value.startswith("/") or any(character.isspace() for character in value):
            raise ValueError(f"{label} must be an absolute ROS name without whitespace")

    @staticmethod
    def _validate_mode(mode: str) -> EdgeTAMMode:
        normalized = str(mode).strip().lower()
        if normalized not in _VALID_MODES:
            raise ValueError(
                "EdgeTAM mode must be exactly 'edgetam' or 'pointcloud'"
            )
        return normalized  # type: ignore[return-value]

    @property
    def active_mode(self) -> EdgeTAMMode:
        with self._lock:
            return self._active_mode

    @property
    def requested_mode(self) -> EdgeTAMMode:
        with self._lock:
            return self._requested_mode

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._node is not None

    def start(self) -> None:
        """Attach the subscriber/client to the application's shared runtime."""

        with self._lock:
            if self._node is not None:
                return

        from diagnostic_msgs.msg import DiagnosticArray
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image, PointCloud2
        from std_srvs.srv import SetBool

        from realtime_safety.ros2_bridge.runtime import (
            acquire_ros2_runtime,
            release_ros2_runtime,
        )

        runtime = acquire_ros2_runtime()
        node: Any | None = None
        try:
            node = Node(self.node_name, context=runtime.context)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            obstacle_cloud_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            debug_image_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            subscription = node.create_subscription(
                DiagnosticArray,
                self.diagnostics_topic,
                self._on_diagnostics,
                qos,
            )
            debug_subscription = (
                node.create_subscription(
                    Image,
                    self.debug_image_topic,
                    self._on_debug_image_message,
                    debug_image_qos,
                )
                if self._on_debug_image is not None
                else None
            )
            obstacle_cloud_subscription = (
                node.create_subscription(
                    PointCloud2,
                    self.obstacle_cloud_topic,
                    self._on_obstacle_cloud_message,
                    obstacle_cloud_qos,
                )
                if self._on_obstacle_cloud is not None
                else None
            )
            client = node.create_client(SetBool, self.service_name)
            timer = node.create_timer(0.1, self._poll_requests)
            runtime.add_node(node)
        except Exception:
            if node is not None:
                node.destroy_node()
            release_ros2_runtime(runtime)
            raise

        with self._lock:
            self._runtime = runtime
            self._node = node
            self._subscription = subscription
            self._debug_subscription = debug_subscription
            self._obstacle_cloud_subscription = obstacle_cloud_subscription
            self._client = client
            self._timer = timer
            self._request_type = SetBool.Request
            self._closed = False
            # The tracker process starts with neural refinement enabled.  A
            # dashboard configured for the MediaPipe/YOLO fallback must send
            # its initial ``false`` request instead of merely labelling the
            # local bridge as pointcloud mode while EdgeTAM keeps consuming
            # GPU and publishing competing masks.
            if not self._mode_confirmed and self._pending is None:
                self._generation += 1
                self._pending = _Request(
                    generation=self._generation,
                    mode=self._requested_mode,
                    deadline=time.monotonic() + self.request_timeout_sec,
                )
                self._state = "loading"
                self._message = f"Requesting initial {self._requested_mode} mode"
            status = self._status_locked()
        self._notify(status)
        self._dispatch_pending_request()

    def _on_debug_image_message(self, message: Any) -> None:
        callback = self._on_debug_image
        if callback is None:
            return
        try:
            import numpy as np

            height = int(message.height)
            width = int(message.width)
            step = int(message.step)
            encoding = str(message.encoding).lower()
            channels = 4 if encoding in {"bgra8", "rgba8"} else 3
            if (
                height <= 0
                or width <= 0
                or channels * width > step
                or encoding not in {"bgr8", "rgb8", "bgra8", "rgba8"}
            ):
                return
            rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
                height, step
            )
            image = rows[:, : width * channels].reshape(
                height, width, channels
            )[..., :3]
            if encoding in {"rgb8", "rgba8"}:
                image = image[..., ::-1]
            callback(np.ascontiguousarray(image))
        except Exception:
            LOGGER.exception("Could not decode EdgeTAM debug image")

    def _on_obstacle_cloud_message(self, message: Any) -> None:
        callback = self._on_obstacle_cloud
        if callback is None:
            return
        try:
            import numpy as np

            # The safety publisher intentionally emits a width-zero cloud
            # when no obstacle is present.  Preserve that clearing signal;
            # the generic PointCloud2 decoder rejects zero-sized layouts as
            # malformed input.
            if int(message.width) * int(message.height) == 0:
                callback(
                    np.empty((0, 3), dtype=np.float32),
                    None,
                    str(message.header.frame_id),
                )
                return
            from realtime_safety.edgetam_tracker.sensor_sync import (
                pointcloud2_to_cloud,
            )

            cloud = pointcloud2_to_cloud(message)
            callback(cloud.points, cloud.colors, str(message.header.frame_id))
        except Exception:
            LOGGER.exception("Could not decode EdgeTAM obstacle cloud")

    def request_mode(self, mode: str) -> bool:
        """Queue a mode change and return immediately.

        A request made before :meth:`start` is retained and dispatched as soon
        as the ROS bridge starts.  A later request supersedes an older pending
        request; an already-sent service call is allowed to finish before the
        newest request is sent so service side effects remain ordered.
        """

        requested = self._validate_mode(mode)
        with self._lock:
            if self._closed:
                raise RuntimeError("EdgeTAM control bridge is closed")
            if (
                self._mode_confirmed
                and requested == self._active_mode
                and self._pending is None
                and self._inflight is None
            ):
                self._requested_mode = requested
                self._state = "active"
                self._message = f"{requested} mode is already active"
                status = self._status_locked()
                dispatch = False
            else:
                self._generation += 1
                request = _Request(
                    generation=self._generation,
                    mode=requested,
                    deadline=time.monotonic() + self.request_timeout_sec,
                )
                self._pending = request
                self._requested_mode = requested
                self._state = "loading"
                self._message = f"Requesting {requested} mode"
                status = self._status_locked()
                dispatch = True
        self._notify(status)
        if dispatch:
            self._dispatch_pending_request()
        return True

    def _dispatch_pending_request(self) -> None:
        with self._lock:
            if (
                self._closed
                or self._client is None
                or self._request_type is None
                or self._pending is None
                or self._inflight is not None
            ):
                return
            client = self._client
            pending = self._pending

        try:
            if not client.service_is_ready():
                return
            request_message = self._request_type()
            request_message.data = pending.mode == "edgetam"
            future = client.call_async(request_message)
        except Exception as exc:
            self._fail_request(pending, f"SetBool request failed: {exc}")
            return

        with self._lock:
            if self._pending is not pending or self._closed:
                cancel = getattr(future, "cancel", None)
                if callable(cancel):
                    cancel()
                return
            self._pending = None
            self._inflight = (pending, future)
        future.add_done_callback(
            lambda completed, sent=pending: self._on_service_result(sent, completed)
        )

    def _poll_requests(self) -> None:
        now = time.monotonic()
        timed_out: _Request | None = None
        future: Any | None = None
        with self._lock:
            if self._closed:
                return
            if self._inflight is not None:
                inflight, future = self._inflight
                if now >= inflight.deadline:
                    self._inflight = None
                    timed_out = inflight
            elif self._pending is not None and now >= self._pending.deadline:
                timed_out = self._pending
                self._pending = None
        if timed_out is not None:
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
            self._fail_request(
                timed_out,
                f"Timed out waiting for {self.service_name}",
                already_removed=True,
            )
        self._dispatch_pending_request()

    def _on_service_result(self, request: _Request, future: Any) -> None:
        try:
            response = future.result()
            success = bool(response.success)
            response_message = str(response.message).strip()
        except Exception as exc:
            success = False
            response_message = f"SetBool response failed: {exc}"

        with self._lock:
            owns_inflight = (
                self._inflight is not None
                and self._inflight[0] is request
                and self._inflight[1] is future
            )
            if owns_inflight:
                self._inflight = None
            current = (
                owns_inflight
                and request.generation == self._generation
                and not self._closed
            )
            if not current:
                status = None
            elif success:
                self._requested_mode = request.mode
                edge_state = self._diagnostics.get("state", "").lower()
                if request.mode == "pointcloud":
                    # Disabling is synchronous on the server: pending contexts
                    # are invalidated before SetBool returns success.
                    self._active_mode = "pointcloud"
                    self._mode_confirmed = True
                    self._state = "active"
                elif edge_state in _EDGE_ACTIVE_STATES:
                    # Enabling is asynchronous, but a ready/degraded
                    # diagnostic may already have arrived before the response.
                    self._active_mode = "edgetam"
                    self._mode_confirmed = True
                    self._state = "active"
                else:
                    # SetBool success for true means "accepted", not "model
                    # ready".  Only diagnostics may make EdgeTAM effective.
                    self._state = "loading"
                self._message = response_message or (
                    f"{request.mode} mode is active"
                    if self._state == "active"
                    else "EdgeTAM enable request accepted; waiting for diagnostics"
                )
                status = self._status_locked()
            else:
                self._state = "error"
                self._message = response_message or "EdgeTAM mode request was rejected"
                status = self._status_locked()
        if status is not None:
            self._notify(status)
        self._dispatch_pending_request()

    def _fail_request(
        self,
        request: _Request,
        message: str,
        *,
        already_removed: bool = False,
    ) -> None:
        with self._lock:
            if not already_removed and self._pending is request:
                self._pending = None
            current = request.generation == self._generation and not self._closed
            if current:
                self._state = "error"
                self._message = message
                status = self._status_locked()
            else:
                status = None
        if status is not None:
            self._notify(status)

    def _on_diagnostics(self, message: Any) -> None:
        diagnostics, edge_state, edge_level, edge_message = (
            self._extract_diagnostics(message)
        )
        normalized_state = edge_state.lower()
        diagnostic_error = diagnostics.get("error", "").strip()

        with self._lock:
            if self._closed:
                return
            self._diagnostics = diagnostics
            request_in_progress = self._pending is not None or self._inflight is not None

            # Before the first command, adopt the node's actual startup mode.
            if self._generation == 0 and not self._mode_confirmed:
                if normalized_state in _EDGE_ACTIVE_STATES:
                    self._active_mode = "edgetam"
                    self._requested_mode = "edgetam"
                    self._mode_confirmed = True
                    self._state = "active"
                elif normalized_state == "disabled":
                    self._active_mode = "pointcloud"
                    self._requested_mode = "pointcloud"
                    self._mode_confirmed = True
                    self._state = "active"
                elif normalized_state == "loading":
                    self._requested_mode = "edgetam"
                    self._state = "loading"
                elif normalized_state == "error":
                    self._requested_mode = "edgetam"
                    self._state = "error"

            if self._requested_mode == "edgetam":
                if normalized_state in _EDGE_ACTIVE_STATES:
                    self._active_mode = "edgetam"
                    self._mode_confirmed = True
                    self._state = "active"
                elif normalized_state == "loading":
                    self._state = "loading"
                elif normalized_state == "error" or edge_level >= 2:
                    self._state = "error"
            elif self._requested_mode == "pointcloud":
                if normalized_state == "disabled":
                    self._active_mode = "pointcloud"
                    self._mode_confirmed = True
                    self._state = "active"
                elif request_in_progress:
                    self._state = "loading"

            if self._state == "error":
                self._message = diagnostic_error or edge_message or "EdgeTAM reported an error"
            elif self._state == "loading":
                self._message = edge_message or f"Loading {self._requested_mode} mode"
            else:
                self._message = edge_message or f"{self._active_mode} mode is active"
            status = self._status_locked()
        self._notify(status)

    @staticmethod
    def _extract_diagnostics(
        message: Any,
    ) -> tuple[dict[str, str], str, int, str]:
        statuses = list(getattr(message, "status", ()) or ())
        edge_status = next(
            (
                status
                for status in statuses
                if str(getattr(status, "hardware_id", "")) == "official_edgetam"
                or str(getattr(status, "name", "")).rstrip("/").endswith("/edgetam")
            ),
            None,
        )
        pipeline_status = next(
            (
                status
                for status in statuses
                if str(getattr(status, "name", "")).rstrip("/").endswith("/pipeline")
            ),
            None,
        )

        diagnostics: dict[str, str] = {}
        if pipeline_status is not None:
            diagnostics["pipeline.level"] = str(
                EdgeTAMControlBridge._diagnostic_level(
                    getattr(pipeline_status, "level", 0)
                )
            )
            diagnostics["pipeline.message"] = str(
                getattr(pipeline_status, "message", "")
            )
            for item in getattr(pipeline_status, "values", ()) or ():
                diagnostics[f"pipeline.{item.key}"] = str(item.value)

        if edge_status is None:
            edge_state = diagnostics.get("pipeline.edge_status", "")
            edge_error = diagnostics.get("pipeline.edge_error", "")
            if edge_error:
                diagnostics["error"] = edge_error
            return diagnostics, edge_state, 0, ""

        edge_level = EdgeTAMControlBridge._diagnostic_level(
            getattr(edge_status, "level", 0)
        )
        edge_message = str(getattr(edge_status, "message", ""))
        diagnostics["edge.level"] = str(edge_level)
        diagnostics["edge.message"] = edge_message
        diagnostics["edge.name"] = str(getattr(edge_status, "name", ""))
        for item in getattr(edge_status, "values", ()) or ():
            diagnostics[str(item.key)] = str(item.value)
        edge_state = diagnostics.get("state", "")
        return diagnostics, edge_state, edge_level, edge_message

    @staticmethod
    def _diagnostic_level(value: Any) -> int:
        """Normalize ROS uint8 constants from both Humble and newer builds."""

        if isinstance(value, (bytes, bytearray)):
            return int(value[0]) if value else 0
        return int(value)

    def _status_locked(self) -> EdgeTAMControlStatus:
        return EdgeTAMControlStatus(
            state=self._state,
            requested_mode=self._requested_mode,
            active_mode=self._active_mode,
            message=self._message,
            diagnostics=dict(self._diagnostics),
        )

    def _notify(self, status: EdgeTAMControlStatus) -> None:
        try:
            self._on_status(status)
        except Exception:
            LOGGER.exception("EdgeTAM status callback failed")

    def close(self) -> None:
        """Detach all ROS entities and release one shared-runtime reference."""

        with self._lock:
            if self._closed and self._node is None:
                return
            self._closed = True
            self._generation += 1
            inflight = self._inflight
            self._pending = None
            self._inflight = None
            runtime = self._runtime
            node = self._node
            subscription = self._subscription
            debug_subscription = self._debug_subscription
            obstacle_cloud_subscription = self._obstacle_cloud_subscription
            client = self._client
            timer = self._timer
            self._runtime = None
            self._node = None
            self._subscription = None
            self._debug_subscription = None
            self._obstacle_cloud_subscription = None
            self._client = None
            self._timer = None
            self._request_type = None

        if inflight is not None:
            cancel = getattr(inflight[1], "cancel", None)
            if callable(cancel):
                cancel()
        if runtime is not None and node is not None:
            runtime.remove_node(node)
        if node is not None and subscription is not None:
            node.destroy_subscription(subscription)
        if node is not None and debug_subscription is not None:
            node.destroy_subscription(debug_subscription)
        if node is not None and obstacle_cloud_subscription is not None:
            node.destroy_subscription(obstacle_cloud_subscription)
        if node is not None and client is not None:
            node.destroy_client(client)
        if node is not None and timer is not None:
            node.destroy_timer(timer)
        if node is not None:
            node.destroy_node()
        if runtime is not None:
            from realtime_safety.ros2_bridge.runtime import release_ros2_runtime

            release_ros2_runtime(runtime)


EdgeTAMController = EdgeTAMControlBridge

__all__ = [
    "EdgeTAMControlBridge",
    "EdgeTAMController",
    "EdgeTAMControlStatus",
    "EdgeTAMMode",
    "EdgeTAMUiState",
]
