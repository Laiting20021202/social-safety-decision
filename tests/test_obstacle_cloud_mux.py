from types import SimpleNamespace

import pytest

from realtime_safety.ros2_bridge.obstacle_cloud_mux import ObstacleCloudMux


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


def test_mux_holds_old_source_until_requested_pipeline_produces_output() -> None:
    statuses = []
    mux = ObstacleCloudMux(on_status=statuses.append, initial_mode="edgetam")
    publisher = _Publisher()
    mux._publisher = publisher
    edge_before = SimpleNamespace(name="edge-before")
    edge_during_switch = SimpleNamespace(name="edge-during-switch")
    yolo_first = SimpleNamespace(name="yolo-first")

    mux._on_source(edge_before, "edge")
    mux.request_mode("yolo")
    mux._on_source(edge_during_switch, "edge")

    assert publisher.messages == [edge_before, edge_during_switch]
    assert mux.active_mode == "edgetam"
    assert mux.requested_mode == "yolo"
    assert statuses[-1].state == "waiting"

    mux._on_source(yolo_first, "yolo")

    assert publisher.messages[-1] is yolo_first
    assert mux.active_mode == "yolo"
    assert statuses[-1].state == "active"


def test_mux_keeps_yolo_live_while_waiting_for_first_edge_cloud() -> None:
    mux = ObstacleCloudMux(initial_mode="edgetam")
    publisher = _Publisher()
    mux._publisher = publisher
    yolo_first = SimpleNamespace(name="yolo-first")
    yolo_during_switch = SimpleNamespace(name="yolo-during-switch")
    edge_first = SimpleNamespace(name="edge-first")

    mux.request_mode("yolo")
    mux._on_source(yolo_first, "yolo")
    mux.request_mode("edgetam")
    mux._on_source(yolo_during_switch, "yolo")

    assert publisher.messages == [yolo_first, yolo_during_switch]
    assert mux.active_mode == "yolo"
    assert mux.requested_mode == "edgetam"

    mux._on_source(edge_first, "edge")

    assert publisher.messages[-1] is edge_first
    assert mux.active_mode == "edgetam"


def test_rapid_switch_cancels_superseded_pending_source() -> None:
    mux = ObstacleCloudMux(initial_mode="edgetam")
    publisher = _Publisher()
    mux._publisher = publisher
    edge_before = SimpleNamespace(name="edge-before")
    late_yolo = SimpleNamespace(name="late-yolo")
    edge_after = SimpleNamespace(name="edge-after")

    mux._on_source(edge_before, "edge")
    mux.request_mode("yolo")
    mux.request_mode("edgetam")
    mux._on_source(late_yolo, "yolo")
    mux._on_source(edge_after, "edge")

    assert publisher.messages == [edge_before, edge_after]
    assert mux.requested_mode == "edgetam"
    assert mux.active_mode == "edgetam"


def test_edgetam_and_pointcloud_share_edge_source_without_output_gap() -> None:
    statuses = []
    mux = ObstacleCloudMux(on_status=statuses.append, initial_mode="edgetam")
    publisher = _Publisher()
    mux._publisher = publisher
    before = SimpleNamespace(name="before")
    during_pointcloud = SimpleNamespace(name="during-pointcloud")
    after_edgetam = SimpleNamespace(name="after-edgetam")

    mux._on_source(before, "edge")

    assert mux.request_mode("pointcloud")
    mux._on_source(during_pointcloud, "edge")
    assert mux.request_mode("edgetam")
    mux._on_source(after_edgetam, "edge")

    assert publisher.messages == [before, during_pointcloud, after_edgetam]
    assert mux.active_mode == "edgetam"
    assert mux.requested_mode == "edgetam"
    assert statuses[-1].state == "active"


def test_mux_rejects_unsafe_topic_alias_and_unknown_mode() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        ObstacleCloudMux(edge_topic="/same", output_topic="/same")
    mux = ObstacleCloudMux()
    with pytest.raises(ValueError, match="edgetam, pointcloud, or yolo"):
        mux.request_mode("sam")


def test_stale_yolo_holds_selected_mode_and_recovers_without_model_jump() -> None:
    statuses = []
    mux = ObstacleCloudMux(
        on_status=statuses.append,
        initial_mode="yolo",
        stale_timeout_sec=1.0,
    )
    publisher = _Publisher()
    mux._publisher = publisher
    edge = SimpleNamespace(name="edge-fallback")
    yolo = SimpleNamespace(name="yolo-live")
    mux._on_source(edge, "edge")
    mux._on_source(yolo, "yolo")
    edge_at = mux._last_received_at["edge"]
    assert edge_at is not None
    mux._last_received_at["yolo"] = edge_at - 2.0

    mux._check_stale(edge_at)

    assert mux.active_mode == "yolo"
    assert mux.requested_mode == "yolo"
    assert publisher.messages[-1] is yolo
    assert statuses[-1].state == "error"
    assert "operator selects another model" in statuses[-1].message

    resumed = SimpleNamespace(name="yolo-resumed")
    mux._on_source(resumed, "yolo")
    assert mux.active_mode == "yolo"
    assert publisher.messages[-1] is resumed


def test_stale_yolo_without_edge_clears_error_when_yolo_resumes() -> None:
    statuses = []
    mux = ObstacleCloudMux(
        on_status=statuses.append,
        initial_mode="yolo",
        stale_timeout_sec=1.0,
    )
    publisher = _Publisher()
    mux._publisher = publisher
    mux._on_source(SimpleNamespace(name="yolo-before-gap"), "yolo")
    received_at = mux._last_received_at["yolo"]
    assert received_at is not None

    mux._check_stale(received_at + 2.0)

    assert statuses[-1].state == "error"
    assert "operator selects another model" in statuses[-1].message

    resumed = SimpleNamespace(name="yolo-after-gap")
    mux._on_source(resumed, "yolo")

    assert publisher.messages[-1] is resumed
    assert statuses[-1].state == "active"
    assert statuses[-1].message == "yolo obstacle cloud resumed"


def test_active_source_recovery_preserves_pending_switch() -> None:
    statuses = []
    mux = ObstacleCloudMux(
        on_status=statuses.append,
        initial_mode="yolo",
        stale_timeout_sec=1.0,
    )
    publisher = _Publisher()
    mux._publisher = publisher
    mux._on_source(SimpleNamespace(name="yolo-before-gap"), "yolo")
    mux.request_mode("edgetam")
    received_at = mux._last_received_at["yolo"]
    assert received_at is not None
    mux._check_stale(received_at + 2.0)

    resumed = SimpleNamespace(name="yolo-resumed")
    mux._on_source(resumed, "yolo")

    assert publisher.messages[-1] is resumed
    assert mux.active_mode == "yolo"
    assert mux.requested_mode == "edgetam"
    assert statuses[-1].state == "waiting"
    assert "waiting for edgetam" in statuses[-1].message
