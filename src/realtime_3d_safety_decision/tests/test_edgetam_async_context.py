from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from types import ModuleType, SimpleNamespace
import sys
import threading
import time

import numpy as np
from builtin_interfaces.msg import Time as RosTime
from diagnostic_msgs.msg import DiagnosticStatus

# Unit-test the ROS adapter without requiring a prior colcon install of this
# repository's generated messages. Standard ROS 2 Python packages remain real.
try:
    import realtime_3d_safety_decision.msg  # type: ignore[import-not-found]
except ImportError:
    package = ModuleType("realtime_3d_safety_decision")
    messages = ModuleType("realtime_3d_safety_decision.msg")
    messages.TrackedObstacle = type("TrackedObstacle", (), {})
    messages.TrackedObstacleArray = type("TrackedObstacleArray", (), {})
    package.msg = messages  # type: ignore[attr-defined]
    sys.modules["realtime_3d_safety_decision"] = package
    sys.modules["realtime_3d_safety_decision.msg"] = messages

from std_msgs.msg import Header

from realtime_safety.edgetam_tracker.edgetam_wrapper import EdgeTAMResult
from realtime_safety.edgetam_tracker.mask_pointcloud_fusion import FusionConfig
from realtime_safety.edgetam_tracker.hand_semantic_gate import HandDetection
from realtime_safety.edgetam_tracker.models import (
    AABB,
    CloudFrame,
    MaskObservation,
    MaskQuality,
    OBB,
    PointCloudQuality,
    ProjectionPrompt,
    TrackEstimate,
    TrackingState,
)
from realtime_safety.edgetam_tracker.quality import (
    ConfidenceConfig,
    MaskQualityConfig,
)
from realtime_safety.edgetam_tracker.tracked_obstacle_node import (
    EdgeTAMPointCloudTrackerNode,
    _EdgeFrameContext,
    _background_output_is_trusted,
    _fresh_measured_obstacle_cloud,
)
from realtime_safety.edgetam_tracker.pointcloud_preprocessor import (
    StaticBackgroundState,
)
import realtime_safety.edgetam_tracker.tracked_obstacle_node as node_module


def test_hand_output_waits_for_enabled_background_calibration() -> None:
    assert not _background_output_is_trusted(
        StaticBackgroundState.CALIBRATING,
        background_enabled=True,
        ray_depth_enabled=True,
        alignment_valid=False,
    )


def test_hand_output_rejects_untrusted_ray_depth_alignment() -> None:
    assert not _background_output_is_trusted(
        StaticBackgroundState.READY,
        background_enabled=True,
        ray_depth_enabled=True,
        alignment_valid=False,
    )
    assert _background_output_is_trusted(
        StaticBackgroundState.READY,
        background_enabled=True,
        ray_depth_enabled=True,
        alignment_valid=True,
    )


def test_disabled_background_can_continue_to_rgb_hand_gate() -> None:
    assert _background_output_is_trusted(
        StaticBackgroundState.DISABLED,
        background_enabled=False,
        ray_depth_enabled=False,
        alignment_valid=False,
    )


def test_alignment_recovery_selects_only_rgb_hand_supported_3d_rays() -> None:
    points = np.arange(18, dtype=np.float32).reshape(6, 3)
    cloud = CloudFrame(
        points=points,
        pixels_uv=np.array(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [-1, -1]],
            dtype=np.int32,
        ),
        image_shape=(2, 3),
        stamp=1.0,
        frame_id="tracking",
    )
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    mask[1, 0] = True
    detection = HandDetection(
        bbox_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
        confidence=0.9,
        mask=mask,
        image_size=(3, 2),
    )

    selected = EdgeTAMPointCloudTrackerNode._cloud_inside_hand_detections(
        cloud, [detection], (2, 3)
    )

    np.testing.assert_array_equal(selected.points, points[[1, 3]])
    np.testing.assert_array_equal(
        selected.pixels_uv, np.array([[1, 0], [0, 1]], dtype=np.int32)
    )


def _header(stamp: float, frame_id: str = "tracking") -> Header:
    result = Header()
    result.stamp.sec = int(stamp)
    result.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    result.frame_id = frame_id
    return result


def test_published_cloud_excludes_cached_points_from_occluded_track() -> None:
    source_points = np.array(
        ((0.0, 0.8, 0.0), (0.1, 0.8, 0.0)), dtype=np.float32
    )
    fresh = TrackEstimate(
        track_id=4,
        state=TrackingState.CONFIRMED,
        position=np.mean(source_points, axis=0),
        velocity=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        aabb=AABB(source_points.min(axis=0), source_points.max(axis=0)),
        obb=OBB(
            np.mean(source_points, axis=0),
            np.array((0.1, 0.02, 0.02), dtype=np.float32),
            np.eye(3),
        ),
        nearest_point=source_points[0],
        nearest_distance=float(np.linalg.norm(source_points[0])),
        point_count=len(source_points),
        hit_count=3,
        missed_count=0,
        age_frames=3,
        first_timestamp=9.8,
        last_measurement_timestamp=10.0,
        filter_timestamp=10.0,
        confidence=0.9,
        pointcloud_quality=PointCloudQuality.GOOD,
        source_points=source_points,
    )
    stale = replace(
        fresh,
        track_id=5,
        state=TrackingState.OCCLUDED,
        missed_count=1,
        last_measurement_timestamp=9.9,
        pointcloud_quality=PointCloudQuality.INVALID,
        source_points=source_points + 1.0,
    )

    points, colors = _fresh_measured_obstacle_cloud([fresh, stale], 10.0)
    np.testing.assert_allclose(points, source_points)
    assert colors.shape == (2, 3)

    empty_points, empty_colors = _fresh_measured_obstacle_cloud(
        [stale], 10.0
    )
    assert empty_points.shape == (0, 3)
    assert empty_colors.shape == (0, 3)


def _context(*, serial: int = 7, stamp: float = 10.0) -> _EdgeFrameContext:
    prompt = ProjectionPrompt(
        track_id=1,
        frame_index=4,
        box_xyxy=np.array([0, 0, 2, 2], dtype=np.float32),
        positive_points=np.array([[1, 1]], dtype=np.float32),
        projection_mask=np.ones((2, 2), dtype=bool),
    )
    track = SimpleNamespace(
        track_id=1,
        first_timestamp=1.0,
        edge_tam_refined=False,
        position=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    return _EdgeFrameContext(
        sequence=9,
        node_generation=3,
        frame_index=4,
        geometry_stamp=stamp,
        rgb_stamp=stamp + 0.01,
        submitted_monotonic=time.monotonic(),
        publication_serial=serial,
        header=_header(stamp),
        rgb_header=_header(stamp + 0.01, "camera_optical"),
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        tracks=(track,),  # type: ignore[arg-type]
        clusters=(),
        raw_cloud=CloudFrame(
            points=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            pixels_uv=np.array([[1, 1]], dtype=np.int32),
            image_shape=(2, 2),
            stamp=stamp,
            frame_id="tracking",
        ),
        prompts={1: prompt},
        tracking_to_camera=np.eye(4, dtype=np.float64),
        fusion_config=FusionConfig(robot_origin=(0.2, 0.0, 0.0)),
    )


def _result(context: _EdgeFrameContext) -> EdgeTAMResult:
    return EdgeTAMResult(
        sequence=context.sequence,
        stream_generation=2,
        frame_index=context.frame_index,
        stamp=context.rgb_stamp,
        ok=True,
        masks={1: np.ones((2, 2), dtype=bool)},
        latency_ms=12.0,
    )


def _node_harness() -> EdgeTAMPointCloudTrackerNode:
    node = object.__new__(EdgeTAMPointCloudTrackerNode)
    node._edge_context_lock = threading.RLock()
    node._edge_enabled = True
    node._publication_lock = threading.RLock()
    node._edge_context_generation = 3
    node._safety_publication_serial = 7
    node._last_safety_output_stamp = 10.0
    node._edge_refined_corrections = 0
    node._edge_stale_results = 0
    node._latest_edge_latency_ms = 0.0
    node._latest_masks = {}
    node._previous_masks = {}
    node._reprompt_reasons = {}
    node._last_reprompt_stamp = {}
    node._frames_without_cluster_points = {}
    node._edge_status = "ready"
    node._edge_error = ""
    node._debug_image_publisher = None
    node._p = lambda name: {
        "sync.sensor_stale_timeout_sec": 0.5,
        "projection.use_projection_mask_prompt": True,
    }[name]
    observation = MaskObservation(
        track_id=1,
        frame_index=4,
        stamp=10.01,
        mask=np.ones((2, 2), dtype=bool),
    )
    node._edge_observations = lambda result, context: ({1: observation}, {})
    node._matching_current_track_ids = lambda context: {1}
    node.quality_updates = 0
    node._update_edge_quality_memory = (
        lambda context, observations, qualities, current_ids: setattr(
            node, "quality_updates", node.quality_updates + 1
        )
    )
    refined = SimpleNamespace(track_id=1, edge_tam_refined=True)
    node._decorate_and_fuse_tracks = (
        lambda *args, **kwargs: [refined]
    )
    node.published = []
    node._publish_safety_outputs = (
        lambda header, tracks: node.published.append(
            (header.stamp.sec + header.stamp.nanosec * 1e-9, tracks)
        )
    )
    return node


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class _ResettableEdge:
    available = True

    def __init__(self) -> None:
        self.reset_count = 0

    def reset_stream(self) -> None:
        self.reset_count += 1


class _RecordingPublisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


def _edge_control_harness(
    *, enabled: bool, loaded: bool = True
) -> tuple[EdgeTAMPointCloudTrackerNode, _ResettableEdge | None]:
    node = object.__new__(EdgeTAMPointCloudTrackerNode)
    edge = _ResettableEdge() if loaded else None
    node._edge_context_lock = threading.RLock()
    node._shutdown = threading.Event()
    node._edge_enabled = enabled
    node._edge_status = "ready" if enabled and loaded else "disabled"
    node._edge_error = "old error"
    node._edge = edge
    node._edge_loader = None
    node._edge_context_generation = 4
    node._edge_contexts = OrderedDict([(9, _context())])
    node._latest_masks = {1: object()}
    node._previous_masks = {1: np.ones((2, 2), dtype=bool)}
    node._reprompt_reasons = {1: "test"}
    node._last_reprompt_stamp = {1: 1.0}
    node._frames_without_cluster_points = {1: 1}
    node._tracker = object()
    logger = _RecordingLogger()
    node.get_logger = lambda: logger
    return node, edge


def test_runtime_disable_keeps_model_and_pointcloud_tracker_but_clears_contexts() -> None:
    node, edge = _edge_control_harness(enabled=True)
    tracker = node._tracker
    response = SimpleNamespace(success=False, message="")

    result = node._set_edgetam_enabled(
        SimpleNamespace(data=False), response
    )

    assert result is response
    assert response.success
    assert "point-cloud tracking remains active" in response.message
    assert node._edge_status == "disabled"
    assert not node._edge_enabled
    assert node._edge is edge
    assert node._tracker is tracker
    assert edge is not None and edge.reset_count == 1
    assert node._edge_context_generation == 5
    assert not node._edge_contexts
    assert not node._latest_masks
    assert not node._previous_masks
    assert not node._reprompt_reasons


def test_runtime_enable_reuses_loaded_model_and_starts_with_fresh_context() -> None:
    node, edge = _edge_control_harness(enabled=False)
    response = SimpleNamespace(success=False, message="")
    loader_calls = 0

    def unexpected_loader() -> None:
        nonlocal loader_calls
        loader_calls += 1

    node._ensure_edgetam_loader = unexpected_loader

    node._set_edgetam_enabled(SimpleNamespace(data=True), response)

    assert response.success
    assert response.message == "EdgeTAM refinement enabled"
    assert node._edge_enabled
    assert node._edge_status == "ready"
    assert node._edge_error == ""
    assert node._edge is edge
    assert edge is not None and edge.reset_count == 1
    assert loader_calls == 0
    assert not node._edge_contexts


def test_runtime_enable_without_model_queues_async_loader() -> None:
    node, _ = _edge_control_harness(enabled=False, loaded=False)
    response = SimpleNamespace(success=False, message="")
    loader_calls = 0

    def record_loader() -> None:
        nonlocal loader_calls
        loader_calls += 1

    node._ensure_edgetam_loader = record_loader

    node._set_edgetam_enabled(SimpleNamespace(data=True), response)

    assert response.success
    assert "loading asynchronously" in response.message
    assert node._edge_enabled
    assert node._edge_status == "loading"
    assert loader_calls == 1


def test_disabled_runtime_rejects_late_async_refinement() -> None:
    node = _node_harness()
    context = _context()
    node._edge_enabled = False
    node._edge_status = "disabled"

    node._process_edge_result(_result(context), context)

    assert node.published == []
    assert node.quality_updates == 0
    assert node._edge_status == "disabled"


def test_diagnostics_explicitly_report_runtime_refinement_disabled() -> None:
    node, _ = _edge_control_harness(enabled=False)
    node._edge_error = ""
    node._edge_refined_corrections = 2
    node._edge_stale_results = 3
    node._latest_edge_latency_ms = 4.0
    node._state_lock = threading.Lock()
    node._pipeline_level = DiagnosticStatus.OK
    node._pipeline_message = "point-cloud tracking active"
    node._diagnostic_values = {}
    node._dropped_bundles = 0
    node._latest_fps = 8.0
    node._latest_latency_ms = 12.0
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: RosTime())
    )
    node.get_fully_qualified_name = lambda: "/edgetam_pointcloud_tracker"
    node._diagnostics_publisher = _RecordingPublisher()
    node._fps_publisher = _RecordingPublisher()
    node._latency_publisher = _RecordingPublisher()

    node._publish_diagnostics()

    diagnostic = node._diagnostics_publisher.messages[0]
    edge_status = diagnostic.status[1]
    values = {item.key: item.value for item in edge_status.values}
    assert edge_status.message == (
        "EdgeTAM refinement disabled; point-cloud tracking remains active"
    )
    assert values["enabled"] == "false"
    assert values["state"] == "disabled"
    assert values["error"] == ""


def test_exact_async_context_can_publish_same_stamp_refined_correction() -> None:
    node = _node_harness()
    context = _context()

    node._process_edge_result(_result(context), context)

    assert [stamp for stamp, _ in node.published] == [10.0]
    assert node._safety_publication_serial == 8
    assert node._last_safety_output_stamp == 10.0
    assert node._edge_refined_corrections == 1
    assert node.quality_updates == 1


def test_stale_result_updates_quality_but_cannot_overwrite_newer_output() -> None:
    node = _node_harness()
    context = _context(serial=7, stamp=10.0)
    # A newer geometry or bounded prediction already became authoritative.
    node._safety_publication_serial = 8
    node._last_safety_output_stamp = 11.0

    node._process_edge_result(_result(context), context)

    assert node.published == []
    assert node._safety_publication_serial == 8
    assert node._last_safety_output_stamp == 11.0
    assert node._edge_refined_corrections == 0
    assert node._edge_stale_results == 1
    assert node.quality_updates == 1


def test_publication_order_is_immediate_then_same_stamp_then_newer() -> None:
    node = _node_harness()
    node._safety_publication_serial = 0
    node._last_safety_output_stamp = None
    node.published = []
    node._publish_safety_outputs = lambda header, tracks: (
        node.published.append(
            header.stamp.sec + header.stamp.nanosec * 1e-9
        )
        or (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )
    )

    immediate = node._publish_new_safety_output(_header(10.0), [])
    assert immediate is not None
    context = _context(serial=immediate[2], stamp=10.0)
    assert node._try_publish_refined_context(
        context, [SimpleNamespace(edge_tam_refined=True)]
    )
    assert node._publish_new_safety_output(_header(11.0), []) is not None

    assert node.published == [10.0, 10.0, 11.0]


def test_new_safety_output_rejects_timestamp_regression() -> None:
    node = _node_harness()
    node._last_safety_output_stamp = 11.0

    result = node._publish_new_safety_output(
        _header(10.0), [SimpleNamespace()]
    )

    assert result is None
    assert node.published == []
    assert node._safety_publication_serial == 7


def test_confirmed_clock_reset_starts_new_publication_stamp_epoch() -> None:
    node = _node_harness()
    node._last_safety_output_stamp = 100.0
    node._last_prediction_publish = 50.0
    node._publish_safety_outputs = lambda header, tracks: (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
    )
    # A small/unconfirmed regression remains blocked.
    assert node._publish_new_safety_output(_header(99.0), []) is None

    node._reset_safety_publication_epoch()
    accepted = node._publish_new_safety_output(_header(0.1), [])

    assert accepted is not None
    assert node._last_safety_output_stamp == 0.1
    assert node._last_prediction_publish == 0.0


def test_mismatched_timed_out_and_missing_context_results_never_publish() -> None:
    node = _node_harness()
    context = _context()
    mismatch = _result(context)
    mismatch.stamp += 0.2

    node._process_edge_result(mismatch, context)
    assert node.published == []
    assert node.quality_updates == 0
    assert node._edge_status == "degraded"

    node = _node_harness()
    timed_out = replace(
        context,
        submitted_monotonic=time.monotonic() - 1.0,
    )
    node._process_edge_result(_result(timed_out), timed_out)
    assert node.published == []
    assert node.quality_updates == 0

    node = _node_harness()
    node._process_edge_result(_result(context), None)
    assert node.published == []
    assert node._edge_stale_results == 1


def test_rgb_resolution_change_discards_incompatible_previous_mask() -> None:
    node = _node_harness()
    node._mask_quality_config = MaskQualityConfig(
        minimum_mask_pixels=1,
        minimum_valid_depth_ratio=0.0,
        minimum_good_valid_depth_ratio=0.0,
    )
    node._previous_masks = {1: np.ones((3, 3), dtype=bool)}
    context = _context()

    observations, quality = (
        EdgeTAMPointCloudTrackerNode._edge_observations(
            node, _result(context), context
        )
    )

    assert set(observations) == {1}
    assert set(quality) == {1}
    assert 1 not in node._previous_masks
    assert node._reprompt_reasons[1] == "rgb_resolution_changed"


def test_reset_id_reuse_does_not_match_old_context_identity() -> None:
    node = _node_harness()
    node._last_processed_stamp = 11.0
    reused = SimpleNamespace(
        track_id=1,
        first_timestamp=11.0,
        state=TrackingState.CONFIRMED,
    )
    node._tracker = SimpleNamespace(
        predict_to=lambda stamp: [reused]
    )

    assert (
        EdgeTAMPointCloudTrackerNode._matching_current_track_ids(
            node, _context()
        )
        == set()
    )


def test_context_invalidation_resets_wrapper_and_all_mask_memory() -> None:
    node = _node_harness()
    reset_calls: list[bool] = []
    node._edge = SimpleNamespace(
        reset_stream=lambda: reset_calls.append(True)
    )
    node._edge_contexts = OrderedDict({9: _context()})
    node._previous_masks = {1: np.ones((2, 2), dtype=bool)}
    node._reprompt_reasons = {1: "old"}
    node._last_reprompt_stamp = {1: 9.0}
    node._frames_without_cluster_points = {1: 2}

    node._invalidate_edge_contexts(reset_wrapper=True)

    assert node._edge_context_generation == 4
    assert not node._edge_contexts
    assert not node._latest_masks
    assert not node._previous_masks
    assert not node._reprompt_reasons
    assert reset_calls == [True]


def test_fast_completion_is_visible_only_after_exact_context_is_stored() -> None:
    node = _node_harness()
    node._edge_contexts = OrderedDict()
    node._edge_context_limit = 4
    node._frame_index = 4
    node._fusion_config = FusionConfig()
    context = _context()

    class _ImmediateEdge:
        available = True
        latest_result: EdgeTAMResult | None = None

        def submit(
            self,
            rgb: np.ndarray,
            frame_index: int,
            stamp: float,
            prompts: list[ProjectionPrompt],
            *,
            active_track_ids: list[int],
        ) -> int:
            self.latest_result = EdgeTAMResult(
                sequence=1,
                stream_generation=0,
                frame_index=frame_index,
                stamp=stamp,
                ok=True,
                masks={1: np.ones((2, 2), dtype=bool)},
            )
            return 1

    node._edge = _ImmediateEdge()
    sequence = node._submit_edgetam(
        np.zeros((2, 2, 3), dtype=np.uint8),
        list(context.prompts.values()),
        rgb_stamp=context.rgb_stamp,
        geometry_stamp=context.geometry_stamp,
        header=context.header,
        rgb_header=context.rgb_header,
        tracks=list(context.tracks),
        clusters=[],
        raw_cloud=context.raw_cloud,
        prompts_by_id=context.prompts,
        tracking_to_camera=context.tracking_to_camera,
        publication_serial=context.publication_serial,
    )
    assert sequence == 1
    assert list(node._edge_contexts) == [1]

    node._last_edge_result_sequence = 0
    node._poll_edge_result()

    assert list(node._edge_contexts) == []
    assert [stamp for stamp, _ in node.published] == [10.0]


def test_submission_context_store_is_bounded_and_copies_rgb() -> None:
    node = _node_harness()
    node._edge_contexts = OrderedDict()
    node._edge_context_limit = 2
    node._frame_index = 4
    node._fusion_config = FusionConfig()
    context = _context()

    class _QueuedEdge:
        available = True

        def __init__(self) -> None:
            self.sequence = 0

        def submit(self, *args, **kwargs) -> int:
            self.sequence += 1
            return self.sequence

    node._edge = _QueuedEdge()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    for offset in range(3):
        node._frame_index = 4 + offset
        node._submit_edgetam(
            rgb,
            list(context.prompts.values()),
            rgb_stamp=context.rgb_stamp + offset,
            geometry_stamp=context.geometry_stamp + offset,
            header=_header(context.geometry_stamp + offset),
            rgb_header=_header(
                context.rgb_stamp + offset, "camera_optical"
            ),
            tracks=list(context.tracks),
            clusters=[],
            raw_cloud=context.raw_cloud,
            prompts_by_id=context.prompts,
            tracking_to_camera=context.tracking_to_camera,
            publication_serial=context.publication_serial + offset,
        )
    rgb[:] = 255

    assert list(node._edge_contexts) == [2, 3]
    assert np.count_nonzero(node._edge_contexts[3].rgb) == 0


def test_async_debug_overlay_uses_exact_rgb_header(
    monkeypatch,
) -> None:
    node = _node_harness()
    published: list[SimpleNamespace] = []
    node._debug_image_publisher = SimpleNamespace(
        publish=published.append
    )
    monkeypatch.setattr(
        node_module,
        "render_debug_image",
        lambda rgb, tracks, prompts, masks, status_text: rgb,
    )
    monkeypatch.setattr(
        node_module,
        "make_rgb_image_message",
        lambda image, *, header, image_type: SimpleNamespace(
            header=header, image=image
        ),
    )
    context = _context(stamp=10.0)
    observation = MaskObservation(
        track_id=1,
        frame_index=4,
        stamp=10.01,
        mask=np.ones((2, 2), dtype=bool),
    )

    node._publish_edge_debug_context(
        context,
        [],
        {1: observation},
        safety_correction_published=False,
    )

    assert len(published) == 1
    assert published[0].header.frame_id == "camera_optical"
    assert published[0].header.stamp.sec == 10
    assert published[0].header.stamp.nanosec == 10_000_000


def test_live_debug_snapshot_clears_stale_mask_with_current_rgb(
    monkeypatch,
) -> None:
    node = _node_harness()
    published: list[SimpleNamespace] = []
    rendered: list[tuple[np.ndarray, str]] = []
    node._debug_image_publisher = SimpleNamespace(publish=published.append)

    def render(rgb, tracks, prompts, masks, status_text):
        assert tracks == []
        assert prompts == {}
        assert masks == {}
        rendered.append((rgb, status_text))
        return rgb

    monkeypatch.setattr(node_module, "render_debug_image", render)
    monkeypatch.setattr(
        node_module,
        "make_rgb_image_message",
        lambda image, *, header, image_type: SimpleNamespace(
            header=header, image=image
        ),
    )
    rgb = np.full((2, 2, 3), 23, dtype=np.uint8)
    header = _header(12.25, "camera_optical")

    node._publish_live_debug_snapshot(
        rgb,
        header,
        status_text="LIVE frame=8 · background alignment untrusted · no mask",
    )

    assert len(published) == 1
    assert published[0].header is header
    np.testing.assert_array_equal(rendered[0][0], rgb)
    assert rendered[0][1].startswith("LIVE frame=8")
    assert rendered[0][1].endswith("no mask")


def test_zero_emergency_distance_never_admits_invalid_origin_cluster() -> None:
    invalid = SimpleNamespace(
        quality=PointCloudQuality.INVALID,
        nearest_distance=0.0,
    )
    good = SimpleNamespace(
        quality=PointCloudQuality.GOOD,
        nearest_distance=2.0,
    )

    usable, emergency = (
        EdgeTAMPointCloudTrackerNode._select_usable_clusters(
            [invalid, good], 0.0
        )
    )

    assert usable == [good]
    assert emergency == []


def test_occluded_confirmed_track_uses_mask_depth_without_claiming_3d_measurement() -> None:
    node = object.__new__(EdgeTAMPointCloudTrackerNode)
    node._clusterer = SimpleNamespace(
        config=SimpleNamespace(depth_axis=1)
    )
    node._fusion_config = FusionConfig(robot_origin=(100.0, 0.0, 0.0))
    node._tracker_config = SimpleNamespace(confirmation_hits=3)
    node._confidence_config = ConfidenceConfig()
    node._p = {
        "tracking.maximum_association_distance": 0.4,
        "safety.base_uncertainty_margin_m": 0.03,
        "safety.maximum_uncertainty_margin_m": 0.25,
    }.__getitem__
    yy, xx = np.indices((5, 5))
    points = np.column_stack(
        (
            (xx.reshape(-1) - 2) * 0.01,
            np.ones(25),
            (yy.reshape(-1) - 2) * 0.01,
        )
    ).astype(np.float32)
    # Same frame-local source ID as a cached point, but a genuinely closer
    # current depth sample. Cross-frame ID dedup must not discard it.
    points[0, 1] = 0.95
    pixels = np.column_stack(
        (xx.reshape(-1), yy.reshape(-1))
    ).astype(np.int32)
    raw_cloud = CloudFrame(
        points=points,
        pixels_uv=pixels,
        image_shape=(5, 5),
        stamp=2.0,
        frame_id="tracking",
    )
    cached = points[[6, 8, 16, 18]]
    track = TrackEstimate(
        track_id=1,
        state=TrackingState.OCCLUDED,
        position=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        velocity=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6),
        aabb=AABB(
            np.array([-0.03, 0.97, -0.03]),
            np.array([0.03, 1.03, 0.03]),
        ),
        obb=OBB(
            np.array([0.0, 1.0, 0.0]),
            np.array([0.06, 0.06, 0.06]),
            np.eye(3),
        ),
        nearest_point=cached[0],
        nearest_distance=float(np.linalg.norm(cached[0])),
        point_count=len(cached),
        hit_count=4,
        missed_count=1,
        age_frames=5,
        first_timestamp=1.0,
        last_measurement_timestamp=1.9,
        filter_timestamp=2.0,
        confidence=0.5,
        pointcloud_quality=PointCloudQuality.INVALID,
        source_points=cached,
        source_indices=np.arange(len(cached), dtype=np.int64),
        uncertainty_margin=0.01,
    )
    mask = MaskObservation(
        track_id=1,
        frame_index=2,
        stamp=2.0,
        mask=np.ones((5, 5), dtype=bool),
        quality=MaskQuality.GOOD,
        quality_score=0.9,
    )
    prompt = ProjectionPrompt(
        track_id=1,
        frame_index=2,
        box_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
        positive_points=np.array([[2, 2]], dtype=np.float32),
        projection_mask=np.ones((5, 5), dtype=bool),
    )
    wrong_point = np.array([[0.0, 1.02, 0.0]], dtype=np.float32)
    wrong_track = replace(
        track,
        track_id=2,
        position=wrong_point[0],
        source_points=wrong_point,
        aabb=AABB(
            np.array([-0.01, 1.01, -0.01]),
            np.array([0.01, 1.03, 0.01]),
        ),
        obb=OBB(
            wrong_point[0],
            np.array([0.02, 0.02, 0.02]),
            np.eye(3),
        ),
        nearest_point=wrong_point[0],
    )
    wrong_nearest_cluster = node._track_geometry_proxy(wrong_track)

    refined = node._decorate_and_fuse_tracks(
        [track],
        [wrong_nearest_cluster],
        raw_cloud,
        {1: prompt},
        {1: mask},
        tracking_to_camera=None,
        fusion_config=FusionConfig(
            erosion_iterations=0,
            minimum_component_pixels=1,
            minimum_fused_points=3,
            aabb_gate_margin=0.03,
            center_gate_margin=0.03,
            robot_origin=(0.0, 0.0, 0.0),
        ),
    )[0]

    assert refined.edge_tam_refined
    assert refined.pointcloud_quality is PointCloudQuality.INVALID
    assert refined.point_count > track.point_count
    assert refined.confidence <= track.confidence
    assert refined.uncertainty_margin >= 0.05
    assert refined.nearest_distance < 2.0
    assert not np.any(
        np.all(np.isclose(refined.source_points, wrong_point[0]), axis=1)
    )
    assert np.any(
        np.all(np.isclose(refined.source_points, points[0]), axis=1)
    )


def test_live_edge_mask_follows_current_predicted_prompt_box() -> None:
    node = object.__new__(EdgeTAMPointCloudTrackerNode)
    node._edge_context_lock = threading.RLock()
    node._frame_index = 12
    previous_mask = np.zeros((80, 120), dtype=bool)
    previous_mask[20:40, 30:50] = True
    previous_prompt = ProjectionPrompt(
        track_id=4,
        frame_index=10,
        box_xyxy=np.array([30, 20, 50, 40], dtype=np.float32),
        positive_points=np.array([[40, 30]], dtype=np.float32),
        projection_mask=previous_mask.copy(),
    )
    current_projection = np.zeros((80, 120), dtype=bool)
    current_projection[24:54, 37:67] = True
    current_prompt = ProjectionPrompt(
        track_id=4,
        frame_index=12,
        box_xyxy=np.array([37, 24, 67, 54], dtype=np.float32),
        positive_points=np.array([[52, 39]], dtype=np.float32),
        projection_mask=current_projection,
    )
    node._previous_masks = {4: previous_mask}
    node._previous_mask_prompts = {4: previous_prompt}

    result = node._live_predicted_masks({4: current_prompt}, stamp=2.0)

    assert set(result) == {4}
    assert result[4].quality is MaskQuality.DEGRADED
    rows, columns = np.nonzero(result[4].mask)
    assert (columns.min(), rows.min()) == (37, 24)
    assert (columns.max() + 1, rows.max() + 1) == (67, 54)


def _stamp(value: float) -> SimpleNamespace:
    return SimpleNamespace(
        sec=int(value),
        nanosec=int(round((value - int(value)) * 1e9)),
    )


def test_camera_info_static_stamp_is_explicit_and_frame_checked() -> None:
    node = object.__new__(EdgeTAMPointCloudTrackerNode)
    parameters = {
        "sync.allow_static_camera_info": True,
        "sync.slop_sec": 0.05,
    }
    node._p = parameters.__getitem__
    info = SimpleNamespace(
        width=640,
        height=480,
        header=SimpleNamespace(
            frame_id="camera_optical", stamp=_stamp(0.0)
        ),
    )
    image = SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera_optical", stamp=_stamp(20.0)
        )
    )
    assert node._camera_info_validation_error(
        info, image, (480, 640)
    ) == ""

    info.header.frame_id = "wrong_camera"
    assert "frame mismatch" in node._camera_info_validation_error(
        info, image, (480, 640)
    )


def test_camera_info_nonzero_stamp_must_be_within_sync_slop() -> None:
    node = object.__new__(EdgeTAMPointCloudTrackerNode)
    node._p = {
        "sync.allow_static_camera_info": True,
        "sync.slop_sec": 0.05,
    }.__getitem__
    info = SimpleNamespace(
        width=640,
        height=480,
        header=SimpleNamespace(
            frame_id="camera_optical", stamp=_stamp(19.8)
        ),
    )
    image = SimpleNamespace(
        header=SimpleNamespace(
            frame_id="camera_optical", stamp=_stamp(20.0)
        )
    )

    assert "timestamp mismatch" in node._camera_info_validation_error(
        info, image, (480, 640)
    )


def test_rgb_colors_require_same_depth_grid_and_frame() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_message = SimpleNamespace(
        header=SimpleNamespace(frame_id="camera_optical")
    )
    depth_message = SimpleNamespace(
        header=SimpleNamespace(frame_id="camera_optical")
    )
    assert EdgeTAMPointCloudTrackerNode._rgb_registered_to_depth(
        rgb, rgb_message, depth_message, (480, 640)
    )
    assert not EdgeTAMPointCloudTrackerNode._rgb_registered_to_depth(
        rgb, rgb_message, depth_message, (240, 320)
    )
    depth_message.header.frame_id = "depth_optical"
    assert not EdgeTAMPointCloudTrackerNode._rgb_registered_to_depth(
        rgb, rgb_message, depth_message, (480, 640)
    )
