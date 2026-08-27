#!/usr/bin/env python3
"""ROS graph smoke test for geometry, prediction-only, and stale behavior.

Run this only after building and sourcing the ament package. It uses synthetic
PointCloud2 input, does not load EdgeTAM, and never publishes to legacy topics.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from realtime_3d_safety_decision.msg import TrackedObstacleArray
from realtime_safety.edgetam_tracker.ros_utils import make_pointcloud2
from realtime_safety.edgetam_tracker.tracked_obstacle_node import (
    EdgeTAMPointCloudTrackerNode,
)


INPUT_TOPIC = "/edgetam_smoke/input_cloud"
OBSTACLE_TOPIC = "/edgetam_smoke/obstacles"
OUTPUT_CLOUD_TOPIC = "/edgetam_smoke/obstacle_cloud"
DIAGNOSTICS_TOPIC = "/edgetam_smoke/diagnostics"


class _Probe(Node):
    def __init__(self) -> None:
        super().__init__("edgetam_tracker_smoke_probe")
        self.publisher = self.create_publisher(
            PointCloud2, INPUT_TOPIC, qos_profile_sensor_data
        )
        self.obstacle_messages: list[
            tuple[float, TrackedObstacleArray]
        ] = []
        self.output_clouds: list[tuple[float, PointCloud2]] = []
        self.diagnostics: list[tuple[float, DiagnosticArray]] = []
        self.create_subscription(
            TrackedObstacleArray,
            OBSTACLE_TOPIC,
            lambda message: self.obstacle_messages.append(
                (time.monotonic(), message)
            ),
            10,
        )
        self.create_subscription(
            PointCloud2,
            OUTPUT_CLOUD_TOPIC,
            lambda message: self.output_clouds.append(
                (time.monotonic(), message)
            ),
            10,
        )
        self.create_subscription(
            DiagnosticArray,
            DIAGNOSTICS_TOPIC,
            lambda message: self.diagnostics.append(
                (time.monotonic(), message)
            ),
            10,
        )

    def publish_cluster(self, frame_index: int) -> None:
        rng = np.random.default_rng(700 + frame_index)
        center = np.array(
            [0.24 + frame_index * 0.005, 0.55, 0.04],
            dtype=np.float32,
        )
        points = center + rng.uniform(
            [-0.06, -0.05, -0.07],
            [0.06, 0.05, 0.07],
            size=(120, 3),
        ).astype(np.float32)
        colors = np.broadcast_to(
            np.array([220, 80, 40], dtype=np.uint8),
            (len(points), 3),
        ).copy()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "smoke_tracking_frame"
        self.publisher.publish(
            make_pointcloud2(
                points,
                colors,
                header=header,
                pointcloud_type=PointCloud2,
                pointfield_type=PointField,
            )
        )


def _parameters() -> list[Parameter]:
    values = {
        "topics.rgb_image": "",
        "topics.depth_image": "",
        "topics.camera_info": "",
        "topics.pointcloud": INPUT_TOPIC,
        "topics.output_obstacles": OBSTACLE_TOPIC,
        "topics.output_obstacle_cloud": OUTPUT_CLOUD_TOPIC,
        "topics.output_diagnostics": DIAGNOSTICS_TOPIC,
        "topics.output_fps": "/edgetam_smoke/fps",
        "topics.output_latency_ms": "/edgetam_smoke/latency",
        "frames.tracking_frame": "smoke_tracking_frame",
        "frames.robot_base_frame": "smoke_tracking_frame",
        "sync.max_data_age_sec": 1.0,
        "sync.pointcloud_fallback_delay_sec": 0.05,
        "sync.sensor_stale_timeout_sec": 0.60,
        "pointcloud.remove_outliers": False,
        "pointcloud.voxel_size": 0.01,
        "clustering.method": "euclidean",
        "clustering.tolerance": 0.09,
        "clustering.min_points": 8,
        "tracking.confirmation_hits": 2,
        "tracking.maximum_association_distance": 0.30,
        "safety.emergency_distance_m": 0.10,
        "edgetam.enabled": False,
        "self_filter.enabled": False,
        "compatibility.publish_legacy_obstacle_alias": False,
        "performance.publish_debug_image": False,
        "performance.publish_debug_cloud": False,
        "performance.publish_markers": False,
        "performance.prediction_publish_rate_hz": 20.0,
        "performance.diagnostics_rate_hz": 10.0,
    }
    return [Parameter(name, value=value) for name, value in values.items()]


def _wait_until(
    predicate: Callable[[], bool],
    timeout: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {description}")


def _pipeline_is_stale(probe: _Probe) -> bool:
    for _, message in reversed(probe.diagnostics):
        for status in message.status:
            if (
                status.name.endswith("/pipeline")
                and status.level == DiagnosticStatus.ERROR
                and "stale" in status.message
            ):
                return True
    return False


def main() -> int:
    rclpy.init()
    tracker: EdgeTAMPointCloudTrackerNode | None = None
    probe: _Probe | None = None
    executor: MultiThreadedExecutor | None = None
    try:
        tracker = EdgeTAMPointCloudTrackerNode(
            parameter_overrides=_parameters()
        )
        probe = _Probe()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(tracker)
        executor.add_node(probe)
        import threading

        spin_thread = threading.Thread(
            target=executor.spin,
            name="ros-smoke-executor",
            daemon=True,
        )
        spin_thread.start()
        time.sleep(0.45)
        for frame_index in range(5):
            probe.publish_cluster(frame_index)
            time.sleep(0.08)

        _wait_until(
            lambda: any(
                obstacle.tracking_state == "CONFIRMED"
                for _, message in probe.obstacle_messages
                for obstacle in message.obstacles
            ),
            3.0,
            "a confirmed point-cloud track",
        )
        confirmed = next(
            obstacle
            for _, message in reversed(probe.obstacle_messages)
            for obstacle in message.obstacles
            if obstacle.tracking_state == "CONFIRMED"
        )
        assert confirmed.point_count > 0
        assert confirmed.prediction_only is False
        assert len(confirmed.predicted_positions) == 3
        assert probe.output_clouds[-1][1].width > 0

        _wait_until(
            lambda: any(
                obstacle.prediction_only
                for _, message in probe.obstacle_messages
                for obstacle in message.obstacles
            ),
            2.0,
            "a bounded prediction-only obstacle",
        )
        predicted = next(
            obstacle
            for _, message in reversed(probe.obstacle_messages)
            for obstacle in message.obstacles
            if obstacle.prediction_only
        )
        assert predicted.track_id == confirmed.track_id
        assert predicted.tracking_state in {"OCCLUDED", "LOST"}
        assert (
            predicted.header.stamp.sec,
            predicted.header.stamp.nanosec,
        ) > (
            predicted.last_measurement_stamp.sec,
            predicted.last_measurement_stamp.nanosec,
        )

        _wait_until(
            lambda: _pipeline_is_stale(probe),
            2.0,
            "ERROR diagnostics after the stale timeout",
        )
        output_count_at_stale = len(probe.obstacle_messages)
        time.sleep(0.25)
        assert len(probe.obstacle_messages) == output_count_at_stale

        print(
            "PASS ROS tracker smoke: "
            f"track_id={confirmed.track_id}, "
            f"measurement_points={confirmed.point_count}, "
            f"output_cloud_points={probe.output_clouds[-1][1].width}, "
            "prediction_only=true, stale_output_stopped=true"
        )
        return 0
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if tracker is not None:
            tracker.destroy_node()
        if probe is not None:
            probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
