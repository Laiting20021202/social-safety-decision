#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure ROS 2 PointCloud2 liveness, point counts, and wire layout."
    )
    parser.add_argument("topic", nargs="?", default="/realtime_safety/yolo_obstacles/pointcloud")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--reliability",
        choices=("best_effort", "reliable"),
        default="best_effort",
        help="Subscriber reliability used for the wire test",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Fail unless at least this many PointCloud2 frames arrive",
    )
    parser.add_argument("--require-nonempty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.min_frames < 1:
        raise ValueError("--min-frames must be at least 1")

    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField

    rclpy.init()
    node = rclpy.create_node("realtime_safety_pointcloud_measurement")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=(
            ReliabilityPolicy.RELIABLE
            if args.reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=DurabilityPolicy.VOLATILE,
    )
    point_counts: list[int] = []
    receive_times: list[float] = []
    invalid_layout = 0
    frame_ids: set[str] = set()
    data_bytes_min: int | None = None
    data_bytes_max = 0

    def receive(message: PointCloud2) -> None:
        nonlocal invalid_layout, data_bytes_min, data_bytes_max
        now = time.perf_counter()
        receive_times.append(now)
        points = int(message.width) * int(message.height)
        point_counts.append(points)
        frame_ids.add(message.header.frame_id)
        data_bytes = len(message.data)
        data_bytes_min = data_bytes if data_bytes_min is None else min(data_bytes_min, data_bytes)
        data_bytes_max = max(data_bytes_max, data_bytes)
        fields = {field.name: field for field in message.fields}
        xyz_layout_ok = all(
            name in fields
            and fields[name].datatype == PointField.FLOAT32
            and fields[name].offset == offset
            for name, offset in (("x", 0), ("y", 4), ("z", 8))
        )
        expected_row_step = int(message.width) * int(message.point_step)
        expected_bytes = int(message.height) * int(message.row_step)
        if not (
            int(message.height) == 1
            and int(message.point_step) >= 12
            and int(message.row_step) == expected_row_step
            and data_bytes == expected_bytes
            and xyz_layout_ok
        ):
            invalid_layout += 1

    subscription = node.create_subscription(PointCloud2, args.topic, receive, qos)
    del subscription
    started = time.perf_counter()
    deadline = started + args.duration
    try:
        while time.perf_counter() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.2, deadline - time.perf_counter()))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    elapsed = time.perf_counter() - started
    receive_span = receive_times[-1] - receive_times[0] if len(receive_times) > 1 else 0.0
    rate = (len(receive_times) - 1) / receive_span if receive_span > 0 else 0.0
    result = {
        "topic": args.topic,
        "subscriber_reliability": args.reliability,
        "measurement_sec": round(elapsed, 3),
        "frames": len(point_counts),
        "rate_hz": round(rate, 3),
        "min_points": min(point_counts, default=0),
        "max_points": max(point_counts, default=0),
        "nonempty_frames": sum(points > 0 for points in point_counts),
        "empty_frames": sum(points == 0 for points in point_counts),
        "min_data_bytes": data_bytes_min or 0,
        "max_data_bytes": data_bytes_max,
        "invalid_layout_frames": invalid_layout,
        "frame_ids": sorted(frame_ids),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if len(point_counts) < args.min_frames or invalid_layout:
        return 1
    if args.require_nonempty and not any(points > 0 for points in point_counts):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
