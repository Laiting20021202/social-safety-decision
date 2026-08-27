from __future__ import annotations

import math
import time
from typing import Any


_MAX_ROS_TIME_SECONDS = 2**31 - 1
_MAX_CLOCK_DOMAIN_SKEW_SECONDS = 24.0 * 60.0 * 60.0


def source_timestamp_or_now(node: Any, value: object) -> Any:
    """Convert a positive finite source timestamp into a ROS Time message.

    ROS inputs already in the node-clock domain are preserved exactly.  Live
    OpenCV/network inputs use ``time.perf_counter()``; those are translated to
    the node-clock domain while retaining their acquisition-time offset.  A
    missing, zero, non-finite, negative, out-of-range, or otherwise unrelated
    clock-domain value falls back to the node clock.

    ``builtin_interfaces/Time.sec`` is a signed 32-bit field.  The node-clock
    message is reused as the concrete type, keeping this helper importable
    without eagerly importing ROS message packages.
    """

    try:
        source_seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        source_seconds = float("nan")

    fallback = node.get_clock().now().to_msg()
    if not math.isfinite(source_seconds) or source_seconds <= 0.0:
        return fallback

    try:
        node_now = float(fallback.sec) + float(fallback.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError, OverflowError):
        return fallback
    monotonic_now = time.perf_counter()
    if abs(source_seconds - node_now) <= _MAX_CLOCK_DOMAIN_SKEW_SECONDS:
        ros_seconds = source_seconds
    elif abs(source_seconds - monotonic_now) <= _MAX_CLOCK_DOMAIN_SKEW_SECONDS:
        ros_seconds = source_seconds + (node_now - monotonic_now)
    else:
        return fallback

    seconds = math.floor(ros_seconds)
    nanoseconds = int(round((ros_seconds - seconds) * 1_000_000_000.0))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    if seconds < 0 or seconds > _MAX_ROS_TIME_SECONDS:
        return fallback

    fallback.sec = int(seconds)
    fallback.nanosec = int(nanoseconds)
    return fallback


def exact_source_timestamp_or_now(node: Any, value: object) -> Any:
    """Preserve an explicitly trusted ROS-source timestamp exactly.

    This is reserved for bridges that obtain ``value`` from a synchronized
    ROS message header.  Unlike :func:`source_timestamp_or_now`, it does not
    compare against the publisher node's clock domain; the bridge may run on
    wall time while forwarding Gazebo simulation-time sensor messages.
    """

    fallback = node.get_clock().now().to_msg()
    try:
        source_seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if (
        not math.isfinite(source_seconds)
        or source_seconds <= 0.0
        or source_seconds > _MAX_ROS_TIME_SECONDS
    ):
        return fallback
    seconds = math.floor(source_seconds)
    nanoseconds = int(round((source_seconds - seconds) * 1_000_000_000.0))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    if seconds > _MAX_ROS_TIME_SECONDS:
        return fallback
    fallback.sec = int(seconds)
    fallback.nanosec = int(nanoseconds)
    return fallback


__all__ = ["exact_source_timestamp_or_now", "source_timestamp_or_now"]
