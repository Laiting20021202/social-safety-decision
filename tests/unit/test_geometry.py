from __future__ import annotations

from packages.common_models import Point2D
from packages.overlay_renderer.geometry import (
    distance_point_to_polygon,
    estimate_time_to_zone_constant_velocity,
    point_in_polygon,
)


def test_point_in_polygon() -> None:
    polygon = [
        Point2D(x=0, y=0),
        Point2D(x=10, y=0),
        Point2D(x=10, y=10),
        Point2D(x=0, y=10),
    ]

    assert point_in_polygon(Point2D(x=5, y=5), polygon)
    assert not point_in_polygon(Point2D(x=15, y=5), polygon)


def test_distance_to_zone() -> None:
    polygon = [
        Point2D(x=10, y=10),
        Point2D(x=20, y=10),
        Point2D(x=20, y=20),
        Point2D(x=10, y=20),
    ]

    assert distance_point_to_polygon(Point2D(x=15, y=15), polygon) == 0.0
    assert distance_point_to_polygon(Point2D(x=5, y=15), polygon) == 5.0


def test_time_to_zone_constant_velocity() -> None:
    polygon = [
        Point2D(x=10, y=0),
        Point2D(x=20, y=0),
        Point2D(x=20, y=10),
        Point2D(x=10, y=10),
    ]

    ttz = estimate_time_to_zone_constant_velocity(
        current=Point2D(x=0, y=5),
        velocity_px_per_sec=Point2D(x=5, y=0),
        polygon=polygon,
        horizon_sec=4,
        step_sec=0.1,
    )

    assert ttz is not None
    assert 1.9 <= ttz <= 2.1


def test_time_to_zone_returns_none_when_parallel() -> None:
    polygon = [
        Point2D(x=10, y=0),
        Point2D(x=20, y=0),
        Point2D(x=20, y=10),
        Point2D(x=10, y=10),
    ]

    assert (
        estimate_time_to_zone_constant_velocity(
            current=Point2D(x=0, y=20),
            velocity_px_per_sec=Point2D(x=5, y=0),
            polygon=polygon,
            horizon_sec=4,
        )
        is None
    )
