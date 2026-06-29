from __future__ import annotations

import math
from typing import cast

from packages.common_models import (
    DirectionLabel,
    DynamicRiskZone,
    MotionEstimate,
    Point2D,
    RiskLevel,
    TrackObservation,
    VQADirectionEstimate,
)


def point_in_polygon(point: Point2D, polygon: list[Point2D]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i, vertex in enumerate(polygon):
        previous = polygon[j]
        intersects = ((vertex.y > point.y) != (previous.y > point.y)) and (
            point.x
            < (previous.x - vertex.x) * (point.y - vertex.y) / (previous.y - vertex.y + 1e-12)
            + vertex.x
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_point_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0 and dy == 0:
        return math.hypot(point.x - start.x, point.y - start.y)
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = Point2D(x=start.x + t * dx, y=start.y + t * dy)
    return math.hypot(point.x - projection.x, point.y - projection.y)


def distance_point_to_polygon(point: Point2D, polygon: list[Point2D]) -> float | None:
    if len(polygon) < 3:
        return None
    if point_in_polygon(point, polygon):
        return 0.0
    distances = [
        distance_point_to_segment(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    return min(distances)


def estimate_time_to_zone_constant_velocity(
    current: Point2D,
    velocity_px_per_sec: Point2D,
    polygon: list[Point2D],
    horizon_sec: float,
    step_sec: float = 0.05,
) -> float | None:
    if len(polygon) < 3:
        return None
    if point_in_polygon(current, polygon):
        return 0.0
    if horizon_sec <= 0 or step_sec <= 0:
        return None
    speed = math.hypot(velocity_px_per_sec.x, velocity_px_per_sec.y)
    if speed < 1e-6:
        return None
    steps = max(1, int(math.ceil(horizon_sec / step_sec)))
    for step in range(1, steps + 1):
        t = min(step * step_sec, horizon_sec)
        predicted = Point2D(
            x=current.x + velocity_px_per_sec.x * t,
            y=current.y + velocity_px_per_sec.y * t,
        )
        if point_in_polygon(predicted, polygon):
            return t
    return None


def ground_contact_point(
    bounding_box: tuple[float, float, float, float] | None = None,
    mask_points: list[Point2D] | None = None,
    class_name: str = "person",
) -> Point2D | None:
    """Estimate the image point where an agent touches the ground."""
    if mask_points:
        max_y = max(point.y for point in mask_points)
        bottom_points = [point for point in mask_points if abs(point.y - max_y) <= 0.5]
        if bottom_points:
            return Point2D(
                x=sum(point.x for point in bottom_points) / len(bottom_points),
                y=max_y,
            )
    if bounding_box is None:
        return None
    x1, y1, x2, y2 = bounding_box
    if class_name in {"car", "bus", "truck", "bicycle", "motorcycle"}:
        return Point2D(x=(x1 + x2) / 2.0, y=y2)
    return Point2D(x=(x1 + x2) / 2.0, y=y2)


def estimate_motion_from_history(
    observations: list[TrackObservation],
    min_observations: int = 3,
    stationary_threshold: float = 0.015,
    max_jump: float = 0.35,
) -> MotionEstimate | None:
    valid = [
        observation
        for observation in observations
        if observation.ground_contact_point is not None or observation.bottom_center is not None
    ]
    if len(valid) < min_observations:
        return None
    valid = sorted(valid, key=lambda item: item.timestamp_sec)
    start = valid[0]
    end = valid[-1]
    start_point = start.ground_contact_point or start.bottom_center
    end_point = end.ground_contact_point or end.bottom_center
    if start_point is None or end_point is None:
        return None
    dt = end.timestamp_sec - start.timestamp_sec
    if dt <= 1e-6:
        return None
    displacements = []
    previous = valid[0]
    previous_point = previous.ground_contact_point or previous.bottom_center
    for observation in valid[1:]:
        current_point = observation.ground_contact_point or observation.bottom_center
        if previous_point is None or current_point is None:
            previous = observation
            previous_point = current_point
            continue
        step_dt = observation.timestamp_sec - previous.timestamp_sec
        if step_dt <= 1e-6:
            previous = observation
            previous_point = current_point
            continue
        dx = current_point.x - previous_point.x
        dy = current_point.y - previous_point.y
        if math.hypot(dx, dy) <= max_jump:
            displacements.append((dx / step_dt, dy / step_dt))
        previous = observation
        previous_point = current_point
    if not displacements:
        return None
    vx = sum(item[0] for item in displacements) / len(displacements)
    vy = sum(item[1] for item in displacements) / len(displacements)
    speed = math.hypot(vx, vy)
    if speed < stationary_threshold:
        vx = 0.0
        vy = 0.0
        speed = 0.0
    angle = direction_angle_deg(vx, vy)
    label = direction_label_from_vector(vx, vy, stationary_threshold=stationary_threshold)
    return MotionEstimate(
        track_id=end.track_id,
        timestamp_sec=end.timestamp_sec,
        velocity_vector=(vx, vy),
        speed=speed,
        speed_unit="normalized/s",
        direction_angle_deg=angle,
        direction_label_geometry=label,
        direction_label_fused=label,
        is_approximate=True,
        velocity_px_per_sec=Point2D(x=vx, y=vy),
        speed_px_per_sec=speed,
        movement_direction="stationary" if speed == 0.0 else "unknown",
        confidence=min(1.0, len(displacements) / 5.0),
    )


def direction_angle_deg(vx: float, vy: float) -> float:
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(vy, vx))


def direction_label_from_vector(
    vx: float,
    vy: float,
    stationary_threshold: float = 0.015,
) -> DirectionLabel:
    speed = math.hypot(vx, vy)
    if speed < stationary_threshold:
        return "stationary"
    horizontal: DirectionLabel = "left" if vx < 0 else "right"
    vertical: DirectionLabel = "away_from_camera" if vy < 0 else "toward_camera"
    if abs(vx) >= abs(vy) * 1.8:
        return horizontal
    if abs(vy) >= abs(vx) * 1.8:
        return vertical
    if vy < 0 and vx < 0:
        return "forward_left"
    if vy < 0 and vx >= 0:
        return "forward_right"
    if vy >= 0 and vx < 0:
        return "backward_left"
    return "backward_right"


def fuse_direction(
    geometry_label: str,
    vqa: VQADirectionEstimate | None,
    geometry_confidence: float,
) -> tuple[DirectionLabel, float, bool]:
    if vqa is None or not vqa.parse_valid or vqa.direction_label == "uncertain":
        return _direction_label(geometry_label), geometry_confidence, False
    if geometry_label in {"stationary", "uncertain"}:
        return vqa.direction_label, vqa.confidence, False
    if geometry_label == vqa.direction_label:
        return (
            _direction_label(geometry_label),
            min(1.0, max(geometry_confidence, vqa.confidence) + 0.15),
            False,
        )
    return "uncertain", min(geometry_confidence, vqa.confidence), True


def predict_constant_velocity_points(
    origin: Point2D,
    velocity: tuple[float, float],
    horizon_sec: float = 3.0,
    steps: int = 6,
) -> list[Point2D]:
    if steps <= 0 or horizon_sec <= 0:
        return [origin]
    vx, vy = velocity
    return [
        Point2D(
            x=origin.x + vx * horizon_sec * index / steps,
            y=origin.y + vy * horizon_sec * index / steps,
        )
        for index in range(steps + 1)
    ]


def swept_corridor_polygon(
    points: list[Point2D],
    half_width: float,
) -> list[Point2D]:
    if not points:
        return []
    if len(points) == 1:
        return circle_polygon(points[0], half_width)
    start = points[0]
    end = points[-1]
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return circle_polygon(start, half_width)
    nx = -dy / length
    ny = dx / length
    left = [Point2D(x=point.x + nx * half_width, y=point.y + ny * half_width) for point in points]
    right = [
        Point2D(x=point.x - nx * half_width, y=point.y - ny * half_width)
        for point in reversed(points)
    ]
    return left + right


def circle_polygon(center: Point2D, radius: float, segments: int = 16) -> list[Point2D]:
    return [
        Point2D(
            x=center.x + math.cos((2.0 * math.pi * index) / segments) * radius,
            y=center.y + math.sin((2.0 * math.pi * index) / segments) * radius,
        )
        for index in range(segments)
    ]


def dynamic_risk_zone(
    track: TrackObservation,
    motion: MotionEstimate | None,
    robot_corridor: list[Point2D],
    prediction_horizon_sec: float = 3.0,
) -> DynamicRiskZone:
    origin = track.ground_contact_point or track.bottom_center or track.centroid
    velocity = motion.velocity_vector if motion and motion.velocity_vector else (0.0, 0.0)
    speed = motion.speed if motion else 0.0
    direction = motion.direction_label_fused if motion else "uncertain"
    uncertainty = 1.0 - (motion.confidence if motion else 0.0)
    base_half_width = 0.035 if track.class_name == "person" else 0.07
    if speed <= 0.015:
        predicted = [origin]
        polygon = circle_polygon(origin, base_half_width + uncertainty * 0.035)
    else:
        predicted = predict_constant_velocity_points(origin, velocity, prediction_horizon_sec)
        polygon = swept_corridor_polygon(
            predicted,
            half_width=base_half_width + min(0.12, uncertainty * 0.08) + min(0.08, speed * 0.04),
        )
    intersects = polygons_intersect(polygon, robot_corridor)
    risk_level: RiskLevel = "low"
    time_to_intersection = None
    if intersects:
        risk_level = "critical" if speed > 0.12 else "warning"
        time_to_intersection = estimate_time_to_zone_constant_velocity(
            current=origin,
            velocity_px_per_sec=Point2D(x=velocity[0], y=velocity[1]),
            polygon=robot_corridor,
            horizon_sec=prediction_horizon_sec,
        )
    elif direction == "uncertain":
        risk_level = "warning"
    return DynamicRiskZone(
        track_id=track.track_id,
        class_name=track.class_name,
        timestamp_sec=track.timestamp_sec,
        prediction_horizon_sec=prediction_horizon_sec,
        predicted_points=predicted,
        risk_polygon=polygon,
        speed=speed,
        direction=direction,
        uncertainty=uncertainty,
        intersects_robot_corridor=intersects,
        risk_level=risk_level,
        time_to_intersection_sec=time_to_intersection,
    )


def _direction_label(value: str) -> DirectionLabel:
    valid = {
        "toward_camera",
        "away_from_camera",
        "left",
        "right",
        "forward_left",
        "forward_right",
        "backward_left",
        "backward_right",
        "stationary",
        "uncertain",
    }
    return cast(DirectionLabel, value if value in valid else "uncertain")


def approximate_bev_point(point: Point2D, image_width: float, image_height: float) -> Point2D:
    if image_width <= 0 or image_height <= 0:
        return Point2D(x=0.5, y=0.5)
    nx = max(0.0, min(1.0, point.x / image_width))
    ny = max(0.0, min(1.0, point.y / image_height))
    ground_y = max(0.0, min(1.0, (ny - 0.45) / 0.55))
    return Point2D(x=nx, y=ground_y)


def polygon_from_image_to_bev(
    polygon: list[Point2D],
    image_width: float,
    image_height: float,
) -> list[Point2D]:
    return [approximate_bev_point(point, image_width, image_height) for point in polygon]


def default_robot_corridor() -> list[Point2D]:
    return [
        Point2D(x=0.42, y=1.0),
        Point2D(x=0.58, y=1.0),
        Point2D(x=0.64, y=0.18),
        Point2D(x=0.36, y=0.18),
    ]


def polygons_intersect(first: list[Point2D], second: list[Point2D]) -> bool:
    if len(first) < 3 or len(second) < 3:
        return False
    if any(point_in_polygon(point, second) for point in first):
        return True
    if any(point_in_polygon(point, first) for point in second):
        return True
    for index, start in enumerate(first):
        end = first[(index + 1) % len(first)]
        for other_index, other_start in enumerate(second):
            other_end = second[(other_index + 1) % len(second)]
            if segments_intersect(start, end, other_start, other_end):
                return True
    return False


def segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    def orientation(p: Point2D, q: Point2D, r: Point2D) -> float:
        return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    def on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
        return (
            min(p.x, r.x) - 1e-9 <= q.x <= max(p.x, r.x) + 1e-9
            and min(p.y, r.y) - 1e-9 <= q.y <= max(p.y, r.y) + 1e-9
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-9 and on_segment(a, c, b):
        return True
    if abs(o2) < 1e-9 and on_segment(a, d, b):
        return True
    if abs(o3) < 1e-9 and on_segment(c, a, d):
        return True
    if abs(o4) < 1e-9 and on_segment(c, b, d):
        return True
    return False


def should_reject_stale_analysis(
    video_timestamp_sec: float,
    analysis_timestamp_sec: float,
    max_age_sec: float = 0.75,
) -> bool:
    return video_timestamp_sec - analysis_timestamp_sec > max_age_sec
