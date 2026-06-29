from __future__ import annotations

from packages.common_models import (
    Point2D,
    RoadSegmentationResult,
    TrackObservation,
    VQADirectionEstimate,
)
from packages.overlay_renderer import (
    approximate_bev_point,
    default_robot_corridor,
    direction_angle_deg,
    direction_label_from_vector,
    dynamic_risk_zone,
    estimate_motion_from_history,
    fuse_direction,
    ground_contact_point,
    parse_vqa_direction_json,
    polygons_intersect,
    predict_constant_velocity_points,
    should_reject_stale_analysis,
    swept_corridor_polygon,
)


def test_road_mask_schema() -> None:
    result = RoadSegmentationResult(
        scenario_id="demo",
        frame_index=1,
        timestamp_sec=0.5,
        source="manual_fallback",
        polygon=[
            Point2D(x=0, y=100),
            Point2D(x=640, y=100),
            Point2D(x=640, y=360),
            Point2D(x=0, y=360),
        ],
        confidence=0.9,
        is_valid=True,
    )
    assert result.source == "manual_fallback"
    assert result.is_valid


def test_ground_contact_point_prefers_bottom_mask_points() -> None:
    point = ground_contact_point(
        bounding_box=(10, 10, 30, 80),
        mask_points=[
            Point2D(x=14, y=78),
            Point2D(x=20, y=80),
            Point2D(x=24, y=80),
        ],
        class_name="person",
    )
    assert point == Point2D(x=22, y=80)


def test_timestamp_velocity_estimation_and_stationary_detection() -> None:
    moving = [
        _observation(1, 0.0, 0.2, 0.8),
        _observation(1, 0.5, 0.25, 0.75),
        _observation(1, 1.0, 0.3, 0.7),
    ]
    motion = estimate_motion_from_history(moving)
    assert motion is not None
    assert motion.speed > 0
    assert motion.speed_unit == "normalized/s"
    assert motion.direction_label_geometry == "forward_right"

    stationary = [
        _observation(1, 0.0, 0.2, 0.8),
        _observation(1, 0.5, 0.201, 0.801),
        _observation(1, 1.0, 0.202, 0.802),
    ]
    motion = estimate_motion_from_history(stationary, stationary_threshold=0.01)
    assert motion is not None
    assert motion.direction_label_geometry == "stationary"


def test_direction_angle_and_label_conversion() -> None:
    assert direction_angle_deg(1.0, 0.0) == 0.0
    assert direction_label_from_vector(-0.2, 0.0) == "left"
    assert direction_label_from_vector(0.2, -0.2) == "forward_right"


def test_vqa_json_parser_and_direction_fusion() -> None:
    parsed = parse_vqa_direction_json(
        """
        {
          "track_id": 1,
          "motion_state": "moving",
          "direction_label": "forward_right",
          "path_relation": "crossing_path",
          "confidence": 0.8,
          "reason": "The highlighted agent moves diagonally."
        }
        """,
        track_id=1,
    )
    assert parsed.parse_valid
    label, confidence, conflict = fuse_direction("forward_right", parsed, 0.6)
    assert label == "forward_right"
    assert confidence > 0.8
    assert not conflict

    invalid = parse_vqa_direction_json("not json", track_id=2)
    assert not invalid.parse_valid

    conflict_vqa = VQADirectionEstimate(
        track_id=1,
        motion_state="moving",
        direction_label="left",
        path_relation="crossing_path",
        confidence=0.7,
        parse_valid=True,
    )
    label, _confidence, conflict = fuse_direction("forward_right", conflict_vqa, 0.6)
    assert label == "uncertain"
    assert conflict


def test_prediction_swept_corridor_and_intersection() -> None:
    origin = Point2D(x=0.45, y=0.9)
    points = predict_constant_velocity_points(origin, (0.02, -0.2), horizon_sec=3.0)
    corridor = swept_corridor_polygon(points, half_width=0.08)
    assert len(points) == 7
    assert len(corridor) >= 4
    assert polygons_intersect(corridor, default_robot_corridor())


def test_dynamic_risk_zone_gets_longer_with_speed_and_intersects_robot_corridor() -> None:
    slow_track = _observation(1, 1.0, 0.5, 0.82)
    fast_track = _observation(1, 1.0, 0.5, 0.5)
    slow = estimate_motion_from_history(
        [_observation(1, 0.0, 0.5, 0.9), _observation(1, 0.5, 0.5, 0.86), slow_track]
    )
    fast = estimate_motion_from_history(
        [_observation(1, 0.0, 0.5, 0.9), _observation(1, 0.5, 0.5, 0.7), fast_track]
    )
    assert slow is not None
    assert fast is not None
    slow_zone = dynamic_risk_zone(slow_track, slow, default_robot_corridor())
    fast_zone = dynamic_risk_zone(fast_track, fast, default_robot_corridor())
    assert len(fast_zone.predicted_points) == len(slow_zone.predicted_points)
    assert fast_zone.predicted_points[-1].y < slow_zone.predicted_points[-1].y
    assert fast_zone.intersects_robot_corridor


def test_approximate_bev_transform_and_stale_analysis_rejection() -> None:
    bev = approximate_bev_point(Point2D(x=320, y=360), 640, 360)
    assert bev == Point2D(x=0.5, y=1.0)
    assert should_reject_stale_analysis(10.0, 9.0, max_age_sec=0.75)
    assert not should_reject_stale_analysis(10.0, 9.5, max_age_sec=0.75)


def _observation(track_id: int, timestamp: float, x: float, y: float) -> TrackObservation:
    point = Point2D(x=x, y=y)
    return TrackObservation(
        track_id=track_id,
        class_name="person",
        timestamp_sec=timestamp,
        frame_index=int(timestamp * 10),
        centroid=point,
        ground_contact_point=point,
        bottom_center=point,
        confidence=0.9,
    )
