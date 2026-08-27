from __future__ import annotations

import numpy as np

from openarm_sorting_task.route_planner import (
    avoidance_route_waypoints,
    guarded_route_needs_restart,
    preserve_home_after_side_failure,
    progressive_pose_tolerance,
    route_progress,
    segment_intersects_box,
    target_contact_point,
    target_hold_sequence,
    touch_then_home_sequence,
    under_route_waypoints,
    waiting_path_should_retry,
)


def test_target_hold_sequence_never_adds_implicit_home() -> None:
    assert target_hold_sequence(("left", "right")) == [
        ("left", "model_target"),
        ("right", "model_target"),
    ]
    assert target_hold_sequence(("left",), "explicit_target") == [
        ("left", "explicit_target")
    ]


def test_waiting_path_retries_when_measured_clearance_is_safe() -> None:
    assert waiting_path_should_retry(
        "SAFE", 0.31, 0.26, coarse_path_blocked=True
    )
    assert waiting_path_should_retry(
        "WARNING", 0.20, 0.26, coarse_path_blocked=False
    )


def test_waiting_path_keeps_waiting_while_close() -> None:
    assert not waiting_path_should_retry(
        "WARNING", 0.20, 0.26, coarse_path_blocked=True
    )
    assert not waiting_path_should_retry(
        "SAFE", 0.25, 0.26, coarse_path_blocked=True
    )


def test_target_commands_touch_then_return_home_without_a_dwell_step() -> None:
    assert touch_then_home_sequence(("left", "right")) == [
        ("left", "model_target"),
        ("left", "home"),
        ("right", "model_target"),
        ("right", "home"),
    ]
    assert touch_then_home_sequence(("left",), "explicit_target") == [
        ("left", "explicit_target"),
        ("left", "home"),
    ]


def test_failed_target_keeps_home_but_drops_stale_same_side_route_steps() -> None:
    sequence = [
        ("left", "under_exit"),
        ("left", "model_target"),
        ("left", "home"),
        ("right", "model_target"),
        ("right", "home"),
    ]
    assert preserve_home_after_side_failure(sequence, "left") == [
        ("left", "home"),
        ("right", "model_target"),
        ("right", "home"),
    ]


def test_pause_cancelled_guarded_leg_restarts_on_replan_only_when_idle() -> None:
    assert guarded_route_needs_restart("REPLAN", True, False)
    assert not guarded_route_needs_restart("REPLAN", True, True)
    assert not guarded_route_needs_restart("PAUSE", True, False)
    assert not guarded_route_needs_restart("REPLAN", False, False)


def test_segment_box_intersection_rejects_visible_clear_path() -> None:
    center = np.array([0.0, 0.0, 0.35])
    size = np.array([0.12, 0.12, 0.12])
    assert segment_intersects_box(
        np.array([-0.4, 0.0, 0.35]), np.array([0.4, 0.0, 0.35]), center, size
    )
    assert not segment_intersects_box(
        np.array([-0.4, 0.3, 0.35]), np.array([0.4, 0.3, 0.35]), center, size
    )


def test_under_route_places_two_points_below_obstacle_in_travel_order() -> None:
    route = under_route_waypoints(
        np.array([-0.4, 0.0, 0.42]),
        np.array([0.4, 0.0, 0.34]),
        np.array([0.0, 0.0, 0.34]),
        np.array([0.14, 0.12, 0.14]),
        collision_margin_m=0.02,
        horizontal_clearance_m=0.08,
        vertical_clearance_m=0.04,
        minimum_tcp_z_m=0.16,
    )
    assert route is not None
    assert route.entry[0] < 0.0 < route.exit[0]
    assert route.entry[2] == route.exit[2]
    assert route.entry[2] < 0.34 - 0.14 / 2.0


def test_under_route_refuses_gap_too_close_to_table() -> None:
    route = under_route_waypoints(
        np.array([-0.4, 0.0, 0.30]),
        np.array([0.4, 0.0, 0.30]),
        np.array([0.0, 0.0, 0.22]),
        np.array([0.14, 0.12, 0.12]),
        collision_margin_m=0.02,
        horizontal_clearance_m=0.08,
        vertical_clearance_m=0.04,
        minimum_tcp_z_m=0.16,
    )
    assert route is None


def test_avoidance_route_uses_upper_guide_when_table_blocks_lower_gap() -> None:
    center = np.array([-0.114, 0.064, 0.261])
    size = np.array([0.252, 0.222, 0.137])
    route = avoidance_route_waypoints(
        np.array([-0.553, 0.139, 1.075]),
        np.array([0.085, 0.179, 0.412]),
        center,
        size,
        collision_margin_m=0.025,
        horizontal_clearance_m=0.03,
        vertical_clearance_m=0.12,
        minimum_tcp_z_m=0.16,
        maximum_tcp_z_m=1.12,
        entry_horizontal_clearance_m=0.12,
        exit_horizontal_clearance_m=0.055,
    )
    assert route is not None
    assert route.strategy == "over"
    assert route.entry[2] > center[2] + size[2] / 2.0
    assert route.exit[2] == route.entry[2]


def test_moving_obstacle_replan_does_not_send_tcp_back_toward_home() -> None:
    start = np.array([-0.05, 0.0, 0.42])
    goal = np.array([0.40, 0.0, 0.32])
    route = avoidance_route_waypoints(
        start,
        goal,
        np.array([0.0, 0.0, 0.34]),
        np.array([0.20, 0.18, 0.16]),
        collision_margin_m=0.025,
        horizontal_clearance_m=0.03,
        vertical_clearance_m=0.12,
        minimum_tcp_z_m=0.16,
        maximum_tcp_z_m=1.12,
        entry_horizontal_clearance_m=0.12,
        exit_horizontal_clearance_m=0.055,
        maximum_backtrack_m=0.02,
    )
    assert route is not None
    assert route.strategy.endswith("_progress")
    assert route_progress(start, goal, route.entry) >= -0.02 - 1e-9
    assert route_progress(start, goal, route.exit) > 0.0


def test_soft_stop_forces_local_detour_for_elbow_only_blocker() -> None:
    start = np.array([-0.50, 0.0, 0.70])
    goal = np.array([0.30, 0.0, 0.35])
    center = np.array([-0.10, 0.35, 0.45])
    route = avoidance_route_waypoints(
        start,
        goal,
        center,
        np.array([0.18, 0.16, 0.14]),
        collision_margin_m=0.025,
        horizontal_clearance_m=0.04,
        vertical_clearance_m=0.12,
        minimum_tcp_z_m=0.16,
        maximum_tcp_z_m=1.12,
        maximum_backtrack_m=0.02,
        force_local_detour=True,
        forced_local_clearance_m=0.06,
    )
    assert route is not None
    assert route.strategy == "side_forced_progress"
    assert route_progress(start, goal, route.entry) >= -1e-9
    assert route_progress(start, goal, route.exit) > 0.0
    # The detour moves away from the obstacle side of the TCP line.
    assert route.entry[1] < start[1] - 0.05


def test_under_route_is_preferred_for_xy_blocker_even_if_tcp_line_is_above() -> None:
    route = under_route_waypoints(
        np.array([-0.59, 0.15, 1.05]),
        np.array([0.02, 0.18, 0.34]),
        np.array([-0.28, 0.18, 0.42]),
        np.array([0.20, 0.21, 0.16]),
        collision_margin_m=0.025,
        horizontal_clearance_m=0.08,
        vertical_clearance_m=0.04,
        minimum_tcp_z_m=0.16,
    )
    assert route is not None
    assert route.entry[2] < 0.42 - 0.16 / 2.0


def test_under_route_supports_safer_entry_and_reachable_exit_clearances() -> None:
    route = under_route_waypoints(
        np.array([-0.59, 0.15, 1.05]),
        np.array([-0.12, 0.18, 0.34]),
        np.array([-0.28, 0.18, 0.42]),
        np.array([0.20, 0.21, 0.16]),
        collision_margin_m=0.025,
        horizontal_clearance_m=0.03,
        vertical_clearance_m=0.025,
        minimum_tcp_z_m=0.16,
        entry_horizontal_clearance_m=0.08,
        exit_horizontal_clearance_m=0.03,
    )
    assert route is not None
    assert abs(route.entry[0] + 0.28) > abs(route.exit[0] + 0.28)


def test_under_route_ends_above_goal_when_goal_is_below_obstacle() -> None:
    goal = np.array([0.005, 0.18, 0.256])
    route = under_route_waypoints(
        np.array([-0.59, 0.16, 1.05]),
        goal,
        np.array([0.001, 0.21, 0.527]),
        np.array([0.265, 0.220, 0.143]),
        collision_margin_m=0.025,
        horizontal_clearance_m=0.03,
        vertical_clearance_m=0.12,
        minimum_tcp_z_m=0.16,
        entry_horizontal_clearance_m=0.12,
        exit_horizontal_clearance_m=0.055,
    )
    assert route is not None
    np.testing.assert_allclose(route.exit[:2], goal[:2])
    assert goal[2] < route.exit[2] < 0.527


def test_target_contact_point_uses_marker_surface_facing_tcp() -> None:
    contact = target_contact_point(
        np.array([0.0, 0.0, 0.25]),
        np.array([-0.5, 0.0, 0.25]),
        0.0275,
    )
    np.testing.assert_allclose(contact, [-0.0275, 0.0, 0.25])


def test_pose_tolerance_starts_precise_and_has_bounded_fallback() -> None:
    assert progressive_pose_tolerance(0.010, 0, 0.020) == 0.010
    assert progressive_pose_tolerance(0.010, 1, 0.020) == 0.015
    assert progressive_pose_tolerance(0.010, 4, 0.020) == 0.020
