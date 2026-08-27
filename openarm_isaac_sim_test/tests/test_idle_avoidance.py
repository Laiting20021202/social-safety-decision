from __future__ import annotations

import numpy as np

from openarm_sorting_task.idle_avoidance import (
    active_escape_can_continue,
    crisis_escape_point,
    obstacle_has_withdrawn,
    obstacle_is_clear_for_recovery,
    point_to_box_distance,
    safety_evade_sequence,
    select_evading_sides,
    select_evading_sides_from_links,
    union_axis_aligned_boxes,
)


def test_active_vertical_escape_is_not_restarted_by_soft_safety_states() -> None:
    assert active_escape_can_continue("PAUSE", "idle_evade", True)
    assert active_escape_can_continue("REPLAN", "idle_evade", True)
    assert active_escape_can_continue("PAUSE", "safety_evade", True)
    assert not active_escape_can_continue("EMERGENCY_STOP", "idle_evade", True)
    assert not active_escape_can_continue("PAUSE", "idle_restore", True)
    assert not active_escape_can_continue("PAUSE", "idle_evade", False)


def test_crisis_escape_moves_up_and_away_from_hand() -> None:
    tcp = np.array([0.10, 0.05, 0.55])
    center = np.array([0.10, -0.05, 0.58])
    point = crisis_escape_point(
        tcp,
        center,
        np.array([0.20, 0.12, 0.16]),
        side="left",
        minimum_lift_m=0.16,
        horizontal_escape_m=0.10,
        vertical_clearance_m=0.14,
        maximum_tcp_z_m=1.12,
    )
    assert point[1] > tcp[1]
    assert np.isclose(point[2], 0.80) or point[2] > 0.80


def test_zero_horizontal_escape_is_a_pure_vertical_lift() -> None:
    tcp = np.array([-0.08, 0.19, 0.41])
    point = crisis_escape_point(
        tcp,
        np.array([-0.08, 0.18, 0.30]),
        np.array([0.12, 0.09, 0.04]),
        side="left",
        minimum_lift_m=0.14,
        horizontal_escape_m=0.0,
        vertical_clearance_m=0.10,
        maximum_tcp_z_m=1.12,
    )
    np.testing.assert_allclose(point[:2], tcp[:2])
    assert np.isclose(point[2], tcp[2] + 0.14)


def test_centered_crisis_escapes_separate_left_and_right_arms() -> None:
    tcp = np.array([0.0, 0.0, 0.60])
    center = np.array([0.0, 0.0, 0.60])
    size = np.array([0.10, 0.10, 0.10])
    left = crisis_escape_point(
        tcp, center, size, side="left", minimum_lift_m=0.10,
        horizontal_escape_m=0.08, vertical_clearance_m=0.08,
        maximum_tcp_z_m=1.12,
    )
    right = crisis_escape_point(
        tcp, center, size, side="right", minimum_lift_m=0.10,
        horizontal_escape_m=0.08, vertical_clearance_m=0.08,
        maximum_tcp_z_m=1.12,
    )
    assert left[1] > 0.0
    assert right[1] < 0.0


def test_point_to_box_distance_is_zero_inside_and_metric_outside() -> None:
    center = np.array([0.0, 0.0, 0.0])
    size = np.array([0.2, 0.4, 0.6])
    assert point_to_box_distance(np.array([0.05, 0.1, 0.2]), center, size) == 0.0
    assert np.isclose(
        point_to_box_distance(np.array([0.3, 0.0, 0.0]), center, size), 0.2
    )


def test_selects_only_arm_inside_tcp_trigger_distance() -> None:
    sides = select_evading_sides(
        {
            "left": np.array([0.1, 0.15, 0.2]),
            "right": np.array([0.8, -0.5, 0.8]),
        },
        np.array([0.1, 0.0, 0.2]),
        np.array([0.1, 0.1, 0.1]),
        trigger_distance_m=0.12,
    )
    assert sides == ("left",)


def test_complete_link_warning_falls_back_to_closest_tcp() -> None:
    sides = select_evading_sides(
        {
            "left": np.array([-0.6, 0.2, 1.0]),
            "right": np.array([-0.4, -0.1, 0.9]),
        },
        np.array([0.0, 0.0, 0.3]),
        np.array([0.2, 0.2, 0.1]),
        trigger_distance_m=0.05,
    )
    assert sides == ("right",)


def test_link_samples_select_arm_near_elbow_even_when_other_tcp_is_closer() -> None:
    center = np.array([0.0, 0.18, 0.55])
    size = np.array([0.10, 0.10, 0.10])
    sides = select_evading_sides_from_links(
        {
            # Left elbow is beside the obstacle; its TCP is intentionally far.
            "left": np.array([[0.0, 0.18, 0.55], [-0.6, 0.5, 0.9]]),
            # Right TCP is closer than the left TCP but its links are clear.
            "right": np.array([[0.0, -0.02, 0.55], [0.0, -0.20, 0.7]]),
        },
        center,
        size,
        trigger_distance_m=0.08,
    )
    assert sides == ("left",)


def test_robot_retreat_does_not_count_as_obstacle_withdrawal() -> None:
    trigger = np.array([0.1, -0.2, 0.3])
    assert not obstacle_has_withdrawn(
        trigger, trigger + np.array([0.02, 0.01, 0.0]), minimum_motion_m=0.18
    )
    assert obstacle_has_withdrawn(
        trigger, trigger + np.array([0.0, 0.25, 0.0]), minimum_motion_m=0.18
    )
    assert obstacle_has_withdrawn(trigger, None, minimum_motion_m=0.18)


def test_estop_recovery_requires_hand_motion_and_link_clearance() -> None:
    trigger = np.array([0.1, -0.2, 0.3])
    assert not obstacle_is_clear_for_recovery(
        trigger,
        trigger + np.array([0.02, 0.0, 0.0]),
        minimum_motion_m=0.08,
        current_distance_m=0.40,
        release_distance_m=0.26,
    )
    assert not obstacle_is_clear_for_recovery(
        trigger,
        trigger + np.array([0.0, 0.20, 0.0]),
        minimum_motion_m=0.08,
        current_distance_m=0.12,
        release_distance_m=0.26,
    )
    assert obstacle_is_clear_for_recovery(
        trigger,
        trigger + np.array([0.0, 0.20, 0.0]),
        minimum_motion_m=0.08,
        current_distance_m=0.30,
        release_distance_m=0.26,
    )
    assert obstacle_is_clear_for_recovery(
        trigger,
        None,
        minimum_motion_m=0.08,
        current_distance_m=0.0,
        release_distance_m=0.26,
    )


def test_compact_hand_boxes_are_consumed_as_one_stable_obstacle() -> None:
    union = union_axis_aligned_boxes(
        [np.array([0.0, -0.1, 0.3]), np.array([0.0, 0.1, 0.3])],
        [np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])],
    )
    assert union is not None
    center, size = union
    np.testing.assert_allclose(center, [0.0, 0.0, 0.3])
    np.testing.assert_allclose(size, [0.1, 0.3, 0.1])


def test_safety_evade_discards_stale_route_and_resumes_target_once() -> None:
    route_kinds = frozenset(
        {"under_approach", "under_entry", "under_exit", "under_recover"}
    )
    sequence = safety_evade_sequence(
        ("left", "under_entry"),
        [
            ("left", "under_exit"),
            ("left", "under_recover"),
            ("left", "model_target"),
            ("left", "home"),
            ("right", "model_target"),
        ],
        ("left", "right"),
        target_kind="model_target",
        route_step_kinds=route_kinds,
    )
    assert sequence == [
        ("left", "safety_evade"),
        ("right", "safety_evade"),
        ("left", "model_target"),
        ("left", "home"),
        ("right", "model_target"),
    ]


def test_safety_evade_preserves_interrupted_home() -> None:
    assert safety_evade_sequence(
        ("right", "home"),
        [("left", "model_target")],
        ("right",),
        target_kind="model_target",
        route_step_kinds=frozenset(),
    ) == [
        ("right", "safety_evade"),
        ("right", "home"),
        ("left", "model_target"),
    ]
