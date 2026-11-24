from __future__ import annotations

from types import SimpleNamespace

import pytest

from openarm_dynamic_avoidance.policy import (
    BLOCKING_STATES,
    effective_velocity_scale,
    hold_recent_obstacle,
    obstacle_affects_motion_corridor,
    obstacle_affects_segment,
    retime_trajectory,
    trajectory_blocked,
)


def _duration(seconds: float) -> SimpleNamespace:
    whole = int(seconds)
    return SimpleNamespace(sec=whole, nanosec=int((seconds - whole) * 1e9))


def test_dynamic_layer_retimes_and_rebases_nominal_trajectory() -> None:
    trajectory = SimpleNamespace(
        joint_names=["j1", "j2"],
        points=[
            SimpleNamespace(
                positions=[0.0, 0.0],
                velocities=[1.0, 2.0],
                accelerations=[2.0, 4.0],
                time_from_start=_duration(0.1),
            ),
            SimpleNamespace(
                positions=[1.0, 2.0],
                velocities=[1.0, 2.0],
                accelerations=[2.0, 4.0],
                time_from_start=_duration(1.0),
            ),
        ],
    )
    result = retime_trajectory(trajectory, 0.5, {"j1": 0.2, "j2": -0.3})

    assert result.points[0].positions == [0.2, -0.3]
    assert result.points[0].velocities == [0.0, 0.0]
    assert result.points[1].time_from_start.sec == 2
    assert result.points[1].velocities == [0.5, 1.0]
    assert result.points[1].accelerations == [0.5, 1.0]
    assert trajectory.points[1].time_from_start.sec == 1


def test_dynamic_layer_rejects_stop_scale_and_empty_trajectory() -> None:
    with pytest.raises(ValueError, match="velocity_scale"):
        retime_trajectory(SimpleNamespace(points=[object()]), 0.0)
    with pytest.raises(ValueError, match="no points"):
        retime_trajectory(SimpleNamespace(points=[]), 1.0)


def test_replanned_trajectory_can_run_slowly_while_replan_state_is_active() -> None:
    assert "PAUSE" in BLOCKING_STATES
    assert "EMERGENCY_STOP" in BLOCKING_STATES
    assert "REPLAN" not in BLOCKING_STATES


def test_dynamic_mode_always_caps_speed_and_honors_stricter_safety_scale() -> None:
    assert effective_velocity_scale("SAFE", 1.0, 0.25) == 0.25
    assert effective_velocity_scale("WARNING", 0.2, 0.25) == 0.2
    assert effective_velocity_scale("REPLAN", 0.4, 0.25) == 0.25
    assert effective_velocity_scale("RECOVER", 0.01, 0.25) == 0.05


def test_dynamic_mode_rejects_invalid_speed_configuration() -> None:
    with pytest.raises(ValueError, match="guarded_route_velocity_scale"):
        effective_velocity_scale("SAFE", 1.0, 0.0)
    with pytest.raises(ValueError, match="safety_velocity_scale"):
        effective_velocity_scale("WARNING", 1.2, 0.25)


def test_collision_checked_under_route_only_overrides_soft_pause() -> None:
    assert trajectory_blocked("PAUSE", guarded_route_active=False)
    assert trajectory_blocked("PAUSE", guarded_route_active=True)
    assert trajectory_blocked("EMERGENCY_STOP", guarded_route_active=True)


def test_single_empty_perception_frame_does_not_clear_live_obstacle() -> None:
    assert hold_recent_obstacle(True, 10.0, 10.1, 0.75)


def test_sustained_empty_perception_clears_obstacle() -> None:
    assert not hold_recent_obstacle(True, 10.0, 12.0, 0.75)
    assert not hold_recent_obstacle(False, 10.0, 10.1, 0.75)


def test_only_obstacle_motion_near_active_tcp_corridor_requires_replan() -> None:
    start = [-0.55, 0.14, 1.07]
    goal = [0.08, 0.18, 0.41]
    assert obstacle_affects_segment(
        start,
        goal,
        [-0.11, 0.06, 0.31],
        [0.25, 0.22, 0.14],
        margin_m=0.10,
    )
    assert not obstacle_affects_segment(
        start,
        goal,
        [-0.10, -0.60, 0.30],
        [0.15, 0.12, 0.10],
        margin_m=0.10,
    )


def test_low_hand_crossing_articulated_xy_corridor_triggers_early_replan() -> None:
    start = [-0.58, 0.15, 1.05]
    goal = [0.08, 0.18, 0.41]
    center = [-0.30, 0.17, 0.43]
    size = [0.18, 0.14, 0.10]
    assert not obstacle_affects_segment(
        start, goal, center, size, margin_m=0.025
    )
    assert obstacle_affects_motion_corridor(
        start, goal, center, size, margin_m=0.025
    )
    assert not obstacle_affects_motion_corridor(
        start,
        goal,
        [-0.30, -0.70, 0.43],
        size,
        margin_m=0.025,
    )


def test_low_hand_does_not_cancel_unrelated_high_arm_motion() -> None:
    start = [-0.55, 0.14, 1.04]
    goal = [-0.25, 0.18, 0.92]
    center = [-0.38, 0.16, 0.10]
    size = [0.32, 0.28, 0.13]
    assert not obstacle_affects_motion_corridor(
        start, goal, center, size, margin_m=0.06
    )
