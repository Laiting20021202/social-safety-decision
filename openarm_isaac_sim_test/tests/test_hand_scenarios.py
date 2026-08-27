import pytest

from openarm_sim.config import load_yaml
from openarm_sim.scenario import HandTrajectory, MotionPhase, bounded_step, load_scenario
from openarm_sim_bringup.gazebo_hand_controller import (
    GazeboHandController,
    bounded_sim_dt,
)


EXPECTED = {
    "no_obstacle",
    "perception_preview",
    "static_blocking",
    "right_side_sweep",
    "reach_for_cube",
    "cross_during_transit",
    "hover_near_end_effector",
    "sudden_intrusion",
    "intrude_and_withdraw",
    "repeated_intrusion",
    "fully_blocked",
}


def test_all_required_scenarios_are_configured() -> None:
    config = load_yaml("config/hand_scenarios.yaml")
    assert set(config["scenarios"]) == EXPECTED
    assert config["speed_profiles_mps"] == [0.01, 0.02, 0.05, 0.1, 0.3, 0.6]


def test_demo_uses_only_a_slightly_enlarged_hand_mesh() -> None:
    defaults = load_yaml("config/hand_scenarios.yaml")["defaults"]
    proxy = defaults["collision_proxy"]

    assert defaults["visual_scale"] == [0.052, 0.052, 0.052]
    assert proxy["forearm_enabled"] is False
    assert proxy["forearm_visual"] is False
    assert proxy["palm_size"] == [0.115, 0.085, 0.030]


def test_gui_auto_sweep_is_lateral_and_inside_manual_workspace() -> None:
    defaults = load_yaml("config/hand_scenarios.yaml")["defaults"]
    assert defaults["auto_sweep_axis"] == "y"
    low, high = defaults["auto_sweep_limits_m"]
    assert low < high
    assert defaults["manual_workspace_min"][1] <= low
    assert high <= defaults["manual_workspace_max"][1]


@pytest.mark.parametrize("name", sorted(EXPECTED - {"no_obstacle"}))
def test_scenario_is_deterministic_and_withdraws(name: str) -> None:
    first = HandTrajectory(load_scenario(name))
    second = HandTrajectory(load_scenario(name))
    trigger = first.trigger_state
    task_state = trigger if trigger else None
    a = first.sample(max(first.start_time, 0.0), task_state)
    b = second.sample(max(second.start_time, 0.0), task_state)
    assert a == b
    first.withdraw(1.0, a.position)
    assert first.sample(1.01).phase in {MotionPhase.MOVING_OUT, MotionPhase.COMPLETE}


def test_no_obstacle_never_enters_workspace() -> None:
    trajectory = HandTrajectory(load_scenario("no_obstacle"))
    assert trajectory.sample(100.0).phase is MotionPhase.COMPLETE


def test_manual_hand_step_never_teleports() -> None:
    result = bounded_step((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.3, 0.02)
    assert result == pytest.approx((0.006, 0.0, 0.0))
    assert bounded_step(result, (0.007, 0.0, 0.0), 0.3, 0.02) == pytest.approx(
        (0.007, 0.0, 0.0)
    )


def test_hand_controller_integrates_simulation_time() -> None:
    assert hasattr(GazeboHandController, "_publish_pose")
    assert bounded_sim_dt(10.0, 10.05) == pytest.approx(0.05)
    assert bounded_sim_dt(10.0, 10.5) == pytest.approx(0.1)
    assert bounded_sim_dt(10.0, 9.0) == 0.0
