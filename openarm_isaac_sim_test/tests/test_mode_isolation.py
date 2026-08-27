import pytest

from openarm_safety_bridge.node import _stale_obstacle_ids
from openarm_sim.contracts import (
    GROUND_TRUTH_TOPICS,
    RuntimeMode,
    assert_mode_isolation,
    subscriptions_for_mode,
)


def test_perception_mode_has_no_ground_truth_subscription() -> None:
    subscriptions = subscriptions_for_mode(RuntimeMode.PERCEPTION)
    assert not set(subscriptions).intersection(GROUND_TRUTH_TOPICS)
    assert_mode_isolation(RuntimeMode.PERCEPTION, subscriptions)


def test_ground_truth_leak_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not subscribe"):
        assert_mode_isolation(RuntimeMode.PERCEPTION, [GROUND_TRUTH_TOPICS[0]])


def test_dynamic_obstacles_are_removed_on_startup_and_source_change() -> None:
    expected = {"ground_truth_hand", "perception_hand_obstacle"}
    assert set(_stale_obstacle_ids("", "perception")) == expected
    assert set(_stale_obstacle_ids("ground_truth", "perception")) == expected
    assert set(_stale_obstacle_ids("perception", "ground_truth")) == expected


def test_reselecting_same_source_preserves_current_dynamic_obstacle() -> None:
    assert _stale_obstacle_ids("perception", "perception") == ()
