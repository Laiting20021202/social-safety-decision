import math

import pytest

from realtime_safety.gui.isaac_joint_control import JointJogState


@pytest.fixture
def state():
    return JointJogState({"openarm_left_joint1": (-1.0, 1.0), "openarm_right_joint1": (-1.0, 1.0)})


def test_unrelated_hand_feedback_cannot_replace_or_refresh_arm_state(state):
    assert state.update(list(state.limits), [0.1, -0.1], 10.0)
    assert not state.update(["joint_index_mcp"], [0.5], 11.9)
    assert state.positions["openarm_left_joint1"] == 0.1
    assert not state.fresh(12.1)


def test_named_jog_preserves_other_joints_and_uses_live_pose(state):
    state.update(list(state.limits), [0.1, -0.1], 10.0)
    assert state.target("openarm_left_joint1", 0.01, 10.5) == pytest.approx(0.11)
    assert state.positions["openarm_right_joint1"] == -0.1


@pytest.mark.parametrize("delta", [math.nan, math.inf, 0, 0.051, -0.051])
def test_invalid_steps_rejected(state, delta):
    state.update(list(state.limits), [0.0, 0.0], 10.0)
    with pytest.raises(ValueError):
        state.target("openarm_left_joint1", delta, 10.1)


def test_stale_feedback_and_joint_limit_rejected(state):
    state.update(list(state.limits), [0.99, 0.0], 10.0)
    with pytest.raises(ValueError):
        state.target("openarm_left_joint1", 0.02, 10.1)
    with pytest.raises(ValueError):
        state.target("openarm_left_joint1", -0.01, 12.1)


def test_malformed_samples_do_not_refresh_feedback(state):
    assert not state.update(list(state.limits), [0.0], 10.0)
    assert not state.update(list(state.limits), [math.nan, 0.0], 10.0)
    assert not state.update(["openarm_left_joint1"] * 2, [0.0, 0.0], 10.0)
    assert state.received_at is None
