from openarm_safety_bridge.policy import SafetyPolicy
from openarm_sim.state_machine import SafetyState


def policy() -> SafetyPolicy:
    return SafetyPolicy(0.35, 0.22, 0.08, 0.40, 1.0, 0.25)


def test_pause_replan_recover_sequence() -> None:
    subject = policy()
    assert subject.observe(0.20, 0.0) is SafetyState.PAUSE
    assert subject.observe(0.50, 0.1) is SafetyState.PAUSE
    assert subject.observe(0.50, 1.2) is SafetyState.REPLAN
    assert subject.observe(0.50, 1.31) is SafetyState.RECOVER
    assert subject.observe(0.50, 1.32) is SafetyState.SAFE


def test_estop_is_latched_until_explicit_reset() -> None:
    subject = policy()
    assert subject.observe(0.01, 0.0) is SafetyState.EMERGENCY_STOP
    assert subject.observe(1.0, 3.0) is SafetyState.EMERGENCY_STOP
    assert subject.reset_estop(3.1) is SafetyState.REPLAN


def test_static_non_emergency_obstacle_advances_from_pause_to_replan() -> None:
    subject = policy()
    assert subject.observe(0.15, 0.0) is SafetyState.PAUSE
    assert subject.observe(0.15, 0.20) is SafetyState.PAUSE
    assert subject.observe(0.15, 0.30) is SafetyState.REPLAN
    assert subject.observe(0.15, 0.60) is SafetyState.REPLAN


def test_single_near_outlier_pauses_without_latching_estop() -> None:
    subject = SafetyPolicy(
        0.25,
        0.12,
        0.04,
        0.30,
        1.0,
        3.2,
        emergency_confirmation_sec=0.20,
    )
    assert subject.observe(0.01, 1.0) is SafetyState.PAUSE
    assert subject.observe(0.40, 1.1) is SafetyState.PAUSE
    assert not subject.estop_latched


def test_persistent_near_contact_latches_estop() -> None:
    subject = SafetyPolicy(
        0.25,
        0.12,
        0.04,
        0.30,
        1.0,
        3.2,
        emergency_confirmation_sec=0.20,
    )
    assert subject.observe(0.01, 1.0) is SafetyState.PAUSE
    assert subject.observe(0.01, 1.21) is SafetyState.EMERGENCY_STOP


def test_checked_escape_gets_bounded_motion_grace_then_estops() -> None:
    subject = SafetyPolicy(
        0.25,
        0.12,
        0.04,
        0.30,
        1.0,
        3.2,
        emergency_confirmation_sec=0.20,
    )
    assert subject.observe(0.01, 1.0) is SafetyState.PAUSE
    assert subject.grant_escape_grace(1.05, 1.0) is SafetyState.PAUSE
    assert subject.observe(0.01, 1.80) is SafetyState.PAUSE
    assert subject.observe(0.01, 2.06) is SafetyState.PAUSE
    assert subject.observe(0.01, 2.27) is SafetyState.EMERGENCY_STOP


def test_escape_grace_cannot_clear_latched_estop() -> None:
    subject = policy()
    assert subject.observe(0.01, 0.0) is SafetyState.EMERGENCY_STOP
    assert subject.grant_escape_grace(0.1, 3.0) is SafetyState.EMERGENCY_STOP
