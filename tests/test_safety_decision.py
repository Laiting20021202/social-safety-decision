import numpy as np

from realtime_safety.config import SafetyConfig
from realtime_safety.pipeline.local_planner import PlannerResult
from realtime_safety.pipeline.safety_decision import SafetyDecisionEngine
from realtime_safety.types import DangerZone, RecommendedAction, SafetyLevel


def zone(level: SafetyLevel) -> DangerZone:
    return DangerZone(
        track_id=1,
        predicted_positions=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        radii=np.array([0.5], dtype=np.float32),
        predicted_direction=np.zeros(3, dtype=np.float32),
        predicted_speed=0.0,
        risk_score=1.0 if level == SafetyLevel.STOP else 0.0,
        closest_predicted_distance=0.0 if level == SafetyLevel.STOP else 2.0,
        ttc=0.5 if level == SafetyLevel.STOP else None,
        risk_level=level,
        dynamic=False,
    )


def planner(action: RecommendedAction = RecommendedAction.CONTINUE) -> PlannerResult:
    return PlannerResult([], None, action)


def test_stop_triggers_immediately_and_requires_safe_updates_to_release() -> None:
    engine = SafetyDecisionEngine(SafetyConfig(release_safe_updates=3))
    stopped = engine.update(0, 0, [], [zone(SafetyLevel.STOP)], planner(), False)
    assert stopped.safety_state == SafetyLevel.STOP
    for index in (1, 2):
        still_stopped = engine.update(index, index, [], [zone(SafetyLevel.SAFE)], planner(), False)
        assert still_stopped.safety_state == SafetyLevel.STOP
    released = engine.update(3, 3, [], [zone(SafetyLevel.SAFE)], planner(), False)
    assert released.safety_state == SafetyLevel.SAFE


def test_invalid_depth_reports_degraded_but_never_masks_stop() -> None:
    engine = SafetyDecisionEngine(SafetyConfig())
    degraded = engine.update(0, 0, [], [], planner(), False, depth_valid=False)
    assert degraded.safety_state == SafetyLevel.DEGRADED
    assert degraded.recommended_action == RecommendedAction.WAIT
    stopped = engine.update(1, 1, [], [zone(SafetyLevel.STOP)], planner(), False, depth_valid=False)
    assert stopped.safety_state == SafetyLevel.STOP


def test_relative_scale_is_labeled_without_fake_metric_validity() -> None:
    snapshot = SafetyDecisionEngine(SafetyConfig()).update(0, 0, [], [], planner(), metric_valid=False)
    assert not snapshot.metric_valid
    assert "RELATIVE_SCALE" in snapshot.degraded_reasons
