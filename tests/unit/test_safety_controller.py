from __future__ import annotations

from packages.common_models import GeometryPrediction
from packages.safety_state_machine import SafetyController


def test_missing_zone_fails_safe() -> None:
    controller = SafetyController()
    decision = controller.decide("demo", 0, 0.0, [], zone_available=False)

    assert decision.recommended_action == "human_review"
    assert decision.hard_rule_triggered


def test_critical_time_to_zone_pauses() -> None:
    controller = SafetyController()
    decision = controller.decide(
        "demo",
        5,
        2.5,
        [GeometryPrediction(track_id=1, time_to_zone_sec=0.5, zone_relation="approaching")],
        zone_available=True,
    )

    assert decision.recommended_action == "pause"
    assert decision.risk_level == "critical"
    assert decision.target_track_ids == [1]
