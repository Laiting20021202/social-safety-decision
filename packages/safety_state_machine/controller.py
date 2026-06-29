from __future__ import annotations

from packages.common_models import GeometryPrediction, SafetyDecision


class SafetyController:
    def __init__(
        self,
        warning_time_to_zone_sec: float = 3.0,
        critical_time_to_zone_sec: float = 1.0,
    ) -> None:
        self.warning_time_to_zone_sec = warning_time_to_zone_sec
        self.critical_time_to_zone_sec = critical_time_to_zone_sec

    def decide(
        self,
        scenario_id: str,
        frame_index: int,
        timestamp_sec: float,
        geometry: list[GeometryPrediction],
        zone_available: bool,
        tracker_valid: bool = True,
    ) -> SafetyDecision:
        if not zone_available:
            return self._fail_safe(scenario_id, frame_index, timestamp_sec, "zone is not set")
        if not tracker_valid:
            return self._fail_safe(
                scenario_id, frame_index, timestamp_sec, "tracking result unavailable or invalid"
            )
        if not geometry:
            return SafetyDecision(
                timestamp_sec=timestamp_sec,
                scenario_id=scenario_id,
                frame_index=frame_index,
                zone_state="clear",
                risk_level="safe",
                recommended_action="continue",
                confidence=0.6,
                reason="No tracked person is near the safety zone.",
                source={"tracker": True, "geometry": True, "vqa": False},
            )

        most_urgent = min(
            geometry,
            key=lambda item: item.time_to_zone_sec
            if item.time_to_zone_sec is not None
            else float("inf"),
        )
        target_ids = [prediction.track_id for prediction in geometry]
        if any(prediction.zone_relation == "inside" for prediction in geometry):
            return SafetyDecision(
                timestamp_sec=timestamp_sec,
                scenario_id=scenario_id,
                frame_index=frame_index,
                zone_state="occupied",
                risk_level="critical",
                recommended_action="pause",
                target_track_ids=target_ids,
                time_to_zone_sec=0.0,
                confidence=0.95,
                reason="A tracked person is inside the safety zone.",
                hard_rule_triggered=True,
                hard_rule_reason="person inside zone",
                source={"tracker": True, "geometry": True, "vqa": False},
            )
        if most_urgent.time_to_zone_sec is not None:
            if most_urgent.time_to_zone_sec <= self.critical_time_to_zone_sec:
                return SafetyDecision(
                    timestamp_sec=timestamp_sec,
                    scenario_id=scenario_id,
                    frame_index=frame_index,
                    zone_state="approaching",
                    risk_level="critical",
                    recommended_action="pause",
                    target_track_ids=[most_urgent.track_id],
                    time_to_zone_sec=most_urgent.time_to_zone_sec,
                    confidence=0.9,
                    reason="Predicted time-to-zone is below critical threshold.",
                    hard_rule_triggered=True,
                    hard_rule_reason="critical time-to-zone",
                    source={"tracker": True, "geometry": True, "vqa": False},
                )
            if most_urgent.time_to_zone_sec <= self.warning_time_to_zone_sec:
                return SafetyDecision(
                    timestamp_sec=timestamp_sec,
                    scenario_id=scenario_id,
                    frame_index=frame_index,
                    zone_state="approaching",
                    risk_level="warning",
                    recommended_action="slow_down",
                    target_track_ids=[most_urgent.track_id],
                    time_to_zone_sec=most_urgent.time_to_zone_sec,
                    confidence=0.85,
                    reason="Predicted time-to-zone is below warning threshold.",
                    hard_rule_triggered=True,
                    hard_rule_reason="warning time-to-zone",
                    source={"tracker": True, "geometry": True, "vqa": False},
                )
        return SafetyDecision(
            timestamp_sec=timestamp_sec,
            scenario_id=scenario_id,
            frame_index=frame_index,
            zone_state="clear",
            risk_level="safe",
            recommended_action="continue",
            target_track_ids=[],
            confidence=0.7,
            reason="Geometry does not predict zone entry within thresholds.",
            source={"tracker": True, "geometry": True, "vqa": False},
        )

    @staticmethod
    def _fail_safe(
        scenario_id: str,
        frame_index: int,
        timestamp_sec: float,
        reason: str,
    ) -> SafetyDecision:
        return SafetyDecision(
            timestamp_sec=timestamp_sec,
            scenario_id=scenario_id,
            frame_index=frame_index,
            zone_state="unknown",
            risk_level="unknown",
            recommended_action="human_review",
            confidence=0.0,
            reason=reason,
            hard_rule_triggered=True,
            hard_rule_reason=reason,
            source={"tracker": False, "geometry": False, "vqa": False},
        )
