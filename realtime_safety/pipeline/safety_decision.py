from __future__ import annotations

from realtime_safety.config import SafetyConfig
from realtime_safety.pipeline.local_planner import PlannerResult
from realtime_safety.types import DangerZone, RecommendedAction, SafetyLevel, SafetySnapshot, Track3DState


_SEVERITY = {SafetyLevel.SAFE: 0, SafetyLevel.CAUTION: 1, SafetyLevel.WARNING: 2, SafetyLevel.STOP: 3}


class SafetyDecisionEngine:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._latched = SafetyLevel.SAFE
        self._release_count = 0

    def reset(self) -> None:
        self._latched = SafetyLevel.SAFE
        self._release_count = 0

    def update(
        self,
        timestamp: float,
        frame_index: int,
        tracks: list[Track3DState],
        zones: list[DangerZone],
        planner: PlannerResult,
        metric_valid: bool,
        depth_valid: bool = True,
        camera_motion_confidence: float = 1.0,
        model_ready: bool = True,
    ) -> SafetySnapshot:
        degraded_reasons: list[str] = []
        if not metric_valid:
            degraded_reasons.append("RELATIVE_SCALE")
        if not depth_valid:
            degraded_reasons.append("DEPTH_INVALID")
        if camera_motion_confidence < 0.2:
            degraded_reasons.append("CAMERA_MOTION_UNCERTAIN")
        if not model_ready:
            degraded_reasons.append("MODEL_NOT_READY")
        raw = max((zone.risk_level for zone in zones), key=lambda level: _SEVERITY[level], default=SafetyLevel.SAFE)
        if planner.action == RecommendedAction.STOP:
            raw = SafetyLevel.STOP
        if not depth_valid or not model_ready:
            raw = SafetyLevel.STOP if raw == SafetyLevel.STOP else SafetyLevel.DEGRADED
        level = self._debounce(raw)
        action = planner.action
        if level == SafetyLevel.STOP:
            action = RecommendedAction.STOP
        elif level == SafetyLevel.WARNING and action == RecommendedAction.CONTINUE:
            action = RecommendedAction.SLOW_DOWN
        elif level == SafetyLevel.DEGRADED:
            action = RecommendedAction.WAIT
        return SafetySnapshot(
            timestamp=float(timestamp),
            frame_index=frame_index,
            safety_state=level,
            recommended_action=action,
            tracks=tracks,
            danger_zones=zones,
            candidates=planner.candidates,
            selected_path=planner.selected,
            metric_valid=metric_valid,
            degraded_reasons=degraded_reasons,
        )

    def _debounce(self, raw: SafetyLevel) -> SafetyLevel:
        if raw == SafetyLevel.DEGRADED:
            self._latched = raw
            self._release_count = 0
            return raw
        if self._latched == SafetyLevel.DEGRADED:
            self._latched = raw
            return raw
        if _SEVERITY[raw] >= _SEVERITY[self._latched]:
            self._latched = raw
            self._release_count = 0
            return raw
        self._release_count += 1
        required = self.config.release_safe_updates if self._latched == SafetyLevel.STOP else max(2, self.config.release_safe_updates // 2)
        if self._release_count >= required:
            self._latched = raw
            self._release_count = 0
        return self._latched
