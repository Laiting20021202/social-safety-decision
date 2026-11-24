from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realtime_safety.config import SafetyConfig
from realtime_safety.types import DangerZone, PathCandidate, RecommendedAction, Track3DState


@dataclass(slots=True)
class PlannerResult:
    candidates: list[PathCandidate]
    selected: PathCandidate | None
    action: RecommendedAction


class LocalSafetyPlanner:
    """Demo forward-corridor sampler; it is not a global navigation planner."""

    def __init__(self, config: SafetyConfig, forward_speed: float = 0.8) -> None:
        self.config = config
        self.forward_speed = forward_speed

    def plan(self, tracks: list[Track3DState], zones: list[DangerZone]) -> PlannerResult:
        times = np.arange(
            0.0,
            self.config.prediction_horizon + self.config.prediction_timestep * 0.5,
            self.config.prediction_timestep,
            dtype=np.float32,
        )
        distance = self.forward_speed * times
        candidates: list[PathCandidate] = []
        for curvature in np.linspace(-1.4, 1.4, 9):
            # Candidate headings fan out from the forward corridor.
            x = curvature * distance
            y = distance
            z = np.full_like(x, 0.03)
            points = np.stack((x, y, z), axis=-1)
            safe, clearance, collision = self._evaluate(points, tracks, zones)
            score = float(clearance + 0.4 * y[-1] - 1.5 * abs(curvature)) if safe else float("-inf")
            name = f"curve_{curvature:+.2f}"
            candidates.append(PathCandidate(points, safe, score, name, collision))
        safe_candidates = [candidate for candidate in candidates if candidate.safe]
        if not safe_candidates:
            return PlannerResult(candidates, None, RecommendedAction.STOP)
        selected = max(safe_candidates, key=lambda candidate: candidate.score)
        lateral = float(selected.points[-1, 0])
        if lateral > 0.15:
            action = RecommendedAction.DETOUR_RIGHT
        elif lateral < -0.15:
            action = RecommendedAction.DETOUR_LEFT
        elif min((zone.closest_predicted_distance for zone in zones), default=10.0) < self.config.warning_clearance * 1.5:
            action = RecommendedAction.SLOW_DOWN
        else:
            action = RecommendedAction.CONTINUE
        return PlannerResult(candidates, selected, action)

    def _evaluate(
        self, path: np.ndarray, tracks: list[Track3DState], zones: list[DangerZone]
    ) -> tuple[bool, float, np.ndarray | None]:
        minimum_clearance = float("inf")
        for track in tracks:
            distances = np.linalg.norm(path[:, :2] - track.position_xyz[None, :2], axis=1)
            clearances = distances - (track.radius + self.config.robot_radius + self.config.safety_margin)
            index = int(np.argmin(clearances))
            minimum_clearance = min(minimum_clearance, float(clearances[index]))
            if track.motion_state == "static" and clearances[index] < 0:
                return False, minimum_clearance, path[index].copy()
        for zone in zones:
            count = min(len(path), len(zone.predicted_positions))
            if count == 0:
                continue
            if zone.dynamic:
                distances = np.linalg.norm(path[:count, :2] - zone.predicted_positions[:count, :2], axis=1)
                clearances = distances - zone.radii[:count]
            else:
                distances = np.linalg.norm(path[:, :2] - zone.predicted_positions[0, :2], axis=1)
                clearances = distances - zone.radii[0]
            index = int(np.argmin(clearances))
            minimum_clearance = min(minimum_clearance, float(clearances[index]))
            if clearances[index] < 0:
                return False, minimum_clearance, path[index].copy()
        return True, minimum_clearance if np.isfinite(minimum_clearance) else 10.0, None
