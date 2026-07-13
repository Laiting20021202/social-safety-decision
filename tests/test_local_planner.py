import numpy as np

from realtime_safety.config import SafetyConfig
from realtime_safety.pipeline.local_planner import LocalSafetyPlanner
from realtime_safety.types import BBox3D, RecommendedAction, Track3DState


def obstacle(x: float, y: float, radius: float = 0.35) -> Track3DState:
    position = np.array([x, y, 0.3], dtype=np.float32)
    return Track3DState(
        track_id=int((x + 2) * 10 + y),
        class_name="box",
        position_xyz=position,
        velocity_xyz=np.zeros(3, dtype=np.float32),
        acceleration_xyz=np.zeros(3, dtype=np.float32),
        covariance=np.eye(6) * 0.01,
        bbox3d=BBox3D(position - radius, position + radius),
        radius=radius,
        hit_count=3,
        missing_count=0,
        last_timestamp=0.0,
        motion_state="static",
        confidence=0.9,
    )


def test_center_obstacle_selects_a_detour() -> None:
    result = LocalSafetyPlanner(SafetyConfig()).plan([obstacle(0.0, 1.2)], [])
    assert result.selected is not None
    assert result.action in (RecommendedAction.DETOUR_LEFT, RecommendedAction.DETOUR_RIGHT)
    assert any(not candidate.safe for candidate in result.candidates)


def test_blocked_corridor_returns_stop() -> None:
    obstacles = [obstacle(x, 0.7, 0.75) for x in (-1.0, 0.0, 1.0)]
    result = LocalSafetyPlanner(SafetyConfig()).plan(obstacles, [])
    assert result.selected is None
    assert result.action == RecommendedAction.STOP
