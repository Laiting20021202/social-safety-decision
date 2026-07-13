import json
from pathlib import Path

import numpy as np

from realtime_safety.export.pointcloud_export import export_ply
from realtime_safety.export.session_logger import SessionLogger
from realtime_safety.types import (
    BBox3D,
    DangerZone,
    PerformanceSnapshot,
    PointCloudFrame,
    RecommendedAction,
    SafetyLevel,
    SafetySnapshot,
    Track3DState,
)


def _safety() -> SafetySnapshot:
    position = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    track = Track3DState(
        1,
        "person",
        position,
        np.array([0.0, -0.5, 0.0], dtype=np.float32),
        np.zeros(3, np.float32),
        np.eye(6),
        BBox3D(position - 0.2, position + 0.2),
        0.3,
        4,
        0,
        1.0,
        "dynamic",
        0.9,
    )
    zone = DangerZone(
        1,
        np.array([position, position + track.velocity_xyz]),
        np.array([0.7, 0.8]),
        np.array([0, -1, 0]),
        0.5,
        0.8,
        0.2,
        1.2,
        SafetyLevel.WARNING,
        True,
    )
    return SafetySnapshot(1.0, 10, SafetyLevel.WARNING, RecommendedAction.SLOW_DOWN, [track], [zone], [], None, False, ["RELATIVE_SCALE"])


def test_jsonl_relative_scale_never_claims_metric_velocity(tmp_path: Path) -> None:
    logger = SessionLogger(tmp_path)
    logger.log(_safety(), "FAST", "fast_depth", "relative", PerformanceSnapshot())
    logger.close()
    record = json.loads((tmp_path / "safety.jsonl").read_text().splitlines()[0])
    assert not record["metric_valid"]
    assert record["objects"][0]["velocity_unit"] == "relative_units/s"
    assert "m/s" not in (tmp_path / "trajectories.csv").read_text()


def test_pointcloud_ply_is_written(tmp_path: Path) -> None:
    points = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    cloud = PointCloudFrame(points, np.array([[255, 0, 0], [0, 255, 0]], np.uint8), np.ones(2), points.reshape(1, 2, 3), 0, 0.0, 0, 1.0, True, "test")
    path = export_ply(cloud, tmp_path / "cloud.ply")
    assert path.is_file()
    assert path.stat().st_size > 100
