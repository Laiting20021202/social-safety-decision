from __future__ import annotations

from pathlib import Path

import numpy as np

from realtime_safety.types import PointCloudFrame


def export_ply(cloud: PointCloudFrame, path: str | Path) -> Path:
    from plyfile import PlyData, PlyElement

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = min(len(cloud.points), len(cloud.colors))
    vertices = np.empty(
        count,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = cloud.points[:count].T
    vertices["red"], vertices["green"], vertices["blue"] = cloud.colors[:count].T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(destination)
    return destination
