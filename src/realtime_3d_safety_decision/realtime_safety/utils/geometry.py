from __future__ import annotations

import numpy as np


def planar_clearance(points: np.ndarray, center_xyz: np.ndarray, radius: float) -> np.ndarray:
    """Signed XY clearance; negative values are inside the inflated obstacle."""
    return np.linalg.norm(np.asarray(points)[..., :2] - np.asarray(center_xyz)[:2], axis=-1) - float(radius)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1), dtype=np.float32)), axis=-1)
    return (homogeneous @ transform.T)[..., :3]
