from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Rotation from Isaac Camera's world convention (+X forward, +Y left, +Z up)
# to REP-103 optical convention (+Z forward, +X right, +Y down), expressed
# as x/y/z/w for ROS messages.
ISAAC_WORLD_LINK_TO_ROS_OPTICAL_XYZW = (-0.5, 0.5, -0.5, 0.5)


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def intrinsics_from_horizontal_fov(width: int, height: int, horizontal_fov_deg: float) -> PinholeIntrinsics:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("horizontal FOV must be between 1 and 179 degrees")
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return PinholeIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        cx=(width - 1.0) / 2.0,
        cy=(height - 1.0) / 2.0,
    )


def back_project_depth(
    depth_m: np.ndarray,
    intrinsics: PinholeIntrinsics,
    rgb: np.ndarray | None = None,
    near_clip: float = 0.0,
    far_clip: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project aligned 32FC1 depth using the ROS optical-frame convention.

    Returned XYZ columns are x-right, y-down, z-forward and contain only finite,
    in-range depth samples. RGB values, when supplied, remain uint8 RGB.
    """

    if depth_m.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth shape {depth_m.shape} does not match "
            f"{(intrinsics.height, intrinsics.width)}"
        )
    if rgb is not None and rgb.shape[:2] != depth_m.shape:
        raise ValueError("RGB and depth must be pixel-aligned")
    v, u = np.indices(depth_m.shape, dtype=np.float32)
    z = depth_m.astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z >= near_clip) & (z <= far_clip) & (z > 0.0)
    x = (u[valid] - intrinsics.cx) * z[valid] / intrinsics.fx
    y = (v[valid] - intrinsics.cy) * z[valid] / intrinsics.fy
    points = np.column_stack((x, y, z[valid])).astype(np.float32, copy=False)
    colors = None if rgb is None else rgb[..., :3][valid].astype(np.uint8, copy=False)
    return points, colors


def camera_world_position(
    workspace_center: tuple[float, float, float] | list[float],
    height_above_table: float,
    horizontal_offset: float,
    lateral_offset: float,
) -> np.ndarray:
    center = np.asarray(workspace_center, dtype=np.float64)
    return center + np.array([-horizontal_offset, lateral_offset, height_above_table])


def camera_aim_direction(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the normalized link-frame +X viewing direction in world axes."""

    delta = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("camera position and target must be distinct finite points")
    return delta / length


def quaternion_wxyz_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert a normalized Isaac w/x/y/z quaternion into a rotation matrix."""

    values = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("camera quaternion must be finite and non-zero")
    w, x, y, z = values / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def optical_points_to_world(
    points: np.ndarray,
    camera_position: np.ndarray,
    camera_orientation_wxyz: np.ndarray,
) -> np.ndarray:
    """Transform REP-103 optical XYZ into the Isaac/ROS world frame."""

    optical = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # Optical x/y/z maps to link -y/-z/+x.
    optical_to_link = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    world_from_link = quaternion_wxyz_matrix(camera_orientation_wxyz)
    rotation = world_from_link @ optical_to_link
    translated = optical @ rotation.T + np.asarray(camera_position, dtype=np.float64)
    return translated.astype(np.float32, copy=False)
