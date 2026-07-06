from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from packages.common_models import FramePacket, Point2D


@dataclass(frozen=True)
class CameraProfile:
    name: str
    base_width: int
    base_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_height_m: float
    pitch_down_deg: float
    lateral_min_m: float = -4.0
    lateral_max_m: float = 4.0
    forward_min_m: float = 0.0
    forward_max_m: float = 12.0


SCAND_AZURE_KINECT_ESTIMATED = CameraProfile(
    name="scand_azure_kinect_estimated_ipm",
    base_width=1280,
    base_height=720,
    fx=608.115906,
    fy=607.871704,
    cx=639.186401,
    cy=363.069092,
    camera_height_m=0.85,
    pitch_down_deg=24.0,
)


def profile_for_frame(frame: FramePacket) -> CameraProfile | None:
    dataset_name = frame.dataset_name.lower()
    source_type = frame.source_type.lower()
    if "scand" in dataset_name or source_type == "local_video":
        return SCAND_AZURE_KINECT_ESTIMATED
    return None


def project_pixel_to_bev_normalized(
    point: Point2D,
    *,
    image_width: int | float,
    image_height: int | float,
    profile: CameraProfile,
) -> Point2D | None:
    ground = project_pixel_to_ground_m(
        point,
        image_width=image_width,
        image_height=image_height,
        profile=profile,
    )
    if ground is None:
        return None
    lateral_m, forward_m = ground
    x = (lateral_m - profile.lateral_min_m) / (
        profile.lateral_max_m - profile.lateral_min_m
    )
    y = (profile.forward_max_m - forward_m) / (
        profile.forward_max_m - profile.forward_min_m
    )
    return Point2D(x=max(0.0, min(1.0, x)), y=max(0.0, min(1.0, y)))


def project_pixel_to_ground_m(
    point: Point2D,
    *,
    image_width: int | float,
    image_height: int | float,
    profile: CameraProfile,
) -> tuple[float, float] | None:
    scaled = _scaled_intrinsics(profile, image_width, image_height)
    x = (point.x - scaled["cx"]) / scaled["fx"]
    y = (point.y - scaled["cy"]) / scaled["fy"]
    base_ray = np.array([x, 1.0, -y], dtype=np.float64)
    ray = _rotation_x(-math.radians(profile.pitch_down_deg)) @ base_ray
    if ray[2] >= -1e-6:
        return None
    scale = -profile.camera_height_m / ray[2]
    lateral_m = float(ray[0] * scale)
    forward_m = float(ray[1] * scale)
    if forward_m < profile.forward_min_m - 0.5 or forward_m > profile.forward_max_m + 2.0:
        return None
    return lateral_m, forward_m


def render_bev_projection(
    image_path: Path,
    output_path: Path,
    *,
    image_width: int,
    image_height: int,
    profile: CameraProfile = SCAND_AZURE_KINECT_ESTIMATED,
    output_width: int = 1000,
    output_height: int = 420,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (image_width, image_height) and image_width > 0 and image_height > 0:
            rgb = rgb.resize((image_width, image_height), Image.Resampling.BILINEAR)
        source = np.asarray(rgb, dtype=np.float32)
    rendered = _render_ground_plane(source, profile, output_width, output_height)
    image = Image.fromarray(rendered.astype(np.uint8), mode="RGB")
    _draw_metric_grid(image, profile)
    image.save(output_path)
    return output_path


def _render_ground_plane(
    source: np.ndarray,
    profile: CameraProfile,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    rows, cols = np.indices((output_height, output_width), dtype=np.float64)
    lateral = profile.lateral_min_m + (cols / max(1, output_width - 1)) * (
        profile.lateral_max_m - profile.lateral_min_m
    )
    forward = profile.forward_max_m - (rows / max(1, output_height - 1)) * (
        profile.forward_max_m - profile.forward_min_m
    )
    robot_vectors = np.stack(
        [
            lateral,
            forward,
            np.full_like(lateral, -profile.camera_height_m),
        ],
        axis=-1,
    )
    base_vectors = robot_vectors @ _rotation_x(math.radians(profile.pitch_down_deg)).T
    camera_x = base_vectors[..., 0]
    camera_y = -base_vectors[..., 2]
    camera_z = base_vectors[..., 1]
    scaled = _scaled_intrinsics(profile, source_width, source_height)
    valid = camera_z > 1e-6
    u = scaled["fx"] * camera_x / np.maximum(camera_z, 1e-6) + scaled["cx"]
    v = scaled["fy"] * camera_y / np.maximum(camera_z, 1e-6) + scaled["cy"]
    valid &= (u >= 0) & (u <= source_width - 1) & (v >= 0) & (v <= source_height - 1)
    background = np.full((output_height, output_width, 3), 244, dtype=np.float32)
    sampled = _bilinear_sample(source, u, v)
    background[valid] = sampled[valid]
    return (background * 0.78 + 255.0 * 0.22).clip(0, 255)


def _bilinear_sample(source: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = source.shape[:2]
    u0 = np.floor(u).astype(np.int64).clip(0, width - 1)
    v0 = np.floor(v).astype(np.int64).clip(0, height - 1)
    u1 = (u0 + 1).clip(0, width - 1)
    v1 = (v0 + 1).clip(0, height - 1)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    top = source[v0, u0] * (1.0 - du) + source[v0, u1] * du
    bottom = source[v1, u0] * (1.0 - du) + source[v1, u1] * du
    return top * (1.0 - dv) + bottom * dv


def _draw_metric_grid(image: Image.Image, profile: CameraProfile) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for lateral in range(math.ceil(profile.lateral_min_m), math.floor(profile.lateral_max_m) + 1):
        x = int(round((lateral - profile.lateral_min_m) / (
            profile.lateral_max_m - profile.lateral_min_m
        ) * width))
        draw.line([(x, 0), (x, height)], fill=(40, 46, 43, 52), width=2 if lateral == 0 else 1)
    for forward in range(math.ceil(profile.forward_min_m), math.floor(profile.forward_max_m) + 1):
        y = int(round((profile.forward_max_m - forward) / (
            profile.forward_max_m - profile.forward_min_m
        ) * height))
        draw.line([(0, y), (width, y)], fill=(40, 46, 43, 52), width=1)
    origin_x = int(round((0.0 - profile.lateral_min_m) / (
        profile.lateral_max_m - profile.lateral_min_m
    ) * width))
    draw.ellipse(
        (origin_x - 11, height - 23, origin_x + 11, height - 1),
        fill=(20, 91, 73, 230),
    )


def _scaled_intrinsics(
    profile: CameraProfile,
    image_width: int | float,
    image_height: int | float,
) -> dict[str, float]:
    sx = float(image_width) / max(1.0, float(profile.base_width))
    sy = float(image_height) / max(1.0, float(profile.base_height))
    return {
        "fx": profile.fx * sx,
        "fy": profile.fy * sy,
        "cx": profile.cx * sx,
        "cy": profile.cy * sy,
    }


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )
