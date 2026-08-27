from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CubeSpec:
    name: str
    color: str
    position: tuple[float, float, float]


def deterministic_cube_layout(scene_config: dict[str, object]) -> list[CubeSpec]:
    targets = scene_config.get("target_cubes")
    if isinstance(targets, dict) and bool(targets.get("enabled", False)):
        models = targets.get("models")
        if not isinstance(models, dict):
            raise ValueError("target_cubes.models must be a mapping")
        specs: list[CubeSpec] = []
        for side in ("left", "right"):
            values = models.get(side)
            if not isinstance(values, dict):
                raise ValueError(f"target_cubes.models.{side} is required")
            position = tuple(float(value) for value in values["position"])
            if len(position) != 3:
                raise ValueError(f"{side} target position must have x/y/z")
            specs.append(
                CubeSpec(name=str(values["name"]), color=side, position=position)
            )
        return specs
    cubes = scene_config["cubes"]
    assert isinstance(cubes, dict)
    seed = int(scene_config["seed"])
    center = np.asarray(cubes["spawn_center"], dtype=float)
    extent = np.asarray(cubes["spawn_extent"], dtype=float)
    count = int(cubes["count_per_color"])
    separation = float(cubes["minimum_separation"])
    size = float(cubes["size"])
    colors = tuple(cubes["colors"].keys())
    rng = np.random.default_rng(seed)
    positions: list[np.ndarray] = []
    total = count * len(colors)
    slots = cubes.get("spawn_slots_xy")
    if slots is not None:
        if len(slots) != total:
            raise ValueError(f"spawn_slots_xy must contain exactly {total} slots")
        jitter = float(cubes.get("slot_jitter_m", 0.0))
        for slot in slots:
            xy = np.asarray(slot, dtype=float) + rng.uniform(-jitter, jitter, size=2)
            positions.append(np.array([xy[0], xy[1], center[2] + size / 2.0]))
        for index, position in enumerate(positions):
            if any(
                np.linalg.norm(position[:2] - other[:2]) < separation
                for other in positions[index + 1 :]
            ):
                raise ValueError("jittered cube slots violate minimum_separation")
    else:
        maximum_attempts = 10_000
        for _ in range(total):
            for _attempt in range(maximum_attempts):
                xy = center[:2] + rng.uniform(-0.5, 0.5, size=2) * extent
                if all(np.linalg.norm(xy - old[:2]) >= separation for old in positions):
                    position = np.array([xy[0], xy[1], center[2] + size / 2.0])
                    positions.append(position)
                    break
            else:
                raise ValueError("cube spawn region is too small for minimum_separation")
    specs: list[CubeSpec] = []
    index = 0
    for color in colors:
        for color_index in range(count):
            specs.append(
                CubeSpec(
                    name=f"{color}_cube_{color_index + 1}",
                    color=color,
                    position=tuple(float(value) for value in positions[index]),
                )
            )
            index += 1
    return specs
