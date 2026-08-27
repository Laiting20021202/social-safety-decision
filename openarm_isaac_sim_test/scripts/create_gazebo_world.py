#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_DIR = Path(os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_DIR))

from openarm_sim.camera_math import camera_world_position
from openarm_sim.config import PROJECT_ROOT, load_yaml
from openarm_sim.scene_model import deterministic_cube_layout


def _text(parent: ET.Element, name: str, value: object) -> ET.Element:
    child = ET.SubElement(parent, name)
    child.text = str(value)
    return child


def _pose(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _material(visual: ET.Element, rgba: tuple[float, float, float, float]) -> None:
    material = ET.SubElement(visual, "material")
    _text(material, "ambient", _pose(rgba))
    _text(material, "diffuse", _pose(rgba))


def _box_link(
    model: ET.Element,
    name: str,
    position: list[float] | tuple[float, ...] | np.ndarray,
    size: list[float] | tuple[float, ...] | np.ndarray,
    color: tuple[float, float, float, float],
    *,
    collision: bool = True,
    mass: float | None = None,
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ET.Element:
    link = ET.SubElement(model, "link", {"name": name})
    _text(link, "pose", _pose((*position, *rpy)))
    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    _text(ET.SubElement(geometry, "box"), "size", _pose(size))
    _material(visual, color)
    if mass is not None:
        sx, sy, sz = (float(value) for value in size)
        inertial = ET.SubElement(link, "inertial")
        _text(inertial, "mass", mass)
        inertia = ET.SubElement(inertial, "inertia")
        _text(inertia, "ixx", mass * (sy * sy + sz * sz) / 12.0)
        _text(inertia, "iyy", mass * (sx * sx + sz * sz) / 12.0)
        _text(inertia, "izz", mass * (sx * sx + sy * sy) / 12.0)
        _text(inertia, "ixy", 0.0)
        _text(inertia, "ixz", 0.0)
        _text(inertia, "iyz", 0.0)
    if collision:
        collision_node = ET.SubElement(link, "collision", {"name": "collision"})
        geometry = ET.SubElement(collision_node, "geometry")
        _text(ET.SubElement(geometry, "box"), "size", _pose(size))
    return link


def _model(world: ET.Element, name: str, *, static: bool) -> ET.Element:
    model = ET.SubElement(world, "model", {"name": name})
    _text(model, "static", str(static).lower())
    return model


def _camera_rpy(position: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    direction = target - position
    direction /= np.linalg.norm(direction)
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    pitch = math.atan2(float(-direction[2]), float(np.linalg.norm(direction[:2])))
    return 0.0, pitch, yaw


def _apply_saved_model_poses(world: ET.Element) -> None:
    """Apply poses captured from Gazebo without changing semantic scene YAML."""

    path = PROJECT_ROOT / "config/gazebo_layout.yaml"
    if not path.is_file():
        return
    document = yaml.safe_load(path.read_text()) or {}
    saved = document.get("models", {})
    if not isinstance(saved, dict):
        raise ValueError("config/gazebo_layout.yaml models must be a mapping")
    for model in world.findall("model"):
        values = saved.get(model.get("name", ""))
        if not isinstance(values, dict):
            continue
        position = values.get("position")
        rpy_deg = values.get("rpy_deg")
        if not (
            isinstance(position, list)
            and len(position) == 3
            and isinstance(rpy_deg, list)
            and len(rpy_deg) == 3
        ):
            raise ValueError(f"invalid saved Gazebo pose for {model.get('name')}")
        pose = model.find("pose")
        if pose is None:
            pose = ET.SubElement(model, "pose")
        pose.text = _pose(
            (*position, *(math.radians(float(value)) for value in rpy_deg))
        )


def build_world() -> ET.ElementTree:
    scene = load_yaml("config/scene.yaml")
    camera = load_yaml("config/camera.yaml")["camera"]
    hand_document = load_yaml("config/hand_scenarios.yaml")
    hand = hand_document["defaults"]
    sdf = ET.Element("sdf", {"version": "1.7"})
    world = ET.SubElement(sdf, "world", {"name": "openarm_sorting"})
    physics = ET.SubElement(world, "physics", {"name": "default_physics", "type": "ode"})
    _text(physics, "max_step_size", "0.001")
    _text(physics, "real_time_update_rate", "1000")
    state_plugin = ET.SubElement(
        world,
        "plugin",
        {"name": "gazebo_ros_state", "filename": "libgazebo_ros_state.so"},
    )
    state_ros = ET.SubElement(state_plugin, "ros")
    _text(state_ros, "namespace", "/gazebo")
    _text(state_plugin, "update_rate", "100")
    scene_node = ET.SubElement(world, "scene")
    _text(scene_node, "ambient", "0.55 0.55 0.55 1")
    _text(scene_node, "background", "0.04 0.04 0.05 1")
    _text(scene_node, "shadows", "false")
    ground = _model(world, "ground_plane", static=True)
    ground_link = ET.SubElement(ground, "link", {"name": "ground"})
    ground_collision = ET.SubElement(ground_link, "collision", {"name": "collision"})
    ground_geometry = ET.SubElement(ground_collision, "geometry")
    ground_plane = ET.SubElement(ground_geometry, "plane")
    _text(ground_plane, "normal", "0 0 1")
    _text(ground_plane, "size", "20 20")
    ground_visual = ET.SubElement(ground_link, "visual", {"name": "visual"})
    ground_geometry = ET.SubElement(ground_visual, "geometry")
    ground_plane = ET.SubElement(ground_geometry, "plane")
    _text(ground_plane, "normal", "0 0 1")
    _text(ground_plane, "size", "20 20")
    _material(ground_visual, (0.18, 0.18, 0.19, 1.0))
    sun = ET.SubElement(world, "light", {"name": "sun", "type": "directional"})
    _text(sun, "pose", "0 0 10 0 0 0")
    _text(sun, "diffuse", "0.8 0.8 0.8 1")
    _text(sun, "specular", "0.2 0.2 0.2 1")
    _text(sun, "direction", "-0.3 0.2 -1")

    table = scene["table"]
    table_yaw = math.radians(float(table.get("yaw_deg", 0.0)))
    table_rotation = np.array(
        [
            [math.cos(table_yaw), -math.sin(table_yaw), 0.0],
            [math.sin(table_yaw), math.cos(table_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    table_model = _model(world, "work_table", static=True)
    _box_link(
        table_model,
        "table",
        table["center"],
        table["size"],
        (*table["color_rgb"], 1.0),
        rpy=(0.0, 0.0, table_yaw),
    )
    edge = table["yellow_edge"]
    edge_offset = table_rotation @ np.array(
        [
            0.0,
            -float(table["size"][1]) / 2.0 + float(edge["width"]) / 2.0,
            (float(table["size"][2]) + float(edge["height"])) / 2.0,
        ]
    )
    edge_position = np.asarray(table["center"], dtype=float) + edge_offset
    _box_link(
        table_model,
        "yellow_edge",
        edge_position,
        [table["size"][0], edge["width"], edge["height"]],
        (*edge["color_rgb"], 1.0),
        rpy=(0.0, 0.0, table_yaw),
    )

    bins = scene["bins"]
    if bool(bins.get("enabled", True)):
        inner_x, inner_y, height = bins["inner_size"]
        wall = bins["wall_thickness"]
        base = bins["base_thickness"]
        bin_colors = scene["cubes"]["colors"]
        for color, center in bins["centers"].items():
            model = _model(world, f"{color}_bin", static=True)
            rgba = (*bin_colors[color], 0.7)
            _box_link(model, "base", [center[0], center[1], scene["table"]["top_z"] + base / 2], [inner_x + 2 * wall, inner_y + 2 * wall, base], rgba)
            for suffix, offset, size in (
                ("left", (-inner_x / 2 - wall / 2, 0), (wall, inner_y + 2 * wall, height)),
                ("right", (inner_x / 2 + wall / 2, 0), (wall, inner_y + 2 * wall, height)),
                ("front", (0, -inner_y / 2 - wall / 2), (inner_x, wall, height)),
                ("back", (0, inner_y / 2 + wall / 2), (inner_x, wall, height)),
            ):
                _box_link(model, suffix, [center[0] + offset[0], center[1] + offset[1], scene["table"]["top_z"] + height / 2], size, rgba)

    targets = scene.get("target_cubes", {})
    if bool(targets.get("enabled", False)):
        cube_size = float(targets["size"])
        for spec in deterministic_cube_layout(scene):
            values = targets["models"][spec.color]
            model = _model(world, spec.name, static=False)
            _text(model, "pose", _pose((*spec.position, 0.0, 0.0, 0.0)))
            link = _box_link(
                model,
                "target",
                [0.0, 0.0, 0.0],
                [cube_size] * 3,
                (*values["color_rgb"], 1.0),
                collision=bool(targets.get("collision_enabled", False)),
            )
            _text(link, "gravity", str(bool(targets.get("gravity_enabled", False))).lower())
            _text(link, "kinematic", "true")
    elif bool(scene["cubes"].get("enabled", True)):
        cube_size = float(scene["cubes"]["size"])
        for spec in deterministic_cube_layout(scene):
            model = _model(world, spec.name, static=False)
            _box_link(
                model,
                "cube",
                spec.position,
                [cube_size] * 3,
                (*scene["cubes"]["colors"][spec.color], 1.0),
                mass=0.08,
            )

    tag = scene["apriltag"]
    marker = _model(world, "apriltag_36h11_0", static=True)
    pixels = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        int(tag["id"]),
        8,
        borderBits=1,
    )
    cell = float(tag["size"]) / 8.0
    _box_link(marker, "quiet_zone", [tag["center"][0], tag["center"][1], tag["center"][2] - 0.00025], [tag["size"] + 0.02, tag["size"] + 0.02, 0.0005], (1.0, 1.0, 1.0, 1.0), collision=False)
    for row in range(8):
        for col in range(8):
            center = [
                tag["center"][0] + (col - 3.5) * cell,
                tag["center"][1] + (3.5 - row) * cell,
                tag["center"][2] + 0.0005,
            ]
            value = 1.0 if int(pixels[row, col]) > 127 else 0.0
            _box_link(marker, f"cell_{row}_{col}", center, [cell, cell, 0.001], (value, value, value, 1.0), collision=False)

    hand_model = _model(world, "human_hand", static=False)
    parked = hand_document["scenarios"]["no_obstacle"]["path_waypoints"][0]
    _text(hand_model, "pose", _pose((*parked, 0.0, 0.0, 0.0)))
    collision_proxy = hand["collision_proxy"]
    hand_link = ET.SubElement(hand_model, "link", {"name": "hand_body"})
    _text(hand_link, "kinematic", "true")
    _text(hand_link, "gravity", "false")
    hand_mesh = (PROJECT_ROOT / str(hand["visual_asset"])).resolve()
    if hand_mesh.is_file():
        hand_visual = ET.SubElement(hand_link, "visual", {"name": "libhand_visual"})
        hand_rpy = tuple(
            math.radians(float(value)) for value in hand.get("visual_rpy_deg", [0, 0, 0])
        )
        _text(hand_visual, "pose", _pose((0.0, 0.0, 0.0, *hand_rpy)))
        hand_geometry = ET.SubElement(hand_visual, "geometry")
        hand_mesh_node = ET.SubElement(hand_geometry, "mesh")
        _text(hand_mesh_node, "uri", hand_mesh.as_uri())
        _text(hand_mesh_node, "scale", _pose(hand.get("visual_scale", [1, 1, 1])))
    else:
        fallback = ET.SubElement(hand_link, "visual", {"name": "missing_mesh_fallback"})
        fallback_geometry = ET.SubElement(fallback, "geometry")
        _text(
            ET.SubElement(fallback_geometry, "box"),
            "size",
            _pose(collision_proxy["palm_size"]),
        )
        _material(fallback, (0.95, 0.62, 0.48, 1.0))
    forearm_length = float(collision_proxy["forearm_length"])
    forearm_radius = float(collision_proxy["forearm_radius"])
    forearm_enabled = bool(collision_proxy.get("forearm_enabled", True))
    if forearm_enabled and bool(collision_proxy.get("forearm_visual", True)):
        segment_count = max(
            int(collision_proxy.get("forearm_visual_segments", 3)), 1
        )
        segment_length = forearm_length / segment_count
        wrist_visual_radius = float(
            collision_proxy.get("forearm_wrist_visual_radius", forearm_radius)
        )
        elbow_radius = float(
            collision_proxy.get("forearm_elbow_radius", forearm_radius)
        )
        skin_rgba = tuple(
            float(value)
            for value in collision_proxy.get(
                "forearm_skin_rgba", [0.72, 0.35, 0.22, 1.0]
            )
        )
        for index in range(segment_count):
            blend = (index + 0.5) / segment_count
            radius = wrist_visual_radius + blend * (
                elbow_radius - wrist_visual_radius
            )
            center_y = -(index + 0.5) * segment_length
            forearm_visual = ET.SubElement(
                hand_link,
                "visual",
                {"name": f"forearm_visual_{index}"},
            )
            _text(
                forearm_visual,
                "pose",
                _pose((0.0, center_y, 0.0, math.pi / 2.0, 0.0, 0.0)),
            )
            visual_geometry = ET.SubElement(forearm_visual, "geometry")
            visual_cylinder = ET.SubElement(visual_geometry, "cylinder")
            _text(visual_cylinder, "radius", radius)
            # A slight overlap avoids visible seams while preserving the
            # configured total forearm length.
            _text(visual_cylinder, "length", segment_length * 1.04)
            _material(forearm_visual, skin_rgba)

    # One continuous conservative collision proxy covers the tapered visual.
    palm_collision = ET.SubElement(hand_link, "collision", {"name": "palm_proxy"})
    palm_geometry = ET.SubElement(palm_collision, "geometry")
    _text(
        ET.SubElement(palm_geometry, "box"),
        "size",
        _pose(collision_proxy["palm_size"]),
    )
    if forearm_enabled:
        forearm_collision = ET.SubElement(
            hand_link, "collision", {"name": "forearm_proxy"}
        )
        _text(
            forearm_collision,
            "pose",
            _pose((0.0, -forearm_length / 2.0, 0.0, math.pi / 2.0, 0.0, 0.0)),
        )
        forearm_collision_geometry = ET.SubElement(forearm_collision, "geometry")
        forearm_collision_cylinder = ET.SubElement(
            forearm_collision_geometry, "cylinder"
        )
        _text(forearm_collision_cylinder, "radius", forearm_radius)
        _text(forearm_collision_cylinder, "length", forearm_length)

    workspace = np.asarray(scene["zones"]["workspace"]["center"], dtype=float)
    explicit_pose = camera.get("world_pose")
    if explicit_pose:
        camera_position = np.asarray(explicit_pose["position"], dtype=float)
        rpy = tuple(math.radians(float(value)) for value in explicit_pose["rpy_deg"])
    else:
        camera_position = camera_world_position(
            workspace,
            float(camera["height_above_table"]),
            float(camera["horizontal_offset_to_workspace_center"]),
            float(camera["lateral_offset"]),
        )
        camera_target = workspace + np.asarray(
            camera.get("aim_offset", [0.0, 0.0, 0.0]), dtype=float
        )
        rpy = _camera_rpy(camera_position, camera_target)
    camera_model = _model(world, "rgbd_sensor", static=True)
    _text(camera_model, "pose", _pose((*camera_position, *rpy)))
    link = ET.SubElement(camera_model, "link", {"name": "rgbd_link"})
    visual = ET.SubElement(link, "visual", {"name": "housing"})
    geometry = ET.SubElement(visual, "geometry")
    _text(ET.SubElement(geometry, "box"), "size", "0.07 0.11 0.05")
    _material(visual, (0.08, 0.08, 0.09, 1.0))
    sensor = ET.SubElement(link, "sensor", {"name": "rgbd", "type": "depth"})
    _text(sensor, "always_on", "true")
    _text(sensor, "update_rate", "15")
    _text(sensor, "visualize", "true")
    camera_node = ET.SubElement(sensor, "camera")
    _text(camera_node, "horizontal_fov", f"{math.radians(float(camera['horizontal_fov_deg'])):.9g}")
    image = ET.SubElement(camera_node, "image")
    _text(image, "width", camera["resolution"]["width"])
    _text(image, "height", camera["resolution"]["height"])
    _text(image, "format", "R8G8B8")
    ET.SubElement(camera_node, "depth_camera")
    clip = ET.SubElement(camera_node, "clip")
    _text(clip, "near", camera["near_clip"])
    _text(clip, "far", camera["far_clip"])
    plugin = ET.SubElement(sensor, "plugin", {"name": "rgbd_ros_camera", "filename": "libgazebo_ros_camera.so"})
    ros = ET.SubElement(plugin, "ros")
    _text(ros, "namespace", "/")
    for source, target in (
        ("rgbd/image_raw", "rgbd/color/image_raw"),
        ("rgbd/camera_info", "rgbd/color/camera_info"),
        ("rgbd/depth/image_raw", "rgbd/depth/image_raw"),
        ("rgbd/depth/camera_info", "rgbd/depth/camera_info"),
        ("rgbd/points", "rgbd/points"),
    ):
        _text(ros, "remapping", f"{source}:={target}")
    _text(plugin, "camera_name", "rgbd")
    _text(plugin, "frame_name", "rgbd_color_optical_frame")
    _text(plugin, "min_depth", camera["near_clip"])
    _text(plugin, "max_depth", camera["far_clip"])
    _apply_saved_model_poses(world)
    return ET.ElementTree(sdf)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the OpenArm Gazebo Phase-1 world")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "gazebo/worlds/openarm_sorting.world")
    args = parser.parse_args()
    tree = build_world()
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
