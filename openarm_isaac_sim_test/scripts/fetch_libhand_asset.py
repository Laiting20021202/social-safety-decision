#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


URL = "https://libhand.org/files/libhand-0.9.tar.gz"
SHA256 = "dad21d2cf170652368bacd584422fe5b4d52f349c6f411d2b546904bfd25d671"
MEMBERS = (
    "libhand-0.9/hand_model/ogre/hand.mesh",
    "libhand-0.9/hand_model/ogre/hand.skeleton",
    "libhand-0.9/hand_model/ogre/hand_texture.png",
    "libhand-0.9/hand_model/ogre/hand_model_license.txt",
)


def _ogre_xml_converter(cache: Path) -> Path:
    installed = shutil.which("OgreXMLConverter")
    if installed:
        return Path(installed)
    local = cache / "ogre-tools/usr/bin/OgreXMLConverter"
    if local.is_file():
        return local
    package_dir = cache / "ogre-package"
    package_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["apt-get", "download", "ogre-1.9-tools"], cwd=package_dir, check=True
    )
    debs = tuple(package_dir.glob("ogre-1.9-tools_*.deb"))
    if len(debs) != 1:
        raise RuntimeError("could not resolve the local ogre-1.9-tools package")
    subprocess.run(
        ["dpkg-deb", "-x", str(debs[0]), str(cache / "ogre-tools")], check=True
    )
    if not local.is_file():
        raise RuntimeError("OgreXMLConverter was not extracted")
    return local


def _download(cache: Path) -> Path:
    archive = cache / "libhand-0.9.tar.gz"
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"LibHand archive checksum mismatch: {digest}")
    return archive


SPREAD_POSE = {
    "finger1joint1": (0.05, -0.30, 0.0),
    "finger2joint1": (0.02, -0.11, 0.0),
    "finger3joint1": (0.02, 0.08, 0.0),
    "finger4joint1": (0.05, 0.25, 0.0),
    "finger5joint1": (0.28, 0.34, 0.18),
    "finger5joint2": (0.18, 0.0, 0.0),
}


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s, one = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.asarray(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ]
    )


def _euler_xyz(bend: float, side: float, twist: float) -> np.ndarray:
    rx = _axis_angle(np.asarray([1.0, 0.0, 0.0]), bend)
    ry = _axis_angle(np.asarray([0.0, 1.0, 0.0]), twist)
    rz = _axis_angle(np.asarray([0.0, 0.0, 1.0]), side)
    return rx @ ry @ rz


def _spread_vertices(
    mesh_root: ET.Element,
    skeleton_xml: Path,
    positions: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    skeleton = ET.parse(skeleton_xml).getroot()
    bones: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    names: dict[str, int] = {}
    for bone in skeleton.findall("./bones/bone"):
        bone_id = int(bone.get("id"))
        name = str(bone.get("name"))
        position_node = bone.find("position")
        rotation_node = bone.find("rotation")
        axis_node = rotation_node.find("axis")
        position = np.asarray(
            [float(position_node.get(axis)) for axis in "xyz"], dtype=float
        )
        rotation = _axis_angle(
            np.asarray([float(axis_node.get(axis)) for axis in "xyz"], dtype=float),
            float(rotation_node.get("angle")),
        )
        bones[bone_id] = (name, position, rotation)
        names[name] = bone_id
    parent: dict[int, int] = {}
    for relation in skeleton.findall("./bonehierarchy/boneparent"):
        parent[names[str(relation.get("bone"))]] = names[str(relation.get("parent"))]

    def local_matrix(bone_id: int, posed: bool) -> np.ndarray:
        name, position, bind_rotation = bones[bone_id]
        rotation = bind_rotation
        if posed and name in SPREAD_POSE:
            rotation = rotation @ _euler_xyz(*SPREAD_POSE[name])
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = position
        return result

    cache: dict[tuple[int, bool], np.ndarray] = {}

    def world_matrix(bone_id: int, posed: bool) -> np.ndarray:
        key = (bone_id, posed)
        if key in cache:
            return cache[key]
        local = local_matrix(bone_id, posed)
        result = (
            world_matrix(parent[bone_id], posed) @ local
            if bone_id in parent
            else local
        )
        cache[key] = result
        return result

    skin = {
        bone_id: world_matrix(bone_id, True)
        @ np.linalg.inv(world_matrix(bone_id, False))
        for bone_id in bones
    }
    posed_positions = np.zeros_like(positions)
    posed_normals = np.zeros_like(normals)
    total_weight = np.zeros(len(positions), dtype=float)
    assignments = mesh_root.find("boneassignments")
    for assignment in assignments or ():
        vertex = int(assignment.get("vertexindex"))
        bone_id = int(assignment.get("boneindex"))
        weight = float(assignment.get("weight"))
        transform = skin[bone_id]
        point = transform @ np.append(positions[vertex], 1.0)
        posed_positions[vertex] += weight * point[:3]
        posed_normals[vertex] += weight * (transform[:3, :3] @ normals[vertex])
        total_weight[vertex] += weight
    unassigned = total_weight <= 1e-8
    posed_positions[unassigned] = positions[unassigned]
    posed_normals[unassigned] = normals[unassigned]
    lengths = np.linalg.norm(posed_normals, axis=1)
    posed_normals /= np.maximum(lengths[:, None], 1e-12)
    return posed_positions, posed_normals


def _convert(xml_path: Path, skeleton_xml: Path, output: Path) -> None:
    root = ET.parse(xml_path).getroot()
    shared = root.find("sharedgeometry")
    if shared is None:
        raise RuntimeError("LibHand mesh has no shared geometry")
    buffers = shared.findall("vertexbuffer")
    positions_buffer = next(item for item in buffers if item.get("positions") == "true")
    texcoord_buffer = next(item for item in buffers if item.get("texture_coords"))
    positions = []
    normals = []
    for vertex in positions_buffer.findall("vertex"):
        position = vertex.find("position")
        normal = vertex.find("normal")
        positions.append(tuple(float(position.get(axis)) for axis in "xyz"))
        normals.append(tuple(float(normal.get(axis)) for axis in "xyz"))
    positions_array, normals_array = _spread_vertices(
        root,
        skeleton_xml,
        np.asarray(positions, dtype=float),
        np.asarray(normals, dtype=float),
    )
    texcoords = []
    for vertex in texcoord_buffer.findall("vertex"):
        texcoord = vertex.find("texcoord")
        texcoords.append((float(texcoord.get("u")), 1.0 - float(texcoord.get("v"))))
    if len(positions) != len(texcoords):
        raise RuntimeError("LibHand position and UV counts differ")
    with (output / "hand.obj").open("w", encoding="utf-8") as stream:
        stream.write("# LibHand 0.9 human hand, CC BY 3.0\nmtllib hand.mtl\n")
        for point in positions_array:
            stream.write(f"v {point[0]:.7g} {point[1]:.7g} {point[2]:.7g}\n")
        for uv in texcoords:
            stream.write(f"vt {uv[0]:.7g} {uv[1]:.7g}\n")
        for normal in normals_array:
            stream.write(f"vn {normal[0]:.7g} {normal[1]:.7g} {normal[2]:.7g}\n")
        submeshes = root.find("submeshes")
        for index, submesh in enumerate(submeshes or ()):
            material = submesh.get("material", f"part_{index}")
            stream.write(f"g {material}\nusemtl {material}\n")
            faces = submesh.find("faces")
            for face in faces or ():
                values = [int(face.get(key)) + 1 for key in ("v1", "v2", "v3")]
                stream.write(
                    "f " + " ".join(f"{value}/{value}/{value}" for value in values) + "\n"
                )
    (output / "hand.mtl").write_text(
        "newmtl skin\nKd 0.95 0.68 0.56\nKa 0.2 0.12 0.1\n"
        "Ks 0.08 0.08 0.08\nmap_Kd hand_texture.png\n\n"
        "newmtl blackness\nKd 0.03 0.02 0.02\nKa 0.01 0.01 0.01\n",
        encoding="utf-8",
    )


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fetch and convert the licensed LibHand mesh")
    parser.add_argument("--output", type=Path, default=project / "assets/hand/libhand")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "hand.obj").is_file() and not args.force:
        print(output / "hand.obj")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    cache = project / ".cache/libhand"
    cache.mkdir(parents=True, exist_ok=True)
    archive = _download(cache)
    with tempfile.TemporaryDirectory(prefix="openarm_libhand_") as directory:
        work = Path(directory)
        with tarfile.open(archive, "r:gz") as bundle:
            available = {member.name: member for member in bundle.getmembers()}
            bundle.extractall(
                work,
                members=[available[name] for name in MEMBERS],
                filter="data",
            )
        source = work / "libhand-0.9/hand_model/ogre"
        converter = _ogre_xml_converter(cache)
        xml_path = source / "hand.mesh.xml"
        subprocess.run([str(converter), "hand.mesh", xml_path.name], cwd=source, check=True)
        skeleton_xml = source / "hand.skeleton.xml"
        subprocess.run(
            [str(converter), "hand.skeleton", skeleton_xml.name], cwd=source, check=True
        )
        _convert(xml_path, skeleton_xml, output)
        shutil.copy2(source / "hand_texture.png", output / "hand_texture.png")
        shutil.copy2(source / "hand_model_license.txt", output / "LICENSE.txt")
    (output / "source.json").write_text(
        json.dumps(
            {
                "name": "LibHand 0.9 Human Hand Model",
                "source": URL,
                "sha256": SHA256,
                "license": "CC-BY-3.0",
                "conversion": "OGRE skinned mesh -> spread-hand Wavefront OBJ",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output / "hand.obj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
