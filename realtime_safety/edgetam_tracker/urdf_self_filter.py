"""URDF collision-mesh self masking; geometry only, without ROS or simulator APIs."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh


@lru_cache(maxsize=64)
def _vertices(filename):
    return np.asarray(trimesh.load(filename, force="mesh", process=False).vertices)


class UrdfSelfFilter:
    """Convex envelope per collision mesh with metric padding.

    Unlike spheres centred on joints, this respects mesh scale, collision
    origin, link rotation, fingers and pedestal. Convex envelopes may mask
    small concavities; padding is explicitly bounded, not a semantic guess.
    Callers provide transforms AT THE IMAGE STAMP, derived from JointState/FK.
    Missing any required transform raises; callers must hold their last output.
    """
    def __init__(self, urdf_path, padding=.008):
        self.padding = float(padding)
        self.parts = []
        for link in ET.parse(urdf_path).getroot().findall("link"):
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename", "")
                if filename.startswith("file://"):
                    filename = unquote(urlparse(filename).path)
                elif not Path(filename).is_absolute():
                    filename = str(Path(urdf_path).parent / filename)
                vertices = _vertices(filename) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                origin = collision.find("origin")
                if origin is not None:
                    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                    vertices = vertices @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                vertices = np.unique(vertices, axis=0)
                hull = ConvexHull(vertices)
                # STL coplanar triangles often duplicate the same plane.
                equations = np.unique(np.round(hull.equations, 7), axis=0)
                self.parts.append((link.get("name"), vertices.min(axis=0), vertices.max(axis=0), equations))
        if not self.parts:
            raise ValueError("URDF has no mesh collision geometry")
        self.link_names = tuple(dict.fromkeys(part[0] for part in self.parts))

    def keep_mask(self, points, transforms):
        missing = set(self.link_names) - transforms.keys()
        if missing:
            raise ValueError("missing image-time link transforms: " + ",".join(sorted(missing)))
        points = np.asarray(points)
        keep = np.isfinite(points).all(axis=1)
        for name, lower, upper, equations in self.parts:
            rotation, origin = transforms[name]
            world_center = (lower + upper) / 2 @ rotation.T + origin
            world_half = np.abs(rotation) @ ((upper - lower) / 2 + self.padding)
            broad_world = np.all(np.abs(points - world_center) <= world_half, axis=1)
            selected = np.flatnonzero(keep & broad_world)
            if not len(selected): continue
            local = (points[selected] - origin) @ rotation
            broad = np.all((local >= lower - self.padding) & (local <= upper + self.padding), axis=1)
            selected = selected[broad]
            local = local[broad]
            # Bound temporary storage for full-resolution RGB-D clouds.
            for start in range(0, len(local), 1024):
                block = local[start:start + 1024]
                inside = np.all(block @ equations[:, :3].T + equations[:, 3] <= self.padding, axis=1)
                keep[selected[start:start + 1024][inside]] = False
        return keep

    def clearance_lower_bound(self, points, transforms, link_names=None):
        """Conservative point-to-convex-mesh distance, in metres.

        Maximum signed supporting-plane distance is <= exact Euclidean
        distance. It respects slender fingers instead of replacing them with
        a uniform 5 cm capsule. Negative means inside a convex envelope.
        """
        points = np.asarray(points).reshape(-1, 3)
        distances = np.full(len(points), np.inf)
        for name, lower, upper, planes in self.parts:
            if link_names is not None and name not in link_names:
                continue
            rotation, origin = transforms[name]
            local = (points - origin) @ rotation
            for start in range(0, len(local), 1024):
                block = local[start:start + 1024]
                values = (block @ planes[:, :3].T + planes[:, 3]).max(axis=1)
                distances[start:start + 1024] = np.minimum(distances[start:start + 1024], values)
        return np.maximum(distances, 0.)
