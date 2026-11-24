from pathlib import Path
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
from realtime_safety.edgetam_tracker.urdf_self_filter import UrdfSelfFilter


def test_joint_rotation_collision_origin_and_nearby_hand(tmp_path):
    mesh = tmp_path / "link.stl"
    trimesh.creation.box(extents=[100, 60, 400]).export(mesh)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f'<robot name="arm"><link name="forearm"><collision><origin xyz="0 0 -.2"/><geometry><mesh filename="{mesh.as_uri()}" scale=".001 .001 .001"/></geometry></collision></link></robot>')
    geometry = UrdfSelfFilter(urdf, padding=.008)
    rotation = Rotation.from_euler("y", 60, degrees=True).as_matrix()
    origin = np.array([.2, .1, .7])
    local = np.array([[.05, 0, -.39], [0, .03, -.2], [.08, 0, -.2]])
    points = local @ rotation.T + origin
    assert geometry.keep_mask(points, {"forearm": (rotation, origin)}).tolist() == [False, False, True]
    np.testing.assert_allclose(geometry.clearance_lower_bound(points, {"forearm": (rotation, origin)}), [0, 0, .03], atol=1e-6)
    with pytest.raises(ValueError, match="missing"):
        geometry.keep_mask(points, {})


def test_actual_openarm_mesh_includes_fingers_and_body():
    path = Path(__file__).resolve().parents[2] / "openarm_sim/gazebo/generated/openarm_gazebo.urdf"
    if not path.exists(): pytest.skip("generated OpenArm asset unavailable")
    geometry = UrdfSelfFilter(path)
    assert "openarm_body_link0" in geometry.link_names
    assert "openarm_left_right_finger" in geometry.link_names
    assert "openarm_right_link7" in geometry.link_names
    assert len(geometry.link_names) == 23
