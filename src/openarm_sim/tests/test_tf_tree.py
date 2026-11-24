import numpy as np

from openarm_sim.camera_math import ISAAC_WORLD_LINK_TO_ROS_OPTICAL_XYZW
from openarm_sim.config import load_yaml
from openarm_sim.contracts import OPTICAL_FRAMES


def test_generic_optical_frames_are_complete() -> None:
    camera = load_yaml("config/camera.yaml")["camera"]
    configured = {camera["link_frame"], camera["color_frame"], camera["depth_frame"]}
    assert configured == set(OPTICAL_FRAMES)
    assert all("realsense" not in frame.lower() for frame in configured)
    assert camera["optical_convention"] == "REP-103_x_right_y_down_z_forward"


def test_isaac_world_camera_axes_map_to_rep103_optical() -> None:
    x, y, z, w = ISAAC_WORLD_LINK_TO_ROS_OPTICAL_XYZW
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(
        rotation,
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    )
