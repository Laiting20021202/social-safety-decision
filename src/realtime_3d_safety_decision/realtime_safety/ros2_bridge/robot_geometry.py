"""Image-time TF adapter for the simulator-independent URDF self filter."""
import numpy as np
from scipy.spatial.transform import Rotation


def robot_keep_mask(geometry, buffer, points, frame_id, stamp):
    return geometry.keep_mask(points, robot_transforms(geometry, buffer, frame_id, stamp))


def robot_transforms(geometry, buffer, frame_id, stamp):
    from rclpy.duration import Duration
    transforms = {}
    for name in geometry.link_names:
        tf = buffer.lookup_transform(frame_id, name, stamp, timeout=Duration(seconds=.03)).transform
        p, q = tf.translation, tf.rotation
        transforms[name] = (
            Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
            np.array([p.x, p.y, p.z]),
        )
    return transforms
