"""Publish metric MASt3R points in a registered optical TF, or in a separate chest preview TF."""
import os
from scipy.spatial.transform import Rotation
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import StaticTransformBroadcaster

from frames import FRAME, SENSOR, fresh, stamp_ns
from registration import BASE, CHEST_PREVIEW, load_registration


class Scene(Node):
    def __init__(self):
        super().__init__('laiting_rviz_cloud')
        transform = load_registration()
        self.frame = 'laiting_d415_registered_optical' if transform is not None else 'laiting_d415_chest_preview'
        topic = '/laiting/scene/points' if transform is not None else '/laiting/preview/points'
        if transform is None:
            transform = CHEST_PREVIEW
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = BASE, self.frame
        t.header.stamp = self.get_clock().now().to_msg()
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = map(float, Rotation.from_matrix(transform[:3,:3]).as_quat())
        v = t.transform.translation
        v.x, v.y, v.z = map(float, transform[:3,3])
        self.tf = StaticTransformBroadcaster(self)
        self.tf.sendTransform(t)
        self.pub = self.create_publisher(PointCloud2, topic, SENSOR)
        fixed_chest = os.environ.get('LAITING_CAMERA_PROFILE') == 'isaac_fixed_openarm_v1' or os.environ.get('LAITING_FIXED_CHEST') == '1'
        source = '/laiting/mast3r/fused_points' if fixed_chest else '/laiting/mast3r/points'
        self.create_subscription(PointCloud2, source, self.on_cloud, SENSOR)
        self.get_logger().info('RViz cloud output: ' + topic)

    def on_cloud(self, m):
        if (m.header.frame_id != FRAME or not fresh(stamp_ns(m.header), 1.5) or m.is_bigendian
                or m.point_step != 16 or m.height != 1 or len(m.data) != 16*m.width
                or [(f.name,f.offset) for f in m.fields] != [('x',0),('y',4),('z',8),('rgb',12)]):
            return
        # Preserve the true ray origin for the MoveIt occupancy map.
        m.header.frame_id = self.frame
        self.pub.publish(m)


if __name__ == '__main__':
    rclpy.init()
    try: rclpy.spin(Scene())
    except KeyboardInterrupt: pass
