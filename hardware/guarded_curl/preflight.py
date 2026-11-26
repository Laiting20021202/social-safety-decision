"""Read ROS endpoints and measurements for a tabletop curl without opening the command channel."""
import argparse
import json
import os
import time

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import JointState, PointCloud2
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from frames import RGBD, SENSOR, decode, fresh, stamp_ns
from registration import curl_span, load_registration, physical_profile_matches, tabletop_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=8)
    args = parser.parse_args()
    if not 2 <= args.seconds <= 60:
        parser.error('choose a 2–60 second observation')
    domain = os.environ.get('ROS_DOMAIN_ID')
    simulation = os.environ.get('LAITING_CAMERA_PROFILE') == 'isaac_fixed_openarm_v1'
    if domain != ('89' if simulation else '88'):
        raise SystemExit('unexpected domain for this backend')
    rclpy.init()
    node = rclpy.create_node('laiting_passive_preflight')
    stamps = {'camera': 0, 'scene': 0, 'joint': 0}
    angles = []
    last_guard = {}
    description = []
    guard_time = 0.
    def joint(m):
        names = [f'openarm_right_joint{i}' for i in range(1,8)]
        pos = dict(zip(m.name, m.position))
        if all(s in pos and np.isfinite(pos[s]) for s in names):
            angles[:] = [pos[s] for s in names]
            stamps['joint'] = stamp_ns(m.header)
    def rgbd(rgb, depth, info):
        _, mm = decode(rgb,depth,info)
        if np.count_nonzero(mm) > 100:
            stamps['camera'] = stamp_ns(info.header)
    def cloud(m):
        if m.header.frame_id=='laiting_d415_registered_optical' and m.width>=200:
            stamps['scene'] = stamp_ns(m.header)
    def guard(m):
        nonlocal guard_time
        last_guard.clear()
        last_guard.update(json.loads(m.data))
        guard_time = time.monotonic()
    RGBD(node,rgbd)
    node.create_subscription(JointState,'/joint_states',joint,SENSOR)
    node.create_subscription(PointCloud2,'/laiting/scene/points',cloud,SENSOR)
    node.create_subscription(String,'/laiting/safety/state',guard,10)
    node.create_subscription(String,'/robot_description',lambda msg:description.append(msg.data),
                             QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    inventory = node.create_client(ListControllers,'/controller_manager/list_controllers')
    call = inventory.call_async(ListControllers.Request()) if inventory.wait_for_service(timeout_sec=2.) else None
    end = time.monotonic()+args.seconds
    while time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=.05)
    ctrl = {v.name:v.state for v in call.result().controller} if call and call.done() and call.result() else {}
    expected = tabletop_start()
    if expected is None:
        expected = np.zeros(7)
    in_pose = bool(angles and np.max(np.abs(np.array(angles)-expected))<=.16)
    if not simulation:
        if expected[0]>.16:
            in_pose = bool(angles and description and physical_profile_matches(description[-1])
                           and -.16<=angles[0]<=.60 and max(map(abs,angles[1:]))<=.16
                           and np.max(np.abs(np.asarray(angles)-expected))<=.04)
        else:
            in_pose = in_pose and max(map(abs,angles))<=.16
    checks = {
        'camera_registered': load_registration() is not None,
        'chest_rgbd_fresh': fresh(stamps['camera'],.36),
        'joint_feedback_fresh': fresh(stamps['joint'],.25),
        'registered_cloud_fresh': fresh(stamps['scene'],1.4),
        'guard_permits': last_guard.get('permit') is True and fresh(last_guard.get('stamp_ns',0),.36)
                         and time.monotonic()-guard_time<.32,
        'arm_at_backend_start_pose': in_pose,
        'elbow_controller_available': ctrl.get('right_joint_trajectory_controller')=='inactive',
        'exclusive_command_topic_available': node.count_publishers('/right_joint_trajectory_controller/joint_trajectory')==0,
    }
    print(json.dumps({'ready': all(checks.values()), 'domain': domain, 'checks':checks,
                      'guard_reason':last_guard.get('reason'), 'controllers':ctrl, 'right_joints_rad':angles,
                      'planned_curl_deg':float(np.rad2deg(curl_span()))},indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
