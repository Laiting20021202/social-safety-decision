"""Check the site-profile handoff using only a loopback ROS graph and mock joints."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

sys.path.insert(0, '/bench')
from registration import CAL, physical_profile_matches, tabletop_start


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '158' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise SystemExit('run in Docker --network none, --tmpfs /results and domain 158')
    CAL.parent.mkdir(parents=True,exist_ok=True)
    CAL.write_text(json.dumps(dict(serial='319522065801',parent='openarm_body_link0',frame='d415_color_optical_frame',
             fit_count=4,holdout_count=3,max_error_m=.002,base_from_camera=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])))
    path=CAL.parent/'tabletop-task.json'
    pose=[.55,0,0,.03,0,0,0]
    record=dict(method='physical_encoders',serial='319522065801',parent='openarm_body_link0',
                active_right_pose=pose,registration_sha256=hashlib.sha256(CAL.read_bytes()).hexdigest())
    path.write_text(json.dumps(record))
    assert tabletop_start().tolist()==pose
    with open('/tmp/tabletop-mock.log','w') as log:
        proc=subprocess.Popen(['ros2','launch','/mock/right.launch.py','mock:=true',f'tabletop_profile:={path}'],
                              stdout=log,stderr=log,start_new_session=True)
        rclpy.init()
        n=rclpy.create_node('check_mock_gateway_profile')
        xml=[]
        n.create_subscription(String,'/robot_description',lambda m:xml.append(m.data),
                              QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        try:
            end=time.monotonic()+15
            while not xml and proc.poll() is None and time.monotonic()<end:
                rclpy.spin_once(n,timeout_sec=.1)
            assert xml and physical_profile_matches(xml[-1]), 'controller did not advertise the measured file'
            hardware=ET.fromstring(xml[-1]).find("./ros2_control[@name='openarm_right_hardware']/hardware")
            assert hardware.findtext('plugin')=='mock_components/GenericSystem'
            assert [float(hardware.findtext(f"param[@name='tabletop_start{i}']")) for i in range(1,8)]==pose
            record['active_right_pose'][0]=.50
            path.write_text(json.dumps(record))
            assert not physical_profile_matches(xml[-1]), 'changed profile must be rejected'
            CAL.write_text(CAL.read_text()+'\n')
            try: tabletop_start()
            except ValueError: pass
            else: raise AssertionError('changed camera registration must invalidate the pose')
            print('Mock NUC advertised all seven angles and matching digest. Pose and calibration changes were rejected.')
        finally:
            n.destroy_node()
            rclpy.shutdown()
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                proc.wait(timeout=8)


if __name__=='__main__':
    main()
