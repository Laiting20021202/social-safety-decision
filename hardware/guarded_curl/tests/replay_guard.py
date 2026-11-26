"""Isolated controller fixture: inject a hand stop, then later interrupt the guard heartbeat."""
import json
import os
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

if os.environ.get('ROS_DOMAIN_ID') != '158' or set(os.listdir('/sys/class/net')) != {'lo'}:
    raise SystemExit('This fixture requires Docker --network none and domain 158')


class Replay(Node):
    def __init__(self):
        super().__init__('isolated_guard_fixture')
        self.q=0.
        self.t=None
        self.phase=0
        self.pub=self.create_publisher(String,'/laiting/safety/state',10)
        self.create_subscription(JointState,'/joint_states',self.joint,10)
        self.create_timer(.04,self.timer)

    def joint(self,m):
        self.q=m.position[m.name.index('openarm_right_joint4')]

    def timer(self):
        now=time.monotonic()
        if self.phase==0 and self.q>.09: self.phase=1; self.t=now
        if self.phase==1 and now-self.t>2.2: self.phase=2
        if self.phase==2 and self.q>.19: self.phase=3; self.t=now
        if self.phase==3 and now-self.t>2.6: self.phase=4
        if self.phase==3 and now-self.t<1.3: return
        reason='clear'
        if self.phase==1: reason='hand_in_workspace' if now-self.t<.9 else 'clear_dwell'
        if self.phase==3: reason='clear_dwell'
        self.pub.publish(String(data=json.dumps(dict(permit=reason=='clear',registered=True,
                         reason=reason,stamp_ns=time.time_ns()))))


rclpy.init()
rclpy.spin(Replay())
