"""Commission a repeatable camera-driven intrusion trial on the Isaac-only namespace and ROS domain."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

sys.path.insert(0,'/demo/guarded')
from repeat import Repeat


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles',type=int,default=2)
    parser.add_argument('--timeout',type=float,default=110.)
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='89' or os.environ.get('LAITING_CAMERA_PROFILE')!='isaac_fixed_openarm_v1':
        raise SystemExit('requires the Isaac-only station on domain 89')
    if args.cycles<2 or not 20<=args.timeout<=3600:
        raise SystemExit('at least two cycles and timeout 20–3600 s required')
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    folder=Path('/results')
    # Associate evidence from each new trial; do not append to a previous measured run.
    with (folder/'commands.csv').open('w') as output:
        node=Repeat(True,output)
        cmd=node.create_publisher(String,'/laiting/isaac/scenario_command',10)
        mode=[]
        node.create_subscription(String,'/laiting/isaac/scenario',lambda m:mode.append(m.data),10)
        try:
            end=time.monotonic()+35
            while not node.clear() or not node.feedback() or cmd.get_subscription_count()!=1:
                if time.monotonic()>end: raise RuntimeError('readiness timeout: '+str(node.guard))
                node.spin(.1)
            cmd.publish(String(data='start'))
            while 'armed' not in mode:
                if time.monotonic()>end: raise RuntimeError('Isaac did not arm intrusion')
                node.spin(.05)
            node.run(args.cycles,args.timeout,True)
        finally:
            cmd.publish(String(data='finish'))
            node.spin(.4)
            node.destroy_node()
            rclpy.try_shutdown()


if __name__=='__main__':main()
