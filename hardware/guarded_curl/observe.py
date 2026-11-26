"""Measure the three-host topics at the workstation without creating a command publisher."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState, PointCloud2
from std_msgs.msg import String

from frames import RGBD, SENSOR, decode, fresh, stamp_ns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--json',type=Path)
    p.add_argument('--chest-only',action='store_true',help='Verify a single chest RGB-D publisher and the fused scene')
    args=p.parse_args()
    rclpy.init()
    n=Node('laiting_three_host_observer')
    seen=defaultdict(dict)
    qualities=[]; regimes=Counter(); names=[]; last={}
    def add(key,stamp):
        if fresh(stamp,2):
            seen[key].setdefault(stamp,time.monotonic())
    def joints(m):
        nonlocal names
        names=list(m.name); add('joint_states',stamp_ns(m.header))
    def rgbd(rgb,depth,info):
        _,mm=decode(rgb,depth,info)
        if np.count_nonzero(mm)>100: add('d415_matched_rgb_depth_info',stamp_ns(info.header))
    def cloud(m,topic):
        if m.point_step==16 and len(m.data)==m.width*16 and m.width>=200:
            add(topic,stamp_ns(m.header))
    def status(m):
        d=json.loads(m.data)
        if d.get('metric') and fresh(d['stamp_ns'],1.5):
            qualities.append([d['points'],d['inlier_fraction'],d['median_error_m'],d.get('fused_points',0)])
    def state(m):
        nonlocal last
        last=json.loads(m.data); regimes[last['reason']]+=1
    descriptions=[]
    n.create_subscription(String,'/robot_description',lambda m: descriptions.append('openarm_right_joint4' in m.data),
                          QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    n.create_subscription(JointState,'/joint_states',joints,SENSOR)
    RGBD(n,rgbd)
    if not args.chest_only:
        n.create_subscription(CompressedImage,'/camera/d435/color/image_raw/compressed',
                              lambda m:add('d435_color',stamp_ns(m.header)),SENSOR)
    for slug in ('mast3r','preview','scene'):
        n.create_subscription(PointCloud2,f'/laiting/{slug}/points',lambda m,s=slug:cloud(m,s),SENSOR)
    n.create_subscription(String,'/laiting/mast3r/status',status,SENSOR)
    n.create_subscription(String,'/laiting/safety/state',state,10)
    start=time.monotonic()
    while time.monotonic()-start<args.seconds:
        rclpy.spin_once(n,timeout_sec=.05)
    def stats(a):
        t=sorted(a.values())
        return {'distinct_frames':len(t),'rate_hz':round((len(t)-1)/(t[-1]-t[0]),2) if len(t)>1 else 0}
    out={'elapsed_s':round(time.monotonic()-start,2),'robot_description':any(descriptions),
         'joint_count':len(names),'topics':{k:stats(a) for k,a in seen.items()},
         'guard_samples':dict(regimes),'latest_guard':last}
    if args.chest_only:
        out['camera_namespaces']=sorted({name.split('/')[2] for name,_ in n.get_topic_names_and_types()
                                         if name.startswith('/camera/') and n.count_publishers(name)>0})
    if qualities:
        q=np.array(qualities)
        out['mast3r']={'frames':len(q),'median_points':int(np.median(q[:,0])),
                       'median_inlier_fraction':round(float(np.median(q[:,1])),4),
                       'median_depth_error_m':round(float(np.median(q[:,2])),4),
                       'median_fused_points':int(np.median(q[:,3]))}
    text=json.dumps(out,indent=2)
    if args.json: args.json.write_text(text+'\n')
    print(text)
    n.destroy_node();rclpy.shutdown()
    required=['joint_states','d415_matched_rgb_depth_info','mast3r']
    if args.chest_only:
        required.append('scene')
        if out['camera_namespaces']!=['d415']:
            raise SystemExit('unexpected set of camera publishers')
    else:
        required.append('d435_color')
    if not (out['robot_description'] and len(names)==16 and qualities and last and
            min(len(seen[k]) for k in required)>2 and max(len(seen[k]) for k in ('preview','scene'))>2):
        raise SystemExit('incomplete live three-host observation')


if __name__=='__main__':
    main()
