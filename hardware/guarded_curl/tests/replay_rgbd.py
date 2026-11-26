"""Replay measured RGB assets with synthetic depth and robot geometry in a network-isolated controller test."""
from array import array
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState, PointCloud2, PointField
from std_msgs.msg import Header, String
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

if os.environ.get('ROS_DOMAIN_ID')!='158' or set(os.listdir('/sys/class/net'))!={'lo'}:
    raise SystemExit('This fixture requires Docker --network none and domain 158')
sys.path.insert(0,'/bench')
from frames import SENSOR, FRAME, ROOT
from registration import CAL, solve

K=np.array([[320.,0.,320.],[0.,320.,240.],[0.,0.,1.]])
R=np.array([[0,0,-1],[1,0,0],[0,-1,0]],float)
T=np.array([1.,-.15,.32])


def simulated_registration():
    dir=CAL.parent
    dir.mkdir(parents=True,exist_ok=True)
    mm=np.full((12,480,640),1000,np.uint16)
    np.savez_compressed(dir/'fixture.npz',depth_mm=mm,k=K,stamp=time.time_ns())
    records=[]
    for i,(u,v) in enumerate([(185,105),(450,110),(185,360),(430,348),(225,172),(405,190),(301,319)]):
        xyz=R@np.array([(u-320)/320,(v-240)/320,1.])+T
        records.append(dict(id=str(i),role='fit' if i<4 else 'holdout',reference='isolated simulated fixture',
                            capture='fixture.npz',pixel=[u,v],base_m=xyz.tolist()))
    manifest=dir/'simulation.json'
    manifest.write_text(json.dumps(dict(landmarks=records)))
    solve(manifest,CAL)


class Frames(Node):
    def __init__(self):
        super().__init__('isolated_rgbd_fixture')
        clear=cv2.imread('/evidence/d415-mount.jpg')
        original=cv2.imread('/evidence/test-inputs/right_hands.jpg')
        if clear is None or original is None: raise RuntimeError('RGB fixtures unavailable')
        # One of the two real photographed hands is projected in front of the mock forearm.
        hand=cv2.resize(original[:,:original.shape[1]//2],(250,270))
        present=clear.copy()
        present[137:407,194:444]=hand
        def encode(ext,pixels):
            ok,buf=cv2.imencode(ext,pixels); assert ok
            return array('B',buf.tobytes())
        self.images=[encode('.jpg',im) for im in (clear,present)]
        self.tf=Buffer(); self.listener=TransformListener(self.tf,self)
        self.info=self.create_publisher(CameraInfo,ROOT+'/color/camera_info',SENSOR)
        self.color=self.create_publisher(CompressedImage,ROOT+'/color/image_raw/compressed',SENSOR)
        self.depth=self.create_publisher(CompressedImage,ROOT+'/aligned_depth_to_color/image_raw/compressedDepth',SENSOR)
        self.cloud=self.create_publisher(PointCloud2,'/laiting/mast3r/points',SENSOR)
        self.metric=self.create_publisher(String,'/laiting/mast3r/status',SENSOR)
        self.create_subscription(JointState,'/joint_states',self.joints,SENSOR)
        self.create_subscription(String,'/laiting/safety/state',self.safety,10)
        self.q=0.; self.phase=0; self.since=0.; self.report=None
        self.create_timer(1/15,self.publish)

    def joints(self,m):
        self.q=m.position[m.name.index('openarm_right_joint4')]

    def safety(self,m):
        s=json.loads(m.data); reason=s['reason']
        if reason!=self.report:
            print('guard',reason,'hands',s['hands'],'q',round(self.q,4),flush=True)
            self.report=reason

    def publish(self):
        now=time.monotonic()
        if self.phase==0 and self.q>.09: self.phase=1; self.since=now
        if self.phase==1 and now-self.since>1.5: self.phase=2
        if self.phase==2 and self.q>.19: self.phase=3; self.since=now
        if self.phase==3 and now-self.since>1.4: self.phase=4
        h=Header(frame_id=FRAME,stamp=self.get_clock().now().to_msg())
        stamp=h.stamp.sec*10**9+h.stamp.nanosec
        if self.phase==3: return
        self.color.publish(CompressedImage(header=h,format='jpeg',data=self.images[self.phase==1]))
        # Render the mock forearm in camera coordinates from the simulated encoders.
        depth=np.full((480,640),1250,np.uint16)
        if self.phase==1: depth[137:407,194:444]=950
        try:
            xyz=[]
            for link in ('openarm_right_link4','openarm_right_link7','openarm_right_fingertip_center'):
                tr=self.tf.lookup_transform('openarm_body_link0',link,Time()).transform.translation
                xyz.append([tr.x,tr.y,tr.z])
            arm=np.vstack([np.linspace(a,b,35) for a,b in zip(xyz,xyz[1:])])
            cam=(arm-T)@R
            for x,y,z in cam:
                u,v=round(320*x/z+320),round(320*y/z+240)
                cv2.circle(depth,(u,v),12,round(1000*z)-10,-1)
        except TransformException:
            pass
        ok,png=cv2.imencode('.png',depth); assert ok
        self.depth.publish(CompressedImage(header=h,format='16UC1; compressedDepth png',data=array('B',png.tobytes())))
        self.info.publish(CameraInfo(header=h,width=640,height=480,k=K.ravel().tolist()))
        data=np.zeros((225,4),dtype='<f4'); data[:,2]=1.6
        m=PointCloud2(header=h,height=1,width=len(data),is_dense=True,point_step=16,row_step=data.nbytes,
                      fields=[PointField(name=k,offset=i*4,datatype=PointField.FLOAT32,count=1)
                              for i,k in enumerate(('x','y','z','rgb'))],data=array('B',data.tobytes()))
        self.cloud.publish(m)
        self.metric.publish(String(data=json.dumps(dict(stamp_ns=stamp,metric=True))))


simulated_registration()
rclpy.init()
rclpy.spin(Frames())
