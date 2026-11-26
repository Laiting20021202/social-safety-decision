"""Observe the D415 and the swept right forearm. Publish a fresh, explicit motion permission."""
from itertools import product
import json
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformException, TransformListener
from realtime_safety.edgetam_tracker.hand_semantic_gate import MediaPipeHandDetector, MediaPipeHandDetectorConfig

from frames import FRAME, RGBD, SENSOR, decode, fresh, stamp_ns
from registration import BASE, curl_span, load_registration, tabletop_start

FRAME_AGE = 0.36
DWELL = 1.3
RADIUS = .18
OFFSETS = np.array(list(product((-1,1), repeat=3))) * RADIUS


def visibility(depth, k, extrinsic, elbow, wrist, tip, axis):
    # Cover every intermediate pose of a +/- 0.46 rad curl, not only the current TCP.
    arm = np.vstack((np.linspace(elbow,wrist,5), np.linspace(wrist,tip,4)))
    angles = np.linspace(-.46,.46,9)
    swept = np.vstack([Rotation.from_rotvec(axis*a).apply(arm-elbow)+elbow for a in angles])
    def project(points):
        cam = (points - extrinsic[:3,3]) @ extrinsic[:3,:3]
        if np.min(cam[:,2]) < .15:
            raise ValueError('workspace_behind_or_too_close')
        return np.column_stack((cam[:,0]*k[0,0]/cam[:,2]+k[0,2],
                                cam[:,1]*k[1,1]/cam[:,2]+k[1,2])), cam[:,2]
    uv,_ = project((swept[:,None,:] + OFFSETS).reshape(-1,3))
    h,w = depth.shape
    if (uv.min(axis=0) < 5).any() or (uv.max(axis=0) >= [w-5,h-5]).any():
        raise ValueError('workspace_out_of_view')
    uv,z = project(arm)
    for (u,v), expected in zip(uv.round().astype(int), z):
        patch = depth[v-5:v+6,u-5:u+6]
        zs = patch[patch>0]*.001
        if len(zs) < 25 or not np.any(np.abs(zs-expected) <= .11) or np.median(zs) < expected-.14:
            raise ValueError('arm_not_visible_or_occluded')
    return swept


def tabletop_visibility(depth,k,extrinsic,elbow,wrist,tip,axis,q,rest,span):
    """Protect the moving tool over the visible table. The chest-mounted camera sits in front of the shoulder."""
    angles=np.linspace(rest-q,rest+span-q,17)
    tool=np.vstack([Rotation.from_rotvec(axis*a).apply(tip-elbow)+elbow for a in angles])
    def project(points):
        cam=(points-extrinsic[:3,3])@extrinsic[:3,:3]
        if np.min(cam[:,2])<.13: raise ValueError('tool_behind_camera')
        uv=np.column_stack((cam[:,0]*k[0,0]/cam[:,2]+k[0,2],cam[:,1]*k[1,1]/cam[:,2]+k[1,2]))
        return uv,cam[:,2]
    corners=np.array(list(product((-.025,.025),repeat=3)))
    uv,_=project((tool[:,None,:]+corners).reshape(-1,3))
    h,w=depth.shape
    if (uv.min(0)<3).any() or (uv.max(0)>=[w-3,h-3]).any():raise ValueError('tool_path_out_of_view')
    uv,z=project(np.asarray([tip]))
    u,v=uv[0].round().astype(int)
    patch=depth[v-8:v+9,u-8:u+9]; zs=patch[patch>0]*.001
    if len(zs)<40 or not np.any(np.abs(zs-z[0])<.085):
        raise ValueError('tool_not_visible_or_occluded')
    # All forearm-to-tip points take part in the proximity test even where the elbow is off-screen.
    arm=np.linspace(elbow,tip,11)
    return np.vstack([Rotation.from_rotvec(axis*a).apply(arm-elbow)+elbow for a in angles])


def hand_in_workspace(detection, mm, k, extrinsic, swept):
    if detection.mask is None:
        return True
    # Carry through all supported foreground depths; reject holes conservatively.
    mask = np.asarray(detection.mask, dtype=bool)
    if mask.shape != mm.shape:
        return True
    v,u = np.nonzero(mask)
    z=mm[v,u].astype(float)*.001
    valid=(z>.15)&(z<4.)
    if len(z)<10 or valid.mean()<.80:
        return True
    u,v,z=u[valid][::4],v[valid][::4],z[valid][::4]
    camera=np.column_stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1],z))
    base=camera@extrinsic[:3,:3].T+extrinsic[:3,3]
    return any(np.any(np.sum((base-point)**2,axis=1) <= RADIUS**2) for point in swept)


class Guard(Node):
    def __init__(self):
        super().__init__('laiting_hand_guard')
        self.tf=Buffer(); self.listener=TransformListener(self.tf,self)
        self.cal=load_registration()
        self.table_pose=tabletop_start()
        self.span=curl_span()
        self.elbow=None
        self.create_subscription(JointState,'/joint_states',self.joints,SENSOR)
        self.detector=MediaPipeHandDetector(MediaPipeHandDetectorConfig(maximum_hands=4))
        self.detector.load()
        self.status=self.create_publisher(String,'/laiting/safety/state',10)
        self.marker=self.create_publisher(Marker,'/laiting/safety/marker',1)
        self.cloudstamp=0
        self.metricstamp=0
        self.last=0
        self.checked_at=0.
        self.clear_since=None
        self.reason='starting'
        self.hand_count=0
        self.permitted=False
        self.create_subscription(PointCloud2,'/laiting/mast3r/points',self.cloud,SENSOR)
        self.create_subscription(String,'/laiting/mast3r/status',self.metric,SENSOR)
        self.frames=RGBD(self,self.images)
        self.create_timer(.08,self.publish)

    def joints(self,m):
        if 'openarm_right_joint4' in m.name and fresh(stamp_ns(m.header),.25):
            self.elbow=(m.position[m.name.index('openarm_right_joint4')],stamp_ns(m.header))

    def cloud(self,m):
        if m.header.frame_id==FRAME and m.width>=200 and fresh(stamp_ns(m.header),1.4):
            self.cloudstamp=max(self.cloudstamp,stamp_ns(m.header))

    def metric(self,m):
        try:
            d=json.loads(m.data)
            if not d.get('metric'):
                # Rejected per-frame fits never update the map. The previous independently
                # verified sample can cover a single missing frame until the existing age limit.
                if d.get('reason') not in ('initializing','tracking_lost','relocalizing') and fresh(self.metricstamp,1.4):
                    return
                self.metricstamp=0
                self.clear_since=None
                self.permitted=False
            else:
                self.metricstamp=int(d['stamp_ns'])
        except (KeyError,ValueError,TypeError):
            self.metricstamp=0
            self.clear_since=None
            self.permitted=False

    def images(self,rgb_msg,depth_msg,info):
        stamp=stamp_ns(info.header)
        prev=self.last
        self.last=stamp
        self.hand_count=0
        reason='clear'
        try:
            rgb,mm=decode(rgb_msg,depth_msg,info)
            if np.percentile(rgb[::8,::8],95)-float(np.percentile(rgb[::8,::8],5))<15:
                raise ValueError('low_image_contrast')
            hands=self.detector.infer_rgb(rgb)
            self.hand_count=len(hands)
            if self.cal is None:
                raise ValueError('hand_seen_unregistered' if hands else 'unverified_registration')
            poses=[]
            for link in ('openarm_right_link4','openarm_right_link7','openarm_right_fingertip_center'):
                pose=self.tf.lookup_transform(BASE,link,Time())
                if abs(stamp_ns(pose.header)-stamp)>220_000_000: raise ValueError('stale_arm_tf')
                t=pose.transform.translation
                poses.append(np.array([t.x,t.y,t.z]))
                if len(poses)==1:
                    q=pose.transform.rotation
                    axis=Rotation.from_quat([q.x,q.y,q.z,q.w]).apply([0.,1.,0.])
            k=np.asarray(info.k).reshape(3,3)
            if self.table_pose is None:
                swept=visibility(mm,k,self.cal,*poses,axis)
            else:
                if self.elbow is None or abs(stamp-self.elbow[1])>250_000_000: raise ValueError('stale_elbow')
                swept=tabletop_visibility(mm,k,self.cal,*poses,axis,self.elbow[0],self.table_pose[3],self.span)
            if any(hand_in_workspace(h,mm,k,self.cal,swept) for h in hands):
                raise ValueError('hand_in_workspace')
            if not fresh(self.cloudstamp,1.4) or not fresh(self.metricstamp,1.4):
                raise ValueError('mast3r_unavailable')
        except (ValueError,RuntimeError,cv2.error,TransformException) as exc:
            reason=str(exc)
        if not fresh(stamp,FRAME_AGE):
            reason='stale_rgbd'
        now=time.monotonic()
        if stamp-prev>int(FRAME_AGE*1e9) or now-self.checked_at>FRAME_AGE:
            self.clear_since=None
        self.checked_at=now
        if reason!='clear':
            self.clear_since=None
        elif self.clear_since is None:
            self.clear_since=now
        self.permitted=self.clear_since is not None and now-self.clear_since>=DWELL
        self.reason=reason
        self.publish()

    def publish(self):
        reason=self.reason
        if not fresh(self.last,FRAME_AGE) or time.monotonic()-self.checked_at>FRAME_AGE:
            reason='no_fresh_rgbd'
        elif reason=='clear' and (not fresh(self.cloudstamp,1.4) or not fresh(self.metricstamp,1.4)):
            reason='mast3r_unavailable'
        if reason!='clear':
            self.permitted=False
            self.clear_since=None
        elif not self.permitted:
            reason='clear_dwell'
        m=Marker()
        m.header.frame_id=BASE
        m.header.stamp=self.get_clock().now().to_msg()
        m.ns='right_curl_interlock'; m.id=0; m.type=Marker.TEXT_VIEW_FACING
        m.pose.orientation.w=1.
        m.pose.position.z=.89; m.scale.z=.043; m.color.a=1.
        m.color.r,m.color.g,m.color.b=(.24,.95,.52) if self.permitted else (1.,.76,.32)
        m.text=('CLEAR' if self.permitted else 'HOLD')+' | '+reason.replace('_',' ')
        self.marker.publish(m)
        self.status.publish(String(data=json.dumps(dict(permit=self.permitted,reason=reason,
            stamp_ns=self.last, hands=self.hand_count, registered=self.cal is not None,
            cloud_stamp_ns=self.cloudstamp))))


if __name__=='__main__':
    rclpy.init()
    node=Guard()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.detector.close()
        node.destroy_node()
        rclpy.try_shutdown()
