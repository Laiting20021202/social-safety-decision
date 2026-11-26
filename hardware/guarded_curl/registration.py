"""Register the D415 to the robot from surveyed RGB-D landmarks, with independent holdouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from frames import FRAME, RGBD, decode, stamp_ns

BASE = 'openarm_body_link0'
SERIAL = '319522065801'
CAL = Path('/results/calibration/d415.json')
# Historical chest calibration from calibration_chest_left_v1_result.json.
# Shown only on the explicitly labelled preview display.
CHEST_PREVIEW = np.array([
    [0.013437746, -0.668401240, 0.743679507, 0.046468091],
    [-0.986168702, -0.131736492, -0.100582240, 0.015518927],
    [0.165199024, -0.732041856, -0.660926625, 0.569121512],
    [0., 0., 0., 1.],
])


def load_registration(path=CAL):
    if os.environ.get('LAITING_CAMERA_PROFILE') == 'isaac_fixed_openarm_v1':
        if os.environ.get('ROS_DOMAIN_ID') != '89':
            raise ValueError('simulated camera registration restricted to Isaac domain 89')
        cfg=json.loads(Path('/results/calibration/isaac.json').read_text())
        mat=np.asarray(cfg['base_from_camera'],float)
        if (cfg.get('method')!='isaac_stage' or cfg.get('profile')!='isaac_fixed_openarm_v1'
                or cfg.get('parent')!=BASE or cfg.get('frame')!=FRAME or mat.shape!=(4,4)
                or cfg.get('optical_sensor_count')!=1
                or cfg.get('camera_prim')!='/World/OpenArm/root_joint/openarm_body_link0/ChestD415'
                or not np.allclose(mat[3],[0,0,0,1]) or not np.allclose(mat[:3,:3].T@mat[:3,:3],np.eye(3))
                or not np.isclose(np.linalg.det(mat[:3,:3]),1)):
            raise ValueError('bad Isaac stage camera transform')
        return mat
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    m = np.array(data['base_from_camera'], dtype=float)
    if (data.get('serial') != SERIAL or data.get('frame') != FRAME or data.get('parent') != BASE
            or data.get('fit_count', 0) < 4 or data.get('holdout_count', 0) < 3
            or not 0 <= data['max_error_m'] <= 0.015 or m.shape != (4, 4) or not np.isfinite(m).all()
            or not np.allclose(m[3], [0, 0, 0, 1]) or not np.isclose(np.linalg.det(m[:3,:3]), 1)
            or not np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3), atol=1e-5)):
        raise ValueError('invalid D415 registration')
    return m


def curl_span():
    if os.environ.get('LAITING_CAMERA_PROFILE') == 'isaac_fixed_openarm_v1':
        if os.environ.get('ROS_DOMAIN_ID') != '89':
            raise ValueError('invalid Isaac domain')
        span = json.loads(Path('/results/calibration/isaac.json').read_text())['curl_degrees']
        if not 0 < span <= 50:
            raise ValueError('unexpected simulated curl amplitude')
        return np.deg2rad(span)
    return np.deg2rad(18.)


def tabletop_start():
    """An explicit, measured work pose is required for a chest-monitored table curl."""
    if os.environ.get('LAITING_CAMERA_PROFILE')=='isaac_fixed_openarm_v1':
        if os.environ.get('ROS_DOMAIN_ID')!='89': raise ValueError('invalid Isaac domain')
        profile=json.loads(Path('/results/calibration/isaac.json').read_text())
    else:
        path=Path('/results/calibration/tabletop-task.json')
        if not path.is_file(): return None
        profile=json.loads(path.read_text())
        if load_registration() is None: raise ValueError('survey the physical chest camera first')
        if (profile.get('method')!='physical_encoders' or profile.get('serial')!=SERIAL
                or profile.get('registration_sha256')!=hashlib.sha256(CAL.read_bytes()).hexdigest()):
            raise ValueError('capture the work pose for the current physical camera registration')
    values=np.asarray(profile.get('active_right_pose',[]),float)
    if values.shape!=(7,) or not np.isfinite(values).all(): raise ValueError('seven measured right joint angles required')
    return values


def physical_profile_matches(description):
    """Match the surveyed file to the profile advertised by the physical gateway."""
    import xml.etree.ElementTree as ET
    path = CAL.parent/'tabletop-task.json'
    if not description or not path.is_file():
        return False
    try:
        profile = ET.fromstring(description).find(
            "./ros2_control[@name='openarm_right_hardware']/hardware/param[@name='tabletop_profile_id']")
    except ET.ParseError:
        return False
    return profile is not None and profile.text == hashlib.sha256(path.read_bytes()).hexdigest()


def capture_pose(output):
    from sensor_msgs.msg import JointState
    from frames import SENSOR, fresh
    import rclpy
    if os.environ.get('ROS_DOMAIN_ID') != '88' or os.environ.get('LAITING_CAMERA_PROFILE'):
        raise ValueError('physical pose capture requires domain 88 with the physical profile')
    if load_registration() is None:
        raise ValueError('solve the physical camera registration first')
    names = [f'openarm_right_joint{i}' for i in range(1, 8)]
    samples = {}
    def collect(msg):
        stamp = stamp_ns(msg.header)
        if not fresh(stamp, .25) or not all(key in msg.name for key in names):
            return
        indices = [msg.name.index(key) for key in names]
        if len(msg.velocity) <= max(indices) or len(msg.position) <= max(indices):
            return
        vel = np.asarray(msg.velocity)[indices]
        pos = np.asarray(msg.position)[indices]
        if np.isfinite(vel).all() and np.isfinite(pos).all() and max(abs(vel)) < .06:
            samples[stamp] = pos
        else:
            samples.clear()
    rclpy.init()
    node = rclpy.create_node('record_tabletop_start')
    try:
        node.create_subscription(JointState, '/joint_states', collect, SENSOR)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and len(samples) < 30:
            rclpy.spin_once(node, timeout_sec=.05)
        if len(samples) < 30 or max(np.ptp(list(samples.values()), axis=0)) > .015:
            raise ValueError('stationary joint feedback required')
        values = np.median(list(samples.values()), axis=0)
        if not -.16 <= values[0] <= .60 or max(abs(values[1:])) > .16 or values[3] < -.02:
            raise ValueError('measured start is outside the supported tabletop envelope')
        record = {'method':'physical_encoders','serial':SERIAL,'parent':BASE,
                  'active_right_pose':values.tolist(), 'last_stamp_ns':max(samples),
                  'registration_sha256':hashlib.sha256(CAL.read_bytes()).hexdigest()}
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(record,indent=2)+'\n')
        temp.replace(output)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def capture(stem):
    import rclpy
    rclpy.init()
    node = rclpy.create_node('d415_survey_capture')
    samples = []
    def add(rgb, depth, info):
        color, mm = decode(rgb, depth, info)
        samples.append((color, mm, np.asarray(info.k).reshape(3,3), stamp_ns(info.header)))
    RGBD(node, add)
    deadline = time.monotonic() + 12
    while len(samples) < 12 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.1)
    if len(samples) < 12:
        raise RuntimeError('12 synchronized D415 frames required')
    color, _, k, stamp = samples[-1]
    np.savez_compressed(str(stem) + '.npz', depth_mm=np.stack([s[1] for s in samples]), k=k, stamp=stamp)
    if not cv2.imwrite(str(stem) + '.jpg', cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
        raise OSError('could not write survey image')
    node.destroy_node()
    rclpy.shutdown()


def solve(manifest, output):
    groups = {'fit': [], 'holdout': []}
    data = json.loads(manifest.read_text())
    seen = set()
    for item in data['landmarks']:
        key = item['id']
        if key in seen or not str(key).strip() or not item['reference'].strip():
            raise ValueError('provide a distinct independently surveyed landmark and its reference')
        seen.add(key)
        uv = np.asarray(item['pixel'])
        base = np.asarray(item['base_m'], dtype=float)
        if uv.shape != (2,) or uv.dtype.kind not in 'iu' or base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError(f'{key}: invalid UV or surveyed point')
        u, v = uv
        with np.load(manifest.parent/item['capture']) as cap:
            depths, k = cap['depth_mm'], cap['k']
            if not (1 <= u < depths.shape[2]-1 and 1 <= v < depths.shape[1]-1):
                raise ValueError(f'{key}: outside the captured image')
            mm = depths[:, v-1:v+2, u-1:u+2]
            valid = mm[(mm > 149) & (mm < 4000)]
            if valid.size < mm.size*.8 or np.ptp(np.percentile(valid, [10,90])) > 12:
                raise ValueError(f'{key}: invalid depth or depth discontinuity')
            z = np.median(valid)*.001
            cam = np.array([(u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z])
            groups[item['role']].append((cam, base))
    if len(groups['fit']) < 4 or len(groups['holdout']) < 3:
        raise ValueError('at least four fit and three independent holdout landmarks required')
    a,b = np.array(groups['fit']).transpose(1,0,2)
    def spread(points):
        return np.linalg.svd(points-points.mean(axis=0), compute_uv=False)[1]
    ac,bc = a-a.mean(axis=0),b-b.mean(axis=0)
    hcam,hbase = np.array(groups['holdout']).transpose(1,0,2)
    if min(spread(a), spread(b), spread(hcam), spread(hbase)) < .045:
        raise ValueError('landmarks must span two dimensions; spread them around the workspace')
    u,_,v = np.linalg.svd(ac.T@bc)
    rot = v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T
    t = b.mean(axis=0)-rot@a.mean(axis=0)
    m=np.eye(4); m[:3,:3]=rot; m[:3,3]=t
    cam=np.concatenate([a,hcam]); base=np.concatenate([b,hbase])
    err=float(np.linalg.norm(cam@rot.T+t-base,axis=1).max())
    if err > .015:
        raise ValueError(f'landmark error {err:.3f} m exceeds 0.015 m')
    record=dict(serial=SERIAL, frame=FRAME, parent=BASE, base_from_camera=m.tolist(),
                fit_count=len(a), holdout_count=len(hcam), max_error_m=err, survey=manifest.name)
    tmp=output.with_suffix('.tmp'); tmp.write_text(json.dumps(record,indent=2)); tmp.replace(output)
    print(f'{output}: {len(a)} fit, {len(hcam)} holdouts, maximum error {err:.4f} m')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('capture').add_argument('stem',type=Path)
    sub.add_parser('capture-pose').add_argument('--output',type=Path,default=CAL.parent/'tabletop-task.json')
    s=sub.add_parser('solve'); s.add_argument('manifest',type=Path); s.add_argument('--output',type=Path,default=CAL)
    args=p.parse_args()
    if args.command=='capture': capture(args.stem)
    elif args.command=='capture-pose': capture_pose(args.output)
    else: solve(args.manifest,args.output)
