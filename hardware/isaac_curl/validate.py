"""Independent triangle-mesh clearances on each recorded PhysX step and interruption acceptance checks."""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def transform(elem):
    matrix=np.eye(4)
    if elem is not None:
        matrix[:3,3]=np.fromstring(elem.get('xyz','0 0 0'),sep=' ')
        matrix[:3,:3]=Rotation.from_euler('xyz',np.fromstring(elem.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return matrix


def surface(vertices,faces):
    model=fcl.BVHModel()
    model.beginModel(len(faces),len(vertices));model.addSubModel(vertices,faces);model.endModel()
    return fcl.CollisionObject(model)


class Geometry:
    def __init__(self,folder):
        root=ET.parse(folder/'assets/openarm.urdf').getroot()
        self.links=[]
        for link in root.findall('link'):
            for col in link.findall('collision'):
                mesh=col.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('unsupported collider on '+link.get('name'))
                geom=trimesh.load_mesh(mesh.get('filename'),process=False)
                verts=np.asarray(geom.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                tr=transform(col.find('origin')); verts=verts@tr[:3,:3].T+tr[:3,3]
                self.links.append((link.get('name'),surface(verts,np.asarray(geom.faces)),verts.min(0),verts.max(0)))
        joints={}
        for j in root.findall('joint'):
            p=j.find('parent').get('link');child=j.find('child').get('link')
            ax=j.find('axis');axis=np.fromstring(ax.get('xyz'),sep=' ') if ax is not None else None
            joints.setdefault(p,[]).append((j.get('name'),j.get('type'),child,axis,transform(j.find('origin')),j.find('mimic')))
        self.order=[]
        def visit(p):
            for j in joints.get(p,[]):self.order.append((p,*j));visit(j[2])
        visit('world')
        hand=np.load(folder/'assets/hand_geometry.npz')
        v=hand['vertices'];self.hand=surface(v,hand['faces']);self.hmin=v.min(0);self.hmax=v.max(0)
        self.request=fcl.DistanceRequest(enable_nearest_points=False)
        self.creq=fcl.CollisionRequest(num_max_contacts=1)
        self.tip_z=None

    def distances(self,row):
        frames={'world':np.eye(4)}
        for p,name,kind,c,axis,origin,mimic in self.order:
            t=frames[p]@origin
            if kind in ('continuous','revolute','prismatic'):
                q=float(row[name]) if name in row else None
                if q is None and mimic is not None:
                    q=float(row[mimic.get('joint')])*float(mimic.get('multiplier',1))+float(mimic.get('offset',0))
                if q is None:raise ValueError('missing PhysX DOF: '+name)
                if kind=='prismatic':t[:3,3]+=t[:3,:3]@(axis*q)
                else:t[:3,:3]=t[:3,:3]@Rotation.from_rotvec(axis*q).as_matrix()
            frames[c]=t
        self.tip_z=frames['openarm_right_fingertip_center'][2,3]
        offset=np.array([float(row['hand_'+a]) for a in 'xyz'])
        self.hand.setTranslation(offset)
        # Prune only on conservative transformed axis-aligned boxes.
        order=[]
        for name,obj,lo,hi in self.links:
            m=frames[name];r=m[:3,:3];center=r@((lo+hi)/2)+m[:3,3];rad=np.abs(r)@((hi-lo)/2)
            gap=np.maximum(0,np.maximum(self.hmin+offset-(center+rad),center-rad-(self.hmax+offset)))
            order.append((np.linalg.norm(gap),name,obj,m))
        shortest=float('inf');closest=''
        for bound,name,obj,m in sorted(order,key=lambda a:a[0]):
            if bound>=shortest: break
            obj.setTransform(fcl.Transform(m[:3,:3],m[:3,3]))
            d=fcl.distance(self.hand,obj,self.request,fcl.DistanceResult())
            if d<=1e-7 or fcl.collide(self.hand,obj,self.creq,fcl.CollisionResult()):
                return 0.,name
            if d<shortest: shortest,closest=d,name
        return float(shortest),closest


def init_worker(folder):
    global geometry
    geometry = Geometry(folder)


def measure(row):
    clearance, link = geometry.distances(row)
    return clearance, link, geometry.tip_z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=Path('/results'))
    args = parser.parse_args()
    folder = args.folder
    with (folder/'physics.csv').open() as file:
        rows = list(csv.DictReader(file))
    with (folder/'commands.csv').open() as file:
        decisions = list(csv.DictReader(file))
    if not rows or not decisions:
        raise SystemExit('missing joint measurements or controller decisions')
    cal = json.loads((folder/'calibration/isaac.json').read_text())
    hand = [r for r in rows if r['guard']=='hand_in_workspace' and int(r['hands'])>0]
    present = [r for r in rows if r['phase']=='present']
    holds = [r for r in decisions if r['event']=='hold' and r['reason']=='hand_in_workspace']
    if not (hand and present and holds):
        raise SystemExit('missing an observed hand, presence interval or camera-triggered hold')
    first = int(present[0]['wall_ns'])
    last = int(present[-1]['wall_ns'])
    hold = int(holds[0]['wall_ns'])
    withdrawn = next(int(r['wall_ns']) for r in rows if r['phase']=='withdrawn')
    replanned = [r for r in decisions if r['event']=='plan' and int(r['wall_ns'])>withdrawn]
    resumed = bool(replanned) and any(r['event']=='stream' and int(r['wall_ns'])>int(replanned[0]['wall_ns']) for r in decisions)
    began = next(int(r['wall_ns']) for r in rows if r['phase']=='entering')
    moving_in_hold = [r for r in decisions if r['event'] in ('step','stream') and hold<int(r['wall_ns'])<=last]
    measured = []
    nearest = []
    heights = []
    with Pool(4, initializer=init_worker, initargs=(folder,)) as pool, (folder/'clearance.csv').open('w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['wall_ns','phase','clearance_m','nearest_robot_link','tcp_height_m'])
        for row,(distance,link,z) in zip(rows,pool.imap(measure, rows, chunksize=96)):
            measured.append(distance)
            nearest.append(link)
            heights.append(z)
            writer.writerow([row['wall_ns'],row['phase'],round(distance,7),link,round(float(z),7)])
    q = 'openarm_right_joint4'
    enc = np.array([float(r[q]) for r in rows])
    interval = [float(r[q]) for r in present if int(r['wall_ns'])>first+250_000_000]
    steps = np.diff([float(r['sim_time']) for r in rows])
    reached = [float(r['encoder_rad']) for r in decisions if r['event']=='reached']
    count = sum(r['event']=='cycle' for r in decisions)
    checks = {
        'single_chest_camera': cal['optical_sensor_count']==1 and cal['camera_prim'].endswith('/openarm_body_link0/ChestD415'),
        'semantic_hold_before_full_reach': began<=hold<=first,
        'held_through_hand_presence': not moving_in_hold and bool(interval) and np.ptp(interval)<.025,
        'replanned_from_encoders_after_withdrawal': resumed,
        'completed_two_measured_cycles': count>=2 and len(reached)==count*2,
        'reached_configured_high_pose': max(enc)-enc[0]>=np.deg2rad(cal['curl_degrees'])-.035,
        'returned_to_start': abs(enc[-1]-enc[0])<.035,
        'all_physics_steps_recorded': min(steps)>0 and max(steps)<.012,
        'all_mesh_separations_above_15_mm': min(measured)>.015,
        'zero_physx_contact_reports': json.loads((folder/'physx-contacts.json').read_text())['count']==0,
    }
    idx = int(np.argmin(measured))
    state = {
        'passed': all(checks.values()), 'checks': {k: bool(v) for k,v in checks.items()},
        'cycles': count, 'measured_steps': len(rows), 'physics_step_s': float(np.median(steps)),
        'max_physics_step_s': float(max(steps)), 'mesh_contact_steps': int(np.sum(np.asarray(measured)<=1e-7)),
        'physx_hand_robot_contacts': json.loads((folder/'physx-contacts.json').read_text())['count'],
        'minimum_mesh_clearance_m': measured[idx], 'link_at_minimum': nearest[idx],
        'phase_at_minimum': rows[idx]['phase'], 'detected_hand_source_frames': len({r['guard_frame_ns'] for r in hand}),
        'hold_before_full_reach_s': (first-hold)*1e-9,
        'hand_present_duration_s': (last-first)*1e-9,
        'elbow_travel_while_hand_present_deg': float(np.rad2deg(np.ptp(interval))) if interval else None,
        'requested_elbow_travel_deg': cal['curl_degrees'],
        'elbow_start_deg': float(np.rad2deg(enc[0])), 'elbow_peak_deg': float(np.rad2deg(enc.max())),
        'elbow_actual_travel_deg': float(np.rad2deg(enc.max()-enc[0])),
        'tcp_start_height_m': heights[0], 'tcp_peak_height_m': max(heights),
        'tcp_lift_m': max(heights)-heights[0], 'encoders_at_leg_ends_rad': reached,
    }
    report = json.dumps(state,indent=2)+'\n'
    (folder/'validation.json').write_text(report)
    print(report,flush=True)
    if not state['passed']:
        raise SystemExit('trial did not satisfy the recorded checks')


if __name__=='__main__':
    main()
