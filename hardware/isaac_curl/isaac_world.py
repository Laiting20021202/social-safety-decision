"""Fixed OpenArm in PhysX, a textured LibHand mesh, a real RTX RGB-D sensor, and simulator-only evidence."""
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time
from array import array

import numpy as np
from isaacsim import SimulationApp

if os.environ.get('ROS_DOMAIN_ID')!='89':
    raise SystemExit('This simulator runs on ROS domain 89')
app=SimulationApp({'headless':True,'width':640,'height':480,'renderer':'RaytracedLighting',
                   'anti_aliasing':0,'multi_gpu':False})
import cv2
import omni.kit.commands
import omni.physx
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import Float64, Header, String
from openarm_sim.assets import RobotAsset
from openarm_sim.isaac_scene import IsaacSceneBuilder
from layout import BASE_FROM_CAMERA, EYE, ROT, FOCAL, WIDTH, HEIGHT, HAND_ROT, HAND_SCALE, HAND_PARK, HAND_REACH, RIGHT_START, CURL_DEGREES

OUT=Path('/run-data'); MODEL=Path('/model'); HAND=Path('/hand')
SENSOR=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)


def obj_mesh():
    points=[]; normals=[]; uv=[]; triangles=[]; dark=[]
    material='skin'
    obj=HAND/'hand.obj'
    with (obj.open() if obj.exists() else gzip.open(HAND/'hand.obj.gz','rt')) as stream:
        for line in stream:
            tokens=line.split()
            if not tokens: continue
            kind=tokens[0]
            if kind=='v': points.append([float(v) for v in tokens[1:4]])
            elif kind=='vn': normals.append([float(v) for v in tokens[1:4]])
            elif kind=='vt': uv.append([float(v) for v in tokens[1:3]])
            elif kind=='usemtl': material=tokens[1]
            elif kind=='f':
                if material=='blackness': dark.append(len(triangles))
                triangles.append([int(v.split('/')[0])-1 for v in tokens[1:]])
    return np.array(points),np.array(normals),np.array(uv),np.asarray(triangles,np.int32),dark


def surface(stage,path,texture,roughness=.65):
    mat=UsdShade.Material.Define(stage,path)
    shade=UsdShade.Shader.Define(stage,path+'/surface');shade.CreateIdAttr('UsdPreviewSurface')
    shade.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
    uv=UsdShade.Shader.Define(stage,path+'/coord');uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex=UsdShade.Shader.Define(stage,path+'/texture');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
    tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    shade.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shade.ConnectableAPI(),'surface')
    return mat


def make_hand(stage):
    p,n,uv,tri,dark=obj_mesh()
    vertices=(p*HAND_SCALE)@HAND_ROT.T
    root=UsdGeom.Xform.Define(stage,'/World/VisitorHand')
    op=root.AddTranslateOp(); op.Set(Gf.Vec3d(*HAND_PARK))
    mesh=UsdGeom.Mesh.Define(stage,'/World/VisitorHand/Skin')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri),3,np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.flatten()))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(n@HAND_ROT.T,np.float32)))
    mesh.SetNormalsInterpolation('vertex');mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex')
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(surface(stage,'/World/Looks/Skin',HAND/'hand_texture.png',.55))
    # PhysX uses the conservative hull; the evaluation additionally checks the geometry of individual fingers.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    # A tapered, fabric-covered forearm joins the scanned wrist and continues out of view during the reach.
    sleeve_pts=[];sleeve_uv=[];faces=[]
    sides=32
    for k,x in enumerate(np.linspace(.072,.49,26)):
        t=k/25.; cuff=x<.112
        fold=.0015*np.sin(k*2.4) if not cuff else .0008*np.cos(k*3.)
        for j in range(sides+1):
            phi=2*np.pi*j/sides
            ry=.036+.027*t+fold;rz=.031+.025*t+fold
            sleeve_pts.append([x, -.004+ry*np.cos(phi), -.003+rz*np.sin(phi)])
            sleeve_uv.append([j/sides,t])
            if k and j:
                i=k*(sides+1)+j
                faces.extend([[i-sides-2,i-1,i],[i-sides-2,i,i-sides-1]])
    for a in range(1,sides-1):
        faces.append([0,a,a+1]); faces.append([25*(sides+1),25*(sides+1)+a+1,25*(sides+1)+a])
    cloth=np.asarray(sleeve_pts)@HAND_ROT.T;faces=np.asarray(faces,np.int32)
    sleeve=UsdGeom.Mesh.Define(stage,'/World/VisitorHand/Sleeve')
    sleeve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cloth.astype(np.float32)))
    sleeve.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
    sleeve.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten()))
    sleeve.CreateSubdivisionSchemeAttr('none');sleeve.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(sleeve).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(np.asarray(sleeve_uv,np.float32)))
    y,x=np.mgrid[:384,:384]
    weave=3.5*np.cos(x*1.5+y*.7)+3*np.sin(y*1.8)+2*np.sin(x*.19-y*.15)
    fabric=np.clip(np.array([95,77,57])+weave[:,:,None],0,255).astype('uint8')
    cv2.imwrite(str(OUT/'assets/sleeve.png'),fabric)
    UsdShade.MaterialBindingAPI.Apply(sleeve.GetPrim()).Bind(surface(stage,'/World/Looks/WovenCotton',OUT/'assets/sleeve.png',.92))
    UsdPhysics.CollisionAPI.Apply(sleeve.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(sleeve.GetPrim()).CreateApproximationAttr('convexHull')
    np.savez_compressed(OUT/'assets'/'hand_geometry.npz',vertices=np.vstack([vertices,cloth]),faces=np.vstack([tri,faces+len(vertices)]))
    rb=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());rb.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.)
    return op


def build():
    enable_extension('isaacsim.asset.importer.urdf')
    world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/120)
    stage=world.stage
    def box(name,xyz,size,color):
        return world.scene.add(FixedCuboid(prim_path='/World/'+name,name=name,position=np.array(xyz),
                                           size=1.,scale=np.array(size),color=np.array(color)))
    box('Floor',[0,0,-.57],[5,5,.06],[.22,.25,.27])
    box('Stand',[0,0,-.275],[.26,.32,.54],[.22,.25,.29])
    # All four optical corner rays intersect the bench at its measured z=.098 m.
    xmin,xmax,ymin,ymax=.145,1.81,-1.00,.62
    box('Worktable',[(xmin+xmax)/2,(ymin+ymax)/2,.073],[xmax-xmin,ymax-ymin,.048],[.26,.29,.28])
    for i,(x,y) in enumerate((x,y) for x in (xmin+.07,xmax-.07) for y in (ymin+.07,ymax-.07)):
        box('TableLeg'+str(i),[x,y,-.235],[.05,.05,.55],[.25,.28,.30])
    gen=np.random.default_rng(17); yy,xx=np.mgrid[:768,:1024]
    weave=1.6*np.sin(yy*.74)+1.4*np.cos(xx*.81)
    diffuse=2.3*np.sin(xx*.019+np.sin(yy*.025))+1.8*np.sin(yy*.035+xx*.015)
    fabric=np.clip(np.array([101,134,50])+(weave+diffuse+gen.normal(0,1.2,xx.shape))[:,:,None],0,255).astype('uint8')
    tex=OUT/'assets'/'table_mat.png';cv2.imwrite(str(tex),fabric)
    panel=UsdGeom.Mesh.Define(stage,'/World/GreenWorktop')
    panel.CreatePointsAttr([(x,y,.098) for x,y in ((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))])
    panel.CreateFaceVertexCountsAttr([4]);panel.CreateFaceVertexIndicesAttr([0,1,2,3]);panel.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(panel).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set([(0,0),(1,0),(1,1),(0,1)])
    UsdShade.MaterialBindingAPI.Apply(panel.GetPrim()).Bind(surface(stage,'/World/Looks/Mat',tex))
    box('WorkPad',[.61,.004,.101],[.26,.22,.006],[.76,.75,.70])
    box('TrayBottom',[.77,.20,.105],[.19,.13,.012],[.39,.27,.16])
    for j,(dx,dy,sx,sy) in enumerate([(-.09,0,.01,.13),(.09,0,.01,.13),(0,-.06,.18,.01),(0,.06,.18,.01)]):
        box('TrayRim'+str(j),[.77+dx,.20+dy,.13],[sx,sy,.048],[.46,.32,.20])
    box('GaugeBase',[.76,-.098,.119],[.068,.055,.04],[.31,.36,.37])
    box('GaugeTop',[.76,-.098,.151],[.045,.043,.024],[.43,.50,.43])
    dome=UsdLux.DomeLight.Define(stage,'/World/Dome');dome.CreateIntensityAttr(550.)
    key=UsdLux.DistantLight.Define(stage,'/World/Key');key.CreateIntensityAttr(1800.)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(24,-38,18))
    builder=IsaacSceneBuilder()
    builder.robot_config['base_position']=[0,0,0]
    # The fixed-base articulation is built from the same site v1.0 URDF as ros2_control and MoveIt.
    robot=builder._add_robot(world,RobotAsset('urdf',OUT/'assets/openarm.urdf',MODEL))
    hand=make_hand(stage)
    bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/OpenArm')) if p.GetName()=='openarm_body_link0' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies)!=1: raise RuntimeError('expected one OpenArm chest body')
    camera_path=str(bodies[0].GetPath())+'/ChestD415'
    cam=Camera(camera_path,resolution=(WIDTH,HEIGHT),frequency=30)
    q=Rotation.from_matrix(ROT).as_quat()
    cam.set_local_pose(translation=EYE,orientation=q[[3,0,1,2]],camera_axes='ros')
    world.reset(); cam.initialize()
    cam.set_horizontal_aperture(2.);cam.set_vertical_aperture(2.*HEIGHT/WIDTH)
    cam.set_focal_length(2.*FOCAL/WIDTH); cam.set_clipping_range(.045,5.)
    cam.add_distance_to_image_plane_to_frame()
    joints=list(robot.dof_names); iq=joints.index('openarm_right_joint4')
    start=np.zeros(len(joints))
    for j,qstart in enumerate(RIGHT_START,1): start[joints.index(f'openarm_right_joint{j}')]=qstart
    robot.set_joint_positions(start);robot.set_joint_velocities(np.zeros_like(start))
    ctl=robot.get_articulation_controller()
    ctl.set_gains(kps=np.full(len(start),450.),kds=np.full(len(start),70.))
    ctl.apply_action(ArticulationAction(joint_positions=start))
    return world,robot,hand,cam,joints,iq,start


def main():
    world,robot,hand,cam,names,iq,targets=build()
    rclpy.init();node=rclpy.create_node('laiting_isaac_physics')
    enc=node.create_publisher(JointState,'/laiting/isaac/encoders',SENSOR)
    rgb=node.create_publisher(CompressedImage,'/camera/d415/color/image_raw/compressed',SENSOR)
    depth=node.create_publisher(CompressedImage,'/camera/d415/aligned_depth_to_color/image_raw/compressedDepth',SENSOR)
    info=node.create_publisher(CameraInfo,'/camera/d415/color/camera_info',SENSOR)
    phasepub=node.create_publisher(String,'/laiting/isaac/scenario',10)
    mode='idle'; started=0.; intruded=False; reach_at=0.; frame_no=0
    guard={}; j4=float(targets[iq]); lastseen=0.
    contacts=[]
    def contact(headers,data):
        if mode in ('idle','complete','preview'): return
        for header in headers:
            if not header.num_contact_data: continue
            a,b=(str(PhysicsSchemaTools.intToSdfPath(i)) for i in (header.actor0,header.actor1))
            if ('VisitorHand' in a and '/OpenArm/' in b) or ('VisitorHand' in b and '/OpenArm/' in a):
                contacts.append(dict(wall_ns=time.time_ns(),a=a,b=b))
    subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
    def command(m):
        nonlocal j4,lastseen
        if np.isfinite(m.data) and -.025<=m.data<=RIGHT_START[3]+np.deg2rad(CURL_DEGREES)+.025:
            j4=m.data; lastseen=time.monotonic()
    def state(m):
        nonlocal guard
        try: guard=json.loads(m.data)
        except ValueError: guard={}
    def scenario(m):
        nonlocal mode,started,intruded,frame_no
        if m.data=='start' and mode in ('idle','complete'):
            contacts.clear();mode='armed';started=time.monotonic();intruded=False;frame_no=0
        elif m.data=='preview' and mode in ('idle','complete'):
            mode='preview';started=time.monotonic()
        elif m.data=='finish': mode='complete'
    node.create_subscription(String,'/laiting/isaac/scenario_command',scenario,10)
    node.create_subscription(Float64,'/laiting/isaac/elbow_target',command,1)
    node.create_subscription(String,'/laiting/safety/state',state,10)
    phase_prev='';pos=HAND_PARK.copy()
    reg=OUT/'calibration';reg.mkdir(exist_ok=True)
    tmp=reg/'isaac.tmp'
    tmp.write_text(json.dumps(dict(method='isaac_stage',profile='isaac_fixed_openarm_v1',
          parent='openarm_body_link0',frame='d415_color_optical_frame',base_from_camera=BASE_FROM_CAMERA.tolist(),
          camera_prim=cam.prim_path, optical_sensor_count=len([p for p in Usd.PrimRange(world.stage.GetPrimAtPath('/World')) if p.IsA(UsdGeom.Camera)]),
          table_top_m=.098, renderer='Isaac RTX', active_right_pose=RIGHT_START, curl_degrees=CURL_DEGREES)))
    tmp.replace(reg/'isaac.json')
    video=None; telemetry=None; frames_file=None; last_render=-1; last_image=0.; frame_time=0.
    print('Isaac ready. Actual articulation: '+str(names),flush=True)
    while app.is_running():
        for _ in range(3): rclpy.spin_once(node,timeout_sec=0.)
        now=time.monotonic()
        q=robot.get_joint_positions(); vel=robot.get_joint_velocities()
        if q is None: continue
        if lastseen and now-lastseen>.35: j4=targets[iq]
        targets[iq]=np.clip(j4,targets[iq]-.14/120,targets[iq]+.14/120)
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=targets))
        if mode=='armed' and not intruded and q[iq]>.078 and vel[iq]>.003:
            mode='entering';reach_at=now;intruded=True
        if mode in ('entering','present','withdrawing'):
            t=now-reach_at
            if t<.95: frac=t/.95;mode='entering'
            elif t<4.15: frac=1.;mode='present'
            elif t<5.2: frac=1-(t-4.15)/1.05;mode='withdrawing'
            else: frac=0.;mode='withdrawn'
            frac=float(np.clip(frac,0,1)); frac=frac*frac*(3-2*frac)
            pos=HAND_PARK+(HAND_REACH-HAND_PARK)*frac
        elif mode=='preview' and now-started<8.: pos=HAND_REACH.copy()
        else:
            if mode=='preview': mode='idle'
            pos=HAND_PARK.copy()
        hand.Set(Gf.Vec3d(*pos))
        do_render=now-last_image>=.060
        world.step(render=do_render)
        q=robot.get_joint_positions(); vel=robot.get_joint_velocities()
        h=Header(stamp=node.get_clock().now().to_msg())
        enc.publish(JointState(header=h,name=names,position=q.astype(float).tolist(),velocity=vel.astype(float).tolist()))
        if mode!=phase_prev:
            print('scenario '+mode,flush=True);phasepub.publish(String(data=mode));phase_prev=mode
        if mode not in ('idle','complete','preview'):
            if telemetry is None:
                telemetry=(OUT/'physics.csv').open('w',newline='',buffering=1)
                writer=csv.writer(telemetry);writer.writerow(['wall_ns','sim_time','phase','hand_x','hand_y','hand_z','command','guard','hands','guard_frame_ns',*names])
                frames_file=(OUT/'frame-times.csv').open('w',newline='')
                frame_writer=csv.writer(frames_file); frame_writer.writerow(['frame','wall_ns'])
                video=cv2.VideoWriter(str(OUT/'isaac-raw.avi'),cv2.VideoWriter_fourcc(*'MJPG'),15.,(WIDTH,HEIGHT))
                if not video.isOpened(): raise RuntimeError('cannot record the chest camera video')
            writer.writerow([time.time_ns(),world.current_time,mode,*pos,float(targets[iq]),guard.get('reason','unknown'),guard.get('hands',0),guard.get('stamp_ns',0),*q])
        elif mode=='complete' and telemetry is not None:
            telemetry.close();telemetry=None;frames_file.close();video.release();video=None
            (OUT/'physx-contacts.json').write_text(json.dumps(dict(count=len(contacts),events=contacts))+'\n')
        if not do_render: continue
        last_image=now
        fr=cam.get_current_frame()
        img=fr.get('rgba');z=fr.get('distance_to_image_plane')
        if img is None or z is None or np.shape(z)!=(HEIGHT,WIDTH):continue
        serial=fr.get('rendering_frame',-1)
        if serial==last_render:continue
        last_render=serial
        stamp=Header(stamp=h.stamp,frame_id='d415_color_optical_frame')
        bgr=cv2.cvtColor(img[:,:,:3],cv2.COLOR_RGB2BGR)
        valid=np.isfinite(z)&(z>.045)&(z<5)
        mm=np.zeros((HEIGHT,WIDTH),np.uint16);mm[valid]=(z[valid]*1000).round().astype(np.uint16)
        ok,im=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,85]);ok2,dp=cv2.imencode('.png',mm,[cv2.IMWRITE_PNG_COMPRESSION,2])
        if not ok or not ok2:continue
        rgb.publish(CompressedImage(header=stamp,format='rgb8; jpeg compressed bgr8',data=array('B',im.tobytes())))
        depth.publish(CompressedImage(header=stamp,format='16UC1; compressedDepth png',data=array('B',bytes(12)+dp.tobytes())))
        intr=cam.get_intrinsics_matrix()
        info.publish(CameraInfo(header=stamp,width=WIDTH,height=HEIGHT,k=intr.ravel().tolist(),distortion_model='plumb_bob',d=[0.]*5))
        if now-frame_time>3:
            cv2.imwrite(str(OUT/'latest.jpg'),bgr);frame_time=now
        if mode=='preview': cv2.imwrite(str(OUT/'hand-preview.jpg'),bgr)
        if video is not None:
            frame_writer.writerow([frame_no,h.stamp.sec*10**9+h.stamp.nanosec])
            frame_no+=1;video.write(bgr)
            if mode=='present' and frame_no%5==0: cv2.imwrite(str(OUT/'hand-in-workspace.jpg'),bgr)
    node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally: app.close()
