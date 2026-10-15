#!/usr/bin/env python3
"""Gazebo-only observer camera for video. Never connected to perception."""
import argparse, math
import rclpy
from gazebo_msgs.srv import SpawnEntity

p=argparse.ArgumentParser()
p.add_argument('--position',type=float,nargs=3,default=[1.0,-1.25,.95])
p.add_argument('--target',type=float,nargs=3,default=[-.12,0,.4])
a=p.parse_args()
d=[t-v for t,v in zip(a.target,a.position)]
rpy=[0, math.atan2(-d[2],math.hypot(d[0],d[1])), math.atan2(d[1],d[0])]
pose=' '.join(map(str,a.position+rpy))
sdf=f'''<sdf version="1.7"><model name="trial_observer"><static>true</static><pose>{pose}</pose><link name="observer_link"><sensor name="recording_camera" type="camera"><always_on>true</always_on><update_rate>15</update_rate><camera><horizontal_fov>0.9</horizontal_fov><image><width>960</width><height>600</height><format>R8G8B8</format></image><clip><near>0.05</near><far>10</far></clip></camera><plugin name="recording_only_camera" filename="libgazebo_ros_camera.so"><ros><namespace>/demo/observer</namespace><remapping>image_raw:=image_raw</remapping><remapping>camera_info:=camera_info</remapping></ros><camera_name>recording_camera</camera_name><frame_name>observer_optical_frame</frame_name></plugin></sensor></link></model></sdf>'''
rclpy.init();n=rclpy.create_node('spawn_recording_camera');client=n.create_client(SpawnEntity,'/spawn_entity')
if not client.wait_for_service(timeout_sec=10):raise RuntimeError('Gazebo spawn service unavailable')
request=SpawnEntity.Request();request.name='trial_observer';request.xml=sdf
future=client.call_async(request);rclpy.spin_until_future_complete(n,future,timeout_sec=15)
if not future.done():raise RuntimeError('Spawn timed out')
result=future.result(); print(result.status_message)
if not result.success and 'already exists' not in result.status_message:raise RuntimeError(result.status_message)
n.destroy_node();rclpy.shutdown()
