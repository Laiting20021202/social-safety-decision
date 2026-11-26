"""Wire the real Humble trajectory controller to Isaac; do not load a CAN plugin."""
import os
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    if os.environ.get('ROS_DOMAIN_ID')!='89':
        raise RuntimeError('Isaac commissioning requires domain 89')
    share=Path(get_package_share_directory('openarm_description'))
    root=ET.fromstring(xacro.process_file(str(share/'urdf/robot/v10.urdf.xacro'),mappings={
        'bimanual':'true','hand':'true','ros2_control':'false'}).toxml())
    robot={'robot_description':ET.tostring(root,encoding='unicode')}
    c=ET.SubElement(root,'ros2_control',name='IsaacEncoders',type='system')
    hardware=ET.SubElement(c,'hardware')
    ET.SubElement(hardware,'plugin').text='laiting_isaac/Bridge'
    layout=json.loads(Path('/results/calibration/isaac.json').read_text())
    upper=layout['active_right_pose'][3]+math.radians(layout['curl_degrees'])+.025
    ET.SubElement(hardware,'param',name='upper_elbow_rad').text=str(upper)
    for side in ('left','right'):
        for suffix in [f'joint{i}' for i in range(1,8)]+['finger_joint1']:
            j=ET.SubElement(c,'joint',name=f'openarm_{side}_{suffix}')
            for field in ('position','velocity'): ET.SubElement(j,'state_interface',name=field)
            if side=='right' and suffix=='joint4':ET.SubElement(j,'command_interface',name='position')
    ctrl={'robot_description':ET.tostring(root,encoding='unicode')}
    return LaunchDescription([
        Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[robot,{'publish_frequency':40.}]),
        Node(package='controller_manager',executable='ros2_control_node',parameters=[ctrl,'/demo/controllers.yaml'],output='screen'),
        Node(package='controller_manager',executable='spawner',arguments=['joint_state_broadcaster']),
        Node(package='controller_manager',executable='spawner',arguments=['right_joint_trajectory_controller','--inactive']),
        IncludeLaunchDescription(PythonLaunchDescriptionSource('/demo/guarded/station.launch.py'),
                                 launch_arguments={'rviz_config':'/demo/isaac/chest.rviz'}.items())])
