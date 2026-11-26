"""Passive startup. MoveIt plans locally; only the guarded client may stream a bounded curl."""
from pathlib import Path
import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
import yaml


def generate_launch_description():
    here = Path(__file__).parent
    share = Path(get_package_share_directory('openarm_description'))
    robot = xacro.process_file(str(share/'urdf/robot/v10.urdf.xacro'), mappings={
        'bimanual':'true','hand':'true','ros2_control':'false'}).toxml()
    srdf = ET.parse('/opt/laiting/moveit/openarm_bimanual.srdf').getroot()
    group = ET.SubElement(srdf,'group',name='right_curl')
    ET.SubElement(group,'joint',name='openarm_right_joint4')
    fixed_chest = os.environ.get('LAITING_CAMERA_PROFILE') == 'isaac_fixed_openarm_v1' or os.environ.get('LAITING_FIXED_CHEST') == '1'
    params = {'robot_description':robot, 'robot_description_semantic':ET.tostring(srdf,encoding='unicode'),
              'planning_pipelines':['ompl'], 'default_planning_pipeline':'ompl',
              'ompl.planning_plugin':'ompl_interface/OMPLPlanner',
              'ompl.request_adapters':'default_planner_request_adapters/AddTimeOptimalParameterization',
              'ompl.planner_configs.RRTConnect.type':'geometric::RRTConnect',
              'ompl.right_curl.planner_configs':['RRTConnect'],
              'robot_description_planning.joint_limits.openarm_right_joint4.has_acceleration_limits':True,
              'robot_description_planning.joint_limits.openarm_right_joint4.max_acceleration':0.14,
              'robot_description_planning.joint_limits.openarm_right_joint4.has_velocity_limits':True,
              'robot_description_planning.joint_limits.openarm_right_joint4.max_velocity':0.08,
              'sensors':['d415_cloud'], 'd415_cloud.sensor_plugin':'occupancy_map_monitor/PointCloudOctomapUpdater',
              'd415_cloud.point_cloud_topic':'/laiting/scene/points','d415_cloud.max_range':2.0,
              'd415_cloud.point_subsample':1,'d415_cloud.padding_offset':0.030 if fixed_chest else 0.015,
              'd415_cloud.padding_scale':1.0,'d415_cloud.max_update_rate':4.0,
              'd415_cloud.filtered_cloud_topic':'/laiting/scene/filtered',
              'octomap_frame':'openarm_body_link0','octomap_resolution':0.0125 if fixed_chest else 0.025,
              'disable_capabilities':'move_group/MoveGroupExecuteTrajectoryAction',
              'robot_description_kinematics':yaml.safe_load(Path('/opt/laiting/moveit/kinematics.yaml').read_text()),
              'allow_trajectory_execution':False,'publish_robot_description_semantic':True,
              'publish_planning_scene':True,'publish_state_updates':True,'use_sim_time':False}
    show=UnlessCondition(LaunchConfiguration('planner_only'))
    return LaunchDescription([
        DeclareLaunchArgument('planner_only',default_value='false'),
        DeclareLaunchArgument('rviz_config',default_value=str(here/'guarded.rviz')),
        Node(package='moveit_ros_move_group',executable='move_group',name='move_group',output='screen',parameters=[params]),
        ExecuteProcess(cmd=['python3',str(here/'scene.py')],condition=show,output='screen'),
        ExecuteProcess(cmd=['python3',str(here/'guard.py')],condition=show,output='screen'),
        Node(package='rviz2',executable='rviz2',name='laiting_guarded_rviz',condition=show,
             arguments=['-d',LaunchConfiguration('rviz_config')],output='screen')])
