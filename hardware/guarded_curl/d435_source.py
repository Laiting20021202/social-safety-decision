"""Prefer the lab's existing D435 ROS source. Use our direct D435 only while that source is absent."""
import os
import signal
import subprocess
import time

import rclpy

SOURCE = '/camera/camera3/color/camera_info'


def stop(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid,signal.SIGINT)
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=4)


def main():
    env=os.environ.copy()
    os.environ['FASTRTPS_DEFAULT_PROFILES_FILE']='/opt/right-curl/camera-source.xml'
    rclpy.init(domain_id=4)
    node=rclpy.create_node('laiting_d435_source_monitor')
    child=None
    mode=None
    try:
        since=time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.25)
            if time.monotonic()-since<3: continue
            desired='relay' if node.count_publishers(SOURCE)>0 else 'direct'
            if desired==mode and child.poll() is None: continue
            stop(child)
            if desired=='relay':
                cmd=['python3','/opt/right-curl/app/camera_relay.py']
            else:
                cmd=['ros2','run','realsense2_camera','realsense2_camera_node','--ros-args',
                     '-r','__ns:=/camera','-r','__node:=d435','--params-file','/opt/laiting-camera-config/d435.yaml']
            node.get_logger().info('D435 source: '+desired)
            child=subprocess.Popen(cmd,env=env,start_new_session=True)
            mode=desired
            since=time.monotonic()
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
        pass
    finally:
        stop(child)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__=='__main__':
    main()
