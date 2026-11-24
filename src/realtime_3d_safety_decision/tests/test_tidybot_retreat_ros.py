"""Opt-in, localhost-only fake plant; never connects to the live domain 0."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import yaml


@pytest.mark.skipif(os.environ.get("RUN_TIDYBOT_ROS_TEST") != "1", reason="opt-in isolated ROS test")
def test_twist_reaches_two_meter_goal_and_stops(tmp_path, monkeypatch):
    # A separate DDS domain and loopback-only environment for BOTH processes.
    monkeypatch.setenv("ROS_DOMAIN_ID", "179")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE", "CYCLONEDDS_URI", "ROS_DISCOVERY_SERVER"):
        monkeypatch.delenv(key, raising=False)
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import PointCloud2, PointField, JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/tidybot_retreat.yaml").read_text())
    # This plant publishes synthetic base-frame points; do not couple it to the
    # checked-in physical camera extrinsics.
    config["cloud_frame"] = "synthetic_tidybot_base"
    config["cloud_to_base"] = np.eye(4).tolist()
    config["verified"] = {k: True for k in config["verified"]}  # Known synthetic plant, not real calibration.
    file = tmp_path / "fake_plant.yaml"
    file.write_text(yaml.safe_dump(config))
    context = Context()
    rclpy.init(context=context)
    node = Node("fake_tidybot_receiver", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    velocity = [0., 0.]
    pose = np.zeros(3)
    statuses = []
    commands = []
    def command(msg):
        velocity[:] = [msg.linear.x, msg.linear.y]
        commands.append(tuple(velocity))
    sub = node.create_subscription(Twist, "/tidybot/cmd_vel", command, 1)
    status_sub = node.create_subscription(String, "/realtime_safety/tidybot_retreat/status",
                                          lambda msg: statuses.append(json.loads(msg.data)), 1)
    joints = node.create_publisher(JointState, "/tidybot/joint_states", qos_profile_sensor_data)
    clouds = node.create_publisher(PointCloud2, "/realtime_safety/pointcloud", qos_profile_sensor_data)
    reset = node.create_client(Trigger, "/tidybot_retreat/reset")
    log = (tmp_path / "controller.log").open("w")
    process = subprocess.Popen([sys.executable, "-m", "realtime_safety.ros2_bridge.tidybot_retreat",
                                "--config", str(file), "--execute"], cwd=root, env=os.environ.copy(),
                               stdout=log, stderr=subprocess.STDOUT)
    start = last = time.monotonic()
    triggered = False
    last_reset = start
    try:
        while time.monotonic() - start < 26:
            now = time.monotonic()
            dt, last = now-last, now
            pose[:2] += np.array(velocity)*dt
            stamp = node.get_clock().now().to_msg()
            joint = JointState()
            joint.header.stamp = stamp
            joint.name = ["joint_x", "joint_y", "joint_th"]
            joint.position = pose.tolist()
            joints.publish(joint)
            if now-start > 2:
                triggered = True
            cloud = PointCloud2()
            cloud.header.stamp = stamp
            cloud.header.frame_id = config["cloud_frame"]
            cloud.height, cloud.width, cloud.point_step, cloud.row_step = 1, 40, 12, 480
            cloud.fields = [PointField(name=k,offset=i*4,datatype=7,count=1) for i,k in enumerate("xyz")]
            cloud.data = np.tile([.6, 0., .75 if triggered else 3.], (40,1)).astype('<f4').tobytes()
            clouds.publish(cloud)
            executor.spin_once(timeout_sec=.025)
            time.sleep(.01)
            # Clear only initial DDS discovery faults while the synthetic scene is clear.
            if not triggered and reset.service_is_ready() and now-last_reset > .2:
                reset.call_async(Trigger.Request()); last_reset = now
            if statuses and statuses[-1]["state"] == "DONE" and commands[-1] == (0.,0.):
                break
            assert process.poll() is None, (tmp_path / "controller.log").read_text()
        assert statuses[-1]["state"] == "DONE", (tmp_path / "controller.log").read_text()
        assert -2.03 <= pose[0] <= -1.97
        assert abs(pose[1]) < .001
        assert any(vx < 0 for vx,vy in commands)
        speed_limit = float(config["controller"]["speed_mps"])
        assert all(abs(vx) <= speed_limit and vy == 0 for vx,vy in commands)
        assert commands[-1] == (0.,0.)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        executor.shutdown(); node.destroy_node(); context.shutdown(); log.close()
