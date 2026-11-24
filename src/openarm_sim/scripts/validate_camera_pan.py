#!/usr/bin/env python3
"""Exercise the real image pipeline while panning Gazebo's existing camera.

This driver alone uses Gazebo model poses to move/reset the camera. It never
publishes obstacle clouds, safety commands, or model events to the controller.
All observation and memory measurements below are real subscribed messages.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml
import trimesh
import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64, String


class Probe(Node):
    def __init__(self):
        super().__init__("camera_pan_validation")
        self.phase = "baseline"
        self.rows = []
        self.models = {}
        self.cloud_count = 0
        self.memory_count = 0
        self.memory_status = ""
        self.safety = ""
        self.task = ""
        self.truth_distance = None
        self.reference = None
        self.hand_reference = None
        self.memory_points = np.empty((0, 3))
        config_root = Path(__file__).resolve().parents[1] / "config"
        self.table_z = float(yaml.safe_load((config_root / "scene.yaml").read_text())["table"]["top_z"])
        hand = yaml.safe_load((config_root / "hand_scenarios.yaml").read_text())["defaults"]
        self.hand_vertices = np.asarray(trimesh.load(config_root.parent / hand["visual_asset"], force="mesh").vertices) * np.asarray(hand["visual_scale"])
        self.hand_vertices = self.hand_vertices @ Rotation.from_euler("xyz", hand["visual_rpy_deg"], degrees=True).as_matrix().T
        self.latest_world = np.empty((0, 3))
        self.setter = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.hand = self.create_publisher(String, "/sim/hand/command", 10)
        self.create_subscription(ModelStates, "/gazebo/model_states", self.models_cb, 10)
        self.create_subscription(PointCloud2, "/perception/obstacles", lambda m: setattr(self, "cloud_count", m.width * m.height), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/openarm/safety/memory_cloud", self.memory_cb, qos_profile_sensor_data)
        for topic, key in [("/openarm/safety/memory_status", "memory_status"), ("/openarm/safety/state", "safety"), ("/openarm/task/state", "task")]:
            self.create_subscription(String, topic, lambda m, k=key: setattr(self, k, m.data), 10)
        self.create_subscription(Float64, "/sim/ground_truth/min_distance", lambda m: setattr(self, "truth_distance", m.data), 10)
        self.create_subscription(PointCloud2, "/realtime_safety/environment_cloud_world", self.world_cb, qos_profile_sensor_data)

    def models_cb(self, message):
        self.models = dict(zip(message.name, message.pose))

    def memory_cb(self, message):
        records = point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)
        self.memory_points = np.column_stack([records[name] for name in ("x", "y", "z")])
        self.memory_count = len(self.memory_points)

    def capture_hand_reference(self):
        # Evaluator only: identify which remembered points belong to the actual
        # visible hand; unrelated table points cannot make retention pass.
        pose = self.models["human_hand"]
        q, p = pose.orientation, pose.position
        vertices = self.hand_vertices @ Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().T + [p.x, p.y, p.z]
        points = self.memory_points
        selected = cKDTree(vertices).query(points)[0] < .025
        self.hand_reference = points[selected].copy()
        if len(self.hand_reference) < 8:
            raise RuntimeError("memory does not contain sufficient points on the actual hand")

    def retained_hand_fraction(self):
        if self.hand_reference is None: return None
        if not len(self.memory_points): return 0.
        return float(np.mean(cKDTree(self.memory_points).query(self.hand_reference)[0] < .02))

    def world_cb(self, message):
        records = point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.column_stack([records[name] for name in ("x", "y", "z")])
        self.latest_world = xyz
        # Stationary tabletop ROI, excluding robot and target-cube surfaces.
        table = xyz[(xyz[:, 0] > -.05) & (np.abs(xyz[:, 1]) < .5) & (np.abs(xyz[:, 2] - self.table_z) < .008)]
        residual = None
        if self.reference is not None and len(table):
            residual = float(np.median(self.reference.query(table, workers=1)[0]))
        self.rows.append(dict(
            wall=time.monotonic(), stamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            phase=self.phase, world_points=len(xyz), observed_points=self.cloud_count,
            memory_points=self.memory_count, memory_status=self.memory_status,
            retained_hand_fraction=self.retained_hand_fraction(),
            safety=self.safety, task=self.task, table_reference_nn_m=residual,
            evaluation_joint_origin_distance_proxy_m=self.truth_distance,
        ))

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.02)

    def command(self, value):
        self.hand.publish(String(data=value))
        self.spin(.15)

    def pan(self, original, start, end, duration):
        q = original.orientation
        base = Rotation.from_quat([q.x, q.y, q.z, q.w])
        started = time.monotonic()
        while True:
            fraction = min(1., (time.monotonic() - started) / duration)
            yaw = start + (end - start) * fraction
            request = SetEntityState.Request()
            request.state.name = "rgbd_sensor"
            request.state.reference_frame = "world"
            request.state.pose = copy.deepcopy(original)
            quat = (Rotation.from_euler("z", yaw, degrees=True) * base).as_quat()
            orientation = request.state.pose.orientation
            orientation.x, orientation.y, orientation.z, orientation.w = map(float, quat)
            future = self.setter.call_async(request)
            deadline = time.monotonic() + 1.
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=.01)
            if not future.done() or not future.result().success:
                raise RuntimeError("camera pose service did not accept pan command")
            if fraction >= 1.: break
            self.spin(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=65.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    original = None
    report = {"passed": False, "controller": "existing geometric avoidance + visibility memory", "angle_deg": args.angle}
    try:
        node.spin(3.)
        if not node.setter.wait_for_service(timeout_sec=3.) or "rgbd_sensor" not in node.models:
            raise RuntimeError("Gazebo camera or SetEntityState unavailable")
        original = copy.deepcopy(node.models["rgbd_sensor"])
        node.command("auto_sweep:off")
        node.command("withdraw")
        node.spin(5.)
        p = node.latest_world
        table = p[(p[:, 0] > -.05) & (np.abs(p[:, 1]) < .5) & (np.abs(p[:, 2] - node.table_z) < .008)]
        if len(table): node.reference = cKDTree(table)
        node.phase = "visible_hand"
        node.command("scenario:perception_preview")
        node.command("trigger")
        deadline = time.monotonic() + 25.
        while time.monotonic() < deadline and (node.cloud_count < 8 or node.memory_count < 8):
            node.spin(.2)
        if node.cloud_count < 8 or node.memory_count < 8:
            raise RuntimeError("real model did not produce observed hand and memory")
        node.command("pause")
        node.spin(2.)
        node.capture_hand_reference()
        for name, start, end in [("pan_left", 0., args.angle), ("pan_right", args.angle, -args.angle), ("return_view", -args.angle, 0.)]:
            node.phase = name
            print(name, flush=True)
            node.pan(original, start, end, 6.)
            node.spin(4.)
        node.phase = "withdraw_reobserve"
        node.command("withdraw")
        node.spin(12.)
        panned = [r for r in node.rows if r["phase"] in {"pan_left", "pan_right"}]
        hidden = [r for r in panned if r["observed_points"] == 0]
        tail = [r for r in node.rows if r["phase"] == "withdraw_reobserve"][-10:]
        report.update(
            image_derived_world_frames=len(node.rows),
            hidden_frames=len(hidden),
            reference_hand_points=len(node.hand_reference),
            hidden_frames_retaining_memory=sum(r["retained_hand_fraction"] >= .8 for r in hidden),
            cleared_after_reobservation=bool(tail and all(r["retained_hand_fraction"] < .1 for r in tail)),
            final_safety=node.safety,
            table_reference_nn_median_m=float(np.median([r["table_reference_nn_m"] for r in panned if r["table_reference_nn_m"] is not None])) if any(r["table_reference_nn_m"] is not None for r in panned) else None,
        )
        report["passed"] = bool(hidden and all(r["retained_hand_fraction"] >= .8 for r in hidden) and report["cleared_after_reobservation"])
    except Exception as error:
        report["error"] = str(error)
    finally:
        if original is not None:
            node.pan(original, 0., 0., .1)
            node.command("withdraw")
        report["samples"] = node.rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
        node.destroy_node()
        rclpy.shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
