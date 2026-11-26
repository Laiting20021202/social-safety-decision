#!/usr/bin/env python3
"""Measure messages delivered to the workstation, including camera calibration and TF."""
import argparse
import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--raw", action="store_true", help="Use uncompressed image topics")
    parser.add_argument("--json", help="Write the same report to a file")
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("workstation_openarm_observer")
    tf = Buffer()
    listener = TransformListener(tf, node)
    channels = {"/joint_states": JointState}
    images = {}
    for camera, streams in {"d435": ["color"], "d415": ["color", "aligned_depth_to_color"]}.items():
        for stream in streams:
            path = f"/camera/{camera}/{stream}/"
            topic = path + "image_raw"
            if not args.raw:
                topic += "/compressed" if stream == "color" else "/compressedDepth"
            channels[topic] = Image if args.raw else CompressedImage
            images[camera, stream] = topic
            channels[path + "camera_info"] = CameraInfo
    report = {name: {"samples": 0} for name in channels}
    stamps = {name: set() for name in images.values()}

    def measure(msg, name):
        info = report[name]
        stamp = Time.from_msg(msg.header.stamp).nanoseconds
        if stamp <= info.get("last_stamp_ns", 0):
            return
        if not info["samples"]:
            info.update(first_stamp_ns=stamp, frame=msg.header.frame_id)
            if isinstance(msg, JointState):
                info["joints"] = sorted(msg.name)
            elif isinstance(msg, CompressedImage):
                data = bytes(msg.data)
                if "compressedDepth" in msg.format:
                    # compressed_depth_image_transport prepends a configuration
                    # header to the lossless 16-bit PNG. Find the PNG signature.
                    offset = data.find(b"\x89PNG\r\n\x1a\n", 0, 32)
                    if offset < 0:
                        return
                    data = data[offset:]
                pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if pixels is None:
                    return
                info.update(size=[pixels.shape[1], pixels.shape[0]], encoding=msg.format)
                if pixels.dtype == np.uint16 and pixels.ndim == 2:
                    info["nonzero_depth_px"] = int(np.count_nonzero(pixels))
            else:
                info["size"] = [msg.width, msg.height]
                if isinstance(msg, Image):
                    info["encoding"] = msg.encoding
                if isinstance(msg, CameraInfo):
                    info["focal_px"] = [msg.k[0], msg.k[4]]
        if isinstance(msg, Image) and msg.encoding == "16UC1" and not info.get("nonzero_depth_px"):
            depth = np.frombuffer(msg.data, dtype=">u2" if msg.is_bigendian else "<u2")
            info["nonzero_depth_px"] = int(np.count_nonzero(depth))
        info["samples"] += 1
        info["last_stamp_ns"] = stamp
        if name in stamps:
            stamps[name].add(stamp)

    for name, cls in channels.items():
        node.create_subscription(cls, name, lambda msg, n=name: measure(msg, n), qos_profile_sensor_data)
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    for info in report.values():
        if info["samples"] > 1:
            dt = (info["last_stamp_ns"] - info["first_stamp_ns"]) / 1e9
            info["received_hz"] = round((info["samples"] - 1) / dt, 1)
            info["last_age_s"] = round((node.get_clock().now().nanoseconds - info["last_stamp_ns"]) / 1e9, 3)
    depth = images["d415", "aligned_depth_to_color"]
    paired = len(stamps[depth] & stamps[images["d415", "color"]])
    output = {"domain": int(node.context.get_domain_id()), "topics": report,
              "d415_same_stamp_rgbd_pairs": paired}
    ok = all(info["samples"] >= 2 and abs(info["last_age_s"]) < 1.5 for info in report.values())
    ok &= len(report["/joint_states"].get("joints", [])) == 16
    ok &= report[depth].get("nonzero_depth_px", 0) > 0 and paired >= 2
    for (camera, stream), topic in images.items():
        intrinsics = report[f"/camera/{camera}/{stream}/camera_info"]
        ok &= report[topic].get("size") == intrinsics.get("size")
        ok &= all(x > 0 for x in intrinsics.get("focal_px", [0]))
    try:
        msg = tf.lookup_transform("world", "openarm_right_hand_tcp", Time())
        xyz = msg.transform.translation
        output["right_hand_tcp_m"] = [xyz.x, xyz.y, xyz.z]
    except TransformException:
        ok = False
    output["verified"] = bool(ok)
    formatted = json.dumps(output, indent=2)
    print(formatted)
    if args.json:
        with open(args.json, "w") as stream:
            print(formatted, file=stream)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
