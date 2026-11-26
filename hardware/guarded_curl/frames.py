"""Fresh, exact-stamped RealSense RGB-D tuples and the compressedDepth wire format."""
from collections import OrderedDict
import time

import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage

FRAME = 'd415_color_optical_frame'
SENSOR = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
ROOT = '/camera/d415'
PNG = b'\x89PNG\r\n\x1a\n'


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def fresh(stamp, seconds):
    return -0.05 <= (time.time_ns() - stamp) / 1e9 <= seconds


class RGBD:
    def __init__(self, node, ready):
        self.pending = OrderedDict()
        self.last = 0
        self.ready = ready
        topics = [(CompressedImage, '/color/image_raw/compressed'),
                  (CompressedImage, '/aligned_depth_to_color/image_raw/compressedDepth'),
                  (CameraInfo, '/color/camera_info')]
        for i, (msg_type, suffix) in enumerate(topics):
            node.create_subscription(msg_type, ROOT + suffix, lambda m, i=i: self.add(i, m), SENSOR)

    def add(self, index, msg):
        stamp = stamp_ns(msg.header)
        if msg.header.frame_id != FRAME or stamp <= self.last or not fresh(stamp, 1.0):
            return
        group = self.pending.setdefault(stamp, [None] * 3)
        group[index] = msg
        if all(m is not None for m in group):
            self.last = stamp
            self.pending.pop(stamp)
            self.ready(*group)
        while len(self.pending) > 16:
            self.pending.popitem(last=False)


def decode(rgb_msg, depth_msg, info):
    if '16UC1' not in depth_msg.format or 'compressedDepth' not in depth_msg.format:
        raise ValueError('D415 depth must be 16UC1 compressedDepth')
    data = np.frombuffer(depth_msg.data, np.uint8)
    offset = bytes(data[:20]).find(PNG)
    if offset not in (0, 12):
        raise ValueError('invalid compressedDepth header')
    depth = cv2.imdecode(data[offset:], cv2.IMREAD_UNCHANGED)
    bgr = cv2.imdecode(np.frombuffer(rgb_msg.data, np.uint8), cv2.IMREAD_COLOR)
    if (depth is None or depth.dtype != np.uint16 or depth.shape != (info.height, info.width)
            or bgr is None or bgr.shape[:2] != depth.shape):
        raise ValueError('RGB, depth, and calibration dimensions disagree')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth
