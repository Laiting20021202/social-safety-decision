#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from social_bev.config import load_config
from social_bev.pipeline import SocialNavigationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional ROS2 node for RGB Social BEV")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import rclpy
        from cv_bridge import CvBridge
        from nav_msgs.msg import OccupancyGrid
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from visualization_msgs.msg import Marker, MarkerArray
    except Exception as exc:
        print(f"ROS2 adapter requires rclpy, standard messages, and cv_bridge: {exc}", file=sys.stderr)
        return 2

    class SocialBEVNode(Node):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__("social_bev_node")
            self.bridge = CvBridge()
            self.pipeline = SocialNavigationPipeline(load_config(args.config), calibration_path=args.calibration)
            self.sub = self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
            self.pub_annotated = self.create_publisher(Image, "/social_bev/annotated", 10)
            self.pub_mask = self.create_publisher(Image, "/social_bev/walkable_mask", 10)
            self.pub_grid = self.create_publisher(OccupancyGrid, "/social_bev/occupancy_grid", 10)
            self.pub_markers = self.create_publisher(MarkerArray, "/social_bev/people_markers", 10)

        def on_image(self, msg: Image) -> None:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            result = self.pipeline.process_frame(frame, timestamp)
            annotated = self.bridge.cv2_to_imgmsg(result.visualization, encoding="bgr8")
            annotated.header = msg.header
            self.pub_annotated.publish(annotated)
            mask = (result.walkable_mask.astype(np.uint8) * 255)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self.pub_mask.publish(mask_msg)
            self.pub_grid.publish(self._grid_msg(result, msg.header))
            self.pub_markers.publish(self._marker_msg(result, msg.header))

        def _grid_msg(self, result, header):  # type: ignore[no-untyped-def]
            grid = OccupancyGrid()
            grid.header = header
            grid.info.width = int(result.bev.occupancy_grid.shape[1])
            grid.info.height = int(result.bev.occupancy_grid.shape[0])
            resolution = float(result.bev.metric_bev and self.pipeline.calibration.bev_config.get("resolution_m_per_pixel", 0.01) or 0.01)
            grid.info.resolution = resolution
            grid.data = result.bev.occupancy_grid.astype(np.int8).reshape(-1).tolist()
            return grid

        def _marker_msg(self, result, header):  # type: ignore[no-untyped-def]
            markers = MarkerArray()
            for track in result.tracks:
                if track.bev_position is None:
                    continue
                marker = Marker()
                marker.header = header
                marker.ns = "people"
                marker.id = int(track.track_id)
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(track.bev_position[0])
                marker.pose.position.y = float(track.bev_position[1])
                marker.pose.position.z = 0.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.25
                marker.color.r = 1.0
                marker.color.g = 0.8
                marker.color.b = 0.2
                marker.color.a = 1.0
                markers.markers.append(marker)
            return markers

    rclpy.init()
    node = SocialBEVNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

