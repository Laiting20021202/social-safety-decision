"""Use Thor's existing MASt3R-SLAM tracker and metric-check each result against aligned D415 depth."""
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from frames import RGBD, SENSOR, decode, fresh, stamp_ns
from thor_geometry import aligned_depth, camera_matrix, cloud_points
from thor_live import cloud_message
from thor_tracker import LiveSlam


def fixed_camera_pointmap(slam, rgb):
    """Infer a fresh pointmap in the optical frame of a stationary chest camera."""
    import torch
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono

    with torch.inference_mode():
        frame = create_frame(slam.frame_id, rgb.astype(np.float32) / 255., slam.pose, device=slam.device)
        slam.frame_id += 1
        xyz, confidence = mast3r_inference_mono(slam.model, frame)
        h, w = frame.uimg.shape[:2]
        return xyz.cpu().numpy(), confidence.cpu().numpy(), w, h


def fused_points(verified, measured_mm, rgb, k):
    """Preserve depth edges and fill model gaps with measured, co-registered D415 pixels."""
    z = measured_mm.astype(np.float32) * .001
    z[(z <= .12) | (z >= 4.)] = np.nan
    agree = np.isfinite(verified) & (np.abs(verified - z) <= .015)
    z[agree] += .2 * (verified[agree] - z[agree])
    return cloud_points(z, rgb, k, stride=5)


def metric_depth(relative, depth_mm):
    z = depth_mm.astype(np.float32) * 0.001
    valid = np.isfinite(relative) & (relative > 0) & (z > 0.15) & (z < 4.0)
    # Alternate columns in a regularly spaced grid for fitting and evaluation.
    fit = np.zeros_like(valid)
    check = np.zeros_like(valid)
    fit[::8, ::16] = True
    check[4::8, 8::16] = True
    fit &= valid
    check &= valid
    if min(fit.sum(), check.sum()) < 200:
        raise ValueError('insufficient independent D415 depth support')
    for mask in (fit, check):
        h, w = mask.shape
        if min(mask[y:y+h//2, x:x+w//2].sum() for y in (0, h//2) for x in (0, w//2)) < 20:
            raise ValueError('depth samples do not cover all four image quadrants')
    scale = float(np.median(z[fit] / relative[fit]))
    meters = relative * scale
    method = 'global_scale'
    # Monocular reconstructions can exhibit smooth local scale drift. Estimate a coarse
    # field solely from fit pixels; withheld metric depths still have to pass below.
    if float(np.median(np.abs(meters[fit]-z[fit]))) > .035:
        h,w=z.shape
        rows,cols=3,4
        scales=np.empty((rows,cols),np.float32)
        for y in range(rows):
            for x in range(cols):
                sub=np.s_[y*h//rows:(y+1)*h//rows, x*w//cols:(x+1)*w//cols]
                ok=fit[sub]
                scales[y,x]=(np.median(z[sub][ok]/relative[sub][ok]) if ok.sum()>=20 else scale)
        # Bound regional corrections; holes or discontinuities are removed by final verification.
        field=cv2.resize(np.clip(scales,.72*scale,1.35*scale),(w,h),interpolation=cv2.INTER_LINEAR)
        local=relative*field
        if np.median(np.abs(local[fit]-z[fit])) < .9*np.median(np.abs(meters[fit]-z[fit])):
            meters=local
            method='regional_scale_3x4'
    residual = np.abs(meters - z)
    inliers = residual < np.maximum(0.04, z * 0.10)
    ratio = float(inliers[check].mean())
    median = float(np.median(residual[check]))
    if not np.isfinite(scale) or scale <= 0 or ratio < 0.70 or median > 0.06:
        raise ValueError(f'metric check failed: inliers={ratio:.2f} median={median:.3f}m')
    # Retain MASt3R's inferred coordinates only where real depth corroborates them.
    return np.where(valid & inliers, meters, np.nan), scale, ratio, median, method


class Cloud(Node):
    def __init__(self):
        super().__init__('laiting_thor_mast3r')
        self.cloud = self.create_publisher(PointCloud2, '/laiting/mast3r/points', SENSOR)
        self.fused = self.create_publisher(PointCloud2, '/laiting/mast3r/fused_points', SENSOR)
        self.fixed = os.environ.get('LAITING_FIXED_CHEST') == '1'
        self.status = self.create_publisher(String, '/laiting/mast3r/status', SENSOR)
        self.slam = LiveSlam('config/base.yaml', 40)
        self.pending = None
        self.event = threading.Condition()
        self.running = True
        self.inputs = RGBD(self, self.offer)
        self.last_stamp = 0
        self.signature = None
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def offer(self, *group):
        with self.event:
            self.pending = group
            self.event.notify()

    def worker(self):
        while self.running:
            with self.event:
                self.event.wait_for(lambda: self.pending is not None or not self.running)
                if not self.running:
                    return
                rgb_msg, depth_msg, info = self.pending
                self.pending = None
            stamp = stamp_ns(info.header)
            report = {'stamp_ns': stamp, 'metric': False, 'points': 0}
            try:
                rgb, depth_mm = decode(rgb_msg, depth_msg, info)
                k = camera_matrix(info, info.width, info.height)
                signature = (info.width, info.height, tuple(info.k))
                if self.signature != signature or stamp - self.last_stamp > 3e9:
                    self.slam.reset()
                self.last_stamp, self.signature = stamp, signature
                if self.fixed:
                    tracked = fixed_camera_pointmap(self.slam, rgb)
                else:
                    tracked, reason = self.slam.process(rgb, k)
                    if tracked is None:
                        raise ValueError(reason)
                xyz, confidence, w, h = tracked
                relative = aligned_depth(xyz, confidence, info.width, info.height, w, h, 1.5)
                depth, scale, ratio, err, method = metric_depth(relative, depth_mm)
                points = cloud_points(depth, rgb, k, stride=8)
                if len(points) < 200 or not fresh(stamp, 1.2):
                    raise ValueError('sparse or stale inference')
                self.cloud.publish(cloud_message(info.header, points))
                fused = fused_points(depth, depth_mm, rgb, k)
                self.fused.publish(cloud_message(info.header, fused))
                report.update(metric=True, points=len(points), scale=scale, inlier_fraction=ratio,
                              median_error_m=err, metric_fit=method, fused_points=len(fused),
                              inference='fixed_camera' if self.fixed else 'tracking',
                              frame=info.header.frame_id, session=self.slam.session_id)
            except (ValueError, RuntimeError, TypeError) as exc:
                report['reason'] = str(exc)
                self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            if self.running and rclpy.ok():
                self.status.publish(String(data=json.dumps(report)))


def main():
    rclpy.init()
    node = Cloud()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with node.event:
            node.running = False
            node.event.notify()
        node.thread.join(timeout=4)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
