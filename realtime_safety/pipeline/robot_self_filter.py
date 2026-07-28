from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np

from realtime_safety.config import SegmentationConfig
from realtime_safety.types import (
    Detection2D,
    FramePacket,
    PointCloudFrame,
    RobotArmState,
)

LOGGER = logging.getLogger(__name__)


class RobotSelfFilter:
    """Remove camera-visible robot pixels from person segmentation masks.

    The green robot base is fixed near the top of the Koch camera image.  Color
    alone is not sufficient because a person may also wear green, so only the
    green connected component that intersects the configured base anchor is
    accepted.  A narrow dilated capsule covers the attached motors and gripper.
    """

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self._last_mask: np.ndarray | None = None
        self._last_core_mask: np.ndarray | None = None
        self._mask_history: deque[np.ndarray] = deque(
            maxlen=max(1, int(config.robot_mask_temporal_frames))
        )
        self._held_frames = 0
        self._last_arm_state: RobotArmState | None = None
        self._center_held_frames = 0
        self._rejected_count = 0
        self._last_overlap = 0.0
        self._last_residual_pixels = 0

    def reset(self) -> None:
        self._last_mask = None
        self._last_core_mask = None
        self._mask_history.clear()
        self._held_frames = 0
        self._last_arm_state = None
        self._center_held_frames = 0

    @property
    def latest_mask(self) -> np.ndarray | None:
        return self._last_mask

    @property
    def latest_core_mask(self) -> np.ndarray | None:
        return self._last_core_mask

    def filter_obstacles(
        self,
        detections: list[Detection2D],
        frame: FramePacket,
    ) -> list[Detection2D]:
        """Remove the visible robot from every accepted obstacle class."""

        if not self.config.robot_self_filter:
            return detections

        robot_mask = self._robot_mask(frame.bgr)
        if not detections:
            return detections
        if robot_mask is None or not np.any(robot_mask):
            return detections

        filtered: list[Detection2D] = []
        for detection in detections:
            if detection.mask is None:
                filtered.append(detection)
                continue
            result = self._remove_robot_pixels(detection, robot_mask)
            if result is not None:
                filtered.append(result)
                continue
            self._rejected_count += 1
            if self._rejected_count == 1 or self._rejected_count % 300 == 0:
                LOGGER.info(
                    "Rejected robot self-detection "
                    "(track=%s, overlap=%.1f%%, residual=%d px, total=%d)",
                    detection.track_id,
                    self._last_overlap * 100.0,
                    self._last_residual_pixels,
                    self._rejected_count,
                )
        return filtered

    def filter_people(
        self,
        detections: list[Detection2D],
        frame: FramePacket,
    ) -> list[Detection2D]:
        """Backward-compatible alias for the all-obstacle self filter."""

        return self.filter_obstacles(detections, frame)

    def estimate_arm_state(
        self,
        frame: FramePacket,
        cloud: PointCloudFrame,
    ) -> RobotArmState | None:
        """Project the anchored green robot component into the 3D pointmap."""

        mask = self._last_core_mask
        if mask is None or not np.any(mask) or cloud.pointmap.size == 0:
            return self._hold_arm_state(frame.source_timestamp)

        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            return self._hold_arm_state(frame.source_timestamp)
        center_xy = np.array((float(xs.mean()), float(ys.mean())), dtype=np.float32)
        height, width = cloud.pointmap.shape[:2]
        sample_mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid = sample_mask & np.isfinite(cloud.pointmap).all(axis=-1)
        valid &= (cloud.pointmap[..., 1] > 0.05) & (cloud.pointmap[..., 1] < 4.0)
        points = np.asarray(cloud.pointmap[valid], dtype=np.float32).reshape(-1, 3)
        if len(points) < 12:
            return self._hold_arm_state(frame.source_timestamp)

        forward = points[:, 1]
        median_depth = float(np.median(forward))
        mad = float(np.median(np.abs(forward - median_depth)))
        depth_gate = max(3.5 * 1.4826 * mad, 0.04)
        points = points[np.abs(forward - median_depth) <= depth_gate]
        if len(points) < 12:
            return self._hold_arm_state(frame.source_timestamp)
        measured = np.median(points, axis=0).astype(np.float32)
        previous = self._last_arm_state
        if previous is not None:
            displacement = measured - previous.center_xyz
            distance = float(np.linalg.norm(displacement))
            if distance > 0.25:
                measured = previous.center_xyz + displacement * (0.25 / distance)
            alpha = float(self.config.robot_center_ema_alpha)
            measured = (
                (1.0 - alpha) * previous.center_xyz + alpha * measured
            ).astype(np.float32)

        confidence = float(
            np.clip(
                0.45
                + 0.25 * min(len(xs) / 600.0, 1.0)
                + 0.30 * min(len(points) / 120.0, 1.0),
                0.0,
                1.0,
            )
        )
        state = RobotArmState(
            center_xyz=measured,
            center_xy=center_xy,
            image_size=(frame.original_width, frame.original_height),
            mask_pixels=int(len(xs)),
            point_count=int(len(points)),
            confidence=confidence,
            timestamp=float(frame.source_timestamp),
        )
        self._last_arm_state = state
        self._center_held_frames = 0
        return state

    def _hold_arm_state(self, timestamp: float) -> RobotArmState | None:
        previous = self._last_arm_state
        if (
            previous is None
            or self._center_held_frames
            >= max(0, int(self.config.robot_center_hold_frames))
        ):
            self._last_arm_state = None
            return None
        self._center_held_frames += 1
        held = RobotArmState(
            center_xyz=previous.center_xyz.copy(),
            center_xy=previous.center_xy.copy(),
            image_size=previous.image_size,
            mask_pixels=previous.mask_pixels,
            point_count=previous.point_count,
            confidence=max(0.05, previous.confidence * 0.92),
            timestamp=float(timestamp),
            held_frames=self._center_held_frames,
        )
        self._last_arm_state = held
        return held

    def _robot_mask(self, bgr: np.ndarray) -> np.ndarray | None:
        if bgr.size == 0:
            return self._held_mask()

        height, width = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.asarray(self.config.robot_green_hsv_lower, dtype=np.uint8)
        upper = np.asarray(self.config.robot_green_hsv_upper, dtype=np.uint8)
        green = cv2.inRange(hsv, lower, upper)
        # A small kernel closes JPEG/motion-blur holes without connecting the
        # textured green-grey table to the robot.
        green = cv2.morphologyEx(
            green,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

        x1, y1, x2, y2 = self._pixel_roi(
            self.config.robot_anchor_roi,
            width,
            height,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
        anchor_labels = labels[y1:y2, x1:x2]
        if anchor_labels.size == 0:
            return self._held_mask()
        values, frequencies = np.unique(anchor_labels[anchor_labels > 0], return_counts=True)
        anchored = {
            int(label)
            for label, frequency in zip(values, frequencies)
            if int(frequency) >= 8 and int(stats[int(label), cv2.CC_STAT_AREA]) >= 40
        }
        previous_core = self._last_core_mask
        if previous_core is not None and previous_core.shape == green.shape:
            link_px = max(0, int(self.config.robot_component_link_px))
            linked_region = (
                cv2.dilate(
                    previous_core.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * link_px + 1, 2 * link_px + 1),
                    ),
                ).astype(bool)
                if link_px
                else previous_core
            )
            linked_labels = labels[linked_region]
            linked_values, linked_frequencies = np.unique(
                linked_labels[linked_labels > 0],
                return_counts=True,
            )
            anchored.update(
                int(label)
                for label, frequency in zip(linked_values, linked_frequencies)
                if int(frequency) >= 8
                and int(stats[int(label), cv2.CC_STAT_AREA]) >= 30
            )
        if not anchored:
            return self._held_mask()

        core = np.isin(labels, tuple(anchored))
        selected = core.astype(np.uint8) * 255
        # At 320x240 this covers the black joint housings surrounding the green
        # casing while remaining much narrower than a hand or torso.
        dilation = max(0, int(self.config.robot_mask_dilation_px))
        if dilation:
            selected = cv2.dilate(
                selected,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * dilation + 1, 2 * dilation + 1),
                ),
            )

        extension = max(0, int(self.config.robot_tip_extension_px))
        if extension:
            selected = self._extend_toward_gripper(
                selected,
                core,
                (x1, y1, x2, y2),
                extension,
                max(3, dilation),
            )

        current = selected.astype(bool)
        self._last_core_mask = core
        self._mask_history.append(current)
        temporal = np.logical_or.reduce(tuple(self._mask_history))
        self._last_mask = temporal
        self._held_frames = 0
        return temporal

    def _held_mask(self) -> np.ndarray | None:
        if (
            self._last_mask is None
            or self._held_frames >= max(0, int(self.config.robot_mask_hold_frames))
        ):
            self._last_mask = None
            self._last_core_mask = None
            self._mask_history.clear()
            return None
        self._held_frames += 1
        return self._last_mask

    @staticmethod
    def _extend_toward_gripper(
        mask: np.ndarray,
        green_mask: np.ndarray,
        anchor_roi: tuple[int, int, int, int],
        extension: int,
        radius: int,
    ) -> np.ndarray:
        ys, xs = np.nonzero(green_mask)
        if len(xs) < 20:
            return mask

        points = np.column_stack((xs, ys)).astype(np.float32)
        center = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
        anchor_center = np.array(
            (
                (anchor_roi[0] + anchor_roi[2] - 1) * 0.5,
                (anchor_roi[1] + anchor_roi[3] - 1) * 0.5,
            ),
            dtype=np.float32,
        )
        if float(np.dot(axis, center - anchor_center)) < 0.0:
            axis *= -1.0

        projection = (points - center) @ axis
        terminal_points = points[projection >= np.percentile(projection, 92.0)]
        if not len(terminal_points):
            return mask
        start = terminal_points.mean(axis=0)
        end = start + axis * float(extension)
        cv2.line(
            mask,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            color=255,
            thickness=2 * radius + 1,
            lineType=cv2.LINE_AA,
        )
        return mask

    def _remove_robot_pixels(
        self,
        detection: Detection2D,
        robot_mask: np.ndarray,
    ) -> Detection2D | None:
        person_mask = np.asarray(detection.mask, dtype=bool)
        if person_mask.shape != robot_mask.shape:
            robot_mask = cv2.resize(
                robot_mask.astype(np.uint8),
                (person_mask.shape[1], person_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        original_pixels = int(np.count_nonzero(person_mask))
        if original_pixels == 0:
            return detection
        overlap_pixels = int(np.count_nonzero(person_mask & robot_mask))
        if overlap_pixels == 0:
            return detection

        overlap = overlap_pixels / original_pixels
        residual = person_mask & ~robot_mask
        residual = self._remove_small_fragments(
            residual,
            int(self.config.robot_min_residual_pixels),
        )
        residual_pixels = int(np.count_nonzero(residual))
        self._last_overlap = overlap
        self._last_residual_pixels = residual_pixels
        # A high overlap alone is not enough to reject: a small real hand can
        # touch/occlude the robot while YOLO includes both in one mask.  Reject
        # only when virtually no meaningful non-robot region remains.
        centroid_x = int(
            np.clip(round(float(detection.centroid_xy[0])), 0, robot_mask.shape[1] - 1)
        )
        centroid_y = int(
            np.clip(round(float(detection.centroid_xy[1])), 0, robot_mask.shape[0] - 1)
        )
        centered_on_robot = bool(robot_mask[centroid_y, centroid_x])
        if (
            detection.class_name != "person"
            and centered_on_robot
            and overlap >= float(self.config.robot_reject_overlap)
        ):
            # The fixed Koch arm is frequently assigned changing COCO object
            # labels (chair, bag, etc.). A non-person instance centered on and
            # mostly composed of the anchored arm is self geometry.
            return None
        if residual_pixels < int(self.config.robot_min_residual_pixels) and (
            overlap >= float(self.config.robot_reject_overlap)
            or (
                centered_on_robot
                and overlap >= float(self.config.robot_reject_min_overlap)
            )
        ):
            return None

        # Preserve a real hand/person that overlaps the robot, but prevent the
        # robot pixels leaked into the YOLO mask from entering its 3D cloud.
        if residual_pixels < int(self.config.robot_min_residual_pixels):
            return detection
        ys, xs = np.nonzero(residual)
        detection.mask = residual
        detection.bbox_xyxy = np.array(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=np.float32,
        )
        detection.centroid_xy = np.array(
            [float(xs.mean()), float(ys.mean())],
            dtype=np.float32,
        )
        return detection

    @staticmethod
    def _remove_small_fragments(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        if count <= 2:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.max(areas))
        threshold = max(int(minimum_pixels), int(round(largest * 0.10)))
        accepted = np.flatnonzero(areas >= threshold) + 1
        return np.isin(labels, accepted)

    @staticmethod
    def _pixel_roi(
        roi: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1 = int(np.clip(round(float(roi[0]) * width), 0, max(width - 1, 0)))
        y1 = int(np.clip(round(float(roi[1]) * height), 0, max(height - 1, 0)))
        x2 = int(np.clip(round(float(roi[2]) * width), x1 + 1, width))
        y2 = int(np.clip(round(float(roi[3]) * height), y1 + 1, height))
        return x1, y1, x2, y2
