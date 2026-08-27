from __future__ import annotations

from enum import Enum
from typing import Iterable


class RuntimeMode(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PERCEPTION = "perception"


CAMERA_TOPICS = (
    "/rgbd/color/image_raw",
    "/rgbd/depth/image_raw",
    "/rgbd/aligned_depth_to_color/image_raw",
    "/rgbd/color/camera_info",
    "/rgbd/depth/camera_info",
    "/rgbd/points",
    "/rgbd/points_world",
)

CORE_TOPICS = ("/clock", "/joint_states", "/tf", "/tf_static")

GROUND_TRUTH_TOPICS = (
    "/sim/ground_truth/hand_pose",
    "/sim/ground_truth/hand_collision",
    "/sim/ground_truth/min_distance",
)

PERCEPTION_INPUT_TOPICS = (
    "/rgbd/color/image_raw",
    "/rgbd/depth/image_raw",
    "/rgbd/color/camera_info",
    "/perception/hand_mask",
)

OPTICAL_FRAMES = (
    "rgbd_link",
    "rgbd_color_optical_frame",
    "rgbd_depth_optical_frame",
)


def subscriptions_for_mode(mode: RuntimeMode | str) -> tuple[str, ...]:
    selected = RuntimeMode(mode)
    if selected is RuntimeMode.GROUND_TRUTH:
        return GROUND_TRUTH_TOPICS
    return ("/perception/obstacles",)


def assert_mode_isolation(mode: RuntimeMode | str, subscriptions: Iterable[str]) -> None:
    selected = RuntimeMode(mode)
    subscription_set = set(subscriptions)
    leaked = subscription_set.intersection(GROUND_TRUTH_TOPICS)
    if selected is RuntimeMode.PERCEPTION and leaked:
        raise ValueError(
            "perception mode must not subscribe to simulator ground truth: "
            + ", ".join(sorted(leaked))
        )
