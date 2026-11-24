"""Point-cloud-first obstacle tracking with optional EdgeTAM refinement.

The modules in this package deliberately keep ROS and EdgeTAM imports optional.
The geometric safety path can therefore be tested and used when RGB or the
segmentation model is unavailable.
"""

from realtime_safety.edgetam_tracker.models import (
    AABB,
    OBB,
    CloudFrame,
    Cluster3D,
    MaskQuality,
    PointCloudQuality,
    TrackEstimate,
    TrackingState,
)

__all__ = [
    "AABB",
    "OBB",
    "CloudFrame",
    "Cluster3D",
    "MaskQuality",
    "PointCloudQuality",
    "TrackEstimate",
    "TrackingState",
]
