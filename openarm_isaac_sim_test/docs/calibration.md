# RGB-D and workspace calibration

All distances are metres. The default camera is a generic pinhole model, not RealSense.
Edit `config/camera.yaml`, or move `rgbd_sensor` directly with Gazebo's translate/rotate
tools; do not rename frames to RealSense conventions. Runtime edits immediately drive the
camera TF and world cloud. Run `scripts/capture_gazebo_layout.py` to persist them in
`config/gazebo_layout.yaml`.

## Initial extrinsics

The camera position is calculated from workspace center, height above table, horizontal
offset, lateral offset, pitch, and yaw. It is mounted above/behind the workspace and pitched
58° downward. The smoke view covers all six cubes, all three bins, the yellow edge, the
calibration marker, and the side hand-entry region.

The Isaac camera looks along local `-Z` with `+Y` up. The bridge publishes a 180° X rotation
from `rgbd_link` to each ROS optical frame so the optical convention is x-right, y-down,
z-forward.

## Intrinsics

For width `w`, height `h`, and horizontal field of view `θ`:

```text
fx = fy = w / (2 tan(θ/2))
cx = (w - 1) / 2
cy = (h - 1) / 2
```

The same intrinsic matrix is used for the aligned depth image and back-projected cloud.
Depth encoding is `32FC1` in metres. RGB/depth/CameraInfo share one timestamp.

## Aligning to the physical station

1. Measure table top height and the camera optical-center pose from the OpenArm base.
2. Update `scene.yaml` and `camera.yaml`; leave all values in metres/degrees as labelled.
3. Replace the visual marker placeholder with a legally sourced detector-valid tag before
   tag-based calibration. The default black/white graphic is deliberately not claimed to be
   a valid AprilTag.
4. Capture an RGB/depth pair with `run_sim.py --capture-dir <dir>`.
5. Check `/rgbd/points` against the table plane in RViz and verify positive optical Z.
6. Record the changed YAML in each evaluator config snapshot.

Noise, latency, and drop are disabled by default. Enable them in YAML or use
`OPENARM_CAMERA_LATENCY_MS` and `OPENARM_CAMERA_FRAME_DROP_PROBABILITY` for isolated fault
runs.
