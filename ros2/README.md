# Optional ROS2 Adapter

This directory contains an optional ROS2 Humble node. The core `social_bev`
package does not depend on ROS2.

Run only in an environment that already has ROS2 Humble, `rclpy`, standard
message packages, and `cv_bridge` installed:

```bash
python ros2/social_bev_node.py --config configs/default.yaml --calibration configs/calibration.yaml
```

Subscriptions:

- `/camera/image_raw`

Publications:

- `/social_bev/annotated`
- `/social_bev/walkable_mask`
- `/social_bev/occupancy_grid`
- `/social_bev/people_markers`

