# Optional ROS 2 bridge

The core application has no ROS dependency. The bridge tails the streaming `safety.jsonl` file and publishes:

- `/realtime_safety/state` (`std_msgs/String`): the complete latest JSON safety update.
- `/realtime_safety/recommended_cmd_vel` (`geometry_msgs/Twist`): conservative demo velocity recommendation.

Run after sourcing a ROS 2 environment:

```bash
python -m realtime_safety.ros2_bridge.safety_bridge_node --jsonl sessions/<session>/safety.jsonl
```

`STOP`, `WAIT`, or `DEGRADED` publishes zero velocity. `SLOW_DOWN` publishes 0.1 m/s only when `metric_valid=true`; otherwise all linear velocities remain zero because relative-scale video is not a metric robot control source.
