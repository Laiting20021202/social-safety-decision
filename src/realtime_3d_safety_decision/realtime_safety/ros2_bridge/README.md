# ROS 2 topics

ROS remains optional unless a ROS publishing flag is selected. The live viewer
can publish reconstructed points directly as:

- `/realtime_safety/pointcloud` (`sensor_msgs/PointCloud2`): latest bounded RGB point cloud.
- `/realtime_safety/yolo_obstacles/pointcloud` (`sensor_msgs/PointCloud2`):
  depth-filtered 3D clusters inside confirmed person masks in the Koch profile.
  ByteTrack keeps IDs, while motion-aligned temporal fusion fills brief mask
  and depth holes. The conservative hold lasts at most 24 reconstruction
  updates; a zero-width message then clears genuinely stale obstacles.

Start the camera input, local ROS preview, LAN GUI, and point-cloud topic
together:

```bash
source /opt/ros/humble/setup.bash
bash scripts/run_koch_stream.sh
```

Or enable it on any source explicitly:

```bash
python app.py \
  --source /path/to/video.mp4 \
  --pointcloud-topic /realtime_safety/pointcloud \
  --yolo-obstacle-pointcloud-topic /realtime_safety/yolo_obstacles/pointcloud \
  --pointcloud-frame-id realtime_safety_frame \
  --ros-domain-id 42
```

The publisher uses reliable, volatile, keep-last-1 QoS. That is compatible with
RViz's default reliable reader and the Koch computer's best-effort sensor-data
reader. The Koch launch script uses domain 42 and `rmw_cyclonedds_cpp`, selects
the LAN interface, enables SPDP multicast for domain-wide discovery, and keeps
`192.168.0.231` as a unicast fallback peer. The default camera path reads compressed MJPEG from
the camera's ROS `web_video_server`, then republishes decoded frames on
`/realtime_safety/camera/image_raw` for local RQT use. This prevents a second
153–230 KB raw DDS stream from crossing Wi-Fi. Direct raw ROS input remains
available with `CAMERA_SOURCE=ros2:///my_camera/image_raw`.
`ParticipantIndex=auto` allows multiple ROS processes on one computer without
a UDP port collision. All endpoints need the same
`ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=0`, and RMW installed. Point coordinates
on the Koch ROS topics use the controller's `camera_y_forward` contract:
`x-right, y-forward/depth, z-down`, in metres. Reconstruction and the GUI stay
internally `z-up`; `run_koch_stream.sh` performs the explicit z-axis conversion
only while packing the ROS messages. Generic launches can select either
`--pointcloud-coordinate-mode internal_z_up` or
`--pointcloud-coordinate-mode camera_y_forward`.

Source `scripts/ros2_lan_env.sh` in every ROS terminal. On another computer,
set `ROS_LAN_PEER=192.168.0.234` first so unicast discovery still works if the
Wi-Fi access point filters multicast.

The LAN launcher targets 12 Hz and uses the latest 8,000 points. PointCloud2
packing and DDS publication use latest-only worker threads, so network latency
cannot build a backlog or stall reconstruction.
Use `bash scripts/run_rviz_pointcloud.sh` to open the supplied RViz view with
the correct fixed frame, topic, packed-RGB transformer, and QoS.

Measure liveness and wire layout for 10 seconds without moving the robot:

```bash
python3 scripts/measure_pointcloud_topic.py \
  /realtime_safety/yolo_obstacles/pointcloud --duration 10
```

An empty scene should report frames at roughly 10–12 Hz with
`max_points=0`; that is a healthy explicit "no obstacle" signal. With a person
in view, add `--require-nonempty`; `max_points` and `nonempty_frames` must both
be positive. Every nonempty message contains finite FLOAT32 `x/y/z` at offsets
0/4/8, a packed RGB field at offset 12, `point_step=16`, and
`len(data)=row_step=width*16`.

The Koch launch also publishes
`/realtime_safety/arm_obstacle_relationships` as reliable
`std_msgs/msg/String` JSON. It is a small, continuous liveness and tracking
contract containing the arm center, obstacle centers, velocity, radius,
missing-count state, and metric distances. Its axes match the Koch
PointCloud2 wire convention: x-right, y-forward, z-down in metres. Inspect it
with:

```bash
ros2 topic hz /realtime_safety/arm_obstacle_relationships
ros2 topic echo --once /realtime_safety/arm_obstacle_relationships
```

The full Koch control-side implementation prompt is in
`docs/KOCH_NUC_AVOIDANCE_PROMPT.md`.

The separate safety bridge tails the streaming `safety.jsonl` file and publishes:

- `/realtime_safety/state` (`std_msgs/String`): the complete latest JSON safety update.
- `/realtime_safety/recommended_cmd_vel` (`geometry_msgs/Twist`): conservative demo velocity recommendation.

Run after sourcing a ROS 2 environment:

```bash
python -m realtime_safety.ros2_bridge.safety_bridge_node --jsonl sessions/<session>/safety.jsonl
```

`STOP`, `WAIT`, or `DEGRADED` publishes zero velocity. `SLOW_DOWN` publishes 0.1 m/s only when `metric_valid=true`; otherwise all linear velocities remain zero because relative-scale video is not a metric robot control source.
