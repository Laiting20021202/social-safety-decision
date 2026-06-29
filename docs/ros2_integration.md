# ROS 2 Integration

ROS 2 Humble integration is optional and not required for Phase 1.

Future subscriptions:

- `/camera/color/image_raw`
- `/camera/depth/image_rect_raw`
- `/camera/color/camera_info`
- `/tf`
- `/tf_static`
- `/odom`

Future publications:

- `/social_safety/zone`
- `/social_safety/tracks`
- `/social_safety/vqa_result`
- `/social_safety/state`
- `/social_safety/diagnostics`
- `/navigation_pause`
- `/social_safety/speed_limit`

The adapter must not publish arbitrary `/cmd_vel`.
