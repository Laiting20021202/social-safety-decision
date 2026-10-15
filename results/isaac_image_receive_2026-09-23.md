# Isaac LAN image receive recovery — 2026-09-23

Scope: receive the current Isaac front camera on the GUI workstation over
Wi-Fi, reconstruct its RGB-estimated point cloud, and monitor OpenArm joints.
No arm commands or avoidance acceptance tests were run.

## Fault and change

The actual camera topic is `/camera/front/color/image_raw` (`Image`); no
compressed-image publisher was present. The actual robot feedback topic is
`/tidybot/joint_states`. The GUI launcher now defaults to raw Image and accepts
`ISAAC_CAMERA_TOPIC` for an explicit alternative.

Before network tuning, a 25-second independent best-effort subscriber received
314 JointStates and 152 CameraInfos, but zero Images and zero generated clouds.
An additional reliable Image subscription received zero messages in 15 seconds.
System IP fragment-reassembly failures increased while small messages arrived.

The user applied the following temporary receiver-side kernel settings through
the local sudo terminal; no persistent sysctl configuration was written:

| Parameter | Before | After |
|---|---:|---:|
| `net.ipv4.ipfrag_time` | 30 seconds | 3 seconds |
| `net.ipv4.ipfrag_high_thresh` | 4194304 bytes | 134217728 bytes |

## Post-change ROS measurement

Independent `rclpy` subscribers using `qos_profile_sensor_data` sampled for
30 seconds while the existing GUI remained running:

| Stream | Messages | Receive rate | Largest inter-message gap |
|---|---:|---:|---:|
| Raw front image | 141 | 4.69 Hz | 1.013 s |
| Generated point cloud | 147 | 4.89 Hz | 1.052 s |
| Tidybot joint state | 351 | 11.69 Hz | 0.388 s |

Images were 640×480 RGB8 (921600 bytes), with 141 distinct timestamps and
141 distinct payload checksums. Every cloud contained 12000 points. Each of
the six 5-second sampling windows contained both images and clouds.
Base joint positions remained constant during this sample; this test does
not prove camera registration or tracking during commanded base motion.

System-wide counter deltas during the sample: `IpReasmReqds +228329`,
`IpReasmOKs +5347`, `IpReasmFails +84`, `UdpRcvbufErrors +0`. These counters
include other traffic and are not per-camera packet-loss measurements.
GUI HTTP returned 200. Image delivery has recovered, but the remaining gaps
and fragment failures mean this is not a long-term network stability result.

A separate 12-second read-only WebSocket client using the installed Viser
protocol received 115 GUI image-data updates and 88 scene point-array updates.
This verifies live GUI delivery beyond ROS input; it does not constitute a
visual inspection of the user's browser or a robot-motion test.

## Restart and rollback

Launch the GUI from the repository root with `bash scripts/run_isaac_gui.sh`.
The kernel tuning lasts only until reboot or another configuration change.
If the receive issue returns after reboot, follow the documented check/tuning
steps in [the LAN guide](../docs/isaac_lan_gui.md).

Restore the receiver's recorded previous settings with:

```bash
sudo sysctl -w net.ipv4.ipfrag_time=30 net.ipv4.ipfrag_high_thresh=4194304
```

The point cloud still comes from RGB depth estimation with unverified metric
scale. Recovery of image delivery does not complete the Isaac avoidance loop.
