---
name: ros2-docker-multiarch
description: Design, implement, and review ROS 2 Humble and multi-architecture Docker deployment for social-safety AMR systems across x86 CPU, x86 NVIDIA CUDA, Jetson arm64 JetPack/L4T, and optional ROS 2 profiles with healthchecks and architecture validation.
---

# ROS 2 Docker Multiarch

Use this workflow when changing Docker, Compose, ROS 2, CUDA, or Jetson deployment files.

## Workflow

1. Identify target profile:
   - `dataset-demo`
   - `cpu-demo`
   - `cuda-full`
   - `jetson`
   - `ros2`
   - `ci`
2. Validate architecture assumptions:
   - do not use x86 CUDA images for Jetson
   - match Jetson images to JetPack/L4T
   - keep ROS 2 Humble in an optional profile
3. Keep datasets and model weights out of images.
4. Use named volumes:
   - `hf_cache`
   - `sam3_weights`
   - `robopoint_weights`
   - `vqa_weights`
   - `dataset_cache`
   - `experiment_outputs`
   - `zone_configs`
5. Add healthchecks for each service.
6. Pass secrets through environment variables, not committed files.
7. Restrict ROS 2 outputs:
   - allow `/navigation_pause`
   - allow `/social_safety/speed_limit`
   - never publish arbitrary `/cmd_vel`

## Required Checks

- compose profile starts only required services
- ROS 2 is not required for dataset playback
- camera devices are not required for dataset playback
- NVIDIA runtime is only required by GPU profiles
- Jetson files document L4T compatibility
- each service has graceful shutdown and restart policy

## Failure Conditions

- model weights baked into image
- dataset baked into image
- token committed to repo
- single CUDA image claimed to support x86 and Jetson
- dataset demo requires ROS 2, RealSense, or `/dev/video`

## Output Format

Report:

- changed files
- target architectures
- required host prerequisites
- build/run commands
- unvalidated profiles and blocking reasons
