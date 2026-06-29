# Jetson Deployment

Jetson support is scaffolded but not validated in Phase 1.

Use Jetson-specific Dockerfiles under `docker/jetson/` and match the base image to the target JetPack/L4T release. Do not assume an x86 CUDA image is valid on arm64 Jetson devices.

When memory is constrained, use lower resolution, reduced tracking FPS, fewer temporal frames, event-triggered VQA, sequential model loading, CPU offload, or quantization only when officially supported by the model.
