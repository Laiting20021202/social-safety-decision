# Jetson Docker Images

These Dockerfiles target NVIDIA Jetson arm64 devices and must be matched to the installed JetPack/L4T version. They are not interchangeable with x86 CUDA images.

Phase 1 only validates the dataset playback path. SAM 3, RoboPoint, and VQA model loading must be validated on the target Jetson hardware before any formal experiment.

Suggested mitigations for memory pressure:

- lower input resolution
- reduce tracking FPS
- reduce temporal VQA frames
- use event-triggered VQA
- sequential model loading
- CPU offload
- quantization only when officially supported
