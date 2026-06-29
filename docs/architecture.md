# Architecture

`social-safety-amr` is a microservice-oriented system. Phase 1 runs without camera, AMR, ROS 2, SAM 3, RoboPoint, or VQA model runtime, while keeping the same boundaries required for later phases.

## Services

- `dataset-service`: dataset discovery, SocialNav-SUB frame loading, playback state, WebSocket updates, and zone persistence.
- `geometry-service`: planned independent service for track history, zone relations, and time-to-zone prediction. Core geometry functions are already implemented as local packages for testing.
- `sam3-service`: planned isolated SAM 3 image segmentation and video tracking runtime.
- `robopoint-service`: planned isolated RoboPoint language grounding runtime.
- `vqa-service`: planned provider interface for Qwen, SmolVLM, and CI-only mock.
- `safety-orchestrator`: planned event-driven controller combining tracking, geometry, VQA, and hard rules.
- `web-gui`: React/Vite TypeScript browser UI. It does not depend on RViz.
- `ros2-adapter`: optional future ROS 2 Humble profile. The main system runs without ROS 2.

## Phase 1 Data Flow

SocialNav-SUB Hugging Face frames or local mirrored frames flow through `HuggingFaceDatasetSource`, into the playback manager, through the dataset-service API/WebSocket, and into the GUI. Manual zone polygons are edited in the browser and persisted under `config/zones/<dataset>/<scenario>.json`.

## Safety Boundary

No VQA/VLM component is allowed to publish arbitrary velocity commands. Future ROS 2 output is limited to safety topics such as `/navigation_pause` and `/social_safety/speed_limit`.
