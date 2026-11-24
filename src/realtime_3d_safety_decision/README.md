# Realtime 3D Safety Decision

This ROS 2 package tracks obstacles in metric 3D point clouds and publishes the obstacle cloud and tracked-object messages. It also contains the perception experiments, web viewer and recorded-data tools used in the earlier submissions.

- `launch/` and `config/` contain the ROS launch descriptions and perception parameters.
- `msg/` defines `TrackedObstacle` and `TrackedObstacleArray`.
- `realtime_safety/` implements camera input, reconstruction, tracking, risk estimation and visualization.
- `scripts/`, `tools/` and `benchmarks/` contain the host stream and evaluation tools.

See the [workspace README](../../README.md) for the Docker pipeline and [code map](../../docs/CODE_MAP.md) for the individual entry points.
