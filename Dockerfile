ARG ROS_IMAGE=ros:humble-ros-base-jammy@sha256:a18afd4a42ad603579d03b9012385dfeaf2e848423f0a6c41699766e7849c86c
ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:4.5.0@sha256:c2f47dc82a7714af08d3766efe80ac9d084c2b37b5d0dfbd074797ec56390fc7

FROM ${ROS_IMAGE} AS ros_distribution

FROM ros_distribution AS control
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-numpy python3-opencv python3-pytest python3-scipy \
    ros-humble-control-msgs ros-humble-gazebo-msgs ros-humble-message-filters \
    ros-humble-moveit-kinematics ros-humble-moveit-planners-ompl \
    ros-humble-moveit-ros-move-group ros-humble-moveit-simple-controller-manager \
    ros-humble-robot-state-publisher ros-humble-rosidl-default-generators \
    ros-humble-sensor-msgs-py ros-humble-tf2-ros-py ros-humble-visualization-msgs ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/social-safety-decision
COPY openarm.repos ./
COPY scripts/fetch_openarm_dependencies.sh scripts/
RUN bash scripts/fetch_openarm_dependencies.sh
COPY src/ src/
COPY docker/ docker/
RUN source /opt/ros/humble/setup.bash && \
    colcon build --merge-install --build-base /tmp/social-safety-build \
      --install-base /opt/social-safety-ros \
      --base-paths src \
      --cmake-args -DBUILD_TESTING=OFF && \
    source /opt/social-safety-ros/setup.bash && \
    description_dir="$PWD/.openarm-vendor/openarm_description" && \
    mkdir -p "$description_dir/generated" && \
    xacro "$description_dir/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro" \
      bimanual:=true no_prefix:=false ros2_control:=false \
      -o "$description_dir/generated/openarm_v10_bimanual.urdf" && \
    rm -rf /tmp/social-safety-build /opt/social-safety-decision/log

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/bin/bash", "/opt/social-safety-decision/docker/entrypoint.sh"]
CMD ["ros2", "launch", "openarm_control_bringup", "control.launch.py"]

FROM ${ISAAC_IMAGE} AS isaac
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=ros_distribution /usr/share/ros-apt-source/ros2.sources /etc/apt/sources.list.d/ros2.sources
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-control-msgs ros-humble-moveit-msgs ros-humble-rclpy \
    ros-humble-rmw-fastrtps-cpp ros-humble-rosgraph-msgs ros-humble-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/social-safety-decision/src/openarm_sim
COPY src/openarm_sim/ .
COPY src/openarm_isaac_bridge/ ../openarm_isaac_bridge/
COPY docker/entrypoint.sh /opt/social-safety-decision/docker/entrypoint.sh
COPY docker/fastdds.xml /opt/social-safety-decision/docker/fastdds.xml
COPY --from=control /opt/social-safety-decision/.openarm-vendor/openarm_description /opt/openarm-description
ENV OPENARM_SIM_ROOT=/opt/social-safety-decision/src/openarm_sim \
    OPENARM_DESCRIPTION_ROOT=/opt/openarm-description \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["/bin/bash", "/opt/social-safety-decision/docker/entrypoint.sh"]
CMD ["/isaac-sim/python.sh", "scripts/run_sim.py", "--headless", "--mode", "perception"]
