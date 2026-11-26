ARG BASE_IMAGE=openarm-dev:latest
FROM ${BASE_IMAGE} AS model
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/right-curl
COPY description/ /opt/right-curl/share/openarm_description/
RUN mkdir -p share/ament_index/resource_index/packages && touch share/ament_index/resource_index/packages/openarm_description
COPY app/env.sh /opt/right-curl/app/
COPY fastdds.xml /opt/right-curl/
ENV ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTRTPS_DEFAULT_PROFILES_FILE=/opt/right-curl/fastdds.xml PYTHONUNBUFFERED=1
ENTRYPOINT ["/bin/bash", "/opt/right-curl/app/env.sh"]

FROM model AS viewer
COPY app/right.rviz app/curl.py app/observe.py /opt/right-curl/app/
RUN mkdir -p lib/openarm_right_curl && install -m 755 app/curl.py lib/openarm_right_curl/curl \
    && touch share/ament_index/resource_index/packages/openarm_right_curl
CMD ["rviz2", "-d", "/opt/right-curl/app/right.rviz", "--ros-args", "-r", "__node:=right_curl_rviz"]

FROM model AS robot
COPY camera-source.xml /opt/right-curl/
COPY camera/ /tmp/camera/
RUN source /opt/ros/humble/setup.bash && MAKEFLAGS=-j3 colcon build --base-paths /tmp/camera \
      --merge-install --build-base /tmp/camera-build --install-base /opt/laiting-camera \
      --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    && rm -rf /tmp/camera /tmp/camera-build log
COPY can/ /tmp/openarm_can/
RUN cmake -S /tmp/openarm_can -B /tmp/can-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build /tmp/can-build -j2 && cmake --install /tmp/can-build && rm -rf /tmp/can-build /tmp/openarm_can
COPY app/CMakeLists.txt app/package.xml app/plugin.xml app/right_hardware.cpp app/right.launch.py app/controllers.yaml app/camera_relay.py /opt/right-curl/app/
RUN source /opt/ros/humble/setup.bash && cmake -S app -B /tmp/right-build \
      -DCMAKE_INSTALL_PREFIX=/opt/right-curl -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && cmake --build /tmp/right-build -j2 && cmake --install /tmp/right-build && rm -rf /tmp/right-build
CMD ["ros2", "launch", "/opt/right-curl/app/right.launch.py"]
