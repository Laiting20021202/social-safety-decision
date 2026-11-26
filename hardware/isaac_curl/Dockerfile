ARG SAFETY_IMAGE=laiting-openarm-safety:humble
FROM ${SAFETY_IMAGE}
SHELL ["/bin/bash","-c"]
COPY CMakeLists.txt package.xml plugin.xml sim_bridge.cpp /tmp/bridge/
RUN source /opt/ros/humble/setup.bash && cmake -S /tmp/bridge -B /tmp/build -DCMAKE_INSTALL_PREFIX=/opt/isaac-hw -DBUILD_TESTING=OFF && cmake --build /tmp/build -j2 && cmake --install /tmp/build && rm -rf /tmp/bridge /tmp/build
RUN python3 -m pip install --no-cache-dir --no-deps python-fcl==0.7.0.8 trimesh==4.10.0
