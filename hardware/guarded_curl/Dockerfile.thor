FROM pure-mast3r-d455-unity:self-contained
COPY . /opt/laiting/
WORKDIR /workspace/thor/MASt3R-SLAM
ENV ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_DEFAULT_PROFILES_FILE=/opt/laiting/fastdds.xml FASTRTPS_DEFAULT_PROFILES_FILE=/opt/laiting/fastdds.xml \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && export PYTHONPATH=/workspace/thor/MASt3R-SLAM:/workspace/thor/MASt3R-SLAM/thirdparty/in3d:$PYTHONPATH && exec python3 /opt/laiting/thor_cloud.py"]
