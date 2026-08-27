cmake_minimum_required(VERSION 3.8)
project(realtime_3d_safety_decision)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

ament_python_install_package(realtime_safety)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedObstacle.msg"
  "msg/TrackedObstacleArray.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

install(PROGRAMS
  scripts/edgetam_pointcloud_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING
  PATTERN "*.launch.py"
  PATTERN "__pycache__" EXCLUDE
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
