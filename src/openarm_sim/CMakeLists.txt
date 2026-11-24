cmake_minimum_required(VERSION 3.8)
project(openarm_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(openarm_sim)

install(DIRECTORY config scripts assets worlds
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "openarm_cache" EXCLUDE
  PATTERN "libhand" EXCLUDE
)

install(DIRECTORY gazebo
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
