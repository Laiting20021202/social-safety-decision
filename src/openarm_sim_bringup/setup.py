from glob import glob

from setuptools import find_packages, setup

package_name = "openarm_sim_bringup"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenArm simulation maintainers",
    maintainer_email="openarm-sim@example.invalid",
    description="OpenArm Isaac Sim and Gazebo launch package.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "isaac_planning_scene = openarm_sim_bringup.isaac_planning_scene:main",
            "gazebo_rgbd_adapter = openarm_sim_bringup.gazebo_rgbd_adapter:main",
            "gazebo_home_pose = openarm_sim_bringup.gazebo_home_pose:main",
            "gazebo_hand_controller = openarm_sim_bringup.gazebo_hand_controller:main",
            "gazebo_openarm_gateway = openarm_sim_bringup.gazebo_openarm_gateway:main",
            "gazebo_scene_pose_sync = openarm_sim_bringup.gazebo_scene_pose_sync:main",
            "gazebo_planning_scene = openarm_sim_bringup.gazebo_planning_scene:main",
        ],
    },
)
