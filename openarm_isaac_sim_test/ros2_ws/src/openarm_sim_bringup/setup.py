import os
from pathlib import Path

from setuptools import find_packages, setup

package_name = "openarm_sim_bringup"
project_root = Path(__file__).resolve().parents[3]
package_root = Path(__file__).resolve().parent


def relative(path: Path) -> str:
    return os.path.relpath(path, package_root)

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [relative(path) for path in (project_root / "launch").glob("*.launch.py")],
        ),
        (
            "share/" + package_name + "/rviz",
            [relative(path) for path in (project_root / "rviz").glob("*.rviz")],
        ),
        (
            "share/" + package_name + "/config",
            [relative(path) for path in (project_root / "config").glob("*.yaml")],
        ),
        (
            "share/" + package_name + "/scripts",
            [relative(path) for path in (project_root / "scripts").glob("*.py")],
        ),
        (
            "share/" + package_name + "/gazebo/worlds",
            [relative(path) for path in (project_root / "gazebo/worlds").glob("*.world")],
        ),
        (
            "share/" + package_name + "/gazebo/generated",
            [relative(path) for path in (project_root / "gazebo/generated").glob("*.urdf")],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenArm simulation maintainers",
    maintainer_email="openarm-sim@example.invalid",
    description="OpenArm Isaac Sim and Gazebo launch package.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gazebo_rgbd_adapter = openarm_sim_bringup.gazebo_rgbd_adapter:main",
            "gazebo_home_pose = openarm_sim_bringup.gazebo_home_pose:main",
            "gazebo_hand_controller = openarm_sim_bringup.gazebo_hand_controller:main",
            "gazebo_openarm_gateway = openarm_sim_bringup.gazebo_openarm_gateway:main",
            "gazebo_scene_pose_sync = openarm_sim_bringup.gazebo_scene_pose_sync:main",
            "gazebo_planning_scene = openarm_sim_bringup.gazebo_planning_scene:main",
        ],
    },
)
