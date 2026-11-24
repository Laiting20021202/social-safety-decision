from setuptools import find_packages, setup

package_name = "openarm_sorting_task"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="OpenArm simulation maintainers",
    maintainer_email="openarm-sim@example.invalid",
    description="MoveIt-based OpenArm sorting state machine.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sorting_task = openarm_sorting_task.node:main",
            "pose_goal = openarm_sorting_task.pose_goal:main",
        ]
    },
)
