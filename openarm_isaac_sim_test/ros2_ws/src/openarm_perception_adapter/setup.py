from setuptools import find_packages, setup

package_name = "openarm_perception_adapter"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="OpenArm simulation maintainers",
    maintainer_email="openarm-sim@example.invalid",
    description="Synchronized RGB-D mask to metric obstacle cloud adapter.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rgbd_mask_adapter = openarm_perception_adapter.node:main",
            "realtime_obstacle_resampler = openarm_perception_adapter.realtime_obstacle_resampler:main",
        ]
    },
)
