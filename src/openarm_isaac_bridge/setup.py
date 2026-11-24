from setuptools import find_packages, setup

package_name = "openarm_isaac_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenArm simulation maintainers",
    maintainer_email="openarm-sim@example.invalid",
    description="ROS 2 runtime bridge embedded in Isaac Sim.",
    license="Apache-2.0",
)

