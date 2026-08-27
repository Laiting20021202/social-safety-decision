from setuptools import find_packages, setup

package_name = "hand_obstacle_controller"
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
    description="Hand scenario control services and keyboard UI.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "hand_controller = hand_obstacle_controller.node:main",
            "hand_keyboard = hand_obstacle_controller.keyboard:main",
        ]
    },
)
