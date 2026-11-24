from setuptools import find_packages, setup

package_name = "openarm_safety_bridge"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="OpenArm simulation maintainers",
    maintainer_email="openarm-sim@example.invalid",
    description="Ground-truth/perception isolated MoveIt safety bridge.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["safety_bridge = openarm_safety_bridge.node:main"]},
)

