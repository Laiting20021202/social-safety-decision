from setuptools import find_packages, setup

package_name = "openarm_dynamic_avoidance"

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
    description="Safety-gated dynamic trajectory layer for OpenArm.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dynamic_avoidance = openarm_dynamic_avoidance.node:main",
        ]
    },
)
