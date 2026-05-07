from setuptools import find_packages, setup

setup(
    packages=find_packages(include=("verl", "verl.*")),
    include_package_data=True,
)
