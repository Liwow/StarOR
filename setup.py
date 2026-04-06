from setuptools import find_packages, setup

setup(
    packages=find_packages(include=("ttrl_or", "ttrl_or.*", "verl", "verl.*")),
    include_package_data=True,
)
