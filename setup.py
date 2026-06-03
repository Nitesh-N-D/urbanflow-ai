from setuptools import find_packages, setup

setup(
    name="gridlock2",
    version="2.0.0",
    description="Bengaluru traffic intelligence pipeline for Flipkart Gridlock 2.0",
    packages=find_packages(where="."),
    python_requires=">=3.10",
)
