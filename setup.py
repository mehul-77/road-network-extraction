from setuptools import setup, find_packages

setup(
    name="road-network-extraction",
    version="1.0.0",
    description="Automated road network feature extraction from satellite imagery for ITS applications",
    author="Mehul",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
