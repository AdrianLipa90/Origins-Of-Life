from setuptools import setup, find_packages

setup(
    name="origins-of-life-holonomic",
    version="1.0.0",
    author="Adrian Lipa",
    author_email="",
    description=(
        "Computational abiogenesis and protocell-emergence simulator with "
        "phenomenological chemistry and experimental geometric operators"
    ),
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.9",
    ],
    extras_require={
        "plot":  ["matplotlib>=3.6"],
        "speed": ["numba>=0.57"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
