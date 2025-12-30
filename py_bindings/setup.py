from setuptools import setup

setup(
    name="pandrs",
    version="0.1.0",
    description="Rust-powered DataFrame library for Python with pandas-like API",
    author="COOLJAPAN OU (Team KitaSan)",
    author_email="contact@cooljapan.tech",
    url="https://github.com/cool-japan/pandrs",
    packages=["pandrs"],
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)