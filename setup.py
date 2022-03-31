from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setup(
    name="tab2grid",
    version="0.0.1",
    author="Miquel Duran-Frigola",
    author_email="miquel@ersilia.io",
    url="https://github.com/ersilia-os/tab2grid",
    description="Griddify high-dimensional tabular data for easy visualization and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages(exclude=("utilities")),
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="data-visualization",
    project_urls={"Source Code": "https://github.com/ersilia-os/tab2grid"},
    include_package_data=True,
)