# Get version
import re

name = "chromatinhd"

import setuptools

VERSIONFILE = "src/" + name + "/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# get long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=name,
    version=version,
    author="Wouter Saelens",
    author_email="wouter.saelens@gmail.com",
    description="Modeling of chromatin + transcriptomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeplanckeLab/ChromatinHD",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",  # --extra-index-url https://download.pytorch.org/whl/cu113
        "torch-scatter",
        # "torch==1.12.1",  # --extra-index-url https://download.pytorch.org/whl/cu113
        # "torch-scatter",  # --find-links https://data.pyg.org/whl/torch-1.12.1+cu113.html
        "scanpy",
        "matplotlib",
        "numpy",
        "seaborn",
        "Cython",
        "fisher",
        "diskcache",
        "appdirs",
        "xarray",
        "pysam",
    ],
    extras_require={
        "full": [],
        "dev": [
            "pre-commit",
            "pytest",
            "coverage",
            "black",
            "pylint",
            "jupytext",
            "pytest",
            "statsmodels",
            "mkdocs",
            "mkdocs-material",
            "mkdocstrings[python]",
            "mkdocs-jupyter",
            "mike",
            "cairosvg",  # for mkdocs social
            "pillow",  # for mkdocs social
            # "faiss-cpu",
        ],
        "eqtl": ["cyvcf2", "xarray"],
    },
)
