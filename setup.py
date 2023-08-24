from setuptools import setup, Extension
import numpy

setup_args = dict(
    ext_modules=[
        Extension(
            "chromatinhd.loaders.fragments_helpers",
            ["src/chromatinhd/loaders/fragments_helpers.c"],
            include_dirs=[numpy.get_include()],
            py_limited_api=True,
        ),
    ]
)
setup(**setup_args)
