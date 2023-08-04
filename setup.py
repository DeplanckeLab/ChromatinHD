from setuptools import setup, Extension
import numpy

setup_args = dict(
    ext_modules=[
        Extension(
            "chromatinhd.models.pred.loader.fragments_helpers",
            ["src/chromatinhd/models/diff/loader/fragments_helpers.c"],
            include_dirs=[numpy.get_include()],
            py_limited_api=True,
        ),
        Extension(
            "chromatinhd.models.pred.loader.fragments_helpers",
            ["src/chromatinhd/models/pred/loader/fragments_helpers.c"],
            include_dirs=[numpy.get_include()],
            py_limited_api=True,
        ),
    ]
)
setup(**setup_args)
