from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='Extract motifs',
    ext_modules=cythonize("extract_motifs.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)

setup(
    name='Extract fragments',
    ext_modules=cythonize("extract_fragments.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
)