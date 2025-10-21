# setup.py
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "pyqreg.c.blas_lapack",
        ["src/pyqreg/c/blas_lapack.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        # TODO: fix type conversion warnings instead of suppressing them
        extra_compile_args=["/wd4244"] if sys.platform == "win32" else [],  # disables C4244
    ),
    Extension(
        "pyqreg.c.cluster_cov",
        ["src/pyqreg/c/cluster_cov.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "pyqreg.c.fit_coefs",
        ["src/pyqreg/c/fit_coefs.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/wd4244"] if sys.platform == "win32" else [],  # disables C4244
    ),
    Extension(
        "pyqreg.c.mat_vec_ops",
        ["src/pyqreg/c/mat_vec_ops.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "pyqreg.c.matrix_opaccum",
        ["src/pyqreg/c/matrix_opaccum.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["/wd4244"] if sys.platform == "win32" else [],  # disables C4244
    ),
    Extension(
        "pyqreg.c.stats",
        ["src/pyqreg/c/stats.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
