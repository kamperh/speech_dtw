from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

ext = [
    Extension("samediff._dtw_cost", ["_dtw_cost.pyx"], include_dirs=[np.get_include()]),
    Extension("samediff._dtw", ["_dtw.pyx"], include_dirs=[np.get_include()]),
    ]

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext,
    packages=["samediff"],
    package_dir={"samediff": "."})

# from Cython.Build import cythonize
# setup(ext_modules=cythonize("_dtw_cost.pyx"))
