from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

ext = [
    # Extension("speech_dtw._dtw_cost", ["speech_dtw/_dtw_cost.pyx"], include_dirs=[np.get_include()]),
    Extension("speech_dtw._dtw", ["speech_dtw/_dtw.pyx"], include_dirs=[np.get_include()]),
    ]

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext,
    packages=["speech_dtw"],
    package_dir={"speech_dtw": "speech_dtw"})

# from Cython.Build import cythonize
# setup(ext_modules=cythonize("_dtw_cost.pyx"))
