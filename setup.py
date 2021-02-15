"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from glob import glob
from os import path

from setuptools import setup
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext

BASE_DIR = path.dirname(path.abspath(__file__))
CHANGELOG_PATH = path.join(BASE_DIR, "CHANGELOG")

with open(CHANGELOG_PATH, "r") as f:
    __version__ = f.readline()
    __version__ = __version__.split()
    __version__ = __version__[1][1:-1]

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ext_modules = [
    Pybind11Extension(
        # Ref: distutils.extension.Extension
        name="yolov4.common._common",
        sources=sorted(glob("c_src/**/*.cpp", recursive=True)),
        include_dirs=["c_src/"],  # -I
        define_macros=[(("VERSION_INFO", __version__))],  # -D<string>=<string>
        undef_macros=[],  # [string] -D<string>
        library_dirs=[],  # [string] -L<string>
        libraries=[],  # [string] -l<string>
        runtime_library_dirs=[],  # [string] -rpath=<string>
        extra_objects=[],  # [string]
        extra_compile_args=[],  # [string]
        extra_link_args=[],  # [string]
    ),
]


setup(
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
