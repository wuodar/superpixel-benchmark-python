import os
import shutil

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

# clean previous build
for root, dirs, files in os.walk(".", topdown=False):
    if root != os.path.join(".", "lib"):
        for name in files:
            if name.endswith((".cpp", ".h", ".so")):
                os.remove(os.path.join(root, name))
    for name in dirs:
        if name in ["build", "cython_debug"]:
            shutil.rmtree(name)

include_dirs = [
    numpy.get_include(),
    "/usr/include/opencv4",
    "/usr/include/opencv4/opencv2",
]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="superpixel_benchmark",
    version="0.0.1",
    cmdclass={"build_ext": build_ext},
    author="Kacper Włodarczyk",
    description="Python wrapper for superpixel-benchmark oryginally written by David Stutz in C++.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=cythonize(
        [
            Extension(
                name="superpixel_tools",
                sources=[os.path.join("src", "superpixel_tools.pyx")],
                include_dirs=include_dirs,
                language="c++",
                extra_compile_args=["-O0"],
            ),
            Extension(
                name="superpixel_benchmark",
                sources=[os.path.join("src", "superpixel_benchmark.pyx")],
                include_dirs=include_dirs,
                language="c++",
                extra_compile_args=["-O0"],
            ),
            Extension(
                name="opencv_mat",
                sources=[os.path.join("src", "opencv_mat.pyx")],
                include_dirs=include_dirs,
                libraries=[
                    "opencv_core",
                    "opencv_imgproc",
                    "opencv_imgcodecs",
                ],
                language="c++",
                extra_compile_args=["-O0"],
            ),
        ],
        gdb_debug=True,
    ),
    platforms=["linux_x86_64"],
)
