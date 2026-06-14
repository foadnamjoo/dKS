from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "dks",
        ["bindings/python/dks_py.cpp"],
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(
    name="dks",
    version="0.1.0",
    description="Multi-dimensional Kolmogorov-Smirnov distance (d=2)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    py_modules=[],
    packages=[],            # pure C-extension: no Python packages to discover
    python_requires=">=3.8",
    install_requires=["numpy"],
)
