from setuptools import setup, Extension
import pybind11

cpp_args = ['/std:c++17', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension(
    'dragonchess',
    sources=['DragonChess.cpp'],
    include_dirs=[pybind11.get_include(), "C:\\Program Files\\boost\\boost_1_81_0"],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='dragonchess',
    version='1.0',
    description='Python package with superfastcode2 C++ extension (PyBind11)',
    ext_modules=[sfc_module],
)