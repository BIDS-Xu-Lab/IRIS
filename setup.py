from setuptools import setup, Extension

import os
import numpy
import sys
import sysconfig
import platform

extra_link_args = []
if platform.system() == 'Darwin':
    extra_link_args.append('-lomp')

LargeVis = Extension('IRIS.Utils',
                    sources = ['LargeVis/LargeVis.cpp', 'LargeVis/LargeVismodule.cpp'],
                    depends=['LargeVis/LargeVis.h'],
                    include_dirs = ['./include', numpy.get_include()],
                    #library_dirs = ['./lib.osx' if platform.system() == 'Darwin' else './lib.linux'],
                    #libraries=['gsl', 'gslcblas'],
                    language='c++',
                    extra_compile_args=[
                        '-DPYTHON',
                        '-pthread',
                        '-std=c++11',
                        '-O3',
                        '-march=native',
                        '-ffast-math'],
                    extra_link_args=extra_link_args,
                    )

setup (name = 'iris-learn',
       #package_data = { 'IRIS': ['include/faiss/*.h', 'lib/*.so']},
       version = '1.0',
       packages=['IRIS'],
       package_dir={'IRIS': '.'},
       description = 'IRIS',
       ext_modules = [LargeVis],
       include_package_data=True,
       setup_requires=['setuptools'],
       install_requires=['numpy', 'scipy'],
       readme = "README.md",
       long_description = open("README.md").read(),
       long_description_content_type = "text/markdown",
       )
	   
