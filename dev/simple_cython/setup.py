""" Module providing Cython compilation instructions for pairwise_sum_cython.pyx.
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(['simple_cython_engine.pyx']))

# compile instructions:
# python setup.py build_ext --inplace
