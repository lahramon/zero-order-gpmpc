#!/usr/bin/env python
from setuptools import setup, find_packages
from os.path import dirname, abspath, basename
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup(
    name='zero_order_gpmpc',
    version='0.0.1',
    author='Amon Lahr',
    author_email='amlahr@ethz.ch',
    description='A short description of your package',
    packages = find_packages(),
    # package_dir={'zero_order_gpmpc': 'src'},
    install_requires=[
        'gpytorch',
        'tqdm',
    ],
)