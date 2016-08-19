#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

# read the version number out of __init__ so I don't have to remember to edit
# it in 2 places.
# This doesn't work when installing from pypi. I guess the files aren't really
# there yet.
# with open('OpticalRS/__init__.py') as f:
#     for line in f:
#         if line.startswith('__version__'):
#             optrs_version = '.'.join( re.findall(r'\d+', line) )
optrs_version = '1.0.1'

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


# readme = open('README.md').read()
# history = open('HISTORY.rst').read().replace('.. :changelog:', '')
readme = 'OpticalRS is a free and open source Python implementation of passive optical remote sensing methods for the derivation of bathymetric maps and maps of submerged habitats.'

requirements = ['numpy',
                'pandas',
                'statsmodels',
                'matplotlib',
                'scikit-image',
                'scikit-learn',
                'GDAL',
                'geopandas',
                'rasterstats',
                'scipy']

#test_requirements = [
#    # TODO: put package test requirements here
#]

setup(
    name='OpticalRS',
    version=optrs_version,
    description='OpticalRS is a Python implementation of optical remote sensing methods for mapping of submerged habitats.',
    long_description=readme,
    author='Jared Kibele',
    author_email='jkibele@gmail.com',
    url='https://github.com/jkibele/OpticalRS',
    packages=find_packages('OpticalRS', exclude=["tests.*", "tests"]),
    # [
    #     'OpticalRS',
    # ],
    package_dir={'OpticalRS':
                 'OpticalRS'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='OpticalRS',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
#        'Programming Language :: Python :: 3',
#        'Programming Language :: Python :: 3.3',
#        'Programming Language :: Python :: 3.4',
    ],
#    test_suite='tests',
#    tests_require=test_requirements
)
