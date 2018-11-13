#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from setuptools import setup, find_packages
import sys
if sys.version_info[0] < 3:
  import __builtin__ as builtins
else:
  import builtins
import cave

long_description = \
"""
Crowded Aperture Variability Extraction
"""

# Setup!
setup(name = 'cave-tools',
      version = cave.__version__,
      description = 'Crowded Aperture Variability Extraction',
      long_description = long_description,
      classifiers = [
                      'License :: OSI Approved :: MIT License',
                      'Programming Language :: Python',
                      'Programming Language :: Python :: 3',
                      'Topic :: Scientific/Engineering :: Astronomy',
                    ],
      url = 'https://github.com/nksaunders/cave',
      author = 'Nicholas Saunders',
      author_email = 'nicholas.k.saunders@nasa.gov',
      license = 'MIT',
      packages = ['cave'],
      install_requires = [
                          'everest-pipeline',
                          'astroML',
                         ],
      include_package_data = True,
      zip_safe = False,
      test_suite='nose.collector',
      tests_require=['nose']
      )
