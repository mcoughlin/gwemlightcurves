#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of the hveto python package.
#
# hveto is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hveto is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hveto.  If not, see <http://www.gnu.org/licenses/>.

"""Setup the gwemlightcurves package
"""

from __future__ import print_function

import os, sys, glob
from distutils.version import LooseVersion

from setuptools import (setup, find_packages,
                        __version__ as setuptools_version)

def get_scripts(scripts_dir='bin'):
    """Get relative file paths for all files under the ``scripts_dir``
    """
    scripts = []
    for (dirname, _, filenames) in os.walk(scripts_dir):
        scripts.extend([os.path.join(dirname, fn) for fn in filenames])
    return scripts

import versioneer
#from setup_utils import (CMDCLASS, get_setup_requires, get_scripts)
__version__ = versioneer.get_version()
CMDCLASS=versioneer.get_cmdclass()

# set basic metadata
PACKAGENAME = 'gwemlightcurves'
DISTNAME = 'gwemlightcurves'
AUTHOR = 'Michael Coughlin'
AUTHOR_EMAIL = 'michael.coughlin@ligo.org'
LICENSE = 'GPLv3'

cmdclass = {}

# import sphinx commands
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    pass
else:
    cmdclass['build_sphinx'] = BuildDoc

# -- dependencies -------------------------------------------------------------

setup_requires = [
    'setuptools',
    'pytest-runner',
]
install_requires = [
    'numpy',
    'scipy',
    'astropy',
    'h5py',
    'pandas',
    'george',
    'h5py',
    'scikit-learn>=0.18',
    'matplotlib',
    'sncosmo',
    'pymultinest',
    'requests',
    'penquins',
    'afterglowpy',
]
tests_require = [
    'pytest'
]
if sys.version_info < (2, 7):
    tests_require.append('unittest2')
extras_require = {
    'doc': [
        'sphinx',
        'numpydoc',
        'sphinx_rtd_theme',
        'sphinxcontrib_programoutput',
        'sphinxcontrib_epydoc',
    ],
}

# -- run setup ----------------------------------------------------------------

packagenames = find_packages()
scripts = glob.glob(os.path.join('bin', '*')) + glob.glob('input/Monica/*') + glob.glob('input/Wolfgang/*') + glob.glob('input/lalsim/*')

setup(name=DISTNAME,
      provides=[PACKAGENAME],
      version=__version__,
      description=None,
      long_description=None,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      scripts=scripts,
      setup_requires=setup_requires,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      use_2to3=True,
      classifiers=[
          'Programming Language :: Python',
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      ],
)
