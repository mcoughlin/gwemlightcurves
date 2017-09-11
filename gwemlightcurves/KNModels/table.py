# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017)
#
# This file is part of gwemlightcurves.
#
# gwemlightcurves is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwemlightcurves is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwemlightcurves.  If not, see <http://www.gnu.org/licenses/>.

"""Extend :mod:`astropy.table` with the `KNTable`
"""

import operator as _operator
from math import ceil

from six import string_types

import numpy

from astropy.table import (Table, Column, vstack)

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['KNTable']


class KNTable(Table):
    """A container for a table of events

    This differs from the basic `~astropy.table.Table` in two ways

    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `KNTable`
    """
    # -- i/o ------------------------------------
    @classmethod
    def model(cls, format_, *args, **kwargs):
        """Fetch a table of events from a database

        Parameters
        ----------

        *args
            all other positional arguments are specific to the
            data format, see below for basic usage

        **kwargs
            all other positional arguments are specific to the
            data format, see the online documentation for more details


        Returns
        -------
        table : `KNTable`
            a table of events recovered from the remote database

        Examples
        --------

        Notes
        -----"""
        # standard registered fetch
        from .io.model import get_model
        model = get_model(format_, cls)
        return model(*args, **kwargs)
