#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Michael Coughlin (2017)
#
# This file is part of gwemopt
#
# gwemopt is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwemopt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwemopt.  If not, see <http://www.gnu.org/licenses/>

"""Gravitational-wave Electromagnetic Optimization
"""

# load tables
from astropy.table import (Column, Table)
from .table import KNTable

# attach unified I/O
from . import io
