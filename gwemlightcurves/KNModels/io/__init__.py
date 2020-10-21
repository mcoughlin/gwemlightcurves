# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Input/output methods for tabular data.
"""

from . import (  # pylint: disable=unused-import
    DiUj2017,  # Dietrich and Ujevic (2017) Model
    SmCh2017, # Smartt et al. (2017) Model
    Me2017, # Metzger (2017) Model
    KaKy2016, # Kawaguchi et al. (2016) Model
    WoKo2017, # Wollaeger et al. (2017) Model
    BaKa2016, # Barnes et al. (2016) Model
    Ka2017, # Kasen (2017) Model
    Ka2017x2, # Kasen (2017) Model (2-component)
    Ka2017inc, # Kasen (2017) Model + inclination
    Ka2017x2inc, # Kasen (2017) Model (2-component) + inclination
    RoFe2017, # Rosswog et al. (2017) Model
    Bu2019, # Bulla (2019) Model (2-component)
    Bu2019inc, # Bulla (2019) Model (2-component)
    Bu2019lf, # Bulla (2019) Model (Dynamic+Wind Lathanide Free)
    Bu2019lr, # Bulla (2019) Model (Dynamic+Wind Lathanide Rich)
    Bu2019lm, # Bulla (2019) Model (Dynamic+Wind Lathanide Middle)
    Bu2019lw, # Bulla (2019) Model (Dynamic 0.005+Wind Lathanide Middle),
    Bu2019re, # Bulla (2019) Model (Red ellipse),
    Bu2019bc, # Bulla (2019) Model (Blue cone),
    Bu2019op, # Bulla (2019) Model (opacity),
    Bu2019ops, # Bulla (2019) Model (opacity slim)
    Bu2019rp, # Bulla (2019) Model (Red ellipse + Blue cone reprocessing)
    Bu2019rps, # Bulla (2019) Model (Red ellipse + Blue cone reprocessing slim)
    Wo2020dyn, # Wollaeger et al. (2017) inspired dynamical geometry, POSSIS simulated 
    Wo2020dw, # Wollaeger et al. (2017) inspired disk wind geometry, POSSIS simulated
    Bu2019nsbh # Bulla (2019) Model (Two component, nsbh)
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
