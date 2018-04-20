# -*- coding: utf-8 -*-
# Modified from: http://www2.yukawa.kyoto-u.ac.jp/~kyohei.kawaguchi/kn_calc_bns1/main.html
# Reference: Dietrich et al. http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1612.03665

import numpy as np
import scipy

def calc_meje(m1,c1,m2,c2):
    """
.. py:function:: calc_meje(m1,mb1,c1,m2,mb2,c2)

    Neutron star mass ejected (M_sun). Masses in solar masses, neutron star radius in meters, and baryon mass, if provided, is in solar masses.

    mass ejecta calculated from

    https://arxiv.org/pdf/1612.03665.pdf#equation.3.1

    constants taken from

    https://arxiv.org/pdf/1612.03665.pdf#equation.3.2

    _a, _b, _c, _d, _n= -1.35695, 6.11252, -49.43355, 16.1144, -2.5484

   :param float m1: mass of larger ns (MSun)
   :param float mb1: baryonic mass of larger ns
   :param float c1: compactness of the larger neutron star
   :param float m2: mass of samller ns (MSun)
   :param float mb2: baryonic of smaller ns
   :param float c2: compactness of the smaller neutron star
   :return: ejecta mass (Msun)
   :rtype: float
    """

    a= -0.0811876
    b= 0.228762
    d= -2.16099
    n= -2.50884

    log10_mej = a*(m1*(1-2*c1)/c1 + m2*(1-2*c2)/c2) + b*(m1*(m2/m1)**n + m2*(m1/m2)**n)+d
    meje_fit = 10**log10_mej     

    return meje_fit

def calc_vej(m1,c1,m2,c2):
    """
.. py:function:: calc_vrho(m1,c1,m2,c2)

    velocity mass ejecta

    https://arxiv.org/pdf/1612.03665.pdf#equation.3.5

    https://arxiv.org/pdf/1612.03665.pdf#equation.3.6

    a = −0.219479,  b= 0.444836,  c=−2.67385

   :param float m1: mass of larger ns (MSun)
   :param float c1: compactness of the larger neutron star
   :param float m2: mass of samller ns (MSun)
   :param float c2: compactness of the smaller neutron star
   :return: velocity of ejecta mass (Msun)
   :rtype: float
    """
    a=-0.329242
    b=0.720385
    c=-1.63255

    return a*(m1/m2)*(1+c*c1) + a*(m2/m1)*(1+c*c2)+b

def calc_qej(m1,c1,m2,c2):
    """
.. py:function:: calc_qej(m1,c1,m2,c2)

    opening  angle theta_ej

    https://arxiv.org/pdf/1612.03665.pdf#equation.3.12

   :param float m1: mass of larger ns (MSun)
   :param float c1: compactness of the larger neutron star
   :param float m2: mass of samller ns (MSun)
   :param float c2: compactness of the smaller neutron star
   :return: opening angle
   :rtype: float
    """

    vrho=calc_vrho(m1,c1,m2,c2)
    vz=calc_vz(m1,c1,m2,c2)
    vrho2=vrho*vrho
    vz2=vz*vz

    tmp1=3.*vz+np.sqrt(9*vz2+4*vrho2)
    qej=((2.0**(4.0/3.0))*vrho2+(2.*vrho2*tmp1)**(2.0/3.0))/((vrho**5.0)*tmp1)**(1.0/3.0)

    return qej

def calc_phej(m1,c1,m2,c2):
    """
.. py:function:: calc_qej(m1,c1,m2,c2)

    opening  angle theta_ej

    θej∈[π/8,3π/8] and φej∈[π,2π], and that θej and φ ej
    are linearly correlated

    https://arxiv.org/pdf/1612.03665.pdf#equation.3.13

   :param float m1: mass of larger ns (MSun)
   :param float c1: compactness of the larger neutron star
   :param float m2: mass of samller ns (MSun)
   :param float c2: compactness of the smaller neutron star
   :return: opening angle
   :rtype: float
    """
    return 4.0*calc_qej(m1,c1,m2,c2)*np.pi/2.0

