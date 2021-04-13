
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab as py

from pylab import pi,arange,zeros,sin,cos,exp,log,sqrt,linalg,linspace,log10
from pylab import figure,clf,rcParams,FixedLocator,subplot,contourf,contour,axis,scatter
from numpy import sign,sqrt
from scipy.optimize import brentq

def CfromLambda(Lambda):
    return 0.371-0.0391*log(Lambda)+0.001056*log(Lambda)**2
    #return 0.360 - 0.0355*log(Lambda) + 0.000705*log(Lambda)**2.

def Risco(chi):
    Z1 = 1.+(1.-chi**2)**(1./3)*((1+chi)**(1./3)+(1-chi)**(1./3))
    Z2 = sqrt(3.*chi**2+Z1**2.)
    return 3.+Z2-sign(chi)*sqrt((3.-Z1)*(3.+Z1+2.*Z2))

def DiskModelEtaPow(Q,C,chi,a,b,c,d):
    rISCO = Risco(chi)
    eta = Q/(1.+Q)**2.
    mass = (a*(eta)**(-1./3.)*(1.-2.*C)-b*(rISCO/eta*C)+c)
    mass = np.array(mass)
    mass[mass<0] = 0.
    return mass**(1.+d)

def FHN18RemnantMass(Q,C,chi):
    return DiskModelEtaPow(Q,C,chi,0.40642158,0.13885773,0.25512517,0.761250847)

def FoucartEjecta(Q,C,chi):
    a1 = 7.11595154e-03
    a2 = 1.43636803e-03
    a4 = -2.76202990e-02
    n1 = 8.63604211e-01
    n2 = 1.68399507e+00
    rISCO = Risco(chi)
    Mej = a1*Q**n1*(1.-2*C)/C-a2*Q**n2*rISCO+a4
    Mej = np.array(Mej)
    Mej[Mej<0] = 0.
    return Mej

def calc_meje(q,chi_eff,c,mns,mbns,f=0.15):

    #mb = mns*(1+0.6*c/(1.-0.5*c))
    mb = mbns
    mdyn = FoucartEjecta(q,c,chi_eff)*mb
    #mwind = f*(FHN18RemnantMass(q,c,chi_eff)-FoucartEjecta(q,c,chi_eff))*mb
    mwind = f*(FHN18RemnantMass(q,c,chi_eff))*mb
    mwind = np.array(mwind)
    mwind[mwind<0] = 0.0
    mtot = mdyn+mwind

    return mtot

def calc_vave(q):
    return 1.5333330951369120e-2*q+0.19066667068621043
