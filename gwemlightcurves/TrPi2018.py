
import os, sys
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy
#import grbpy
import afterglowpy as grbpy
from functools import wraps
import signal, errno

class TimeoutException(Exception):   # Custom exception class
  pass

def TimeoutHandler(signum, frame):   # Custom signal handler
  raise TimeoutException

def fluxDensity(t, nu, jetType, specType, *Y):
    mJy = grbpy.fluxDensity(t, nu, jetType, specType, *Y)
    return mJy

def lightcurve(tini,tmax,dt,theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B):

    day = 86400.0
    tt = np.arange(tini,tmax+dt,dt)
    ta = 1e-1 * day
    tb = 1.0e3 * day
    t = np.logspace(np.log10(ta), np.log10(tb), base=10.0, num=10)
    lbol = 1e43*np.ones(tt.shape)

    jetType = 0
    specType = 0
    ksiN = 1.0
    dL = 3.09e19
    b = 6
    L0 = 0.0
    q = 0.0
    ts = 0.0

    Y = np.array([theta_v, E0, theta_c, theta_w, b, L0, q, ts, n, p, epsilon_E, epsilon_B, ksiN, dL])

    filts = ["u","g","r","i","z","y","J","H","K"]
    lambdas = np.array([3561.8,4866.46,6214.6,6389.4,7127.0,7544.6,8679.5,9633.3,12350.0,16620.0,21590.0])*1e-10
    nu_0s = 3e8/lambdas

    mag = []
    for filt, nu_0 in zip(filts,nu_0s):

        nu = nu_0*np.ones(t.shape)

        try:
            mJy = fluxDensity(t, nu, jetType, specType, *Y)
        except TimeoutException:
            mJy = np.zeros(t.shape) 

        Jy = 1e-3 * mJy
        mag_d = -48.6 + -1*np.log10(Jy/1e23)*2.5

        ii = np.where(np.isfinite(mag_d))[0]
        if len(ii) >= 2:
            f = interp.interp1d(t[ii]/day, mag_d[ii], fill_value='extrapolate')
            maginterp = f(tt)
        else:
            maginterp = np.nan*np.ones(tt.shape)

        mag.append(maginterp)

    return tt, lbol, np.array(mag)

