
import numpy as np
from gwemlightcurves import Global

def myprior_KaKy2016(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*6.0 + 3.0
        cube[2] = cube[2]*0.75
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*0.1 + 0.1
        cube[6] = cube[6]*np.pi/2
        cube[7] = cube[7]*2*np.pi
        cube[8] = cube[8]*2*Global.ZPRange - Global.ZPRange

def myprior_KaKy2016_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*np.pi/2
        cube[4] = cube[4]*2*np.pi
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_KaKy2016_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*6.0 + 3.0
        cube[2] = cube[2]*0.75
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.1 + 0.1
        cube[5] = cube[5]*np.pi/2
        cube[6] = cube[6]*2*np.pi
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_Me2017(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*10.0
        cube[8] = cube[8]*3.0 - 1.0
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_WoKo2017(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*180.0
        cube[8] = cube[8]*3.0 - 1.0
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_SmCh2017(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*10.0 - 5.0
        cube[8] = cube[8]*3.0 - 1.0
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_Me2017_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*10.0
        cube[6] = cube[6]*3.0 - 1.0
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_WoKo2017_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*180.0
        cube[6] = cube[6]*3.0 - 1.0
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_SmCh2017_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*10.0 - 5.0
        cube[6] = cube[6]*3.0 - 1.0
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_Me2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*10.0
        cube[4] = cube[4]*3.0 - 1.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_WoKo2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*180.0
        cube[4] = cube[4]*3.0 - 1.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_SmCh2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        #cube[2] = cube[2]*1.0
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*10.0 - 5.0
        cube[4] = cube[4]*3.0 - 1.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_DiUj2017(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*np.pi/2
        cube[8] = cube[8]*2*np.pi
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_DiUj2017_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*np.pi/2
        cube[6] = cube[6]*2*np.pi
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_DiUj2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*np.pi/2
        cube[4] = cube[4]*2*np.pi
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_sn(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*10.0
        cube[2] = cube[2]*10.0
        cube[3] = cube[3]*10.0
        cube[4] = cube[4]*10.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange



