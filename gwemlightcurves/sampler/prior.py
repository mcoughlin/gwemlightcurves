
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
        cube[7] = cube[7]*4.0 + 1.0
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

def myprior_BaKa2016(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*5.0 - 5.0
        cube[8] = cube[8]*2*Global.ZPRange - Global.ZPRange

def myprior_RoFe2017(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*1.0
        cube[8] = cube[8]*2*Global.ZPRange - Global.ZPRange   

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
        cube[5] = cube[5]*4.0 + 1.0
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

def myprior_BaKa2016_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*5.0 - 5.0
        cube[6] = cube[6]*2*Global.ZPRange - Global.ZPRange

def myprior_RoFe2017_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*1.0
        cube[6] = cube[6]*2*Global.ZPRange - Global.ZPRange

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
        #cube[1] = cube[1]*4.0 - 5.0
        cube[1] = cube[1]*6.0 - 5.0
        #cube[2] = cube[2]*0.3
        cube[2] = cube[2]*0.25 + 0.05
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*4.0 + 1.0
        cube[4] = cube[4]*3.0 - 1.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_Me2017_A_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*4.0 - 5.0
        cube[1] = cube[1]*6.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*4.0 + 1.0
        cube[4] = cube[4]*3.0 - 1.0
        cube[5] = cube[5]*10.0
        cube[6] = cube[6]*2*Global.ZPRange - Global.ZPRange

def myprior_Me2017x2_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*4.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*4.0 + 1.0
        cube[4] = cube[4]*2.0 - 0.0
        cube[5] = cube[5]*4.0 - 5.0
        cube[6] = cube[6]*0.3
        #cube[6] = cube[6]*1.0
        cube[7] = cube[7]*4.0 + 1.0
        cube[8] = cube[8]*1.0 - 1.0
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_WoKo2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*180.0
        cube[4] = cube[4]*3.0 - 1.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_BaKa2016_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*2.0 - 3.0
        cube[1] = cube[1]*3.0 - 3.0
        #cube[1] = cube[1]*4.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        if not Global.Xlan==0:
            cube[3] = cube[3]*0.1 + Global.Xlan - 0.05  
        else:
            cube[3] = cube[3]*8.0 - 9.0
        #cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        cube[4] = cube[4]*1.0

def myprior_Ka2017inc_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*4.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*8.0 - 9.0
        cube[4] = cube[4]*180.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017_A_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*4.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*8.0 - 9.0
        cube[4] = cube[4]*10.0 - 0.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017x2_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*4.0 - 5.0
        #cube[2] = cube[2]*0.1
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*5.0 - 5.0
        cube[4] = cube[4]*4.0 - 5.0
        #cube[5] = cube[5]*0.2 + 0.1
        cube[5] = cube[5]*0.3
        cube[6] = cube[6]*5.0 - 5.0
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017x2inc_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*4.0 - 5.0
        #cube[2] = cube[2]*0.1
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*5.0 - 5.0
        cube[4] = cube[4]*4.0 - 5.0
        #cube[5] = cube[5]*0.2 + 0.1
        cube[5] = cube[5]*0.3
        cube[6] = cube[6]*5.0 - 5.0
        cube[7] = cube[7]*180.0
        cube[8] = cube[8]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017x2_ejecta_sigma(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*4.0 - 5.0
        #cube[2] = cube[2]*0.1
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*5.0 - 5.0
        cube[4] = cube[4]*4.0 - 5.0
        #cube[5] = cube[5]*0.2 + 0.1
        cube[5] = cube[5]*0.3
        cube[6] = cube[6]*5.0 - 5.0
        cube[7] = cube[7]*2.0
        cube[8] = cube[8]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017x3_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.1 + 0.2
        cube[3] = cube[3]*2.0 - 2.0
        cube[4] = cube[4]*5.0 - 5.0
        cube[5] = cube[5]*0.1 + 0.2
        cube[6] = cube[6]*4.0 - 5.0
        cube[7] = cube[7]*5.0 - 5.0
        cube[8] = cube[8]*0.2
        cube[9] = cube[9]*5.0 - 5.0
        cube[10] = cube[10]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017x3inc_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*4.0 - 5.0
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*5.0 - 5.0
        cube[4] = cube[4]*4.0 - 5.0
        cube[5] = cube[5]*0.3
        cube[6] = cube[6]*5.0 - 5.0
        cube[7] = cube[7]*4.0 - 5.0
        cube[8] = cube[8]*0.3
        cube[9] = cube[9]*5.0 - 5.0
        cube[10] = cube[10]*180.0
        cube[11] = cube[11]*2*Global.ZPRange - Global.ZPRange

def myprior_Bu2019_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*2.0 - 3.0
        if not Global.T==0:
            cube[2] = cube[2]*0.1 + Global.T - 0.05
        else:
            cube[2] = cube[2]*1.0 + 3.0
        #cube[3] = cube[3]*2*Global.ZPRange - Global.ZPRange
        cube[3] = cube[3]*1.0

def myprior_Bu2019inc_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*2.0 - 3.0
        if not Global.phi==-1:
            cube[2] = cube[2]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            cube[2] = cube[2]*15.0 + 15.0
        if not Global.theta==-1:
            cube[3] = cube[3]*0.1 + Global.theta - 0.05
        else:
            #cube[3] = cube[3]*90.0
            cube[3] = cube[3]*15.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019op_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*4.0 + 1.0
        cube[2] = cube[2]*0.7 - 0.7
        cube[3] = cube[3]*4.0 + 1.0
        cube[4] = cube[4]*1.0 - 1.0
        cube[5] = cube[5]*1.0

def myprior_Bu2019ops_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*4.0
        cube[2] = cube[2]*4.0
        cube[3] = cube[3]*1.0 - 1.0
        cube[4] = cube[4]*1.0

def myprior_Bu2019lf_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*2.0 - 3.0
        cube[2] = cube[2]*2.0 - 3.0
        if not Global.phi==-1:
            cube[3] = cube[3]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            cube[3] = cube[3]*30.0 + 30.0
        cube[4] = cube[4]*90.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019lr_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        cube[1] = cube[1]*2.0 - 3.0
        cube[2] = cube[2]*2.0 - 3.0
        if not Global.phi==-1:
            cube[3] = cube[3]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            cube[3] = cube[3]*30.0 + 30.0
        cube[4] = cube[4]*90.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019lm_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        if not Global.mdyn==-1:
            cube[1] = np.log10(cube[1]*0.0002 + Global.mdyn - 0.0001)
        else:
            cube[1] = cube[1]*2.0 - 3.0
        cube[2] = cube[2]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        if not Global.phi==-1:
            cube[3] = cube[3]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            #cube[3] = cube[3]*30.0 + 30.0
            cube[3] = cube[3]*60.0 + 15.0
        if not Global.theta==-1:
            cube[4] = cube[4]*0.1 + Global.theta - 0.05
        else:
            cube[4] = cube[4]*90.0
        #cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange
        cube[5] = cube[5]*1.0

def myprior_Bu2019rb_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        if not Global.phi==-1:
            cube[3] = cube[3]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            #cube[3] = cube[3]*30.0 + 30.0
            cube[3] = cube[3]*60.0 + 15.0
        if not Global.theta==-1:
            cube[4] = cube[4]*0.1 + Global.theta - 0.05
        else:
            cube[4] = cube[4]*90.0
        cube[5] = cube[5]*9 + 1.0
        cube[6] = cube[6]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019rp_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        if not Global.phi==-1:
            cube[3] = cube[3]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            #cube[3] = cube[3]*30.0 + 30.0
            cube[3] = cube[3]*60.0 + 15.0
        if not Global.theta==-1:
            cube[4] = cube[4]*0.1 + Global.theta - 0.05
        else:
            cube[4] = cube[4]*90.0
        cube[5] = cube[5]*9 + 1.0
        cube[6] = cube[6]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019rps_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        cube[3] = cube[3]*9 + 1.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Wo2020_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*3.0 - 3.0
        cube[3] = cube[3]*0.061 + 0.012 # extends +/- 0.002 outside grid
        #cube[3] = cube[3]*0.1 #extends outside grid
        cube[4] = cube[4]*1.75 + 0.25
        cube[5] = cube[5]*0.25 + 0.1
        if not Global.theta==-1:
            cube[6] = cube[6]*0.1 + Global.theta - 0.05
        else:
            cube[6] = cube[6]*90.0
        cube[7] = cube[7]*2*Global.ZPRange - Global.ZPRange

def myprior_Wo2020dyn_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*0.061 + 0.012 # extends +/- 0.002 outside grid
        #cube[2] = cube[2]*0.1 #extends outside grid
        cube[3] = cube[3]*1.75 + 0.25
        if not Global.theta==-1:
            cube[4] = cube[4]*0.1 + Global.theta - 0.05
        else:
            cube[4] = cube[4]*90.0
        cube[5] = cube[5]*2*Global.ZPRange - Global.ZPRange

def myprior_Wo2020dw_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*0.25 + 0.1
        if not Global.theta==-1:
            cube[3] = cube[3]*0.1 + Global.theta - 0.05
        else:
            cube[3] = cube[3]*90.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange

def myprior_Bu2019re_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        cube[1] = cube[1]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        if not Global.theta==-1:
            cube[2] = cube[2]*0.1 + Global.theta - 0.05
        else:
            cube[2] = cube[2]*90.0
        cube[3] = cube[3]*9 + 1.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019bc_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0 
        cube[1] = cube[1]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        if not Global.phi==-1:
            cube[2] = cube[2]*0.1 + Global.phi - 0.05
        else:
            #cube[3] = cube[3]*30.0
            #cube[3] = cube[3]*90.0
            #cube[3] = cube[3]*30.0 + 30.0
            cube[2] = cube[2]*60.0 + 15.0
        if not Global.theta==-1:
            cube[3] = cube[3]*0.1 + Global.theta - 0.05
        else:
            cube[3] = cube[3]*90.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        #cube[4] = cube[4]*1.0

def myprior_Bu2019nsbh_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*2.5 - 3.0
        cube[2] = cube[2]*2.5 - 3.0
        if not Global.theta==-1:
            cube[3] = cube[3]*0.1 + Global.theta - 0.05
        else:
            cube[3] = cube[3]*90.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        #cube[5] = cube[5]*1.0

def myprior_Bu2019lw_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*3.0 - 3.0
        #cube[2] = cube[2]*2.5 - 3.0
        if not Global.phi==-1:
            cube[2] = cube[2]*0.1 + Global.phi - 0.05
        else:
            cube[2] = cube[2]*30.0 + 30.0
        if not Global.theta==-1:
            cube[3] = cube[3]*0.1 + Global.theta - 0.05
        else:
            cube[3] = cube[3]*90.0
        #cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange
        cube[4] = cube[4]*1.0

def myprior_Bu2019inc_TrPi2018(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*3.0 - 3.0
        if not Global.phi==-1:
            cube[2] = cube[2]*0.1 + Global.phi - 0.05
        else:
            cube[2] = cube[2]*90.0
        cube[3] = cube[3]*np.pi/4.0
        cube[4] = cube[4]*6.0 + 49.0
        cube[5] = cube[5]*np.pi/4.0
        cube[6] = cube[6]*np.pi/4.0
        cube[7] = cube[7]*4.0 - 4.0
        cube[8] = cube[8]*0.4 + 2.1
        cube[9] = cube[9]*4.0 - 4.0
        cube[10] = cube[10]*4.0 - 4.0
        cube[11] = cube[11]*2*Global.ZPRange - Global.ZPRange

def myprior_RoFe2017_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.3
        #cube[2] = cube[2]*1.0
        cube[3] = cube[3]*1.0
        cube[4] = cube[4]*2*Global.ZPRange - Global.ZPRange

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

def myprior_boxfit(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*np.pi/4.0
        cube[2] = cube[2]*4.0 + 49.0
        cube[3] = cube[3]*4.0 - 4.0
        cube[4] = cube[4]*np.pi/4.0
        cube[5] = cube[5]*0.1 + 2.1
        cube[6] = cube[6]*3.0 - 4.0
        cube[7] = cube[7]*3.0 - 4.0
        cube[8] = cube[8]*4.0 - 4.0
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_TrPi2018(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*np.pi/4.0
        cube[2] = cube[2]*6.0 + 49.0
        cube[3] = cube[3]*np.pi/4.0
        cube[4] = cube[4]*np.pi/4.0
        cube[5] = cube[5]*4.0 - 4.0
        cube[6] = cube[6]*0.4 + 2.1
        cube[7] = cube[7]*4.0 - 4.0
        cube[8] = cube[8]*4.0 - 4.0
        cube[9] = cube[9]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017_TrPi2018(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        #cube[1] = cube[1]*5.0 - 5.0
        cube[1] = cube[1]*3.0 - 3.0
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*8.0 - 9.0
        cube[4] = cube[4]*np.pi/4.0
        cube[5] = cube[5]*6.0 + 49.0
        cube[6] = cube[6]*np.pi/4.0
        cube[7] = cube[7]*np.pi/4.0
        cube[8] = cube[8]*4.0 - 4.0
        cube[9] = cube[9]*0.4 + 2.1
        cube[10] = cube[10]*4.0 - 4.0
        cube[11] = cube[11]*4.0 - 4.0
        cube[12] = cube[12]*2*Global.ZPRange - Global.ZPRange

def myprior_Ka2017_TrPi2018_A(cube, ndim, nparams):
        cube[0] = cube[0]*2*Global.T0Range - Global.T0Range
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*0.3
        cube[3] = cube[3]*8.0 - 9.0
        cube[4] = cube[4]*np.pi/4.0
        cube[5] = cube[5]*6.0 + 49.0
        cube[6] = cube[6]*np.pi/4.0
        cube[7] = cube[7]*np.pi/4.0
        cube[8] = cube[8]*4.0 - 4.0
        cube[9] = cube[9]*0.4 + 2.1
        cube[10] = cube[10]*4.0 - 4.0
        cube[11] = cube[11]*4.0 - 4.0
        cube[12] = cube[12]*10.0 - 0.0
        cube[13] = cube[13]*2*Global.ZPRange - Global.ZPRange
