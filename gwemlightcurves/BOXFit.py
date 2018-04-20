
import os, sys
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy

def lightcurve(boxfitDir,tini,tmax,dt,theta_0,E,n,theta_obs,p,epsilon_B,epsilon_E,ksi_N):

    tt = np.arange(tini,tmax,dt)
    lbol = 1e43*np.ones(tt.shape)

    filts = ["u","g","r","i","z","y","J","H","K"]
    lambdas = np.array([3561.8,4866.46,6214.6,6389.4,7127.0,7544.6,8679.5,9633.3,12350.0,16620.0,21590.0])*1e-10
    nu_0s = 3e8/lambdas

    exampleIni = "%s/boxfit.ini"%boxfitDir
    mag = []
    for filt, nu_0 in zip(filts,nu_0s):
        ini = []
        for line in open(exampleIni).readlines():
             line = line.replace("xxx_boxfitDir",boxfitDir)
             line = line.replace("xxx_nu0","%.5e"%nu_0)
             line = line.replace("xxx_theta0","%.5f"%theta_0)
             line = line.replace("xxx_E","%.5e"%E)
             line = line.replace("xxx_n","%.5f"%n)
             line = line.replace("xxx_theta_obs","%.5f"%theta_obs)
             line = line.replace("xxx_p","%.5f"%p)
             line = line.replace("xxx_epsilon_B","%.5e"%epsilon_B)
             line = line.replace("xxx_epsilon_E","%.5e"%epsilon_E)
             line = line.replace("xxx_ksi_N","%.5f"%ksi_N)
             ini.append(line)
        paramFile = "boxfitsettings.txt" 
        open(paramFile,'w').write("".join(ini))
        system_command = "mpiexec %s/boxfit > %s/out"%(boxfitDir,boxfitDir)
        os.system(system_command)

        filename = "%s/out"%(boxfitDir)
        data_out = np.loadtxt(filename,delimiter=",")
 
        t = data_out[:,1]/86400.0
        mJy = data_out[:,3]
        Jy = 1e-3 * mJy
        mag_d = -48.6 + -1*np.log10(Jy/1e23)*2.5

        ii = np.where(np.isfinite(mag_d))[0]
        if len(ii) >= 2:
            f = interp.interp1d(t[ii], mag_d[ii], fill_value='extrapolate')
            maginterp = f(tt)
        else:
            maginterp = np.nan*np.ones(tt.shape)

        mag.append(maginterp)

    return tt, lbol, mag

