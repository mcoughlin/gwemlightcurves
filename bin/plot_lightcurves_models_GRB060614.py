
import os, sys, glob, pickle
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

from astropy.table import Table, Column

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

import corner

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

plotDir = '../plots/gws/GRB060614'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

errorbudget = 0.01

plotDir1 = '../plots/gws/Ka2017_A_FixZPT0/V_R_I/0_10/ejecta/GRB060614/0.01/'
pcklFile = os.path.join(plotDir1,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data1, tmag1, lbol1, mag1, t0_best1, zp_best1, n_params1, labels1, best1, truths1) = pickle.load(f)
f.close()

plotDir2 = '../plots/gws/TrPi2018_FixZPT0/V_R_I/0_10/GRB060614/0.01/'
pcklFile = os.path.join(plotDir2,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data2, tmag2, lbol2, mag2, t0_best2, zp_best2, n_params2, labels2, best2, truths2) = pickle.load(f)
f.close()

plotDir3 = '../plots/gws/Ka2017_TrPi2018_A_FixZPT0/V_R_I/0_10/GRB060614/0.01/'
pcklFile = os.path.join(plotDir3,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data3, tmag3, lbol3, mag3, t0_best3, zp_best3, n_params3, labels3, best3, truths3) = pickle.load(f)
f.close()

tmag1 = tmag1 + t0_best1
tmag2 = tmag2 + t0_best2
tmag3 = tmag3 + t0_best3

title_fontsize = 30
label_fontsize = 30

#filts = ["u","g","r","i","z","y","J","H","K"]
filts = ["V","R","I"]
colors=cm.jet(np.linspace(0,1,len(filts)))
tini, tmax, dt = 0.0, 21.0, 0.1    
tt = np.arange(tini,tmax,dt)

color1 = 'coral'
color2 = 'cornflowerblue'
color3 = 'palegreen'

plotName = "%s/models_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,18))
plt.figure(figsize=(20,12))

tini, tmax, dt = 0.0, 7.0, 0.1
tt = np.arange(tini,tmax,dt)

cnt = 0
for filt, color in zip(filts,colors):
    cnt = cnt+1
    if cnt == 1:
        ax1 = plt.subplot(len(filts),1,cnt)
    else:
        ax2 = plt.subplot(len(filts),1,cnt,sharex=ax1,sharey=ax1)

    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color, markersize=16)

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=16)

    magave1 = lightcurve_utils.get_mag(mag1,filt)
    magave2 = lightcurve_utils.get_mag(mag2,filt)
    magave3 = lightcurve_utils.get_mag(mag3,filt)

    ii = np.where(~np.isnan(magave1))[0]
    f = interp.interp1d(tmag1[ii], magave1[ii], fill_value='extrapolate')
    maginterp1 = f(tt)
    plt.plot(tt,maginterp1+zp_best1,'--',c=color1,linewidth=2,label='Kilonova')
    plt.plot(tt,maginterp1+zp_best1-errorbudget,'-',c=color1,linewidth=2)
    plt.plot(tt,maginterp1+zp_best1+errorbudget,'-',c=color1,linewidth=2)
    plt.fill_between(tt,maginterp1+zp_best1-errorbudget,maginterp1+zp_best1+errorbudget,facecolor=color1,alpha=0.2)

    ii = np.where(~np.isnan(magave2))[0]
    f = interp.interp1d(tmag2[ii], magave2[ii], fill_value='extrapolate')
    maginterp2 = f(tt)
    plt.plot(tt,maginterp2+zp_best2,'--',c=color2,linewidth=2,label='Afterglow')
    plt.plot(tt,maginterp2+zp_best2-errorbudget,'-',c=color2,linewidth=2)
    plt.plot(tt,maginterp2+zp_best2+errorbudget,'-',c=color2,linewidth=2)
    plt.fill_between(tt,maginterp2+zp_best2-errorbudget,maginterp2+zp_best2+errorbudget,facecolor=color2,alpha=0.2)

    ii = np.where(~np.isnan(magave3))[0]
    f = interp.interp1d(tmag3[ii], magave3[ii], fill_value='extrapolate')
    maginterp3 = f(tt)
    plt.plot(tt,maginterp3+zp_best3,'--',c=color3,linewidth=2,label='Kilonova+Afterglow')
    plt.plot(tt,maginterp3+zp_best3-errorbudget,'-',c=color3,linewidth=2)
    plt.plot(tt,maginterp3+zp_best3+errorbudget,'-',c=color3,linewidth=2)
    plt.fill_between(tt,maginterp3+zp_best3-errorbudget,maginterp3+zp_best3+errorbudget,facecolor=color3,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 7.0])
    plt.ylim([-20.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-20,-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, prop={'size':38})
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=48)
plt.savefig(plotName, bbox_inches='tight')
plt.close()
