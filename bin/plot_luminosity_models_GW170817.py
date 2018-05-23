
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

import pandas as pd

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

plotDir = '../plots/gws/Ka2017_combine'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

errorbudget = 0.01

plotDir1 = '../plots/gws_luminosity/Ka2017_FixZPT0/0_14/GW170817_Lbol/0.01/'
pcklFile = os.path.join(plotDir1,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data1, tmag1, lbol1, mag1, t0_best1, zp_best1, n_params1, labels1, best1, truths1) = pickle.load(f)
f.close()

plotDir2 = '../plots/gws_luminosity/Ka2017x2_FixZPT0/0_14/GW170817_Lbol/0.01/'
pcklFile = os.path.join(plotDir2,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data2, tmag2, lbol2, mag2, t0_best2, zp_best2, n_params2, labels2, best2, truths2) = pickle.load(f)
f.close()

tmag1 = tmag1 + t0_best1
tmag2 = tmag2 + t0_best2

color2 = 'coral'
color1 = 'cornflowerblue'

plotName = "%s/lbol.pdf"%(plotDir)
fig = plt.figure(figsize=(15,10))

t, y, sigma_y = data_out["tt"], data_out["Lbol"], data_out["Lbol_err"]
sigma_y_up, sigma_y_down = data_out["Lbol_err_up"], data_out["Lbol_err_down"]
idx = np.where(~np.isnan(y))[0]
t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
idx = np.where(sigma_y > y)[0]
sigma_y[idx] = 0.99*y[idx]
yerr = np.vstack((sigma_y_down, sigma_y_up))
plt.errorbar(t,y,yerr,fmt='o',c='k')

tini, tmax, dt = 0.0, 14.0, 0.1
tt = np.arange(tini,tmax,dt)

ii = np.where(~np.isnan(lbol1))[0]
f = interp.interp1d(tmag1[ii], np.log10(lbol1[ii]), fill_value='extrapolate')
lbolinterp = 10**f(tt)
zp_factor = 10**(zp_best1/-2.5)
plt.loglog(tt,zp_factor*lbolinterp,'-',c=color1,linewidth=4,label='1 Component')
plt.fill_between(tt,zp_factor*lbolinterp/(1+errorbudget),zp_factor*lbolinterp*(1+errorbudget),facecolor=color1,alpha=0.2)

ii = np.where(~np.isnan(lbol1))[0]
f = interp.interp1d(tmag2[ii], np.log10(lbol2[ii]), fill_value='extrapolate')
lbolinterp = 10**f(tt)
zp_factor = 10**(zp_best2/-2.5)
plt.loglog(tt,zp_factor*lbolinterp,'--',c=color2,linewidth=4,label='2 Component')
plt.fill_between(tt,zp_factor*lbolinterp/(1+errorbudget),zp_factor*lbolinterp*(1+errorbudget),facecolor=color2,alpha=0.2)

#plt.xlim([10**-2,50])
#plt.ylim([10.0**39,10.0**45])
plt.xlim([4*10**-1,14])
plt.ylim([10.0**40,2*10.0**42])
plt.xlabel('Time [days]',fontsize=30)
plt.ylabel('Bolometric Luminosity [erg/s]',fontsize=30)
plt.legend(loc="best",prop={'size':30},numpoints=1)
plt.grid()

plt.savefig(plotName)
plt.close()

values = np.vstack((t,y,sigma_y_down, sigma_y_up))
df = pd.DataFrame(values.T,columns=["Phase [days]","L [erg/s]","\sigma L^+ [erg/s]","\sigma L^- [erg/s]"])
print(df.to_latex(index=False,formatters=[lambda x: '%.2f' %x,lambda x: '%.1e' %x,lambda x: '%.1e' %x,lambda x: '%.1e' %x]))
