
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

from statsmodels.nonparametric.smoothers_lowess import lowess

plotDir = '../plots/gws/GRB130603B'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

errorbudget = 1.00

plotDir1 = '../plots/gws/Ka2017_FixZPT0/g_V_F606W_r_i_z_J_F160W_K/0_10/ejecta/GRB130603B/1.00/'
pcklFile = os.path.join(plotDir1,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data1, tmag1, lbol1, mag1, t0_best1, zp_best1, n_params1, labels1, best1, truths1) = pickle.load(f)
f.close()

plotDir2 = '../plots/gws/TrPi2018_FixZPT0/g_V_F606W_r_i_z_J_F160W_K/0_10/GRB130603B/1.00/'
pcklFile = os.path.join(plotDir2,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data2, tmag2, lbol2, mag2, t0_best2, zp_best2, n_params2, labels2, best2, truths2) = pickle.load(f)
f.close()

plotDir3 = '../plots/gws/Ka2017_TrPi2018_FixZPT0/g_V_F606W_r_i_z_J_F160W_K/0_10/GRB130603B/1.00/'
pcklFile = os.path.join(plotDir3,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data3, tmag3, lbol3, mag3, t0_best3, zp_best3, n_params3, labels3, best3, truths3) = pickle.load(f)
f.close()

tmag1 = tmag1 + t0_best1
tmag2 = tmag2 + t0_best2
tmag3 = tmag3 + t0_best3

idx = np.argmax(data3[:,-1])
mej = 10**data3[idx,1]
vej = data3[idx,2]
Xlan = 10**data3[idx,3]

#mej = 0.1
#vej = 0.2
#Xlan = 0.1

samples = {}
samples['tini'] = 0.1
samples['tmax'] = 14.0
samples['dt'] = 0.1
samples['Xlan'] = Xlan
samples['mej'] = mej
samples['vej'] = vej
ModelPath = '%s/svdmodels'%('../output')
kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False
t = Table()
for key, val in samples.iteritems():
    t.add_column(Column(data=[val],name=key))
samples = t
model = 'Ka2017'
model_table = KNTable.model(model, samples, **kwargs)
tmag4, lbol4, mag4 = model_table["t"][0], model_table["lbol"][0], model_table["mag"][0]
zp_best4 = 0.0

title_fontsize = 30
label_fontsize = 30

#filts = ["u","g","r","i","z","y","J","H","K"]
filts = ["g","V","F606W","r","i","z","J","F160W","K"]
colors=cm.jet(np.linspace(0,1,len(filts)))
tini, tmax, dt = 0.0, 21.0, 0.1    
tt = np.arange(tini,tmax,dt)

color1 = 'coral'
color2 = 'cornflowerblue'
color3 = 'forestgreen'
color4 = 'darkmagenta'

plotName = "%s/models_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,28))
#plt.figure(figsize=(28,28))
plt.figure(figsize=(28,46))

tini, tmax, dt = 0.0, 14.0, 0.1
tt = np.arange(tini,tmax,dt)

cnt = 0
for filt, color in zip(filts,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

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
    magave4 = lightcurve_utils.get_mag(mag4,filt)

    frac = 0.10
    ii = np.where(~np.isnan(magave1))[0]
    f = interp.interp1d(tmag1[ii], magave1[ii], fill_value='extrapolate')
    maginterp1 = f(tt)
    maginterp1 = lowess(maginterp1.T, tt, is_sorted=True, frac=frac, it=0)[:,1]
    plt.plot(tt,maginterp1+zp_best1,'--',c=color1,linewidth=2,label='Kilonova only')
    #plt.plot(tt,maginterp1+zp_best1-errorbudget,'-',c=color1,linewidth=2)
    #plt.plot(tt,maginterp1+zp_best1+errorbudget,'-',c=color1,linewidth=2)
    plt.fill_between(tt,maginterp1+zp_best1-errorbudget,maginterp1+zp_best1+errorbudget,facecolor=color1,alpha=0.2)

    ii = np.where(~np.isnan(magave2))[0]
    f = interp.interp1d(tmag2[ii], magave2[ii], fill_value='extrapolate')
    maginterp2 = f(tt)
    maginterp2 = lowess(maginterp2.T, tt, is_sorted=True, frac=frac, it=0)[:,1]
    plt.plot(tt,maginterp2+zp_best2,'--',c=color2,linewidth=2,label='Afterglow only')
    #plt.plot(tt,maginterp2+zp_best2-errorbudget,'-',c=color2,linewidth=2)
    #plt.plot(tt,maginterp2+zp_best2+errorbudget,'-',c=color2,linewidth=2)
    plt.fill_between(tt,maginterp2+zp_best2-errorbudget,maginterp2+zp_best2+errorbudget,facecolor=color2,alpha=0.2)

    ii = np.where(~np.isnan(magave3))[0]
    f = interp.interp1d(tmag3[ii], magave3[ii], fill_value='extrapolate')
    maginterp3 = f(tt)
    maginterp3 = lowess(maginterp3.T, tt, is_sorted=True, frac=frac, it=0)[:,1]
    plt.plot(tt,maginterp3+zp_best3,'--',c=color3,linewidth=2,label='Kilonova+Afterglow')
    #plt.plot(tt,maginterp3+zp_best3-errorbudget,'-',c=color3,linewidth=2)
    #plt.plot(tt,maginterp3+zp_best3+errorbudget,'-',c=color3,linewidth=2)
    plt.fill_between(tt,maginterp3+zp_best3-errorbudget,maginterp3+zp_best3+errorbudget,facecolor=color3,alpha=0.2)

    ii = np.where(~np.isnan(magave4))[0]
    f = interp.interp1d(tmag4[ii], magave4[ii], fill_value='extrapolate')
    maginterp4 = f(tt) 
    maginterp4 = lowess(maginterp4.T, tt, is_sorted=True, frac=frac, it=0)[:,1]
    plt.plot(tt,maginterp4+zp_best4,'--',c=color4,linewidth=2,label='Kilonova Contribution')
    #plt.plot(tt,maginterp4+zp_best4-errorbudget,'-',c=color4,linewidth=2)
    #plt.plot(tt,maginterp4+zp_best4+errorbudget,'-',c=color4,linewidth=2)
    plt.fill_between(tt,maginterp4+zp_best4-errorbudget,maginterp4+zp_best4+errorbudget,facecolor=color4,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=28,rotation=0,labelpad=48)
    plt.xlim([0.0, 10.0])
    plt.ylim([-22.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid(alpha=1.0,linewidth=3)

    if cnt == 1:
        ax1.set_yticks([-22,-18,-14,-10])
        ax1.set_yticks([-21,-20,-19,-17,-16,-15,-13,-12,-11],minor=True)
        plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, prop={'size':30})
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tick_params(direction='out', length=15, width=3, colors='k', labelsize = 14)
    plt.tick_params(direction='out', which = "minor", length=8, width=1.5, colors='k', labelsize = 14)

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

ax1.set_zorder(1)
plt.xlabel('Rest frame time since burst [days]',fontsize=48)
plt.savefig(plotName, bbox_inches='tight')
plt.close()
