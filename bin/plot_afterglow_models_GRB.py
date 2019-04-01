
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

plotDir = '../plots/gws/afterglow_GRB'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

basedir = "../plots/gws/Ka2017_A_FixZPT0"
dataDirs, grbs = [], []
basedirs = glob.glob(os.path.join(basedir,'*'))
for b in basedirs:
    thisdir1 = os.path.join(b,"0_10/ejecta")
    thisdir2 = os.path.join(b,"0_14/ejecta")
    grbdir = glob.glob(os.path.join(thisdir1,'*')) + glob.glob(os.path.join(thisdir2,'*')) 
    if len(grbdir) == 0: continue
    grbdir = grbdir[0]
    grb = grbdir.split("/")[-1]
    grbdir = os.path.join(grbdir,'1.00')

    dataDirs.append(grbdir)
    grbs.append(grb)

#grbs_skip = ["GW170817","GRB060614","GRB050709","GRB130603B"]
grbs_include = ["GW170817","GRB060614","GRB050709","GRB130603B"]

nsamples = 100
data_out = {}
for grb, dataDir in zip(grbs,dataDirs): 
    if not grb in grbs_include: continue
    if not grb in data_out:
        data_out[grb] = {}
    multifile = lightcurve_utils.get_post_file(dataDir)
    data = np.loadtxt(multifile)
    #if (data.size > 0) and (not grb in grbs_skip): 
    if data.size > 0: 
        data_out[grb]["KN"] = data
        data_out[grb]["KN_samples"] = KNTable.read_multinest_samples(multifile,'Ka2017_A')
        data_out[grb]["KN_samples"] = data_out[grb]["KN_samples"].downsample(Nsamples=nsamples)

ModelPath = '%s/svdmodels'%('../output')
kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False

filts = ["u","g","r","i","z","y","J","H","K"]
errorbudget = 0.0
tini, tmax, dt = 0.1, 14.0, 0.1

for grb in data_out.keys():
    print('Generating lightcurves for %s'%grb)
    if "KN_samples" in data_out[grb]:
        samples = data_out[grb]["KN_samples"]
        samples['tini'] = tini
        samples['tmax'] = tmax
        samples['dt'] = dt
        data_out[grb]["KN_model"] = KNTable.model('Ka2017', samples, **kwargs)
        for ii,row in enumerate(data_out[grb]["KN_model"]):
            A = data_out[grb]["KN_model"][ii]["A"]
            dm = -2.5*np.log10(A)
            data_out[grb]["KN_model"][ii]["mag"] = data_out[grb]["KN_model"][ii]["mag"] + dm
            data_out[grb]["KN_model"][ii]["lbol"] = data_out[grb]["KN_model"][ii]["lbol"]*A
        data_out[grb]["KN_med"] = lightcurve_utils.get_med(data_out[grb]["KN_model"],errorbudget = errorbudget)

tt = np.arange(tini,tmax+dt,dt)
#colors = ['coral','cornflowerblue','palegreen','goldenrod']
keys = data_out.keys()
colors=cm.rainbow(np.linspace(0,1,len(keys)))
colors = ['coral','cornflowerblue','forestgreen','darkmagenta']
plotName = "%s/mag_panels.pdf"%(plotDir)
plt.figure(figsize=(20,48))

cnt = 0
for filt in filts:
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    for ii, grb in enumerate(data_out.keys()):
        legend_name = grb

        if "KN_med" in data_out[grb]:
            magmed = data_out[grb]["KN_med"][filt]["50"]
            magmax = data_out[grb]["KN_med"][filt]["95"]
            magmin = data_out[grb]["KN_med"][filt]["5"]

            plt.plot(tt,magmed,'--',c=colors[ii],linewidth=4,label=legend_name)
            #plt.plot(tt,magmin,'-',c=colors[ii],linewidth=4)
            #plt.plot(tt,magmax,'-',c=colors[ii],linewidth=4)
            plt.fill_between(tt,magmin,magmax,facecolor=colors[ii],edgecolor=colors[ii],alpha=0.2,linewidth=3)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 14.0])
    plt.ylim([-20.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    if cnt == 1:
        ax1.set_yticks([-20,-17,-14,-11])
        ax1.set_yticks([-19,-18,-16,-15,-13,-12],minor=True)
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':40},numpoints=1,shadow=True, fancybox=True)

        #ax3 = ax1.twinx()   # mirror them
        #ax3.set_yticks([16,12,8,4,0])
        #app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(opts.distance*1e6) - 1))
        #ax3.set_yticklabels(app.astype(int))

        plt.tick_params(direction='out', length=15, width=3, colors='k', labelsize = 14)
        plt.tick_params(direction='out', which = "minor", length=8, width=1.5, colors='k', labelsize = 14)
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
    else:
        #ax4 = ax2.twinx()   # mirror them
        #ax4.set_yticks([16,12,8,4,0])
        #app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(opts.distance*1e6) - 1))
        #ax4.set_yticklabels(app.astype(int))

        plt.tick_params(direction='out', length=15, width=3, colors='k', labelsize = 14)
        plt.tick_params(direction='out', which = "minor", length=8, width=1.5, colors='k', labelsize = 14)
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

    if (not cnt == len(filts)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
ax2.set_xlabel('Time [days]',fontsize=48,labelpad=30)
plt.savefig(plotName)
plt.close()

