
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

plotDir = '../plots/gws/masses_GRB'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

dataDirsKN = ['../plots/gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/','../plots/gws/Ka2017_FixZPT0/r_J_H_K/0_10/ejecta/GRB150101B/1.00/','../plots/gws/Ka2017_FixZPT0/V_R_F814W/0_10/ejecta/GRB050709/1.00/','../plots/gws/Ka2017_FixZPT0/g_V_F606W_r_i_z_J_F160W_K/0_10/ejecta/GRB130603B/1.00/']

dataDirsKNAG = ['../plots/gws/Ka2017_TrPi2018_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/GW170817/1.00/','../plots/gws/Ka2017_TrPi2018_FixZPT0/r_J_H_K/0_10/GRB151.00B/1.00/','../plots/gws/Ka2017_TrPi2018_FixZPT0/V_R_F814W/0_10/GRB050709/1.00/','../plots/gws/Ka2017_TrPi2018_FixZPT0/g_V_F606W_r_i_z_J_F160W_K/0_10/GRB130603B/1.00/']

grbs = ["GW170817","GRB150101B","GRB050709","GRB130603B"]

nsamples = 3
data_out = {}
for grb, dataDirKN, dataDirKNAG in zip(grbs,dataDirsKN,dataDirsKNAG): 
    if not grb in data_out:
        data_out[grb] = {}
    multifile = lightcurve_utils.get_post_file(dataDirKN)
    data = np.loadtxt(multifile)
    if (data.size > 0) and (grb in ["GW170817","GRB150101B","GRB050709"]): 
        data_out[grb]["KN"] = data
        data_out[grb]["KN_samples"] = KNTable.read_multinest_samples(multifile,'Ka2017')
        #data_out[grb]["KN_samples"] = data_out[grb]["KN_samples"].downsample(Nsamples=nsamples)

    multifile = lightcurve_utils.get_post_file(dataDirKNAG)
    data = np.loadtxt(multifile)
    if (data.size > 0) and (grb in ["GRB130603B"]): 
        data_out[grb]["KNAG"] = data
        data_out[grb]["KNAG_samples"] = KNTable.read_multinest_samples(multifile,'Ka2017_TrPi2018')
        #data_out[grb]["KNAG_samples"] = data_out[grb]["KNAG_samples"].downsample(Nsamples=nsamples)

ModelPath = '%s/svdmodels'%('../output')
kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False

filts = ["u","g","r","i","z","y","J","H","K"]
errorbudget = 0.0
tini, tmax, dt = 0.1, 14.0, 0.1

for grb, dataDirKN, dataDirKNAG in zip(grbs,dataDirsKN,dataDirsKNAG):
    if "KN_samples" in data_out[grb]:
        samples = data_out[grb]["KN_samples"]
        samples['tini'] = tini
        samples['tmax'] = tmax
        samples['dt'] = dt
        data_out[grb]["KN_model"] = KNTable.model('Ka2017', samples, **kwargs)
        data_out[grb]["KN_med"] = lightcurve_utils.get_med(data_out[grb]["KN_model"],errorbudget = errorbudget)

    if "KNAG_samples" in data_out[grb]:
        samples = data_out[grb]["KNAG_samples"]
        samples['tini'] = tini
        samples['tmax'] = tmax
        samples['dt'] = dt
        data_out[grb]["KNAG_model"] = KNTable.model('Ka2017', samples, **kwargs)
        data_out[grb]["KNAG_med"] = lightcurve_utils.get_med(data_out[grb]["KNAG_model"],errorbudget = errorbudget)

tt = np.arange(tini,tmax+dt,dt)
colors = ['coral','cornflowerblue','palegreen','goldenrod']
colors = ['coral','cornflowerblue','limegreen','goldenrod']
plotName = "%s/mag_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,28))
plt.figure(figsize=(20,36))

cnt = 0
for filt in filts:
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    for ii, grb in enumerate(grbs):
        legend_name = grb

        if "KN_med" in data_out[grb]:
            magmed = data_out[grb]["KN_med"][filt]["50"]
            magmax = data_out[grb]["KN_med"][filt]["90"]
            magmin = data_out[grb]["KN_med"][filt]["10"]

            plt.plot(tt,magmed,'--',c=colors[ii],linewidth=4,label=legend_name)
            #plt.plot(tt,magmin,'-',c=colors[ii],linewidth=4)
            #plt.plot(tt,magmax,'-',c=colors[ii],linewidth=4)
            plt.fill_between(tt,magmin,magmax,facecolor=colors[ii],edgecolor=colors[ii],alpha=0.2,linewidth=3)

        if "KNAG_med" in data_out[grb]:
            magmed = data_out[grb]["KNAG_med"][filt]["50"]
            magmax = data_out[grb]["KNAG_med"][filt]["90"]
            magmin = data_out[grb]["KNAG_med"][filt]["10"]

            plt.plot(tt,magmed,'--',c=colors[ii],linewidth=4,label=legend_name)
            #plt.plot(tt,magmin,'-',c=colors[ii],linewidth=4)
            #plt.plot(tt,magmax,'-',c=colors[ii],linewidth=4)
            plt.fill_between(tt,magmin,magmax,facecolor=colors[ii],edgecolor=colors[ii],alpha=0.2,linewidth=3)
    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 14.0])
    plt.ylim([-18.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':40},numpoints=1,shadow=True, fancybox=True)

        #ax3 = ax1.twinx()   # mirror them
        #ax3.set_yticks([16,12,8,4,0])
        #app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(opts.distance*1e6) - 1))
        #ax3.set_yticklabels(app.astype(int))

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
    else:
        #ax4 = ax2.twinx()   # mirror them
        #ax4.set_yticks([16,12,8,4,0])
        #app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(opts.distance*1e6) - 1))
        #ax4.set_yticklabels(app.astype(int))

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

    if (not cnt == len(filts)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
ax2.set_xlabel('Rest frame time since burst [days]',fontsize=48,labelpad=30)
plt.savefig(plotName)
plt.close()

bounds = [-2.1,-0.9]
xlims = [-2.0,-1.0]
ylims = [1e-2,1]

plotName = "%s/masses.pdf"%(plotDir)
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
cnt = 0
for grb in data_out.iterkeys():
    color = colors[cnt]
    if "KN" in data_out[grb] and grb in ["GW170817","GRB150101B","GRB050709"]:
        samples = data_out[grb]["KN"][:,1]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
        hist1 = hist1 / np.sum(hist1)
        plt.step(bins, hist1, '-', where='mid', label=grb, color = color)
    if "KNAG" in data_out[grb] and grb in ["GRB130603B"]:
        samples = data_out[grb]["KNAG"][:,1]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
        hist1 = hist1 / np.sum(hist1)
        plt.step(bins, hist1, '.-', where='mid', color = color, label=grb)
    cnt = cnt + 1

plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale("log")
plt.savefig(plotName, bbox_inches='tight')
plt.close()

bounds = [-0.1,0.31]
xlims = [0.0,0.3]
ylims = [1e-2,1]

plotName = "%s/vej.pdf"%(plotDir)
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
cnt = 0
for grb in data_out.iterkeys():
    color = colors[cnt]
    if "KN" in data_out[grb] and grb in ["GW170817","GRB150101B","GRB050709"]:
        samples = data_out[grb]["KN"][:,2]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
        hist1 = hist1 / np.sum(hist1)
        plt.step(bins, hist1, '-', where='mid', label=grb, color = color)
    if "KNAG" in data_out[grb] and grb in ["GRB130603B"]:
        samples = data_out[grb]["KNAG"][:,2]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
        hist1 = hist1 / np.sum(hist1)
        plt.step(bins, hist1, '.-', where='mid', color = color, label=grb)
    cnt = cnt + 1

plt.xlabel(r"$v_{\rm ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale("log")
plt.savefig(plotName, bbox_inches='tight')
plt.close()

bounds = [-9.1,-0.9]
xlims = [-9.0,-1.0]
ylims = [1e-2,1]

plotName = "%s/Xlan.pdf"%(plotDir)
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
cnt = 0
for grb in data_out.iterkeys():
    color = colors[cnt]
    if "KN" in data_out[grb] and grb in ["GW170817","GRB150101B","GRB050709"]:
        samples = data_out[grb]["KN"][:,3]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
        hist1 = hist1 / np.sum(hist1)
        plt.step(bins, hist1, '-', where='mid', label=grb, color = color)
    if "KNAG" in data_out[grb] and grb in ["GRB130603B"]:
        samples = data_out[grb]["KNAG"][:,3]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
        hist1 = hist1 / np.sum(hist1)
        plt.step(bins, hist1, '.-', where='mid', color = color, label=grb)
    cnt = cnt + 1

plt.xlabel(r"${\rm log}_{10} (X_{\rm lan})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale("log")
plt.savefig(plotName, bbox_inches='tight')
plt.close()

