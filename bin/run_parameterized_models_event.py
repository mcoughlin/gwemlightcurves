#!/usr/bin/env python

# ---- Import standard modules to the python path.

import os, sys, copy
import numpy as np
import argparse

from scipy.interpolate import interpolate as interp
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves import __version__


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.add_argument("-o","--outputDir",default="../output")
    parser.add_argument("-p","--plotDir",default="../plots")
    parser.add_argument("-d","--dataDir",default="../data")
    parser.add_argument("--posterior_samples", default="../data/event_data/G298048.dat")
    parser.add_argument("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_argument("-m","--model",default="DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017", help="DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017")
    parser.add_argument("--name",default="G298048")

    parser.add_argument("--doEvent",  action="store_true", default=False)
    parser.add_argument("-e","--event",default="G298048_PS1_GROND_SOFI")
    #parser.add_argument("-e","--event",default="G298048_XSH_PESSTO")
    #parser.add_argument("-e","--event",default="G298048_20170822")
    #parser.add_argument("-e","--event",default="G298048_PESSTO_20170818,G298048_PESSTO_20170819,G298048_PESSTO_20170820,G298048_PESSTO_20170821,G298048_XSH_20170819,G298048_XSH_20170821")
    parser.add_argument("--distance",default=40.0,type=float)
    parser.add_argument("--T0",default=57982.5285236896,type=float)

    args = parser.parse_args()
 
    return args

def hist_results(samples,Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-3
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

def get_legend(model):

    if model == "DiUj2017":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "KaKy2016":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Me2017":
        legend_name = "Metzger (2017)"
    elif model == "SmCh2017":
        legend_name = "Smartt et al. (2017)"
    elif model == "WoKo2017":
        legend_name = "Wollaeger et al. (2017)"

    return legend_name

# Parse command line
opts = parse_commandline()

models = opts.model.split(",")
for model in models:
    if not model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017"]:
        print "Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017"
        exit(0)

lightcurvesDir = opts.lightcurvesDir

# These are the default values supplied with respect to generating lightcurves
tini = 0.1
tmax = 50.0
dt = 0.1

vmin = 0.02
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 1

beta = 3.0
kappa_r = 10.0
slope_r = -1.2
theta_r = 0.0

# read in samples
samples = KNTable.read_samples(opts.posterior_samples)

print "m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"]))
print "m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"]))

# Calc lambdas
samples = samples.calc_tidal_lambda(remove_negative_lambda=True)
# Calc compactness
samples = samples.calc_compactness()
# Calc baryonic mass
samples = samples.calc_baryonic_mass()
#samples = samples.downsample()

#add default values from above to table
samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt
samples['vmin'] = vmin
samples['th'] = th
samples['ph'] = ph
samples['kappa'] = kappa
samples['eps'] = eps
samples['alp'] = alp
samples['eth'] = eth
samples['flgbct'] = flgbct
samples['beta'] = beta
samples['kappa_r'] = kappa_r
samples['slope_r'] = slope_r
samples['theta_r'] = theta_r

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
model_tables = {}
for model in models:
    model_tables[model] = KNTable.model(model, samples)

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,"_".join(models))
plotDir = os.path.join(plotDir,"event")
plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]

tini, tmax, dt = 0.1, 50.0, 0.1
tt = np.arange(tini,tmax+dt,dt)

mag_all = {}
lbol_all = {}

for model in models:
    mag_all[model] = {}
    lbol_all[model] = {}

    lbol_all[model] = np.empty((0,len(tt)), float)
    for filt, color, magidx in zip(filts,colors,magidxs):
        mag_all[model][filt] = np.empty((0,len(tt)))

for model in models:
    for row in model_tables[model]:
        t, lbol, mag = row["t"], row["lbol"], row["mag"]

        if np.sum(lbol) == 0.0:
            #print "No luminosity..."
            continue

        allfilts = True
        for filt, color, magidx in zip(filts,colors,magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0]
            if len(idx) == 0:
                allfilts = False
                break
        if not allfilts: continue
        for filt, color, magidx in zip(filts,colors,magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0]
            f = interp.interp1d(t[idx], mag[magidx][idx], fill_value='extrapolate')
            maginterp = f(tt)
            mag_all[model][filt] = np.append(mag_all[model][filt],[maginterp],axis=0)
        idx = np.where((~np.isnan(np.log10(lbol))) & ~(lbol==0))[0]
        f = interp.interp1d(t[idx], np.log10(lbol[idx]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbol_all[model] = np.append(lbol_all[model],[lbolinterp],axis=0)

if opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
    data_out = lightcurve_utils.loadEvent(filename)
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key][:,0] = data_out[key][:,0] - opts.T0
            data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

linestyles = ['-', '-.', ':','--','-']

plotName = "%s/mag.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    maglen, ttlen = lbol_all[model].shape
    for jj in xrange(maglen):
        for filt, color, magidx in zip(filts,colors,magidxs):
            if cnt == 0 and ii == 0:
                plt.plot(tt,mag_all[model][filt][jj,:],alpha=0.2,c=color,label=filt,linestyle=linestyles[ii])
            else:
                plt.plot(tt,mag_all[model][filt][jj,:],alpha=0.2,c=color,linestyle=linestyles[ii])
        cnt = cnt + 1
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

filts = ["g","r","i","z","y","J","H","K"]
#filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]
magidxs = [1,2,3,4,5,6,7,8]
colors_names=cm.rainbow(np.linspace(0,1,len(models)))

plotName = "%s/mag_panels.pdf"%(plotDir)
plt.figure(figsize=(20,18))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if opts.doEvent:
        if not filt in data_out: continue
        samples = data_out[filt]
        t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

        idx = np.where(np.isfinite(sigma_y))[0]
        plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c='k')
        idx = np.where(~np.isfinite(sigma_y))[0]
        plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c='k')

    for ii, model in enumerate(models):
        legend_name = get_legend(model)

        magmed = np.median(mag_all[model][filt],axis=0)
        magmax = np.max(mag_all[model][filt],axis=0)
        magmin = np.min(mag_all[model][filt],axis=0)

        plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
        plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],alpha=0.2)
    plt.ylabel('%s'%filt,fontsize=24,rotation=0,labelpad=20)
    plt.xlim([0.0, 14.0])
    plt.ylim([-18.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-14,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':24},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=24)
plt.savefig(plotName)
plt.close()

plotName = "%s/iminusg.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    legend_name = get_legend(model)

    magmed = np.median(mag_all[model]["i"]-mag_all[model]["g"],axis=0)
    magmax = np.max(mag_all[model]["i"]-mag_all[model]["g"],axis=0)
    magmin = np.min(mag_all[model]["i"]-mag_all[model]["g"],axis=0)

    plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlim([0.0, 14.0])
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/Lbol.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    legend_name = get_legend(model)

    lbolmed = np.median(lbol_all[model],axis=0)
    lbolmax = np.max(lbol_all[model],axis=0)
    lbolmin = np.min(lbol_all[model],axis=0)
    plt.loglog(tt,lbolmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,lbolmin,lbolmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlim([0.0, 50.0])
plt.legend(loc="best")
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.savefig(plotName)
plt.close()

bounds = [-3.0,-1.0]
xlims = [-3.0,-1.0]
ylims = [1e-1,10]

plotName = "%s/mej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins, hist1 = hist_results(np.log10(model_tables[model]["mej"]),Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

bounds = [0.0,1.0]
xlims = [0.0,1.0]
ylims = [1e-1,20]

plotName = "%s/vej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins, hist1 = hist_results(model_tables[model]["vej"],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)

plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

bounds = [0.0,2.0]
xlims = [0.0,2.0]
ylims = [1e-1,10]

plotName = "%s/masses.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins1, hist1 = hist_results(model_tables[model]["m1"],Nbins=25,bounds=bounds)
    plt.semilogy(bins1,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    bins2, hist2 = hist_results(model_tables[model]["m2"],Nbins=25,bounds=bounds)
    plt.semilogy(bins2,hist2,'--',color=colors_names[ii],linewidth=3)
plt.xlabel(r"Masses",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()        
