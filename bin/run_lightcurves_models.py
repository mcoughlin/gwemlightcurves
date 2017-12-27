
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

import corner

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-n","--name",default="PS1-13cyr")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)
    parser.add_option("--doCoverage",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False) 
    parser.add_option("--doWaveformExtrapolate",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("-m","--model",default="KaKy2016")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("-f","--filters",default="g,r,i,z")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017x2","RoFe2017"]:
    print "Model must be either: DiUj2017,KaKy2016,Me2017,Me2017x2,SmCh2017,WoKo2017,BaKa2016, Ka2017, Ka2017x2, RoFe2017"
    exit(0)

if opts.doFixZPT0:
    ZPRange = 0.1
    T0Range = 0.1
else:
    ZPRange = 5.0
    T0Range = 14.0

filters = opts.filters.split(",")

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models'
elif opts.doGoingTheDistance:
    basename = 'going-the-distance'
elif opts.doMassGap:
    basename = 'massgap'
else:
    basename = 'gws'
plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_EOSFit_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
else:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s'%opts.model)
plotDir = os.path.join(plotDir,"_".join(filters))
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
if opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017x2","RoFe2017"]:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
if opts.doReduced:
    plotDir = os.path.join(plotDir,"%s_reduced"%opts.name)
else:
    plotDir = os.path.join(plotDir,opts.name)
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

dataDir = opts.dataDir
lightcurvesDir = opts.lightcurvesDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%lightcurvesDir
elif opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.name)
else:
    filename = "%s/lightcurves.tmp"%lightcurvesDir

errorbudget = opts.errorbudget
mint = opts.tmin
maxt = opts.tmax
dt = opts.dt
tt = np.arange(mint,maxt,dt)

if opts.doModels or opts.doGoingTheDistance or opts.doMassGap:
    if opts.doModels:
        data_out = lightcurve_utils.loadModels(opts.outputDir,opts.name)
        if not opts.name in data_out:
            print "%s not in file..."%opts.name
            exit(0)

        data_out = data_out[opts.name]

    elif opts.doGoingTheDistance or opts.doMassGap:

        truths = {}
        if opts.doGoingTheDistance:
            data_out = lightcurve_utils.going_the_distance(opts.dataDir,opts.name)
        elif opts.doMassGap:
            data_out, truths = lightcurve_utils.massgap(opts.dataDir,opts.name)

        if "m1" in truths:
            eta = lightcurve_utils.q2eta(truths["q"])
            m1, m2 = truths["m1"], truths["m2"]
            mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
            q = 1/q 
            chi_eff = truths["a1"]
            chi_eff = 0.75
            
            eta = lightcurve_utils.q2eta(np.mean(data_out["q"]))
            m1, m2 = lightcurve_utils.mc2ms(np.mean(data_out["mc"]), eta)
            mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
            q = 1/q

        else:
            eta = lightcurve_utils.q2eta(data_out["q"])
            m1, m2 = lightcurve_utils.mc2ms(data_out["mc"], eta)
            q = m2/m1
            mc = data_out["mc"]

            m1, m2 = np.mean(m1), np.mean(m2)
            chi_eff = 0.0       

        c1, c2 = 0.147, 0.147
        mb1, mb2 = lightcurve_utils.EOSfit(m1,c1), lightcurve_utils.EOSfit(m2,c2)
        th = 0.2
        ph = 3.14

        if m1 > 3:
            from gwemlightcurves.EjectaFits.KaKy2016 import calc_meje, calc_vave
            mej = calc_meje(q,chi_eff,c2,mb2,m2)
            vej = calc_vave(q)

        else:
            from gwemlightcurves.EjectaFits.DiUj2017 import calc_meje, calc_vej
            mej = calc_meje(m1,mb1,c1,m2,mb2,c2)
            vej = calc_vej(m1,c1,m2,c2)

        filename = os.path.join(plotDir,'truth_mej_vej.dat')
        fid = open(filename,'w+')
        fid.write('%.5f %.5f\n'%(mej,vej))
        fid.close()

        if m1 > 3:
            filename = os.path.join(plotDir,'truth.dat')
            fid = open(filename,'w+')
            fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(q,chi_eff,c2,mb2,m2))
            fid.close()

            t, lbol, mag = KaKy2016_model(q,chi_eff,m2,mb2,c2,th,ph) 

        else:
            filename = os.path.join(plotDir,'truth.dat')
            fid = open(filename,'w+')
            fid.write('%.5f %.5f %.5f %.5f\n'%(m1,c1,m2,c2))
            fid.close()

            t, lbol, mag = DiUj2017_model(m1,mb1,c1,m2,mb2,c2,th,ph)

        data_out = {}
        data_out["t"] = t
        data_out["u"] = mag[0]
        data_out["g"] = mag[1]
        data_out["r"] = mag[2]
        data_out["i"] = mag[3]
        data_out["z"] = mag[4]
        data_out["y"] = mag[5]
        data_out["J"] = mag[6]
        data_out["H"] = mag[7]
        data_out["K"] = mag[8]

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key] = np.vstack((data_out["t"],data_out[key],errorbudget*np.ones(data_out["t"].shape))).T

    idxs = np.intersect1d(np.where(data_out["t"]>=mint)[0],np.where(data_out["t"]<=maxt)[0])
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    tt = np.arange(mint,maxt,dt)
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:

            ii = np.where(np.isfinite(data_out[key][:,1]))[0]
            f = interp.interp1d(data_out[key][ii,0], data_out[key][ii,1], fill_value=np.nan, bounds_error=False)
            maginterp = f(tt)

            data_out[key] = np.vstack((tt,maginterp,errorbudget*np.ones(tt.shape))).T
           

    del data_out["t"]

    if opts.doReduced:
        tt = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
        for ii,key in enumerate(data_out.iterkeys()):
            maginterp = np.interp(tt,data_out[key][:,0],data_out[key][:,1],left=np.nan, right=np.nan)
            data_out[key] = np.vstack((tt,maginterp,errorbudget*np.ones(tt.shape))).T

    for ii,key in enumerate(data_out.iterkeys()):
        if ii == 0:
            samples = data_out[key].copy()
        else:
            samples = np.vstack((samples,data_out[key].copy()))

    #idx = np.argmin(samples[:,0])
    #t0_save = samples[idx,0] -  1.0
    #samples[:,0] = samples[:,0] - t0_save
    idx = np.argsort(samples[:,0])
    samples = samples[idx,:]

    #for ii,key in enumerate(data_out.iterkeys()):
    #    data_out[key][:,0] = data_out[key][:,0] - t0_save

else:
    if opts.doEvent:
        data_out = lightcurve_utils.loadEvent(filename)
    else:
        data_out = lightcurve_utils.loadLightcurves(filename)
        if not opts.name in data_out:
            print "%s not in file..."%opts.name
            exit(0)

        data_out = data_out[opts.name]

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key][:,0] = data_out[key][:,0] - opts.T0
            data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            idxs = np.intersect1d(np.where(data_out[key][:,0]>=mint)[0],np.where(data_out[key][:,0]<=maxt)[0])
            data_out[key] = data_out[key][idxs,:]

    for ii,key in enumerate(data_out.iterkeys()):
        idxs = np.where(~np.isnan(data_out[key][:,2]))[0]
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    for ii,key in enumerate(data_out.keys()):
        if not key in filters:
            del data_out[key]

    for ii,key in enumerate(data_out.iterkeys()):
        if ii == 0:
            samples = data_out[key].copy()
        else:
            samples = np.vstack((samples,data_out[key].copy()))

    idx = np.argmin(samples[:,0])
    samples = samples[idx,:]

    filename = os.path.join(plotDir,'truth_mej_vej.dat')
    fid = open(filename,'w+')
    fid.write('%.5f %.5f\n'%(np.nan,np.nan))
    fid.close()

    if opts.model == "KaKy2016":
        filename = os.path.join(plotDir,'truth.dat')
        fid = open(filename,'w+')
        fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(np.nan,np.nan,np.nan,np.nan,np.nan))
        fid.close()
    else:
        filename = os.path.join(plotDir,'truth.dat')
        fid = open(filename,'w+')
        fid.write('%.5f %.5f %.5f %.5f\n'%(np.nan,np.nan,np.nan,np.nan))
        fid.close()

Global.data_out = data_out
Global.errorbudget = errorbudget
Global.ZPRange = ZPRange
Global.T0Range = T0Range
Global.doLightcurves = 1
Global.filters = filters
Global.doWaveformExtrapolate = opts.doWaveformExtrapolate

if opts.model == "Ka2017" or opts.model == "Ka2017x2":
    ModelPath = '%s/svdmodels'%(opts.outputDir)

    modelfile = os.path.join(ModelPath,'Ka2017_mag.pkl')
    with open(modelfile, 'rb') as handle:
        svd_mag_model = pickle.load(handle)
    Global.svd_mag_model = svd_mag_model    

    modelfile = os.path.join(ModelPath,'Ka2017_lbol.pkl')
    with open(modelfile, 'rb') as handle:
        svd_lbol_model = pickle.load(handle)
    Global.svd_lbol_model = svd_lbol_model

data, tmag, lbol, mag, t0_best, zp_best, n_params, labels, best = run.multinest(opts,plotDir)
truths = lightcurve_utils.get_truths(opts.name,opts.model,n_params,opts.doEjecta)

if n_params >= 8:
    title_fontsize = 26
    label_fontsize = 30
else:
    title_fontsize = 24
    label_fontsize = 28

plotName = "%s/corner.pdf"%(plotDir)
if opts.doFixZPT0:
    figure = corner.corner(data[:,1:-2], labels=labels[1:-1],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".1f",
                       truths=truths[1:-1], smooth=3)
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".1f",
                       truths=truths, smooth=3)
if n_params >= 8:
    figure.set_size_inches(18.0,18.0)
else:
    figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

tmag = tmag + t0_best

if opts.filters == "c,o":
    filts = ["c","o"]
    #colors = ["y","g","b","c","k","pink","orange","purple"]
    #colors = ["purple","y","g","b","c","k","pink","orange"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [9,10]
    tini, tmax, dt = opts.tmin, opts.tmax, 0.1
else:
    filts = ["u","g","r","i","z","y","J","H","K"]
    #colors = ["y","g","b","c","k","pink","orange","purple"]
    #colors = ["purple","y","g","b","c","k","pink","orange"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [0,1,2,3,4,5,6,7,8]
    tini, tmax, dt = 0.0, 21.0, 0.1    
tt = np.arange(tini,tmax,dt)

plotName = "%s/lightcurve.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
    if not filt in filters: continue
    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)

    if filt == "w":
        magave = (mag[1]+mag[2]+mag[3])/3.0
    elif filt == "c":
        magave = (mag[1]+mag[2])/2.0
    elif filt == "o":
        magave = (mag[2]+mag[3])/2.0
    else:
        magave = mag[magidx]

    ii = np.where(~np.isnan(magave))[0]
    f = interp.interp1d(tmag[ii], magave[ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'k--',linewidth=2)

if opts.filters == "c,o":
    plt.xlim([opts.tmin-2, opts.tmax+2])
    plt.ylim([-20.0,-5.0])
elif opts.model == "SN":
    plt.xlim([0.0, 10.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "Me2017":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "Me2017x2":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "SmCh2017":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "DiUj2017":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "KaKy2016":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "WoKo2016":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
else:
    plt.xlim([1.0, 18.0])
    plt.ylim([-20.0,-5.0])

plt.xlabel('Time [days]',fontsize=24)
plt.ylabel('Absolute Magnitude',fontsize=24)
plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/lightcurve_zoom.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,label='%s-band'%filt)

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=10)

    if filt == "w":
        magave = (mag[1]+mag[2]+mag[3])/3.0
    elif filt == "c":
        magave = (mag[1]+mag[2])/2.0
    elif filt == "o":
        magave = (mag[2]+mag[3])/2.0
    else:
        magave = mag[magidx]

    ii = np.where(~np.isnan(magave))[0]
    f = interp.interp1d(tmag[ii], magave[ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2)
    plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

if opts.filters == "c,o":
    plt.xlim([opts.tmin-2, opts.tmax+2])
    plt.ylim([-20.0,-5.0])
elif opts.model == "SN":
    plt.xlim([0.0, 10.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "Me2017":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "Me2017x2":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "SmCh2017":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "DiUj2017":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "KaKy2016":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
elif opts.model == "WoKo2016":
    plt.xlim([0.0, 18.0])
    plt.ylim([-20.0,-5.0])
else:
    plt.xlim([1.0, 18.0])
    plt.ylim([-20.0,-5.0])

plt.xlabel('Time [days]',fontsize=24)
plt.ylabel('Absolute Magnitude',fontsize=24)
#plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

