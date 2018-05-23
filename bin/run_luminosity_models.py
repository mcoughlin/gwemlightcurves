
import os, sys, glob, copy, pickle
import optparse
import numpy as np

from scipy.interpolate import interpolate as interp
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
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("-m","--model",default="KaKy2016")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)
    parser.add_option("--n_live_points",default=100,type=int)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017x2","RoFe2017"]:
    print "Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017,BaKa2016, Ka2017, Ka2017x2, RoFe2017"
    exit(0)

if opts.doFixZPT0:
    ZPRange = 0.1
    T0Range = 0.1
else:
    ZPRange = 50.0
    T0Range = 5.0

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models_luminosity'
elif opts.doGoingTheDistance:
    basename = 'going-the-distance_luminosity'
elif opts.doMassGap:
    basename = 'massgap_luminosity'
else:
    basename = 'gws_luminosity'
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
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
if opts.model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017"]:
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

if opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.name)

errorbudget = opts.errorbudget
mint = opts.tmin
maxt = opts.tmax
dt = opts.dt

if opts.doModels or opts.doGoingTheDistance or opts.doMassGap:
    if opts.doModels:
        data_out = lightcurve_utils.loadModelsLbol(opts.outputDir,opts.name)
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
            mej = KaKy2016KilonovaLightcurve.calc_meje(q,chi_eff,c2,mb2,m2)
            vej = KaKy2016KilonovaLightcurve.calc_vave(q)
        else:
            mej = DiUj2017KilonovaLightcurve.calc_meje(m1,mb1,c1,m2,mb2,c2)
            vej = DiUj2017KilonovaLightcurve.calc_vej(m1,c1,m2,c2)

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
        data_out["Lbol"] = lbol

    idxs = np.intersect1d(np.where(data_out["tt"]>=mint)[0],np.where(data_out["tt"]<=maxt)[0])
    for ii,key in enumerate(data_out.iterkeys()):
        data_out[key] = data_out[key][idxs]

    tt = np.arange(mint,maxt,dt)
    ii = np.where(np.isfinite(data_out["Lbol"]))[0]
    f = interp.interp1d(data_out["tt"][ii], data_out["Lbol"][ii], fill_value=np.nan, bounds_error=False)
    Lbolinterp = f(tt)
   
    data_out["tt"] = tt 
    data_out["Lbol"] = Lbolinterp

    if opts.doReduced:
        tt = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
        ii = np.where(np.isfinite(data_out["Lbol"]))[0]
        f = interp.interp1d(data_out["tt"][ii], data_out["Lbol"][ii], fill_value=np.nan, bounds_error=False)
        Lbolinterp = f(tt)
        data_out["tt"] = tt          
        data_out["Lbol"] = Lbolinterp

    data_out["Lbol_err"] = np.zeros(data_out["Lbol"].shape)
else:
    if opts.doEvent:
        data_out = lightcurve_utils.loadEventLbol(filename)
    else:
        print "Not implemented..."

    #data_out["tt"] = data_out["tt"] - opts.T0
    data_out_exclude = copy.deepcopy(data_out)

    idxs = np.intersect1d(np.where(data_out["tt"]>=mint)[0],np.where(data_out["tt"]<=maxt)[0])
    idxs_exclude = np.union1d(np.where(data_out["tt"]<mint)[0],np.where(data_out["tt"]>maxt)[0])
    for ii,key in enumerate(data_out.iterkeys()):
        data_out[key] = data_out[key][idxs]
        data_out_exclude[key] = data_out_exclude[key][idxs_exclude]

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
Global.doLuminosity = 1

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

pcklFile = os.path.join(plotDir,"data.pkl")
f = open(pcklFile, 'wb')
pickle.dump((data_out, data, tmag, lbol, mag, t0_best, zp_best, n_params, labels, best,truths), f)
f.close()

if n_params >= 8:
    title_fontsize = 26
    label_fontsize = 30
else:
    title_fontsize = 30
    label_fontsize = 30

plotName = "%s/corner.pdf"%(plotDir)
if opts.doFixZPT0:
    figure = corner.corner(data[:,1:-2], labels=labels[1:-1],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       truths=truths[1:-1], smooth=3, levels=[0.68, 0.95, 0.997])
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       truths=truths, smooth=3, levels=[0.68, 0.95, 0.997])
if n_params >= 10:
    figure.set_size_inches(40.0,40.0)
elif n_params >= 6:
    figure.set_size_inches(22.0,22.0)
else:
    figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

tmag = tmag + t0_best

plotName = "%s/lbol.pdf"%(plotDir)
fig = plt.figure(figsize=(10,8))
fig.set_size_inches(14.0,14.0)

t, y, sigma_y = data_out["tt"], data_out["Lbol"], data_out["Lbol_err"]
idx = np.where(~np.isnan(y))[0]
t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
idx = np.where(sigma_y > y)[0]
sigma_y[idx] = 0.99*y[idx]
plt.errorbar(t,y,sigma_y,fmt='o',c='b')

t, y, sigma_y = data_out_exclude["tt"], data_out_exclude["Lbol"], data_out_exclude["Lbol_err"]
idx = np.where(~np.isnan(y))[0]
t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
idx = np.where(sigma_y > y)[0]
sigma_y[idx] = 0.99*y[idx]
plt.errorbar(t,y,sigma_y,fmt='x',c='b')

tini, tmax, dt = 0.0, 14.0, 0.1
tt = np.arange(tini,tmax,dt)

ii = np.where(~np.isnan(lbol))[0]
f = interp.interp1d(tmag[ii], np.log10(lbol[ii]), fill_value='extrapolate')
lbolinterp = 10**f(tt)
zp_factor = 10**(zp_best/-2.5)
plt.loglog(tt,zp_factor*lbolinterp,'k--',linewidth=2)
plt.fill_between(tt,zp_factor*lbolinterp/(1+errorbudget),zp_factor*lbolinterp*(1+errorbudget),facecolor='k',alpha=0.2)

#plt.xlim([10**-2,50])
#plt.ylim([10.0**39,10.0**45])
plt.xlim([4*10**-1,14])
plt.ylim([10.0**40,2*10.0**42])
plt.xlabel('Time [days]',fontsize=30)
plt.ylabel('Bolometric Luminosity [erg/s]',fontsize=30)
plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()

if opts.doFixZPT0:
    yvals = np.logspace(np.log10(3*10.0**39),np.log10(10.0**40),len(labels[1:-1]))
    for label, best_value, yval in zip(labels[1:-1],best[1:-1],yvals):
        plt.text(2*10**-1, yval, r'%s = %.3f'%(label,best_value), fontsize=15)

plt.savefig(plotName)
plt.close()

