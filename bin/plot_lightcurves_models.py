#!/usr/bin/python

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
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

from chainconsumer import ChainConsumer
import corner

from gwemlightcurves import lightcurve_utils, ztf_utils, Global

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-n","--name",default="GW170817")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--doFixedLimit",  action="store_true", default=False)
    parser.add_option("--limits",default="20.4,20.4")
    parser.add_option("--doZTF",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--distance_uncertainty",default=-1.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)
    parser.add_option("--doCoverage",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False) 
    parser.add_option("--doFitSigma",  action="store_true", default=False)
    parser.add_option("--doWaveformExtrapolate",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doBNSFit",  action="store_true", default=False)
    parser.add_option("-m","--model",default="KaKy2016")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("-f","--filters",default="g,r,i,z")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)
    parser.add_option("--n_live_points",default=100,type=int)
    parser.add_option("--n_coeff",default=10,type=int)
    parser.add_option("--evidence_tolerance",default=0.5,type=float)
    parser.add_option("--max_iter",default=0,type=int)

    parser.add_option("--doFixXlan",  action="store_true", default=False) 
    parser.add_option("--Xlan",default=1e-9,type=float) 
    parser.add_option("--doFixT",  action="store_true", default=False)
    parser.add_option("--T",default=1e4,type=float)
    parser.add_option("--doFixPhi",  action="store_true", default=False)
    parser.add_option("--phi",default=0.0,type=float)

    parser.add_option("--colormodel",default="a2.0")

    parser.add_option("--username",default="username")
    parser.add_option("--password",default="password")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

if not opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017_A","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017_A","Ka2017inc","Ka2017x2","Ka2017x2inc","Ka2017x3","Ka2017x3inc","RoFe2017","BoxFit","TrPi2018","Ka2017_TrPi2018","Ka2017_TrPi2018_A","Bu2019","Bu2019inc","Bu2019lf","Bu2019lr","Bu2019lm","Bu2019inc_TrPi2018","Bu2019rp","Bu2019rps"]:
    print("Model must be either: DiUj2017,KaKy2016,Me2017,Me2017_A,Me2017x2,SmCh2017,WoKo2017,BaKa2016, Ka2017, Ka2017inc, Ka2017_A, Ka2017x2, Ka2017x2inc, Ka2017x3, Ka2017x3inc, RoFe2017, BoxFit, TrPi2018, Ka2017_TrPi2018, Ka2017_TrPi2018_A, Bu2019, Bu2019inc, Bu2019lf, Bu2019lr, Bu2019lm, Bu2019inc_TrPi2018,Bu2019rp,Bu2019rps")
    exit(0)

if opts.doFixZPT0:
    ZPRange = 0.1
    T0Range = 0.1
else:
    ZPRange = 5.0
    T0Range = 0.1

if opts.distance_uncertainty > 0:
    ZPRange = np.abs(5*(opts.distance_uncertainty/opts.distance)/np.log(10))

filters = opts.filters.split(",")
limits = [float(x) for x in opts.limits.split(",")]
colormodel = opts.colormodel.split(",")
if len(colormodel) == 1:
    colormodel = colormodel[0]

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models'
elif opts.doGoingTheDistance:
    basename = 'going-the-distance'
elif opts.doMassGap:
    basename = 'massgap'
elif opts.doFixedLimit:
    basename = 'limits'
else:
    basename = 'gws'
plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_EOSFit_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
elif opts.doBNSFit:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_BNSFit_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s_BNSFit'%opts.model)
else:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s'%opts.model)
if opts.model in ["Ka2017inc","Ka2017x2inc","Ka2017x3inc"]:
    plotDir = os.path.join(plotDir,'%s'%("_".join(colormodel)))
plotDir = os.path.join(plotDir,"_".join(filters))
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
if opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017_A","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017inc","Ka2017_A","Ka2017x2","Ka2017x2inc","Ka2017x3","Ka2017x3inc", "RoFe2017","Bu2019","Bu2019inc","Bu2019rp","Bu2019rps"]:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
if opts.doReduced:
    plotDir = os.path.join(plotDir,"%s_reduced"%opts.name)
else:
    if "knova2D" in opts.name:
        nameSplit = opts.name.split("_")
        nameCombine = "_".join(nameSplit[-2:])
        plotDir = os.path.join(plotDir,nameCombine)
    elif opts.name == "knova_d1_n10_m0.040_vk0.10_fd1.0_Xlan1e-2.0":
        plotDir = os.path.join(plotDir,'sphere')
    else:
        plotDir = os.path.join(plotDir,opts.name)
if opts.doFixXlan:
    plotDir = os.path.join(plotDir,"%.2f"% (np.log10(opts.Xlan)))
if opts.doFixT:
    plotDir = os.path.join(plotDir,"%.2f"% (np.log10(opts.T)))
if opts.doFixPhi:
    plotDir = os.path.join(plotDir,"%.2f"% (opts.phi))
if opts.doFitSigma:
    plotDir = os.path.join(plotDir,"fit")
else:
    plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

dataDir = opts.dataDir
lightcurvesDir = opts.lightcurvesDir

multifile = lightcurve_utils.get_post_file(plotDir)
data = np.loadtxt(multifile)

plotName = "%s/corner.pdf"%(plotDir)

labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej,dyn} / M_\odot)$",r"${\rm log}_{10} (M_{\rm ej,wind} / M_\odot)$",r"$\Phi {\rm [deg]}$",r"$\Theta_{\rm obs} {\rm [deg]}$","ZP"]
n_params = len(labels)

if n_params >= 6:
    title_fontsize = 36
    label_fontsize = 36
else:
    title_fontsize = 30
    label_fontsize = 30

# If you pass in parameter labels and only one chain, you can also get parameter bounds
#c = ChainConsumer().add_chain( data[:,1:-2], parameters=labels[1:-1])
#c.configure(diagonal_tick_labels=False, tick_font_size=label_fontsize, label_font_size=label_fontsize, max_ticks=4, colors="#FF7F50", smooth=0, kde=[0.3,0.3,0.3,0.3,0.3,0.3], linewidths=2, summary=True, bar_shade=True, statistics="max_symmetric")
#fig = c.plotter.plot(figsize="column")

ranges = [0.99, 0.99, 0.99, 0.99]
kwargs = dict(bins=20, smooth=3, label_kwargs=dict(fontsize=label_fontsize),
              show_titles=True,
              title_kwargs=dict(fontsize=title_fontsize, pad=20),
              range=ranges,
              color='#0072C1',
              truth_color='tab:orange', quantiles=[0.05, 0.5, 0.95],
              labelpad = 0.01,
              #levels=(0.68, 0.95),
              levels=[0.10, 0.32, 0.68, 0.90],
              plot_density=False, plot_datapoints=False, fill_contours=True,
              max_n_ticks=4, min_n_ticks=3)

fig = corner.corner(data[:,1:-2], labels=labels[1:-1], **kwargs)

if n_params >= 10:
    fig.set_size_inches(40.0,40.0)
elif n_params > 6:
    fig.set_size_inches(24.0,24.0)
else:
    fig.set_size_inches(20.0,20.0)
plt.savefig(plotName)
plt.close()

