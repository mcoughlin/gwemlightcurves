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
plt.rcParams['text.usetex'] = True

import corner

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
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

if not opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017_A","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017_A","Ka2017inc","Ka2017x2","Ka2017x2inc","Ka2017x3","Ka2017x3inc","RoFe2017","BoxFit","TrPi2018","Ka2017_TrPi2018","Ka2017_TrPi2018_A","Bu2019","Bu2019inc","Bu2019lf","Bu2019lr","Bu2019inc_TrPi2018","Bu2019rp","Bu2019rps","Wo2020","Wo2020dw","Wo2020dyn","Bu2019nsbh","Bu2019rpd"]:
    print("Model must be either: DiUj2017,KaKy2016,Me2017,Me2017_A,Me2017x2,SmCh2017,WoKo2017,BaKa2016, Ka2017, Ka2017inc, Ka2017_A, Ka2017x2, Ka2017x2inc, Ka2017x3, Ka2017x3inc, RoFe2017, BoxFit, TrPi2018, Ka2017_TrPi2018, Ka2017_TrPi2018_A, Bu2019, Bu2019inc, Bu2019lf, Bu2019lr, Bu2019inc_TrPi2018, Bu2019rp, Bu2019rps, Wo2020, Wo2020dw, Wo2020dyn, Bu2019nsbh, Bu2019rpd")
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
if opts.model in ["DiUj2017","KaKy2016","Me2017","Me2017_A","Me2017x2","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017inc","Ka2017_A","Ka2017x2","Ka2017x2inc","Ka2017x3","Ka2017x3inc", "RoFe2017","Bu2019","Bu2019inc","Bu2019rp","Bu2019rps","Wo2020","Wo2020dw","Wo2020dyn","Bu2019nsbh"]:
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

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%lightcurvesDir
elif opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.name)
else:
    filename = "%s/lightcurves.tmp"%lightcurvesDir

if opts.doZTF:
    if not os.path.isfile(filename):
        ztf_utils.get_ztf(filename,opts.name,opts.username,opts.password)

errorbudget = opts.errorbudget
mint = opts.tmin
maxt = opts.tmax
dt = opts.dt
tt = np.arange(mint,maxt,dt)

if opts.doModels or opts.doGoingTheDistance or opts.doMassGap or opts.doFixedLimit:
    if opts.doModels:
        data_out = lightcurve_utils.loadModels(opts.outputDir,opts.name)
        if not opts.name in data_out:
            print("%s not in file..."%opts.name)
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

    elif opts.doFixedLimit:
        tt = np.arange(mint,maxt,dt)
        data_out = {}
        data_out["t"] = tt
        for filt, limit in zip(filters,limits):
            data_out[filt] = limit*np.ones(tt.shape)

    for ii,key in enumerate(list(data_out.iterkeys())):
        if key == "t":
            continue
        else:
            data_out[key] = np.vstack((data_out["t"],data_out[key],errorbudget*np.ones(data_out["t"].shape))).T

    idxs = np.intersect1d(np.where(data_out["t"]>=mint)[0],np.where(data_out["t"]<=maxt)[0])
    for ii,key in enumerate(list(data_out.iterkeys())):
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    tt = np.arange(mint,maxt,dt)
    for ii,key in enumerate(list(data_out.iterkeys())):
        if key == "t":
            continue
        else:
            ii = np.where(np.isfinite(data_out[key][:,1]))[0]
            if opts.doWaveformExtrapolate:
                f = interp.interp1d(data_out[key][ii,0], data_out[key][ii,1], fill_value='extrapolate', bounds_error=False)
            else:
                f = interp.interp1d(data_out[key][ii,0], data_out[key][ii,1], fill_value=np.nan, bounds_error=False)
            maginterp = f(tt)

            if opts.doFixedLimit:
                data_out[key] = np.vstack((tt,maginterp,np.inf*np.ones(tt.shape))).T
                data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)
            else:
                data_out[key] = np.vstack((tt,maginterp,errorbudget*np.ones(tt.shape))).T
           
    del data_out["t"]

    if opts.doReduced:
        tt = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
        for ii,key in enumerate(list(data_out.iterkeys())):
            maginterp = np.interp(tt,data_out[key][:,0],data_out[key][:,1],left=np.nan, right=np.nan)
            data_out[key] = np.vstack((tt,maginterp,errorbudget*np.ones(tt.shape))).T

    for ii,key in enumerate(list(data_out.iterkeys())):
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
            print("%s not in file..."%opts.name)
            exit(0)

        data_out = data_out[opts.name]

    for ii,key in enumerate(list(data_out.keys())):
        if key == "t":
            continue
        else:
            data_out[key][:,0] = data_out[key][:,0] - opts.T0
            if not opts.doZTF:
                data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

    for ii,key in enumerate(list(data_out.keys())):
        if key == "t":
            continue
        else:
            idxs = np.intersect1d(np.where(data_out[key][:,0]>=mint)[0],np.where(data_out[key][:,0]<=maxt)[0])
            data_out[key] = data_out[key][idxs,:]

    for ii,key in enumerate(list(data_out.keys())):
        idxs = np.where(~np.isnan(data_out[key][:,2]))[0]
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    for ii,key in enumerate(list(data_out.keys())):
        if not key in filters:
            del data_out[key]

    for ii,key in enumerate(list(data_out.keys())):
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
Global.n_coeff = opts.n_coeff

if opts.doFixXlan:
    Global.Xlan = np.log10(opts.Xlan)

if opts.doFixT:
    Global.T = np.log10(opts.T)

if opts.doFixXlan:
    Global.phi = opts.phi

if opts.model in ["Ka2017", "Ka2017inc", "Ka2017_A", "Ka2017x2", "Ka2017x2inc", "Ka2017x3", "Ka2017x3inc", "Ka2017_TrPi2018", "Ka2017_TrPi2018_A", "Bu2019", "Bu2019inc", "Bu2019inc_TrPi2018", "Bu2019lf", "Bu2019lr","Bu2019rp","Bu2019rps","Wo2020","Wo2020dw","Wo2020dyn","Bu2019nsbh","Bu2019rpd"]:
    ModelPath = '%s/svdmodels'%(opts.outputDir)

    if opts.model == "Bu2019":
        modelfile = os.path.join(ModelPath,'Bu2019_phi%d_mag.pkl' % opts.phi)
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model    
            
        modelfile = os.path.join(ModelPath,'Bu2019_phi%d_lbol.pkl' % opts.phi)
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model in ["Bu2019inc","Bu2019inc_TrPi2018"]:
        modelfile = os.path.join(ModelPath,'Bu2019inc_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model    
            
        modelfile = os.path.join(ModelPath,'Bu2019inc_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Bu2019lf":
        modelfile = os.path.join(ModelPath,'Bu2019lf_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath,'Bu2019lf_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Bu2019lr":
        modelfile = os.path.join(ModelPath,'Bu2019lr_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath,'Bu2019lr_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Bu2019rp":
        modelfile = os.path.join(ModelPath,'Bu2019rp_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath,'Bu2019rp_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Bu2019rps":
        modelfile = os.path.join(ModelPath,'Bu2019rps_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath,'Bu2019rps_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Bu2019rpd":
        modelfile = os.path.join(ModelPath,'Bu2019rpd_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath,'Bu2019rpd_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model
    elif opts.model == "Wo2020dyn":
        modelfile = os.path.join(ModelPath,'Wo2020dyn_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath, 'Wo2020dyn_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Wo2020dw":
        modelfile = os.path.join(ModelPath,'Wo2020dw_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath, 'Wo2020dw_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Wo2020":
        modelfile = os.path.join(ModelPath,'Wo2020_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath, 'Wo2020_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    elif opts.model == "Bu2019nsbh":
        modelfile = os.path.join(ModelPath,'Bu2019nsbh_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model

        modelfile = os.path.join(ModelPath, 'Bu2019nsbh_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model
    else:
        modelfile = os.path.join(ModelPath,'Ka2017_mag.pkl')
        with open(modelfile, 'rb') as handle:
            svd_mag_model = pickle.load(handle)
        Global.svd_mag_model = svd_mag_model    
    
        modelfile = os.path.join(ModelPath,'Ka2017_lbol.pkl')
        with open(modelfile, 'rb') as handle:
            svd_lbol_model = pickle.load(handle)
        Global.svd_lbol_model = svd_lbol_model

    if opts.model in ["Ka2017inc"]:
        modelfile = os.path.join(ModelPath,'%s.pkl' % colormodel)
        with open(modelfile, 'rb') as handle:
            svd_mag_color_model = pickle.load(handle)
        Global.svd_mag_color_model = svd_mag_color_model
    elif opts.model in ["Ka2017x2inc","Ka2017x3inc"]:
        Global.svd_mag_color_models = []
        for colorm in colormodel:
            if colorm == "a1.0":
                Global.svd_mag_color_models.append("a1.0")
            else:
                modelfile = os.path.join(ModelPath,'%s.pkl' % colorm)
                with open(modelfile, 'rb') as handle:
                    svd_mag_color_model = pickle.load(handle)
                Global.svd_mag_color_models.append(svd_mag_color_model)

data, tmag, lbol, mag, t0_best, zp_best, n_params, labels, best = run.multinest(opts,plotDir)
truths = lightcurve_utils.get_truths(opts.name,opts.model,n_params,opts.doEjecta)

pcklFile = os.path.join(plotDir,"data.pkl")
f = open(pcklFile, 'wb')
pickle.dump((data_out, data, tmag, lbol, mag, t0_best, zp_best, n_params, labels, best,truths), f)
f.close()

if n_params >= 6:
    title_fontsize = 36
    label_fontsize = 36
else:
    title_fontsize = 30
    label_fontsize = 30

plotName = "%s/corner.pdf"%(plotDir)
if opts.doFixZPT0:
    figure = corner.corner(data[:,1:-2], labels=labels[1:-1],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       truths=truths[1:-1], smooth=3,
                       color="coral")
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       truths=truths, smooth=3,
                       color="coral")
if n_params >= 10:
    figure.set_size_inches(40.0,40.0)
elif n_params >= 6:
    figure.set_size_inches(24.0,24.0)
else:
    figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

tmag = tmag + t0_best

#colors=cm.rainbow(np.linspace(0,1,len(filters)))
colors=cm.Spectral(np.linspace(0,1,len(filters)))[::-1]

tini, tmax, dt = opts.tmin, opts.tmax, 0.1
tt = np.arange(tini,tmax,dt)

plotName = "%s/lightcurve.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color in zip(filters,colors):
    if not filt in filters: continue
    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)
    magave = lightcurve_utils.get_mag(mag,filt)

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
elif opts.model == "Me2017_A":
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
elif opts.model == "TrPi2018":
    plt.xlim([0.0, 5.0])
    plt.ylim([-40.0,-10.0])
elif opts.model == "Ka2017_A":
    plt.xlim([0.0, 14.0])
    plt.ylim([-30.0,0.0])
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
for filt, color in zip(filters,colors):
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

    magave = lightcurve_utils.get_mag(mag,filt)

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
elif opts.model == "Me2017_A":
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
elif opts.model == "TrPi2018":
    plt.xlim([0.0, 5.0])
    plt.ylim([-40.0,-10.0])
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

color2 = 'coral'
color1 = 'cornflowerblue'

plotName = "%s/models_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,18))
plt.figure(figsize=(20,28))

tini, tmax, dt = 0.0, 21.0, 0.1
tt = np.arange(tini,tmax,dt)

cnt = 0
for filt, color in zip(filters,colors):
    cnt = cnt+1
    if cnt == 1:
        ax1 = plt.subplot(len(filters),1,cnt)
    else:
        ax2 = plt.subplot(len(filters),1,cnt,sharex=ax1,sharey=ax1)

    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color, markersize=16, label='%s-band'%filt, markeredgecolor='k')

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=16, markeredgecolor='k')

    magave = lightcurve_utils.get_mag(mag,filt)
    ii = np.where(~np.isnan(magave))[0]
    f = interp.interp1d(tmag[ii], magave[ii], fill_value='extrapolate')
    maginterp = f(tt)
    #plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=3)
    #plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

    plt.plot(tt,maginterp+zp_best,'--',c=color2,linewidth=3)
    plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color2,alpha=0.2)

    plt.ylabel(r'$%s$'%filt,fontsize=52,rotation=0,labelpad=40)

    if opts.name == "GW170817":
        if opts.model in ["Ka2017inc","Ka2017x2","Ka2017x2inc"]:
            plt.xlim([0.0, 14.0])
            plt.ylim([-17.0,-11.0])
        elif opts.model in ["Bu2019","Bu2019inc"]:
            plt.xlim([0.0, 14.0])
            plt.ylim([-17.0,-11.0])
        else:
            plt.xlim([0.0, 18.0])
            plt.ylim([-18.0,-10.0])
    elif opts.doZTF:
        plt.xlim([0.0, 7.0])
        plt.ylim([18.0,22.0])
    else:
        plt.xlim([0.0, 10.0])
        plt.ylim([-22.0,-8.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        if opts.name == "GW170817":
            ax1.set_yticks([-18,-16,-14,-12,-10])
        else:
            ax1.set_yticks([-22,-18,-14,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filters):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

ax1.set_zorder(1)
plt.xlabel(r'$\textrm{Time [days]}$',fontsize=52)
fig.text(0.001, 0.5, r'$\textrm{Photometric Band [mag]}$', va='center', rotation='vertical', fontsize=52)
#plt.tight_layout()
plt.tight_layout(rect=[0.03, 0.00, 0.96, 0.95])
plt.savefig(plotName)
plt.close()
