
import os, sys, glob
from time import time
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

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

from astropy.modeling.models import BlackBody1D
from astropy.modeling.blackbody import FLAM
from astropy import units as u
from astropy.visualization import quantity_support

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")

    parser.add_option("-n","--name",default="G298048_XSH_20170821")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--doEventPhot",  action="store_true", default=False)
    parser.add_option("--doRun",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=1.0,type=float)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-m","--model",default="BlackBody")
    parser.add_option("-e","--errorbudget",default=1.0,type=float)

    opts, args = parser.parse_args()

    return opts

def get_radius(T,F):

    T = np.array(T)
    F = np.array(F)

    SBc = 5.67*10**(-5)
    R  = np.sqrt( (F/(4*np.pi*SBc*T**4)))
    return R

def spec_model(T,F):

    bb = BlackBody1D(temperature=T*u.K,bolometric_flux=F*u.erg/(u.cm**2 * u.s))
    wav = np.arange(1000, 110000) * u.AA
    flux = bb(wav).to(FLAM, u.spectral_density(wav))

    return wav, flux

def myloglike_BlackBody(cube, ndim, nparams):

    T = cube[0]
    F = 10**(cube[1])

    wav1, flux1 = spec_model(T,F)

    return calc_prob(wav1, flux1)

def myloglike_BlackBodyx2(cube, ndim, nparams):

    T1 = cube[0]
    F1 = 10**(cube[1])
    T2 = cube[2]
    F2 = 10**(cube[3])

    if F2 > F1:
        wav1, flux1 = spec_model(T1,F1)
        wav2, flux2 = spec_model(T2,F2)
        prob = calc_prob(wav1, flux1+flux2)
    else:
        prob = -np.inf

    return prob

def calc_prob(wav1, flux1):

    wav2, flux2, error = data_out["lambda"], data_out["data"], data_out["error"]
    sigma = np.abs(error/(flux2*np.log(10)))

    flux1 = np.log10(np.abs(flux1.value))
    flux2 = np.log10(np.abs(flux2))

    f = interp.interp1d(wav1,flux1)
    flux1new = f(wav2)

    chisquarevals = ((flux1new-flux2)/sigma)**2

    chisquaresum = np.sum(chisquarevals)
    chisquaresum = (1/float(len(chisquarevals)-1))*chisquaresum
    chisquare = chisquaresum

    if np.isnan(chisquare):
        prob = -np.inf
    else:
        prob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)

    if np.isnan(prob):
        prob = -np.inf

    if prob == 0.0:
        prob = -np.inf

    #if np.isfinite(prob):
    #    print T, F, prob

    return prob

def myprior_BlackBody(cube, ndim, nparams):
    cube[0] = cube[0]*20000.0
    cube[1] = cube[1]*10.0 - 15.0

def myprior_BlackBodyx2(cube, ndim, nparams):
    cube[0] = cube[0]*20000.0
    cube[1] = cube[1]*10.0 - 15.0
    cube[2] = cube[2]*20000.0
    cube[3] = cube[3]*10.0 - 15.0

def plot_results(plotDir):
    multifile = lightcurve_utils.get_post_file(plotDir)
    data = np.loadtxt(multifile)

    a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='%s/2-'%plotDir)
    s = a.get_stats()

    if opts.model == "BlackBody":
        T = data[:,0]
        F = data[:,1]
        loglikelihood = data[:,2]
        idx = np.argmax(loglikelihood)

        T_best = data[idx,0]
        F_best = data[idx,1]
        truths = [np.nan,np.nan]

        wav, flux = spec_model(T_best,10**F_best)
        spec_best = {}
        spec_best["lambda"] = wav
        spec_best["data"] = flux
    elif opts.model == "BlackBodyx2":
        T1 = data[:,0]
        F1 = data[:,1]
        T2 = data[:,2]
        F2 = data[:,3]
        loglikelihood = data[:,4]
        idx = np.argmax(loglikelihood)

        T1_best = data[idx,0]
        F1_best = data[idx,1]
        T2_best = data[idx,2]
        F2_best = data[idx,3]
        truths = [np.nan,np.nan,np.nan,np.nan]

        wav1, flux1 = spec_model(T1_best,10**F1_best)
        wav2, flux2 = spec_model(T2_best,10**F2_best)
        spec_best = {}
        spec_best["lambda"] = wav1
        spec_best["data"] = flux1+flux2

    if n_params >= 8:
        title_fontsize = 26
        label_fontsize = 30
    else:
        title_fontsize = 24
        label_fontsize = 28

    plotName = "%s/corner.pdf"%(plotDir)
    figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                   truths=truths)
    if n_params >= 8:
        figure.set_size_inches(18.0,18.0)
    else:
        figure.set_size_inches(14.0,14.0)
    plt.savefig(plotName)
    plt.close()

    plotName = "%s/spec.pdf"%(plotDir)
    plt.figure(figsize=(14,12))
    plt.loglog(spec_best["lambda"],spec_best["data"],'k--',linewidth=2)
    plt.errorbar(data_out["lambda"],np.abs(data_out["data"]),yerr=data_out["error"],fmt='ro',linewidth=2)
    plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
    plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
    #plt.legend(loc="best",prop={'size':16},numpoints=1)
    ymin = np.min(np.abs(data_out["data"])-2*data_out["error"])
    ymax = np.max(np.abs(data_out["data"])+2*data_out["error"])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.savefig(plotName)
    plt.close()
    
    filename = os.path.join(plotDir,'evidence.dat')
    fid = open(filename,'w+')
    fid.write("%.15e %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))
    fid.close()

    if opts.model == "BlackBody":
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j in zip(T,F):
            fid.write('%.5f %.5f\n'%(i,j))
        fid.close()
    
        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f\n'%(T_best,F_best))
        fid.close()
    elif opts.model == "BlackBodyx2":
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l in zip(T1,F1,T2,F2):
            fid.write('%.5f %.5f %.5f %.5f\n'%(i,j,k,l))
        fid.close()
    
        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f\n'%(T1_best,F1_best,T2_best,F2_best))
        fid.close()

# Parse command line
opts = parse_commandline()

if not opts.model in ["BlackBody","BlackBodyx2"]:
   print "Model must be either: BlackBody,BlackBodyx2"
   exit(0)

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models_spec'
else:
    basename = 'gws_spec'
plotDir = os.path.join(baseplotDir,basename)
plotDir = os.path.join(plotDir,opts.model)
plotDir = os.path.join(plotDir,opts.name)
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

dataDir = opts.dataDir
lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir

errorbudget = opts.errorbudget
n_live_points = 1000
evidence_tolerance = 0.5

if opts.doModels:
    data_out = lightcurve_utils.loadModelsSpec(opts.outputDir,opts.name)
    keys = data_out.keys()
    if not opts.name in data_out:
        print "%s not in file..."%opts.name
        exit(0)

    data_out = data_out[opts.name]
    f = interp.interp2d(data_out["t"],data_out["lambda"],data_out["data"].T)
    xnew = opts.T0
    ynew = data_out["lambda"]
    znew = f(xnew,ynew)
    data_out = {}
    data_out["lambda"] = ynew
    data_out["data"] = np.squeeze(znew)

elif opts.doEvent:
    #events = opts.name.split(",")
    #data_out = {}
    #for event in events:
    #    filename = "%s/%s.dat"%(spectraDir,event)
    #    data_out_event = lightcurve_utils.loadEventSpec(filename)
    #    data_out[event] = data_out_event

    filename = "%s/%s.dat"%(spectraDir,opts.name)
    data_out = lightcurve_utils.loadEventSpec(filename)

elif opts.doEventPhot:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.name)
    data = lightcurve_utils.loadEventPhot(filename)

if opts.doEventPhot:
    if opts.doRun:
        for key in data.iterkeys():
            data_out = {}
            data_out["lambda"] = data[key]["wavelengths"]
            data_out["data"] = data[key]["flam"]
            data_out["error"] = data[key]["flamerr"]
    
            plotDirPhase=os.path.join(plotDir,'%.5f'%key)
            if not os.path.isdir(plotDirPhase):
                os.makedirs(plotDirPhase)
            print plotDirPhase
    
            if opts.model == "BlackBody": 
                parameters = ["T","F"]
                labels = [r"$T$",r"$F$"]
                n_params = len(parameters)
                pymultinest.run(myloglike_BlackBody, myprior_BlackBody, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDirPhase, evidence_tolerance = evidence_tolerance, multimodal = False)
            elif opts.model == "BlackBodyx2": 
                parameters = ["T1","F1","T2","F2"]
                labels = [r"$T_1$",r"$F_1$",r"$T_2$",r"$F_2$"]
                n_params = len(parameters)
                pymultinest.run(myloglike_BlackBodyx2, myprior_BlackBodyx2, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDirPhase, evidence_tolerance = evidence_tolerance, multimodal = False)
    
            plot_results(plotDirPhase)

    MPCtoCM = 3.086e24
    dist = opts.distance*MPCtoCM
    distconv = 4*np.pi*dist**2
 
    if opts.model == "BlackBody":
        phases, Ts, Tslow, Tshigh, Fs, Fslow, Fshigh, Rs, Rslow, Rshigh = [], [], [], [], [], [], [], [], [], []
        evidences, evidenceserr = [], []
        keys = sorted(data.iterkeys())
        for key in keys:
            plotDirPhase=os.path.join(plotDir,'%.5f'%key)
            samplesFile = os.path.join(plotDirPhase,"samples.dat")
            evidenceFile = os.path.join(plotDirPhase,"evidence.dat")
            if not os.path.isfile(samplesFile): continue
            if not os.path.isfile(evidenceFile): continue   

            data_out = np.loadtxt(samplesFile)
            phases.append(key)
            T_low, T_median, T_high = np.percentile(data_out[:,0], [16, 50, 84])
            F_low, F_median, F_high = np.percentile(10**data_out[:,1]*distconv, [16, 50, 84])
            R_low, R_median, R_high = np.percentile(get_radius(data_out[:,0],10**data_out[:,1]*distconv), [16, 50, 84])

            Ts.append(T_median)
            Tslow.append(T_median-T_low)
            Tshigh.append(T_high-T_median)
            Fs.append(F_median)
            Fslow.append(F_median-F_low)
            Fshigh.append(F_high-F_median)       
            Rs.append(R_median)
            Rslow.append(R_median-R_low)
            Rshigh.append(R_high-R_median) 

            data_out = np.loadtxt(samplesFile)
            evidences.append(data_out[0,0])
            evidenceserr.append(data_out[1,0])

        plotName = "%s/combined.pdf"%(plotDir)
        f, axarr = plt.subplots(3, sharex=True,figsize=(14,12))
        axarr[0].errorbar(phases, Ts, yerr=[Tslow,Tshigh], fmt='ko')
        axarr[0].set_ylabel('Temperature [K]',fontsize=18)
        axarr[0].grid()
        axarr[1].errorbar(phases, Fs, yerr=[Fslow,Fshigh], fmt='ko')
        axarr[1].set_ylabel('Fluence [erg/s]',fontsize=18)
        axarr[1].grid()
        axarr[1].set_yscale("log")
        axarr[2].errorbar(phases, Rs, yerr=[Rslow,Rshigh], fmt='ko')
        axarr[2].set_ylabel('Radius [cm]',fontsize=18)
        axarr[2].grid()
        axarr[2].set_yscale("log")
        plt.xlabel(r'Phase [days]',fontsize=24)
        plt.savefig(plotName)
        plt.close()        

        plotName = "%s/evidences.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        plt.errorbar(phases, evidences, yerr = evidenceserr, fmt='ro')
        plt.xlabel(r'Phase [days]',fontsize=24)
        plt.ylabel('log(Evidence)')
        plt.savefig(plotName)
        plt.close()

        filename = "%s/combined.dat"%(plotDir)
        fid = open(filename,'w')
        fid.write('# Phase (days) T (K) eT (K) R (cm) eR (cm) L (erg/s) eL (erg/s)\n')
        for phase, T, dT, F, dF, R, dR in zip(phases,Ts,Tshigh,Fs,Fshigh,Rs,Rshigh):
            fid.write('%.3f %.3f %.3f %.3e %.3e %.3e %.3e\n'%(phase,T,dT,R,dR,F,dF))
        fid.close()

    elif opts.model == "BlackBodyx2":
        phases, T1s, T1slow, T1shigh, F1s, F1slow, F1shigh, R1s, R1slow, R1shigh, T2s, T2slow, T2shigh, F2s, F2slow, F2shigh, R2s, R2slow, R2shigh = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        evidences, evidenceserr = [], []
        keys = sorted(data.iterkeys())
        for key in keys:
            plotDirPhase=os.path.join(plotDir,'%.5f'%key)
            samplesFile = os.path.join(plotDirPhase,"samples.dat")
            evidenceFile = os.path.join(plotDirPhase,"evidence.dat")
            if not os.path.isfile(samplesFile): continue
            if not os.path.isfile(evidenceFile): continue

            data_out = np.loadtxt(samplesFile)
            phases.append(key)
            T1_low, T1_median, T1_high = np.percentile(data_out[:,0], [16, 50, 84])
            F1_low, F1_median, F1_high = np.percentile(10**data_out[:,1]*distconv, [16, 50, 84])
            R1_low, R1_median, R1_high = np.percentile(get_radius(data_out[:,0],10**data_out[:,1]*distconv), [16, 50, 84])
            T2_low, T2_median, T2_high = np.percentile(data_out[:,2], [16, 50, 84])
            F2_low, F2_median, F2_high = np.percentile(10**data_out[:,3]*distconv, [16, 50, 84])
            R2_low, R2_median, R2_high = np.percentile(get_radius(data_out[:,2],10**data_out[:,3]*distconv), [16, 50, 84])

            T1s.append(T1_median)
            T1slow.append(T1_median-T1_low)
            T1shigh.append(T1_high-T1_median)
            F1s.append(F1_median)
            F1slow.append(F1_median-F1_low)
            F1shigh.append(F1_high-F1_median)
            R1s.append(R1_median)
            R1slow.append(R1_median-R1_low)
            R1shigh.append(R1_high-R1_median)
            T2s.append(T2_median)
            T2slow.append(T2_median-T2_low)
            T2shigh.append(T2_high-T2_median)
            F2s.append(F2_median)
            F2slow.append(F2_median-F2_low)
            F2shigh.append(F2_high-F2_median)
            R2s.append(R2_median)
            R2slow.append(R2_median-R2_low)
            R2shigh.append(R2_high-R2_median)

            data_out = np.loadtxt(samplesFile)
            evidences.append(data_out[0,0])
            evidenceserr.append(data_out[1,0])

        plotName = "%s/combined.pdf"%(plotDir)
        f, axarr = plt.subplots(3, sharex=True,figsize=(14,12))
        axarr[0].errorbar(phases, T1s, yerr=[T1slow,T1shigh], fmt='ko')
        axarr[0].errorbar(phases, T2s, yerr=[T2slow,T2shigh], fmt='bx')
        axarr[0].set_ylabel('Temperature [K]',fontsize=18)
        axarr[0].grid()
        axarr[1].errorbar(phases, F1s, yerr=[F1slow,F1shigh], fmt='ko')
        axarr[1].errorbar(phases, F2s, yerr=[F2slow,F2shigh], fmt='bx')
        axarr[1].set_ylabel('Fluence [erg/s]',fontsize=18)
        axarr[1].grid()
        axarr[1].set_yscale("log")
        axarr[2].errorbar(phases, R1s, yerr=[R1slow,R1shigh], fmt='ko')
        axarr[2].errorbar(phases, R2s, yerr=[R2slow,R2shigh], fmt='bx')
        axarr[2].set_ylabel('Radius [cm]',fontsize=18)
        axarr[2].grid()
        axarr[2].set_yscale("log")
        plt.xlabel(r'Phase [days]',fontsize=24)
        plt.savefig(plotName)
        plt.close()
 
        plotName = "%s/evidences.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        plt.errorbar(phases, evidences, yerr = evidenceserr, fmt='ro')
        plt.xlabel(r'Phase [days]',fontsize=24)
        plt.ylabel('log(Evidence)')
        plt.savefig(plotName)
        plt.close()

        filename = "%s/combined.dat"%(plotDir)
        fid = open(filename,'w')
        fid.write('# Phase (days) T1 (K) eT1 (K) R1 (cm) eR1 (cm) L1 (erg/s) eL1 (erg/s) T2 (K) eT2 (K) R2 (cm) eR2 (cm) L2 (erg/s) eL2 (erg/s)\n')
        for phase, T1, dT1, F1, dF1, R1, dR1, T2, dT2, F2, dF2, R2, dR2 in zip(phases,T1s,T1shigh,F1s,F1shigh,R1s,R1shigh,T2s,T2shigh,F2s,F2shigh,R2s,R2shigh):
            fid.write('%.3f %.3f %.3f %.3e %.3e %.3e %.3e %.3f %.3f %.3e %.3e %.3e %.3e\n'%(phase,T1,dT1,R1,dR1,F1,dF1,T2,dT2,R2,dR2,F2,dF2))
        fid.close()

else:
    if opts.model == "BlackBody":
        parameters = ["T","F"]
        labels = [r"$T$",r"$F$"]
        n_params = len(parameters)
        pymultinest.run(myloglike_BlackBody, myprior_BlackBody, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model == "BlackBodyx2":
        parameters = ["T1","F1","T2","F2"]
        labels = [r"$T_1$",r"$F_1$",r"$T_2$",r"$F_2$"]
        n_params = len(parameters)
        pymultinest.run(myloglike_BlackBodyx2, myprior_BlackBodyx2, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

    plot_results(plotDir)
