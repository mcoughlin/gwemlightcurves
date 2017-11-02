
import os, sys, glob
from time import time
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats, scipy.signal

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

import corner

from astropy.time import Time

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
    parser.add_option("-s","--spectraDir",default="../spectra")

    parser.add_option("-n","--name",default="rpft_m005_v2")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    #parser.add_option("--T0",default="1,2,3,4,5,6,7")
    parser.add_option("--T0",default=57982.5285236896,type=float)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-m","--model",default="barnes_kilonova_spectra")

    opts, args = parser.parse_args()

    return opts

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def spec_model(specs,wavelengths,t0,model):

        spec = specs[speckeys[model]]

        f = spec["f"]
        xnew = t0
        ynew = wavelengths
        znew = f(xnew,ynew)

        spec1 = np.squeeze(znew)
        #spec1 = scipy.signal.medfilt(spec1,kernel_size=21)
        spec1 = spec1/np.sum(spec1)

        return spec1

def myloglike(cube, ndim, nparams):

        t0 = cube[0]
        model = int(np.round(cube[1]))

        nspecs = len(speckeys)

        if model > nspecs-1:
            prob = -np.inf
            return prob

        prob = 0
        for key in data_out.iterkeys():
            dt = float(key)
            spec1 = spec_model(specs,data_out[key]["lambda"],t0+dt,model)
            spec2 = data_out[key]["data"]/np.sum(data_out[key]["data"])

            crosscorr = np.abs(np.corrcoef(spec1,spec2)[0,1])
            #chisquare = 1 - crosscorr
            chisquare = crosscorr

            if np.isnan(chisquare):
                prob = -np.inf
            else:
                #prob = prob + scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)
                #prob = prob + 1/chisquare
                prob = prob + chisquare

        if np.isnan(prob):
            prob = -np.inf

        if prob == 0.0:
            prob = -np.inf

        #if np.isfinite(prob):
        #    print t0, model, prob

        return prob

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*50.0
        cube[1] = cube[1]*50.0

# Parse command line
opts = parse_commandline()

if not opts.model in ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog","kasen_kilonova_survey"]:
   print "Model must be either: barnes_kilonova_spectra,ns_merger_spectra,kilonova_wind_spectra, kasen_kilonova_survey, or macronovae-rosswog"
   exit(0)

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models_spec'
else:
    basename = 'gws_spec'
plotDir = os.path.join(baseplotDir,basename)
plotDir = os.path.join(plotDir,opts.model)
plotDir = os.path.join(plotDir,opts.name)
if opts.doModels:
    plotDir = os.path.join(plotDir,"_".join(opts.T0.split(",")))
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

dataDir = opts.dataDir
lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir

fileDir = os.path.join(opts.outputDir,opts.model)
filenames = glob.glob('%s/*_spec.dat'%fileDir)
specs, names = lightcurve_utils.read_files_spec(filenames)
speckeys = specs.keys()
for key in speckeys:
    f = interp.interp2d(specs[key]["t"],specs[key]["lambda"],specs[key]["data"].T)
    specs[key]["f"] = f

n_live_points = 1000
evidence_tolerance = 0.5

if opts.doModels:
    data_out_all = lightcurve_utils.loadModelsSpec(opts.outputDir,opts.name)
    keys = data_out_all.keys()
    if not opts.name in data_out_all:
        print "%s not in file..."%opts.name
        exit(0)

    data_out_full = data_out_all[opts.name]
    f = interp.interp2d(data_out_full["t"],data_out_full["lambda"],data_out_full["data"].T)

    T0s = opts.T0.split(",")
    data_out = {}
    for T0 in T0s:

        xnew = float(T0)
        ynew = data_out_full["lambda"]
        znew = f(xnew,ynew)
        data_out[T0] = {}
        data_out[T0]["lambda"] = ynew
        data_out[T0]["data"] = np.squeeze(znew)

elif opts.doEvent:
    filename = "../spectra/spectra_index.dat"
    lines = [line.rstrip('\n') for line in open(filename)]
    filenames = []
    T0s = []
    for line in lines:
        lineSplit = line.split(" ")
        if not lineSplit[0] == opts.name: continue
        filename = "%s/%s"%(spectraDir,lineSplit[1])
        filenames.append(filename)
        mjd = Time(lineSplit[2], format='isot').mjd
        T0s.append(mjd-opts.T0)

    names = opts.name.split(",")
    data_out = {}
    for filename,T0 in zip(filenames,T0s):
        data_out_temp = lightcurve_utils.loadEventSpec(filename)
        data_out[str(T0)] = data_out_temp
        #data_out[str(T0)]["data"] = scipy.signal.medfilt(data_out[str(T0)]["data"],kernel_size=21)

else:
    print "Must enable --doModels or --doEvent"
    exit(0)

parameters = ["t0","model"]
labels = [r"$T_0$",r"Model"]
n_params = len(parameters)
pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

#multifile= os.path.join(plotDir,'2-.txt')
multifile = lightcurve_utils.get_post_file(plotDir)
data = np.loadtxt(multifile)

t0 = data[:,0]
model = data[:,1]
loglikelihood = data[:,2]
idx = np.argmax(loglikelihood)

t0_best = data[idx,0]
model_best = int(np.round(data[idx,1]))
truths = [np.nan,np.nan]

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

spec_best = {}
for key in data_out.keys():
    dt = float(key)
    znew = spec_model(specs,data_out[key]["lambda"],t0_best+dt,model_best)

    spec_best[key] = {}
    spec_best[key]["lambda"] = data_out[key]["lambda"]
    spec_best[key]["data"] = znew

    print "Cross correlation: %s %.5f"%(key,np.corrcoef(data_out[key]["data"],spec_best[key]["data"])[0,1])

plotName = "%s/spec.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for key in data_out.keys():
    plt.loglog(data_out[key]["lambda"],data_out[key]["data"],'r-',linewidth=2)
    plt.loglog(spec_best[key]["lambda"],spec_best[key]["data"]*np.max(data_out[key]["data"])/np.max(spec_best[key]["data"]),'k--',linewidth=2)

plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
#plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.savefig(plotName)
plt.close()

filename = os.path.join(plotDir,'samples.dat')
fid = open(filename,'w+')
for i, j in zip(t0,model):
    fid.write('%.5f %.5f\n'%(i,j))
fid.close()

filename = os.path.join(plotDir,'best.dat')
fid = open(filename,'w')
fid.write('%.5f %.5f\n'%(t0_best,model_best))
fid.close()

