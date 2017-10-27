
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
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=1.0,type=float)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-m","--model",default="BlackBody1D")
    parser.add_option("-e","--errorbudget",default=1.0,type=float)

    opts, args = parser.parse_args()

    return opts

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def spec_model(T,F):

        bb = BlackBody1D(temperature=T*u.K,bolometric_flux=F*u.erg/(u.cm**2 * u.s))
        wav = np.arange(1000, 110000) * u.AA
        flux = bb(wav).to(FLAM, u.spectral_density(wav))

        return wav, flux

def myloglike(cube, ndim, nparams):

        T = cube[0]
        F = 10**(cube[1])

        wav1, flux1 = spec_model(T,F)
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

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*10000.0
        cube[1] = cube[1]*10.0 - 20.0

# Parse command line
opts = parse_commandline()

if not opts.model in ["BlackBody1D"]:
   print "Model must be either: BlackBody1D"
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

parameters = ["T","F"]
labels = [r"$T$",r"F"]
n_params = len(parameters)
pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='%s/2-'%plotDir)
s = a.get_stats()

import json
# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
            json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
            json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

#multifile= os.path.join(plotDir,'2-.txt')
multifile = get_post_file(plotDir)
data = np.loadtxt(multifile)

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
plt.figure(figsize=(10,8))
plt.loglog(data_out["lambda"],np.abs(data_out["data"]),'r-',linewidth=2)
plt.loglog(spec_best["lambda"],spec_best["data"],'k--',linewidth=2)
plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
#plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.ylim([np.min(np.abs(data_out["data"])),np.max(np.abs(data_out["data"]))])
plt.grid()
plt.savefig(plotName)
plt.close()

filename = os.path.join(plotDir,'samples.dat')
fid = open(filename,'w+')
for i, j in zip(T,F):
    fid.write('%.5f %.5f\n'%(i,j))
fid.close()

filename = os.path.join(plotDir,'best.dat')
fid = open(filename,'w')
fid.write('%.5f %.5f\n'%(T_best,F_best))
fid.close()
