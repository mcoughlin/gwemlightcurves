
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
from gwemlightcurves import BHNSKilonovaLightcurve, BNSKilonovaLightcurve, SALT2
from gwemlightcurves import lightcurve_utils

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
    parser.add_option("--T0",default=1.0,type=float)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-m","--model",default="barnes_kilonova_spectra")
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

def spec_model(specs,t0,model):

        #model = 12
        #t0 = 1.0
        #print speckeys[model]
 
        spec = specs[speckeys[model]]

        #f = interp.interp2d(spec["t"],spec["lambda"],spec["data"].T)
        f = spec["f"]
        xnew = t0
        ynew = data_out["lambda"]
        znew = f(xnew,ynew)

        #znew[znew == 0.0] = np.max(znew)/1e10
        spec1 = znew/np.sum(znew)
        spec1 = np.squeeze(spec1)

        return spec1

def myloglike(cube, ndim, nparams):

        t0 = cube[0]
        model = int(np.round(cube[1]))

        nspecs = len(speckeys)

        if model > nspecs-1:
            prob = -np.inf
            return prob

        spec1 = spec_model(specs,t0,model)
        spec2 = data_out["data"]/np.sum(data_out["data"])

        sigma = np.sqrt(np.max(spec1)**2 + np.max(spec2)**2)
        #sigma = np.sqrt(errorbudget**2)
        chisquarevals = np.zeros(spec1.shape)
        chisquarevals = ((spec1-spec2)/sigma)**2 * np.abs(spec2)

        chisquaresum = np.sum(chisquarevals)
        chisquaresum = (1/float(len(chisquarevals)-1))*chisquaresum
        chisquare = chisquaresum

        #print chisquaresum
        #exit(0)

        if np.isnan(chisquare):
            prob = -np.inf
        else:
            prob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)

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

if not opts.model in ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog"]:
   print "Model must be either: barnes_kilonova_spectra,ns_merger_spectra,kilonova_wind_spectra, or macronovae-rosswog"
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

if opts.doEvent:
    filename = "%s/%s.dat"%(spectraDir,opts.name)

fileDir = os.path.join(opts.outputDir,opts.model)
filenames = glob.glob('%s/*_spec.dat'%fileDir)
specs, names = lightcurve_utils.read_files_spec(filenames)
speckeys = specs.keys()
for key in speckeys:
    f = interp.interp2d(specs[key]["t"],specs[key]["lambda"],specs[key]["data"].T)
    specs[key]["f"] = f

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
    data_out = lightcurve_utils.loadEventSpec(filename)

parameters = ["t0","model"]
labels = [r"$T_0$",r"Model"]
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

t0 = data[:,0]
model = data[:,1]
loglikelihood = data[:,2]
idx = np.argmax(loglikelihood)

t0_best = data[idx,0]
model_best = int(np.round(data[idx,1]))
truths = [np.nan,np.nan]

znew = spec_model(specs,t0_best,model_best)
spec_best = {}
spec_best["lambda"] = data_out["lambda"]
spec_best["data"] = znew

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
plt.loglog(data_out["lambda"],data_out["data"],'r-',linewidth=2)
plt.loglog(spec_best["lambda"],spec_best["data"]*np.max(data_out["data"])/np.max(spec_best["data"]),'k--',linewidth=2)
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
