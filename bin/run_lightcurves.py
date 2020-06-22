
import os, sys
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

import pymultinest
import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="output")
    parser.add_option("-p","--plotDir",default="plots")
    parser.add_option("-d","--dataDir",default="lightcurves")
    parser.add_option("-n","--name",default="PS1-13cyr")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*50.0 - 25.0
        cube[1] = cube[1]*10.0 - 5.0
        cube[2] = cube[2]*10.0 - 5.0
        cube[3] = cube[3]*20.0 - 10.0

def foft_model(t,c,b,tc,t0):
    flux = 10**c * ((t/t0)**b)/(1 + np.exp((t-t0)/tc))
    flux = -2.5*np.log10(flux)
    return flux

def myloglike(cube, ndim, nparams):
        c = cube[0]
        b = cube[1]
        tc = cube[2]
        t0 = cube[3]

        t = samples[:,0]
        y = samples[:,1]
        sigma_y = samples[:,2]

        foft = foft_model(t,c,b,tc,t0)

        chisquare = np.sum(((y-foft)/sigma_y)**2)
        prob = -chisquare/2.0
        #prob = chisquare

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print c,b,tc,t0, prob
        return prob

def loadLightcurves(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    lines = filter(None,lines)
    
    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        numid = float(lineSplit[0])
        psid = lineSplit[1]
        filt = lineSplit[2]
        mjd = float(lineSplit[3])
        mag = float(lineSplit[4])
        dmag = float(lineSplit[5])
    
        if not psid in data:
            data[psid] = {}
        if not filt in data[psid]:
            data[psid][filt] = np.empty((0,3), float)
        data[psid][filt] = np.append(data[psid][filt],np.array([[mjd,mag,dmag]]),axis=0)

    return data

def loadModels(name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_AB","BHNS"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)"]

    filenames = []
    legend_names = []
    for ii,model in enumerate(models):
        filename = '%s/%s/%s.dat'%(outputDir,model,name)
        if not os.path.isfile(filename):
            continue
        filenames.append(filename)
        legend_names.append(models_ref[ii])
        break
    mags, names = lightcurve_utils.read_files(filenames)

    return mags

# Parse command line
opts = parse_commandline()

baseoutputDir = opts.outputDir
if not os.path.isdir(baseoutputDir):
    os.mkdir(baseoutputDir)
outputDir = os.path.join(baseoutputDir,'lightcurves')
outputDir = baseoutputDir
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'lightcurves')
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
dataDir = opts.dataDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%dataDir
else:
    filename = "%s/lightcurves.tmp"%dataDir

if opts.doModels:
    data_out = loadModels(opts.name)
    print data_out
    print stop

else:
    data_out = loadLightcurves(filename)
    if not opts.name in data_out:
        print "%s not in file..."%opts.name
        exit(0)

    data_out = data_out[opts.name]

    for ii,key in enumerate(list(data_out.iterkeys())):
        if ii == 0:
            samples = data_out[key].copy()
        else:
            samples = np.vstack((samples,data_out[key].copy()))
    idx = np.argmin(samples[:,0])
    t0_save = samples[idx,0] -  1.0
    samples[:,0] = samples[:,0] - t0_save
    idx = np.argsort(samples[:,0])
    samples = samples[idx,:]

parameters = ["c","b","tc","t0"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.001, multimodal = False)

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

multifile= os.path.join(plotDir,'2-.txt')
data = np.loadtxt(multifile)

loglikelihood = -(1/2.0)*data[:,1]
idx = np.argmax(loglikelihood)

c = data[:,2]
b = data[:,3]
tc = data[:,4]
t0 = data[:,5]

c_best = data[idx,2]
b_best = data[idx,3]
tc_best = data[idx,4]
t0_best = data[idx,5]

foft = foft_model(samples[:,0],c_best,b_best,tc_best,t0_best)
#foft = foft_model(samples[:,0],-10,1.0,5.0,1.0)

plotName = "%s/lightcurve.pdf"%(plotDir)
plt.figure()
plt.plot(samples[:,0]+t0_save,foft,'gray')
if "g" in data_out:
    plt.errorbar(data_out["g"][:,0],data_out["g"][:,1],data_out["g"][:,2],fmt='yo',label='g-band')
if "r" in data_out:
    plt.errorbar(data_out["r"][:,0],data_out["r"][:,1],data_out["r"][:,2],fmt='go',label='r-band')
if "i" in data_out:
    plt.errorbar(data_out["i"][:,0],data_out["i"][:,1],data_out["i"][:,2],fmt='bo',label='i-band')
if "z" in data_out:
    plt.errorbar(data_out["z"][:,0],data_out["z"][:,1],data_out["z"][:,2],fmt='co',label='z-band')
if "y" in data_out:
    plt.errorbar(data_out["y"][:,0],data_out["y"][:,1],data_out["y"][:,2],fmt='ko',label='y-band')
if "w" in data_out:
    plt.errorbar(data_out["w"][:,0],data_out["w"][:,1],data_out["w"][:,2],fmt='mo',label='w-band')
plt.xlabel('Time [days]')
plt.ylabel('AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

filename = os.path.join(plotDir,'samples.dat')
fid = open(filename,'w+')
for i, j, k, l in zip(c,b,tc,t0):
    fid.write('%.5f %.5f %.5f %.5f\n'%(i,j,k,l))
fid.close()

filename = os.path.join(plotDir,'best.dat')
fid = open(filename,'w')
fid.write('%.5f %.5f %.5f %.5f\n'%(c_best,b_best,tc_best,t0_best))
fid.close()


