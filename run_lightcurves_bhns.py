
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

import BHNSKilonovaLightcurve

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

    opts, args = parser.parse_args()

    return opts

def bhns_model(q,chi,c,mb):

    i = 60.0

    #if opts.eos == "APR4":
    #    c = 0.180
    #    mb = 1.50
    #elif opts.eos == "ALF2":
    #    c = 0.161
    #    mb = 1.49
    #elif opts.eos == "H4":
    #    c = 0.147
    #    mb = 1.47
    #elif opts.eos == "MS1":
    #    c = 0.138
    #    mb = 1.46
    
    mns = 1.35
    
    tini = 0.1
    tmax = 21.0
    dt = 0.1
    
    vave = 0.267
    vmin = 0.02
    th = 0.2
    ph = 3.14
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5
    
    t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi,i,c,mb,mns)

    return t, lbol, mag

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*50.0 - 25.0
        cube[1] = cube[1]*9.0 + 1.0
        cube[2] = cube[2]*0.9
        cube[3] = cube[3]*0.1 + 0.1
        cube[4] = cube[4]*0.04 + 1.46
        cube[5] = cube[5]*50.0

def foft_model(t,c,b,tc,t0):
    flux = 10**c * ((t/t0)**b)/(1 + np.exp((t-t0)/tc))
    flux = -2.5*np.log10(flux)
    return flux

def myloglike(cube, ndim, nparams):
        t0 = cube[0]
        q = cube[1]
        chi = cube[2]
        c = cube[3]
        mb = cube[4]
        zp = cube[5]

        tmag, lbol, mag = bhns_model(q,chi,c,mb)
        if np.sum(lbol) == 0.0:
            prob = -np.inf
            return prob
        tmag = tmag + t0

        count = 0
        chisquare = np.nan
        for key in data_out:
            samples = data_out[key]
            t = samples[:,0]
            y = samples[:,1]
            sigma_y = samples[:,2]

            if key == "g":
                maginterp = np.interp(t,tmag,mag[1])
            elif key == "r":
                maginterp = np.interp(t,tmag,mag[2])
            elif key == "i":
                maginterp = np.interp(t,tmag,mag[3])
            elif key == "z":
                maginterp = np.interp(t,tmag,mag[4])
            elif key == "w":
                maginterp = np.interp(t,tmag,(mag[1]+mag[2]+mag[3])/3.0)
            else:
                continue
            maginterp = maginterp + zp
            if np.isnan(np.sum(maginterp)):
                chisquare = np.nan
                break

            if count == 0:
                chisquare = np.sum(((y-maginterp)/sigma_y)**2)
            else:
                chisquare = chisquare + np.sum(((y-maginterp)/sigma_y)**2)
            count = count + 1
        if np.isnan(chisquare): 
            prob = -np.inf
        else:
            prob = -chisquare/2.0
            #prob = chisquare

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print t0, q,chi,c,mb,zp, prob
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

# Parse command line
opts = parse_commandline()

baseoutputDir = opts.outputDir
if not os.path.isdir(baseoutputDir):
    os.mkdir(baseoutputDir)
outputDir = os.path.join(baseoutputDir,'lightcurves_BHNS')
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'lightcurves_BHNS')
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
data_out = loadLightcurves(filename)
if not opts.name in data_out:
    print "%s not in file..."%opts.name
    exit(0)

data_out = data_out[opts.name]

#q = 3.0
#chi = 0.5
#c = 0.161
#mb = 1.49
#t, lbol, mag = bhns_model(q,chi,c,mb)

for ii,key in enumerate(data_out.iterkeys()):
    if ii == 0:
        samples = data_out[key].copy()
    else:
        samples = np.vstack((samples,data_out[key].copy()))
idx = np.argmin(samples[:,0])
t0_save = samples[idx,0] -  1.0
samples[:,0] = samples[:,0] - t0_save
idx = np.argsort(samples[:,0])
samples = samples[idx,:]

for ii,key in enumerate(data_out.iterkeys()):
    data_out[key][:,0] = data_out[key][:,0] - t0_save

parameters = ["t0","q","chi","c","mb","zp"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = 1000, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.5, multimodal = False)

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

parameters = ["t0","q","chi","c","mb","zp"]

t0 = data[:,2]
q = data[:,3]
chi = data[:,4]
c = data[:,5]
mb = data[:,6]
zp = data[:,7]

t0_best = data[idx,2]
q_best = data[idx,3]
chi_best = data[idx,4]
c_best = data[idx,5]
mb_best = data[idx,6]
zp_best = data[idx,7]

tmag, lbol, mag = bhns_model(q_best,chi_best,c_best,mb_best)

plotName = "%s/lightcurve.pdf"%(plotDir)
plt.figure()
if "g" in data_out:
    plt.errorbar(data_out["g"][:,0],data_out["g"][:,1],data_out["g"][:,2],fmt='yo',label='g-band')
if "r" in data_out:
    plt.errorbar(data_out["r"][:,0],data_out["r"][:,1],data_out["r"][:,2],fmt='go',label='r-band')
if "i" in data_out:
    plt.errorbar(data_out["i"][:,0],data_out["i"][:,1],data_out["i"][:,2],fmt='bo',label='i-band')
if "z" in data_out:
    plt.errorbar(data_out["z"][:,0],data_out["z"][:,1],data_out["z"][:,2],fmt='co',label='z-band')
if "y" in data_out:
    plt.errorbar(data_out["y"][:,0],data_out["y"][:,1],data_out["y"][:,2],fmt='ko',label='k-band')
if "w" in data_out:
    plt.errorbar(data_out["w"][:,0],data_out["w"][:,1],data_out["w"][:,2],fmt='mo',label='w-band')

plt.plot(tmag+t0_best,mag[1]+zp_best,'y--',label='model g-band')
plt.plot(tmag+t0_best,mag[2]+zp_best,'g--',label='model r-band')
plt.plot(tmag+t0_best,mag[3]+zp_best,'b--',label='model i-band')
plt.plot(tmag+t0_best,mag[4]+zp_best,'c--',label='model z-band')
plt.plot(tmag+t0_best,(mag[1]+mag[2]+mag[3])/3.0+zp_best,'m--',label='model w-band')

plt.xlabel('Time [days]')
plt.ylabel('AB Magnitude')
plt.legend(loc="best",prop={'size':6})
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

filename = os.path.join(plotDir,'samples.dat')
fid = open(filename,'w+')
for i, j, k, l,m,n in zip(t0,q,chi,c,mb,zp):
    fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
fid.close()

filename = os.path.join(plotDir,'best.dat')
fid = open(filename,'w')
fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,q_best,chi_best,c_best,mb_best,zp_best))
fid.close()


