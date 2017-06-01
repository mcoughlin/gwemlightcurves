
import os, sys, glob
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

import pymultinest
import BHNSKilonovaLightcurve, BNSKilonovaLightcurve
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
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("-m","--model",default="BHNS")

    opts, args = parser.parse_args()

    return opts

def hist_results(samples):

    bins = np.linspace(np.min(samples),np.max(samples),11)
    hist1, bin_edges = np.histogram(samples, bins=bins)
    hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

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
    tmax = 50.0
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

def bns_model(m1,m2,c,mb):

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

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    vave = 0.267
    vmin = 0.02
    th = 0.2
    ph = 3.14
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5

    flgbct = 0

    t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb,c,m2,mb,c,flgbct)

    return t, lbol, mag

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def myprior_bhns(cube, ndim, nparams):
        cube[0] = cube[0]*40.0 - 20.0
        cube[1] = cube[1]*9.0 + 1.0
        cube[2] = cube[2]*0.9
        cube[3] = cube[3]*0.1 + 0.1
        cube[4] = cube[4]*0.04 + 1.46
        cube[5] = cube[5]*20.0 - 10.0

def myprior_bns(cube, ndim, nparams):
        cube[0] = cube[0]*40.0 - 20.0
        cube[1] = cube[1]*0.1 + 1.3
        cube[2] = cube[2]*0.1 + 1.3
        cube[3] = cube[3]*0.1 + 0.1
        cube[4] = cube[4]*0.04 + 1.46
        cube[5] = cube[5]*20.0 - 10.0

def foft_model(t,c,b,tc,t0):
    flux = 10**c * ((t/t0)**b)/(1 + np.exp((t-t0)/tc))
    flux = -2.5*np.log10(flux)
    return flux

def addconst(array):
    idx = np.where(np.isnan(array))[0]
    idx_diff = np.diff(idx)
    idx_loc = np.where(idx_diff > 1)[0]

    if len(idx_loc) == 0:
        return array
    else:
        idx_loc = idx_loc[0]
    array_copy = array.copy()
    idx_low = idx[idx_loc]
    idx_high = idx[idx_loc+1]
    array_copy[idx_low] = array_copy[idx_low+1]
    array_copy[idx_high] = array_copy[idx_high-1]
   
    return array_copy

def myloglike_bns(cube, ndim, nparams):
        t0 = cube[0]
        m1 = cube[1]
        m2 = cube[2]
        c = cube[3]
        mb = cube[4]
        zp = cube[5]

        #c = 0.147
        #mb = 1.47
        #m1 = 1.35
        #m2 = 1.35
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = bns_model(m1,m2,c,mb)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_bhns(cube, ndim, nparams):
        t0 = cube[0]
        q = cube[1]
        chi = cube[2]
        c = cube[3]
        mb = cube[4]
        zp = cube[5]

        #c = 0.161
        #mb = 1.49
        #q = 3.0
        #chi = 0.3
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = bhns_model(q,chi,c,mb)

        prob = calc_prob(tmag, lbol, mag, t0,zp)

        return prob

def calc_prob(tmag, lbol, mag, t0,zp): 

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

            idx = np.where(~np.isnan(y))[0]
            t = t[idx]
            y = y[idx]
            sigma_y = sigma_y[idx]

            if key == "g":
                maginterp = np.interp(t,tmag,addconst(mag[1]),left=np.nan, right=np.nan)
            elif key == "r":
                maginterp = np.interp(t,tmag,addconst(mag[2]),left=np.nan, right=np.nan)
            elif key == "i":
                maginterp = np.interp(t,tmag,addconst(mag[3]),left=np.nan, right=np.nan)
            elif key == "z":
                maginterp = np.interp(t,tmag,addconst(mag[4]),left=np.nan, right=np.nan)
            elif key == "w":
                maginterp = np.interp(t,tmag,addconst((mag[1]+mag[2]+mag[3])/3.0),left=np.nan, right=np.nan)
            else:
                continue

            maginterp = maginterp + zp
            chisquarevals = ((y-maginterp)/sigma_y)**2
            #idx = np.where(~np.isnan(chisquarevals))[0]
            #chisquarevals = chisquarevals[idx] 
            chisquaresum = np.sum(chisquarevals)

            if np.isnan(chisquaresum):
                chisquare = np.nan
                break

            if count == 0:
                chisquare = chisquaresum
            else:
                chisquare = chisquare + chisquaresum
            #count = count + len(chisquarevals)
            count = count + 1

        if np.isnan(chisquare): 
            prob = -np.inf
        else:
            prob = scipy.stats.chi2.logpdf(chisquare, count, loc=0, scale=1)
            #prob = -chisquare/2.0
            #prob = chisquare
            #prob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)

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

def loadModels(name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_AB","BHNS","BNS"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich et al. (2016)"]

    filenames = []
    legend_names = []
    for ii,model in enumerate(models):
        filename = '%s/%s/%s.dat'%('output',model,name)
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
outputDir = os.path.join(baseoutputDir,'models')
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)
outputDir = os.path.join(baseoutputDir,'models/%s'%opts.model)
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

if opts.doReduced:
    outputDir = os.path.join(outputDir,"%s_reduced"%opts.name)
else:
    outputDir = os.path.join(outputDir,opts.name)
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'models')
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
plotDir = os.path.join(baseplotDir,'models/%s'%opts.model)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

if opts.doReduced:
    plotDir = os.path.join(plotDir,"%s_reduced"%opts.name)
else:
    plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
dataDir = opts.dataDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%dataDir
else:
    filename = "%s/lightcurves.tmp"%dataDir

errorbudget = 1.0
maxt = 14.0

if opts.doModels:
    data_out = loadModels(opts.name)
    if not opts.name in data_out:
        print "%s not in file..."%opts.name
        exit(0)

    data_out = data_out[opts.name]

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key] = np.vstack((data_out["t"],data_out[key],errorbudget*np.ones(data_out["t"].shape))).T

    idxs = np.where(data_out["t"]<=maxt)[0]
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    if opts.doReduced:
        ts = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 5, 10])
        idxs = []
        for t in ts:
            idxs.append(np.argmin(np.abs(data_out["t"]-t)))
        for ii,key in enumerate(data_out.iterkeys()):
            if key == "t":
                continue
            else:
                data_out[key] = data_out[key][idxs,:]

    del data_out["t"]

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
    data_out = loadLightcurves(filename)
    if not opts.name in data_out:
        print "%s not in file..."%opts.name
        exit(0)

    data_out = data_out[opts.name]

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

if opts.model == "BHNS":
    pymultinest.run(myloglike_bhns, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = 100, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.5, multimodal = False)
elif opts.model == "BNS":
    pymultinest.run(myloglike_bns, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = 100, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = 0.5, multimodal = False)

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

#loglikelihood = -(1/2.0)*data[:,1]
#idx = np.argmax(loglikelihood)

if opts.model == "BHNS":

    t0 = data[:,0]
    q = data[:,1]
    chi = data[:,2]
    c = data[:,3]
    mb = data[:,4]
    zp = data[:,5]
    loglikelihood = data[:,6]
    idx = np.argmax(loglikelihood)

    t0_best = data[idx,0]
    q_best = data[idx,1]
    chi_best = data[idx,2]
    c_best = data[idx,3]
    mb_best = data[idx,4]
    zp_best = data[idx,5]

    tmag, lbol, mag = bhns_model(q_best,chi_best,c_best,mb_best)

elif opts.model == "BNS":

    t0 = data[:,0]
    m1 = data[:,1]
    m2 = data[:,2]
    c = data[:,3]
    mb = data[:,4]
    zp = data[:,5]
    loglikelihood = data[:,6]
    idx = np.argmax(loglikelihood)

    t0_best = data[idx,0]
    m1_best = data[idx,1]
    m2_best = data[idx,2]
    c_best = data[idx,3]
    mb_best = data[idx,4]
    zp_best = data[idx,5]

    tmag, lbol, mag = bns_model(m1_best,m2_best,c_best,mb_best)

tmag = tmag + t0_best

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

plt.plot(tmag,mag[1]+zp_best,'y--',label='model g-band')
plt.plot(tmag,mag[2]+zp_best,'g--',label='model r-band')
plt.plot(tmag,mag[3]+zp_best,'b--',label='model i-band')
plt.plot(tmag,mag[4]+zp_best,'c--',label='model z-band')
plt.plot(tmag,(mag[1]+mag[2]+mag[3])/3.0+zp_best,'m--',label='model w-band')

plt.xlim([0.0, 21.0])
plt.ylim([-15.0,5.0])

plt.xlabel('Time [days]')
plt.ylabel('AB Magnitude')
plt.legend(loc="best",prop={'size':6})
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

if opts.model == "BHNS":
    filename = os.path.join(plotDir,'samples.dat')
    fid = open(filename,'w+')
    for i, j, k, l,m,n in zip(t0,q,chi,c,mb,zp):
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
    fid.close()

    filename = os.path.join(plotDir,'best.dat')
    fid = open(filename,'w')
    fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,q_best,chi_best,c_best,mb_best,zp_best))
    fid.close()

elif opts.model == "BNS":
    filename = os.path.join(plotDir,'samples.dat')
    fid = open(filename,'w+')
    for i, j, k, l,m,n in zip(t0,m1,m2,c,mb,zp):
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
    fid.close()

    filename = os.path.join(plotDir,'best.dat')
    fid = open(filename,'w')
    fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,m2_best,c_best,mb_best,zp_best))
    fid.close()

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(t0)
plt.plot(bins1, hist1)
plt.xlabel('t0 [days]')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(plotDir,'t0.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(c)
plt.plot(bins1, hist1)
plt.xlabel('c')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(plotDir,'c.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(mb)
plt.plot(bins1, hist1)
plt.xlabel('mb')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(plotDir,'mb.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

plt.figure(figsize=(12,10))
bins1, hist1 = hist_results(zp)
plt.plot(bins1, hist1)
plt.xlabel('Zero Point [mag]')
plt.ylabel('Probability Density Function')
plt.show()
plotName = os.path.join(plotDir,'zp.pdf')
plt.savefig(plotName,dpi=200)
plt.close('all')

if opts.model == "BHNS":
    plt.figure(figsize=(12,10))
    bins1, hist1 = hist_results(q)
    plt.plot(bins1, hist1)
    plt.xlabel('Mass Ratio')
    plt.ylabel('Probability Density Function')
    plt.show()
    plotName = os.path.join(plotDir,'q.pdf')
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plt.figure(figsize=(12,10))
    bins1, hist1 = hist_results(chi)
    plt.plot(bins1, hist1)
    plt.xlabel('chi')
    plt.ylabel('Probability Density Function')
    plt.show()
    plotName = os.path.join(plotDir,'chi.pdf')
    plt.savefig(plotName,dpi=200)
    plt.close('all')
elif opts.model == "BNS":
    plt.figure(figsize=(12,10))
    bins1, hist1 = hist_results(m1)
    plt.plot(bins1, hist1)
    plt.xlabel('Mass 1 [solar masses]')
    plt.ylabel('Probability Density Function')
    plt.show()
    plotName = os.path.join(plotDir,'q.pdf')
    plt.savefig(plotName,dpi=200)
    plt.close('all')

    plt.figure(figsize=(12,10))
    bins1, hist1 = hist_results(m2)
    plt.plot(bins1, hist1)
    plt.xlabel('Mass 2 [solar masses]')
    plt.ylabel('Probability Density Function')
    plt.show()
    plotName = os.path.join(plotDir,'chi.pdf')
    plt.savefig(plotName,dpi=200)
    plt.close('all')

