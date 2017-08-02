
import os, sys, glob
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

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
    parser.add_option("-d","--dataDir",default="../lightcurves")
    parser.add_option("-n","--name",default="PS1-13cyr")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False) 
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)

    opts, args = parser.parse_args()

    return opts

def plot_results(samples,label,plotName):

    plt.figure(figsize=(12,10))
    bins1, hist1 = hist_results(samples)
    plt.plot(bins1, hist1)
    plt.xlabel(label)
    plt.ylabel('Probability Density Function')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

def hist_results(samples):

    bins = np.linspace(np.min(samples),np.max(samples),11)
    hist1, bin_edges = np.histogram(samples, bins=bins)
    hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

def bhns_model(q,chi_eff,mns,mb,c,th,ph):

    tini = 0.1
    tmax = 50.0
    dt = 0.1
    
    vmin = 0.00
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5
    
    t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c,mb,mns)

    return t, lbol, mag

def bhns_model_ejecta(mej,vej,th,ph):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    vmin = 0.00
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5

    t, lbol, mag = BHNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth)

    return t, lbol, mag

def bns_model(m1,mb1,c1,m2,mb2,c2,th,ph):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    vmin = 0.00
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5

    flgbct = 1

    t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c1,m2,mb2,c2,flgbct)

    return t, lbol, mag

def bns_model_ejecta(mej,vej,th,ph):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    vave = 0.267
    vmin = 0.00
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5

    flgbct = 1

    t, lbol, mag = BNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth,flgbct)

    return t, lbol, mag

def sn_model(z,t0,x0,x1,c):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    t, lbol, mag = SALT2.lightcurve(tini,tmax,dt,z,t0,x0,x1,c)

    return t, lbol, mag

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def myprior_bhns(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*9.0 + 1.0
        cube[2] = cube[2]*2.0 - 1.0
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*0.17 + 0.08
        cube[6] = cube[6]*100.0 - 50.0

def myprior_bhns_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*np.pi/2
        cube[4] = cube[4]*2*np.pi
        cube[5] = cube[5]*100.0 - 50.0

def myprior_bns(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.17 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.17 + 0.08
        cube[7] = cube[7]*100.0 - 50.0

def myprior_bns_fixZPT0(cube, ndim, nparams):
        cube[0] = 0.0
        cube[1] = cube[1]*2.0 + 1
        cube[2] = cube[2]*2.0 + 1
        cube[3] = cube[3]*0.17 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = 0.0

def myprior_bns_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*np.pi/2
        cube[4] = cube[4]*2*np.pi
        cube[5] = cube[5]*100.0 - 50.0

def myprior_bns_ejecta_fixZPT0(cube, ndim, nparams):
        cube[0] = 0.0
        cube[1] = cube[1]*4.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*0.17 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = 0.0

def myprior_sn(cube, ndim, nparams):
        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*10.0
        cube[2] = cube[2]*10.0
        cube[3] = cube[3]*10.0
        cube[4] = cube[4]*10.0
        cube[5] = cube[5]*100.0 - 50.0

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

def findconst(array):
    idx = np.where(~np.isnan(array))[0]
    if len(idx) == 0:
        return np.nan
    else:
        return array[idx[-1]]
    
def myloglike_bns(cube, ndim, nparams):

        t0 = cube[0]
        m1 = cube[1]
        mb1 = cube[2]
        c1 = cube[3]
        m2 = cube[4]
        mb2 = cube[5]
        c2 = cube[6]
        zp = cube[7]

        #c = 0.147
        #mb = 1.47
        #m1 = 1.35
        #m2 = 1.35
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_bns_ejecta(cube, ndim, nparams):
        t0 = cube[0]
        mej = 10**cube[1]
        vej = cube[2]
        th = cube[3]
        ph = cube[4]
        zp = cube[5]

        #c = 0.147
        #mb = 1.47
        #m1 = 1.35
        #m2 = 1.35
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = bns_model_ejecta(mej,vej,th,ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_bhns(cube, ndim, nparams):
        t0 = cube[0]
        q = cube[1]
        chi_eff = cube[2]
        mns = cube[3]
        mb = cube[4]
        c = cube[5]
        zp = cube[6]

        #c = 0.161
        #mb = 1.49
        #q = 3.0
        #chi = 0.3
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = bhns_model(q, chi_eff, mns, mb, c)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_bhns_ejecta(cube, ndim, nparams):
        t0 = cube[0]
        mej = 10**cube[1]
        vej = cube[2]
        th = cube[3]
        ph = cube[4]
        zp = cube[5]

        #c = 0.161
        #mb = 1.49
        #q = 3.0
        #chi = 0.3
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = bhns_model_ejecta(mej,vej,th,ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_sn(cube, ndim, nparams):
        t0 = cube[0]
        z = cube[1]
        x0 = cube[2]
        x1 = cube[3]
        c = cube[4]
        zp = cube[5]

        #z = 0.5
        #x0 = 1.0
        #x1 = 1.0
        #c = 1.0
        #t0 = 0.0
        #zp = 0.0

        tmag, lbol, mag = sn_model(z, 0.0 ,x0,x1,c)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def calc_prob(tmag, lbol, mag, t0, zp): 

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
                #maginterp = np.interp(t,tmag,addconst(mag[1]),left=np.nan, right=np.nan)
                ii = np.where(~np.isnan(mag[1]))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    f = interp.interp1d(tmag[ii], mag[1][ii], fill_value='extrapolate')
                    maginterp = f(t)
            elif key == "r":
                #maginterp = np.interp(t,tmag,addconst(mag[2]),left=np.nan, right=np.nan)
                ii = np.where(~np.isnan(mag[2]))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    f = interp.interp1d(tmag[ii], mag[2][ii], fill_value='extrapolate')
                    maginterp = f(t)
            elif key == "i":
                #maginterp = np.interp(t,tmag,addconst(mag[3]),left=np.nan, right=np.nan)

                ii = np.where(~np.isnan(mag[3]))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    f = interp.interp1d(tmag[ii], mag[3][ii], fill_value='extrapolate')
                    maginterp = f(t)
            elif key == "z":
                #maginterp = np.interp(t,tmag,addconst(mag[4]),left=np.nan, right=np.nan)
                ii = np.where(~np.isnan(mag[4]))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    f = interp.interp1d(tmag[ii], mag[4][ii], fill_value='extrapolate')
                    maginterp = f(t)
            elif key == "w":
                #maginterp = np.interp(t,tmag,addconst((mag[1]+mag[2]+mag[3])/3.0),left=np.nan, right=np.nan)
                magave = (mag[1]+mag[2]+mag[3])/3.0
                ii = np.where(~np.isnan(magave))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    f = interp.interp1d(tmag[ii], magave[ii], fill_value='extrapolate')
                    maginterp = f(t)
            else:
                continue

            maginterp = maginterp + zp
            chisquarevals = ((y-maginterp)/sigma_y)**2
            idx = np.where(~np.isnan(chisquarevals))[0]
            #if float(len(idx))/float(len(chisquarevals)) > 0.95:
            #    chisquarevals = chisquarevals[idx] 
            chisquaresum = np.sum(chisquarevals)

            if np.isnan(chisquaresum):
                chisquare = np.nan
                break

            chisquaresum = (1/float(len(chisquarevals)-1))*chisquaresum
            if count == 0:
                chisquare = chisquaresum
            else:
                chisquare = chisquare + chisquaresum
            #count = count + len(chisquarevals)
            count = count + 1

        if np.isnan(chisquare): 
            prob = -np.inf
        else:
            #prob = scipy.stats.chi2.logpdf(chisquare, count, loc=0, scale=1)
            #prob = -chisquare/2.0
            #prob = chisquare
            prob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print t0, zp, prob
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

def loadModels(outputDir,name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich et al. (2016)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)"]

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

def get_truths(name,model):
    truths = []
    for ii in xrange(n_params):
        truths.append(False)

    if not model in ["BHNS", "BNS"]:
        return truths        

    if name == "BNS_H4M005V20":
        truths = [0,np.log10(0.005),0.2,0.2,3.14,0.0]
    elif name == "BHNS_H4M005V20":
        truths = [0,np.log10(0.005),0.2,0.2,3.14,0.0]
    elif name == "rpft_m005_v2":
        truths = [0,np.log10(0.005),0.2,False,False,False]
    elif name == "APR4-1215_k1":
        truths = [0,np.log10(0.009),0.24,False,False,0.0]
    elif name == "APR4-1314_k1":
        truths = [0,np.log10(0.008),0.22,False,False,0.0]
    elif name == "H4-1215_k1":
        truths = [0,np.log10(0.004),0.21,False,False,0.0]
    elif name == "H4-1314_k1":
        truths = [0,np.log10(0.0007),0.17,False,False,0.0]
    elif name == "Sly-135_k1":
        truths = [0,np.log10(0.02),False,False,False,0.0]
    elif name == "APR4Q3a75_k1":
        truths = [0,np.log10(0.01),0.24,False,False,0.0]
    elif name == "H4Q3a75_k1":
        truths = [0,np.log10(0.05),0.21,False,False,0.0]
    elif name == "MS1Q3a75_k1":
        truths = [0,np.log10(0.07),0.25,False,False,0.0]
    elif name == "MS1Q7a75_k1":
        truths = [0,np.log10(0.06),0.25,False,False,0.0]
    elif name == "SED_nsbh1":
        truths = [0,np.log10(0.04),0.2,False,False,0.0]
    elif name == "SED_ns12ns12_kappa10":
        truths = [0,np.log10(0.0079), 0.12,False,False,False]
    return truths

# Parse command line
opts = parse_commandline()

if not opts.model in ["BHNS", "BNS", "SN"]:
   print "Model must be either: BHNS, BNS, SN"
   exit(0)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'models')
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
if opts.doFixZPT0:
    plotDir = os.path.join(baseplotDir,'models/%s_FixZPT0'%opts.model)
else:
    plotDir = os.path.join(baseplotDir,'models/%s'%opts.model)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
if opts.model in ["BNS","BHNS"]:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
if opts.doReduced:
    plotDir = os.path.join(plotDir,"%s_reduced"%opts.name)
else:
    plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

dataDir = opts.dataDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%dataDir
else:
    filename = "%s/lightcurves.tmp"%dataDir

errorbudget = opts.errorbudget
mint = 0.05
maxt = 7.0
dt = 0.05
n_live_points = 1000
evidence_tolerance = 0.5

if opts.doModels:
    data_out = loadModels(opts.outputDir,opts.name)
    if not opts.name in data_out:
        print "%s not in file..."%opts.name
        exit(0)

    data_out = data_out[opts.name]

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key] = np.vstack((data_out["t"],data_out[key],errorbudget*np.ones(data_out["t"].shape))).T

    idxs = np.intersect1d(np.where(data_out["t"]>=mint)[0],np.where(data_out["t"]<=maxt)[0])
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    tt = np.arange(mint,maxt,dt)
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:

            ii = np.where(np.isfinite(data_out[key][:,1]))[0]
            f = interp.interp1d(data_out[key][ii,0], data_out[key][ii,1], fill_value=np.nan, bounds_error=False)
            maginterp = f(tt)

            data_out[key] = np.vstack((tt,maginterp,errorbudget*np.ones(tt.shape))).T
           

    del data_out["t"]

    if opts.doReduced:
        tt = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
        for ii,key in enumerate(data_out.iterkeys()):
            maginterp = np.interp(tt,data_out[key][:,0],data_out[key][:,1],left=np.nan, right=np.nan)
            data_out[key] = np.vstack((tt,maginterp,errorbudget*np.ones(tt.shape))).T

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

if opts.model in ["BHNS","BNS"]:

    if opts.doMasses:
        if opts.model == "BHNS":
            parameters = ["t0","q","chi_eff","mns","mb","c","th","ph","zp"]
            labels = [r"$T_0$",r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm ns}$",r"$M_{\rm b}$",r"$C$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.model == "BNS":
            parameters = ["t0","m1","mb1","c1","m2","mb2","c2","th","ph","zp"]
            labels = [r"$T_0$",r"$m_{\rm 1}$",r"$m_{\rm b1}$",r"$C_{\rm 1}$",r"$m_{\rm 2}$",r"$m_{\rm b2}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.doEjecta:
        if opts.model == "BHNS":
            parameters = ["t0","mej","vej","th","ph","zp"]
            labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns_ejecta, myprior_bhns_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.model == "BNS":
            parameters = ["t0","mej","vej","th","ph","zp"]
            labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_ejecta, myprior_bns_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    else:
        print "Enable --doEjecta or --doMasses"
        exit(0)

elif opts.model in ["SN"]:

    parameters = ["t0","z","x0","x1","c","zp"]
    labels = [r"$T_0$", r"$z$", r"$x_0$", r"$x_1$",r"$c$","ZP"]
    n_params = len(parameters)

    pymultinest.run(myloglike_sn, myprior_sn, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)


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
    if opts.doMasses:
        t0 = data[:,0]
        q = data[:,1]
        chi_eff = data[:,2]
        mns = data[:,3]
        mb = data[:,4]
        c = data[:,5]
        th = data[:,6]
        ph = data[:,7]
        zp = data[:,8]
        loglikelihood = data[:,9]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        q_best = data[idx,1]
        chi_best = data[idx,2]
        mns_best = data[idx,3]
        mb_best = data[idx,4]
        c_best = data[idx,5]
        th_best = data[idx,6]
        ph_best = data[idx,7]
        zp_best = data[idx,8]

        tmag, lbol, mag = bhns_model(q_best,chi_best,mns_best,mb_best,c_best,th_best,ph_best)
    elif opts.doEjecta:
        t0 = data[:,0]
        mej = 10**data[:,1]
        vej = data[:,2]
        th = data[:,3]
        ph = data[:,4]
        zp = data[:,5]
        loglikelihood = data[:,6]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        mej_best = 10**data[idx,1]
        vej_best = data[idx,2]
        th_best = data[idx,3]
        ph_best = data[idx,4]
        zp_best = data[idx,5]

        tmag, lbol, mag = bhns_model_ejecta(mej_best,vej_best,th_best,ph_best)

elif opts.model == "BNS":

    if opts.doMasses:
        t0 = data[:,0]
        m1 = data[:,1]
        mb1 = data[:,2]
        c1 = data[:,3]
        m2 = data[:,4]
        mb2 = data[:,5]
        c2 = data[:,6]
        th = data[:,7]
        ph = data[:,8]
        zp = data[:,9]
        loglikelihood = data[:,10]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        m1_best = data[idx,1]
        mb1_best = data[idx,2]
        c1_best = data[idx,3]
        m2_best = data[idx,4]
        mb2_best = data[idx,5]
        c2_best = data[idx,6]
        th_best = data[idx,7]
        ph_best = data[idx,8]
        zp_best = data[idx,9]

        tmag, lbol, mag = bns_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,th_best,ph_best)
    elif opts.doEjecta:
        t0 = data[:,0]
        mej = 10**data[:,1]
        vej = data[:,2]
        th = data[:,3]
        ph = data[:,4]
        zp = data[:,5]
        loglikelihood = data[:,6]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        mej_best = 10**data[idx,1]
        vej_best = data[idx,2]
        th_best = data[idx,3]
        ph_best = data[idx,4]
        zp_best = data[idx,5]

        tmag, lbol, mag = bns_model_ejecta(mej_best,vej_best,th_best,ph_best)

elif opts.model == "SN":

    t0 = data[:,0]
    z = data[:,1]
    x0 = data[:,2]
    x1 = data[:,3]
    c = data[:,4]
    zp = data[:,5]
    loglikelihood = data[:,6]
    idx = np.argmax(loglikelihood)

    t0_best = data[idx,0]
    z_best = data[idx,1]
    x0_best = data[idx,2]
    x1_best = data[idx,3]
    c_best = data[idx,4]
    zp_best = data[idx,5]

    #t0_best = 0.0
    #z_best = 0.5
    #x0_best = 1.0
    #x1_best = 1.0
    #c_best = 1.0
    #zp_best = 0.0

    tmag, lbol, mag = sn_model(z_best,0.0,x0_best,x1_best,c_best)

truths = get_truths(opts.name,opts.model)

plotName = "%s/corner.pdf"%(plotDir)
if opts.doFixZPT0:
    figure = corner.corner(data[:,1:5], labels=labels[1:5],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=False, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".1f",
                       truths=truths[1:5])
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".1f",
                       truths=truths)
figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

tmag = tmag + t0_best

filts = ["g","r","i","z","y"]
colors = ["y","g","b","c","k"]
magidxs = [1,2,3,4,5]

plotName = "%s/lightcurve.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
    if not filt in data_out: continue
    samples = data_out[filt]
    t = samples[:,0]
    y = samples[:,1]
    sigma_y = samples[:,2]
    
    idx = np.where(~np.isnan(y))[0]
    t = t[idx]
    y = y[idx]
    sigma_y = sigma_y[idx]

    plt.errorbar(t,y,sigma_y,fmt='%so'%color,label='%s-band'%filt)
    #plt.plot(tmag,mag[magidx]+zp_best,'k--')

    tini = np.min(t)
    tmax = 10.0
    dt = 0.1
    tt = np.arange(tini,tmax,dt)

    ii = np.where(~np.isnan(mag[magidx]))[0]
    f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'k--',linewidth=2)

if opts.model == "SN":
    plt.xlim([0.0, 10.0])
    #plt.ylim([-15.0,5.0])
else:
    plt.xlim([1.0, 8.0])
    #plt.ylim([-16.0,3.0])

plt.xlabel('Time [days]',fontsize=24)
plt.ylabel('Absolute Magnitude',fontsize=24)
plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

if opts.model == "BHNS":
    if opts.doMasses:
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l,m,n,o in zip(t0,q,chi,mns,mb,c,zp):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,q_best,chi_best,mns_best,mb_best,c_best,zp_best))
        fid.close()
    elif opts.doEjecta:
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l in zip(t0,mej,vej,zp):
            fid.write('%.5f %.5f %.5f %.5f\n'%(i,j,k,l))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,zp_best))
        fid.close()

elif opts.model == "BNS":
    if opts.doMasses:
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l, m, n, o, p in zip(t0,m1,mb1,c1,m2,mb2,c2,zp):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n,o,p))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,zp_best))
        fid.close()
    elif opts.doEjecta:
        filename = os.path.join(plotDir,'samples.dat')
        fid = open(filename,'w+')
        for i, j, k, l in zip(t0,mej,vej,zp):
            fid.write('%.5f %.5f %.5f %.5f\n'%(i,j,k,l))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,zp_best))
        fid.close()

elif opts.model == "SN":
    filename = os.path.join(plotDir,'samples.dat')
    fid = open(filename,'w+')
    for i, j, k, l,m,n in zip(t0,z,x0,x1,c,zp):
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
    fid.close()

    filename = os.path.join(plotDir,'best.dat')
    fid = open(filename,'w')
    fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,z_best,x0_best,x1_best,c_best,zp_best))
    fid.close()

