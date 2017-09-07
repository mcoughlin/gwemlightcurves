
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
from matplotlib.pyplot import cm 

import corner

import pymultinest
from gwemlightcurves import BHNSKilonovaLightcurve, BNSKilonovaLightcurve, SALT2, BlueKilonovaLightcurve
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
    parser.add_option("-n","--name",default="PS1-13cyr")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)
    parser.add_option("--doCoverage",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False) 
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("-f","--filters",default="g,r,i,z")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)

    opts, args = parser.parse_args()

    return opts

def norm_sym_ratio(eta): 
    # Assume floating point precision issues
    #if np.any(np.isclose(eta, 0.25)):
    #eta[np.isclose(eta, 0.25)] = 0.25

    # Assert phyisicality
    assert np.all(eta <= 0.25)
 
    return np.sqrt(1 - 4. * eta)

def q2eta(q):
    return q/(1+q)**2
 
def mc2ms(mc,eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    root = np.sqrt(0.25-eta)
    fraction = (0.5+root) / (0.5-root)
    invfraction = 1/fraction

    m2= mc * np.power((1+fraction),0.2) / np.power(fraction,0.6)

    m1= mc* np.power(1+invfraction,0.2) / np.power(invfraction,0.6)
    return (m1,m2)

def ms2mc(m1,m2):
    eta = m1*m2/( (m1+m2)*(m1+m2) )
    mchirp = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))
    q = m2/m1

    return (mchirp,eta,q)

def hist_results(samples,Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-3
    #hist1 = hist1 / float(np.sum(hist1))
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

def blue_model(m1,mb1,c1,m2,mb2,c2,beta,kappa_r):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    t, lbol, mag = BlueKilonovaLightcurve.lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2)

    return t, lbol, mag

def blue_model_ejecta(mej,vej,beta,kappa_r):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    t, lbol, mag = BlueKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,beta,kappa_r)

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
        cube[1] = cube[1]*6.0 + 3.0
        cube[2] = cube[2]*0.75
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*0.1 + 0.1
        cube[6] = cube[6]*np.pi/2
        cube[7] = cube[7]*2*np.pi
        cube[8] = cube[8]*100.0 - 50.0

def myprior_bhns_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*np.pi/2
        cube[4] = cube[4]*2*np.pi
        cube[5] = cube[5]*100.0 - 50.0

def myprior_bhns_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*6.0 + 3.0
        cube[2] = cube[2]*0.75
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.1 + 0.1
        cube[5] = cube[5]*np.pi/2
        cube[6] = cube[6]*2*np.pi
        cube[7] = cube[7]*100.0 - 50.0

def prior_bns(m1,mb1,c1,m2,mb2,c2):
        if m1 < m2:
            return 0.0
        else:
            return 1.0

def prior_bhns(q,chi_eff,mns,mb,c):
        return 1.0

def myprior_blue(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*10.0
        cube[8] = cube[8]*50.0
        cube[9] = cube[9]*100.0 - 50.0

def myprior_blue_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*10.0
        cube[6] = cube[6]*50.0
        cube[7] = cube[7]*100.0 - 50.0

def myprior_blue_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*5.0 - 5.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*10.0
        cube[4] = cube[4]*50.0
        cube[5] = cube[5]*100.0 - 50.0

def myprior_bns(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*2.0 + 1.0
        cube[6] = cube[6]*0.16 + 0.08
        cube[7] = cube[7]*np.pi/2
        cube[8] = cube[8]*2*np.pi
        cube[9] = cube[9]*100.0 - 50.0

def myprior_bns_EOSFit(cube, ndim, nparams):

        cube[0] = cube[0]*10.0 - 5.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.16 + 0.08
        cube[5] = cube[5]*np.pi/2
        cube[6] = cube[6]*2*np.pi
        cube[7] = cube[7]*100.0 - 50.0

def myprior_bhns_ejecta_fixZPT0(cube, ndim, nparams):
        cube[0] = 0.0
        cube[1] = cube[1]*2.0 + 1
        cube[2] = cube[2]*2.0 + 1
        cube[3] = cube[3]*0.16 + 0.08
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
        cube[3] = cube[3]*0.16 + 0.08
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

def myloglike_blue(cube, ndim, nparams):

        t0 = cube[0]
        m1 = cube[1]
        mb1 = cube[2]
        c1 = cube[3]
        m2 = cube[4]
        mb2 = cube[5]
        c2 = cube[6]
        beta = cube[7]
        kappa_r = cube[8]
        zp = cube[9]

        tmag, lbol, mag = blue_model(m1,mb1,c1,m2,mb2,c2,beta,kappa_r)

        prob = calc_prob(tmag, lbol, mag, t0, zp)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_blue_ejecta(cube, ndim, nparams):
        t0 = cube[0]
        mej = 10**cube[1]
        vej = cube[2]
        beta = cube[3]
        kappa_r = cube[4]
        zp = cube[5]

        tmag, lbol, mag = blue_model_ejecta(mej,vej,beta,kappa_r)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_blue_EOSFit(cube, ndim, nparams):

        t0 = cube[0]
        m1 = cube[1]
        c1 = cube[2]
        m2 = cube[3]
        c2 = cube[4]
        beta = cube[5]
        kappa_r = cube[6]
        zp = cube[7]

        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)

        tmag, lbol, mag = blue_model(m1,mb1,c1,m2,mb2,c2,beta,kappa_r)

        prob = calc_prob(tmag, lbol, mag, t0, zp)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob
    
def myloglike_bns(cube, ndim, nparams):

        t0 = cube[0]
        m1 = cube[1]
        mb1 = cube[2]
        c1 = cube[3]
        m2 = cube[4]
        mb2 = cube[5]
        c2 = cube[6]
        th = cube[7]
        ph = cube[8]
        zp = cube[9]

        tmag, lbol, mag = bns_model(m1,mb1,c1,m2,mb2,c2,th,ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_ejecta(cube, ndim, nparams):
        t0 = cube[0]
        mej = 10**cube[1]
        vej = cube[2]
        th = cube[3]
        ph = cube[4]
        zp = cube[5]

        tmag, lbol, mag = bns_model_ejecta(mej,vej,th,ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_bns_EOSFit(cube, ndim, nparams):

        t0 = cube[0]
        m1 = cube[1]
        c1 = cube[2]
        m2 = cube[3]
        c2 = cube[4]
        th = cube[5]
        ph = cube[6]
        zp = cube[7]

        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)

        tmag, lbol, mag = bns_model(m1,mb1,c1,m2,mb2,c2,th,ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns(cube, ndim, nparams):
        t0 = cube[0]
        q = cube[1]
        chi_eff = cube[2]
        mns = cube[3]
        mb = cube[4]
        c = cube[5]
        th = cube[6]
        ph = cube[7]
        zp = cube[8]

        tmag, lbol, mag = bhns_model(q, chi_eff, mns, mb, c, th, ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_ejecta(cube, ndim, nparams):
        t0 = cube[0]
        mej = 10**cube[1]
        vej = cube[2]
        th = cube[3]
        ph = cube[4]
        zp = cube[5]

        tmag, lbol, mag = bhns_model_ejecta(mej,vej,th,ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

        return prob

def myloglike_bhns_EOSFit(cube, ndim, nparams):
        t0 = cube[0]
        q = cube[1]
        chi_eff = cube[2]
        mns = cube[3]
        c = cube[4]
        th = cube[5]
        ph = cube[6]
        zp = cube[7]

        mb = EOSfit(mns,c)

        tmag, lbol, mag = bhns_model(q, chi_eff, mns, mb, c, th, ph)

        prob = calc_prob(tmag, lbol, mag, t0, zp)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

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
        gaussprob = np.nan
        for key in data_out:
            samples = data_out[key]
            t = samples[:,0]
            y = samples[:,1]
            sigma_y = samples[:,2]

            idx = np.where(~np.isnan(y))[0]
            t = t[idx]
            y = y[idx]
            sigma_y = sigma_y[idx]

            if not key in filters: continue

            keyslist = ["u","g","r","i","z","y","J","H","K"]

            if key in keyslist:
                idx = keyslist.index(key)
                ii = np.where(~np.isnan(mag[idx]))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    f = interp.interp1d(tmag[ii], mag[idx][ii], fill_value='extrapolate')
                    maginterp = f(t)
            elif key == "w":
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

            sigma = np.sqrt(errorbudget**2 + sigma_y**2)
            chisquarevals = np.zeros(y.shape)
            chisquarevals = ((y-maginterp)/sigma)**2
            idx = np.where(~np.isfinite(sigma))[0]
            if len(idx) > 0:
                gaussprobvals = scipy.stats.norm.cdf(y[idx], maginterp[idx], errorbudget)
                gaussprobsum = np.sum(np.log(gaussprobvals)) 
            else:
                gaussprobsum = 0.0

            chisquaresum = np.sum(chisquarevals)
            if np.isnan(chisquaresum):
                chisquare = np.nan
                break

            if not float(len(chisquarevals)-1) == 0:
                chisquaresum = (1/float(len(chisquarevals)-1))*chisquaresum

            if count == 0:
                chisquare = chisquaresum
                gaussprob = gaussprobsum
            else:
                chisquare = chisquare + chisquaresum
                gaussprob = gaussprob + gaussprobsum
            #count = count + len(chisquarevals)
            count = count + 1

        if np.isnan(chisquare): 
            prob = -np.inf
        else:
            #prob = scipy.stats.chi2.logpdf(chisquare, count, loc=0, scale=1)
            #prob = -chisquare/2.0
            #prob = chisquare
            chiprob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)
            prob = chiprob + gaussprob

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print t0, zp, prob
        return prob

def get_truths(name,model):
    truths = []
    for ii in xrange(n_params):
        truths.append(False)

    if not model in ["BHNS", "BNS"]:
        return truths        

    if not opts.doEjecta:
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

def EOSfit(mns,c):
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb

# Parse command line
opts = parse_commandline()

if not opts.model in ["BHNS", "BNS", "SN","Blue"]:
   print "Model must be either: BHNS, BNS, SN, Blue"
   exit(0)

filters = opts.filters.split(",")

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models'
elif opts.doGoingTheDistance:
    basename = 'going-the-distance'
elif opts.doMassGap:
    basename = 'massgap'
else:
    basename = 'gws'
plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
else:
    plotDir = os.path.join(plotDir,'%s'%opts.model)
plotDir = os.path.join(plotDir,"_".join(filters))
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
if opts.model in ["BNS","BHNS","Blue"]:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
if opts.doReduced:
    plotDir = os.path.join(plotDir,"%s_reduced"%opts.name)
else:
    plotDir = os.path.join(plotDir,opts.name)
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

errorbudget = opts.errorbudget
mint = opts.tmin
maxt = opts.tmax
dt = opts.dt
n_live_points = 1000
evidence_tolerance = 0.5

if opts.doModels or opts.doGoingTheDistance or opts.doMassGap:
    if opts.doModels:
        data_out = lightcurve_utils.loadModels(opts.outputDir,opts.name)
        if not opts.name in data_out:
            print "%s not in file..."%opts.name
            exit(0)

        data_out = data_out[opts.name]

    elif opts.doGoingTheDistance or opts.doMassGap:

        truths = {}
        if opts.doGoingTheDistance:
            data_out = lightcurve_utils.going_the_distance(opts.dataDir,opts.name)
        elif opts.doMassGap:
            data_out, truths = lightcurve_utils.massgap(opts.dataDir,opts.name)

        if "m1" in truths:
            eta = q2eta(truths["q"])
            m1, m2 = truths["m1"], truths["m2"]
            mchirp,eta,q = ms2mc(m1,m2)
            q = 1/q 
            chi_eff = truths["a1"]
        else:
            eta = q2eta(data_out["q"])
            m1, m2 = mc2ms(data_out["mc"], eta)
            q = m2/m1
            mc = data_out["mc"]

            m1, m2 = np.mean(m1), np.mean(m2)
            chi_eff = 0.0       

        c1, c2 = 0.147, 0.147
        mb1, mb2 = EOSfit(m1,c1), EOSfit(m2,c2)
        th = 0.2
        ph = 3.14

        if m1 > 3:
            mej = BHNSKilonovaLightcurve.calc_meje(q,chi_eff,c2,mb2,m2)
            vej = BHNSKilonovaLightcurve.calc_vave(q)
        else:
            mej = BNSKilonovaLightcurve.calc_meje(m1,mb1,c1,m2,mb2,c2)
            vej = BNSKilonovaLightcurve.calc_vej(m1,c1,m2,c2)

        filename = os.path.join(plotDir,'truth_mej_vej.dat')
        fid = open(filename,'w+')
        fid.write('%.5f %.5f\n'%(mej,vej))
        fid.close()

        if m1 > 3:
            filename = os.path.join(plotDir,'truth.dat')
            fid = open(filename,'w+')
            fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(q,chi_eff,c2,mb2,m2))
            fid.close()

            t, lbol, mag = bhns_model(q,chi_eff,m2,mb2,c2,th,ph) 

        else:
            filename = os.path.join(plotDir,'truth.dat')
            fid = open(filename,'w+')
            fid.write('%.5f %.5f %.5f %.5f\n'%(m1,c1,m2,c2))
            fid.close()

            t, lbol, mag = bns_model(m1,mb1,c1,m2,mb2,c2,th,ph)

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
    if opts.doEvent:
        data_out = lightcurve_utils.loadEvent(filename)
    else:
        data_out = lightcurve_utils.loadLightcurves(filename)
        if not opts.name in data_out:
            print "%s not in file..."%opts.name
            exit(0)

        data_out = data_out[opts.name]

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key][:,0] = data_out[key][:,0] - opts.T0
            data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            idxs = np.intersect1d(np.where(data_out[key][:,0]>=mint)[0],np.where(data_out[key][:,0]<=maxt)[0])
            data_out[key] = data_out[key][idxs,:]

    for ii,key in enumerate(data_out.iterkeys()):
        idxs = np.where(~np.isnan(data_out[key][:,2]))[0]
        if key == "t":
            continue
        else:
            data_out[key] = data_out[key][idxs,:]

    for ii,key in enumerate(data_out.keys()):
        if not key in filters:
            del data_out[key]

    for ii,key in enumerate(data_out.iterkeys()):
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

    if opts.model == "BHNS":
        filename = os.path.join(plotDir,'truth.dat')
        fid = open(filename,'w+')
        fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(np.nan,np.nan,np.nan,np.nan,np.nan))
        fid.close()
    else:
        filename = os.path.join(plotDir,'truth.dat')
        fid = open(filename,'w+')
        fid.write('%.5f %.5f %.5f %.5f\n'%(np.nan,np.nan,np.nan,np.nan))
        fid.close()

if opts.model in ["BHNS","BNS","Blue"]:

    if opts.doMasses:
        if opts.model == "BHNS":
            if opts.doEOSFit:
                parameters = ["t0","q","chi_eff","mns","c","th","ph","zp"]
                labels = [r"$T_0$",r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm ns}$",r"$C$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_bhns_EOSFit, myprior_bhns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
            else:
                parameters = ["t0","q","chi_eff","mns","mb","c","th","ph","zp"]
                labels = [r"$T_0$",r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm ns}$",r"$M_{\rm b}$",r"$C$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_bhns, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.model == "BNS":
            if opts.doEOSFit:
                parameters = ["t0","m1","c1","m2","c2","th","ph","zp"]
                labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_bns_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
            else:
                parameters = ["t0","m1","mb1","c1","m2","mb2","c2","th","ph","zp"]
                labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_bns, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.model == "Blue":
            if opts.doEOSFit:
                parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"$M_{\rm 1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$C_{\rm 2}$",r"$\beta$",r"$\kappa_{\rm r}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_blue_EOSFit, myprior_blue_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
            else:
                parameters = ["t0","m1","mb1","c1","m2","mb2","c2","beta","kappa_r","zp"]
                labels = [r"$T_0$",r"$M_{\rm 1}$",r"$M_{\rm b1}$",r"$C_{\rm 1}$",r"$M_{\rm 2}$",r"$M_{\rm b2}$",r"$C_{\rm 2}$",r"$\beta$",r"$\kappa_{\rm r}$","ZP"]
                n_params = len(parameters)
                pymultinest.run(myloglike_blue, myprior_blue, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
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
        elif opts.model == "Blue":
            parameters = ["t0","mej","vej","beta","kappa_r","zp"]
            labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"$\beta$",r"$\kappa_{\rm r}$","ZP"]
            n_params = len(parameters)
            pymultinest.run(myloglike_blue_ejecta, myprior_blue_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
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
        if opts.doEOSFit:
            t0 = data[:,0]
            q = data[:,1]
            chi_eff = data[:,2]
            mns = data[:,3]
            c = data[:,4]
            th = data[:,5]
            ph = data[:,6]
            zp = data[:,7]
            loglikelihood = data[:,8]
            idx = np.argmax(loglikelihood)
            mb = EOSfit(mns,c)

            t0_best = data[idx,0]
            q_best = data[idx,1]
            chi_best = data[idx,2]
            mns_best = data[idx,3]
            c_best = data[idx,4]
            th_best = data[idx,5]
            ph_best = data[idx,6]
            zp_best = data[idx,7]
            mb_best = mb[idx]

            tmag, lbol, mag = bhns_model(q_best,chi_best,mns_best,mb_best,c_best,th_best,ph_best)
        else:
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
        if opts.doEOSFit:

            t0 = data[:,0]
            m1 = data[:,1]
            c1 = data[:,2]
            m2 = data[:,3]
            c2 = data[:,4]
            th = data[:,5]
            ph = data[:,6]
            zp = data[:,7]
            loglikelihood = data[:,8]
            idx = np.argmax(loglikelihood)
            mb1 = EOSfit(m1,c1)
            mb2 = EOSfit(m2,c2)

            t0_best = data[idx,0]
            m1_best = data[idx,1]
            c1_best = data[idx,2]
            m2_best = data[idx,3]
            c2_best = data[idx,4]
            th_best = data[idx,5]
            ph_best = data[idx,6]
            zp_best = data[idx,7]
            mb1_best = mb1[idx]
            mb2_best = mb2[idx]

            data_new = np.zeros(data.shape)
            parameters = ["t0","m1","c1","m2","c2","th","ph","zp"]
            labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$",r"$\theta_{\rm ej}$",r"$\phi_{\rm ej}$","ZP"]
            mchirp,eta,q = ms2mc(data[:,1],data[:,3])
            data_new[:,0] = data[:,0]
            data_new[:,1] = 1/q
            data_new[:,2] = mchirp
            data_new[:,3] = data[:,2]
            data_new[:,4] = data[:,4]
            data_new[:,5] = data[:,5]
            data_new[:,6] = data[:,6]
            data_new[:,7] = data[:,7]
            data = data_new

        else:
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

elif opts.model == "Blue":

    if opts.doMasses:
        if opts.doEOSFit:

            t0 = data[:,0]
            m1 = data[:,1]
            c1 = data[:,2]
            m2 = data[:,3]
            c2 = data[:,4]
            beta = data[:,5]
            kappa_r = data[:,6]
            zp = data[:,7]
            loglikelihood = data[:,8]
            idx = np.argmax(loglikelihood)
            mb1 = EOSfit(m1,c1)
            mb2 = EOSfit(m2,c2)

            t0_best = data[idx,0]
            m1_best = data[idx,1]
            c1_best = data[idx,2]
            m2_best = data[idx,3]
            c2_best = data[idx,4]
            beta_best = data[idx,5]
            kappa_r_best = data[idx,6]
            zp_best = data[idx,7]
            mb1_best = mb1[idx]
            mb2_best = mb2[idx]

            data_new = np.zeros(data.shape)
            parameters = ["t0","m1","c1","m2","c2","beta","kappa_r","zp"]
            labels = [r"$T_0$",r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$",r"$\beta$",r"$\kappa_{\rm r}$","ZP"]
            mchirp,eta,q = ms2mc(data[:,1],data[:,3])
            data_new[:,0] = data[:,0]
            data_new[:,1] = 1/q
            data_new[:,2] = mchirp
            data_new[:,3] = data[:,2]
            data_new[:,4] = data[:,4]
            data_new[:,5] = data[:,5]
            data_new[:,6] = data[:,6]
            data_new[:,7] = data[:,7]
            data = data_new

        else:
            t0 = data[:,0]
            m1 = data[:,1]
            mb1 = data[:,2]
            c1 = data[:,3]
            m2 = data[:,4]
            mb2 = data[:,5]
            c2 = data[:,6]
            beta = data[:,7]
            kappa_r = data[:,8]
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
            beta_best = data[idx,7]
            kappa_r_best = data[idx,8]
            zp_best = data[idx,9]

        tmag, lbol, mag = blue_model(m1_best,mb1_best,c1_best,m2_best,mb2_best,c2_best,beta_best,kappa_r_best)

    elif opts.doEjecta:
        t0 = data[:,0]
        mej = 10**data[:,1]
        vej = data[:,2]
        beta = data[:,3]
        kappa_r = data[:,4]
        zp = data[:,5]
        loglikelihood = data[:,6]
        idx = np.argmax(loglikelihood)

        t0_best = data[idx,0]
        mej_best = 10**data[idx,1]
        vej_best = data[idx,2]
        beta_best = data[idx,3]
        kappa_r_best = data[idx,4]
        zp_best = data[idx,5]

        tmag, lbol, mag = blue_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best)

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

    tmag, lbol, mag = sn_model(z_best,0.0,x0_best,x1_best,c_best)

truths = get_truths(opts.name,opts.model)

if n_params >= 8:
    title_fontsize = 26
    label_fontsize = 30
else:
    title_fontsize = 24
    label_fontsize = 28

plotName = "%s/corner.pdf"%(plotDir)
if opts.doFixZPT0:
    figure = corner.corner(data[:,1:5], labels=labels[1:5],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=False, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".1f",
                       truths=truths[1:5])
else:
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

tmag = tmag + t0_best

filts = ["u","g","r","i","z","y","J","H","K"]
#colors = ["y","g","b","c","k","pink","orange","purple"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]

plotName = "%s/lightcurve.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)

    #tini, tmax, dt = np.min(t), 10.0, 0.1
    tini, tmax, dt = 0.0, 14.0, 0.1
    tt = np.arange(tini,tmax,dt)

    ii = np.where(~np.isnan(mag[magidx]))[0]
    f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'k--',linewidth=2)

if opts.model == "SN":
    plt.xlim([0.0, 10.0])
else:
    plt.xlim([1.0, 8.0])

plt.xlabel('Time [days]',fontsize=24)
plt.ylabel('Absolute Magnitude',fontsize=24)
plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/lightcurve_zoom.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
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

    #tini, tmax, dt = np.min(t), 14.0, 0.1
    tini, tmax, dt = 0.0, 14.0, 0.1
    tt = np.arange(tini,tmax,dt)

    ii = np.where(~np.isnan(mag[magidx]))[0]
    f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2)
    plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

if opts.model == "SN":
    plt.xlim([0.0, 10.0])
else:
    if opts.model == "Blue":
        plt.xlim([0.0, 14.0])
        plt.ylim([-20.0,-5.0])
    elif opts.model == "BNS":
        plt.xlim([0.0, 14.0])
        plt.ylim([-20.0,-5.0])
    elif opts.model == "BHNS":
        plt.xlim([0.0, 14.0])
        plt.ylim([-20.0,-5.0])
    else:
        plt.xlim([1.0, 7.0])
        plt.ylim([-16.0,-10.0])

plt.xlabel('Time [days]',fontsize=24)
plt.ylabel('Absolute Magnitude',fontsize=24)
plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/lightcurve_zoom_optical.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
    if not filt in data_out: continue
    if not filt in ["u","g","r","i","z","y"]: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)

    #tini, tmax, dt = np.min(t), 10.0, 0.1
    tini, tmax, dt = 0.0, 14.0, 0.1
    tt = np.arange(tini,tmax,dt)

    ii = np.where(~np.isnan(mag[magidx]))[0]
    f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2)
    plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

if opts.model == "SN":
    plt.xlim([0.0, 10.0])
else:
    plt.xlim([0.5, 7.0])
    plt.ylim([-20.0,-10.0])

plt.xlabel('Time [days]',fontsize=24)
plt.ylabel('Absolute Magnitude',fontsize=24)
plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/lightcurve_zoom_nir.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for filt, color, magidx in zip(filts,colors,magidxs):
    if not filt in data_out: continue
    if not filt in ["J","H","K"]: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)

    #tini, tmax, dt = np.min(t), 10.0, 0.1
    tini, tmax, dt = 0.0, 14.0, 0.1
    tt = np.arange(tini,tmax,dt)

    ii = np.where(~np.isnan(mag[magidx]))[0]
    f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2)
    plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

if opts.model == "SN":
    plt.xlim([0.0, 10.0])
else:
    plt.xlim([0.5, 7.0])
    plt.ylim([-20.0,-10.0])

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
        for i, j, k, l,m,n,o in zip(t0,q,chi_eff,mns,mb,c,zp):
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
        for i, j, k, l, m, n in zip(t0,mej,vej,th,ph,zp):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,th_best,ph_best,zp_best))
        fid.close()

elif opts.model == "Blue":
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
        for i, j, k, l, m, n in zip(t0,mej,vej,beta,kappa_r,zp):
            fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(i,j,k,l,m,n))
        fid.close()

        filename = os.path.join(plotDir,'best.dat')
        fid = open(filename,'w')
        fid.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'%(t0_best,mej_best,vej_best,beta_best,kappa_r_best,zp_best))
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

