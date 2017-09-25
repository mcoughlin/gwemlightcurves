
import numpy as np
import scipy.stats
from scipy.interpolate import interpolate as interp
from gwemlightcurves import lightcurve_utils, Global
from .model import *

def prior_DiUj2017(m1,mb1,c1,m2,mb2,c2):
    if m1 < m2:
        return 0.0
    else:
        return 1.0

def prior_KaKy2016(q,chi_eff,mns,mb,c):
    return 1.0

def myloglike_RoFe2017(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    mb1 = cube[2]
    c1 = cube[3]
    m2 = cube[4]
    mb2 = cube[5]
    c2 = cube[6]
    Ye = cube[7]
    zp = cube[8]

    tmag, lbol, mag = RoFe2017_model(m1,mb1,c1,m2,mb2,c2,Ye)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_RoFe2017_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    Ye = cube[3]
    zp = cube[4]

    tmag, lbol, mag = RoFe2017_model_ejecta(mej,vej,Ye)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_RoFe2017_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    Ye = cube[5]
    zp = cube[6]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = RoFe2017_model(m1,mb1,c1,m2,mb2,c2,Ye)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_Ka2017(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    mb1 = cube[2]
    c1 = cube[3]
    m2 = cube[4]
    mb2 = cube[5]
    c2 = cube[6]
    Xlan = 10**cube[7]
    zp = cube[8]

    tmag, lbol, mag = Ka2017_model(m1,mb1,c1,m2,mb2,c2,Xlan)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_Ka2017_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    Xlan = 10**cube[3]
    zp = cube[4]

    tmag, lbol, mag = Ka2017_model_ejecta(mej,vej,Xlan)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_Ka2017_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    Xlan = 10**cube[5]
    zp = cube[6]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = Ka2017_model(m1,mb1,c1,m2,mb2,c2,Xlan)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_Me2017(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    mb1 = cube[2]
    c1 = cube[3]
    m2 = cube[4]
    mb2 = cube[5]
    c2 = cube[6]
    beta = cube[7]
    kappa_r = 10**cube[8]
    zp = cube[9]

    tmag, lbol, mag = Me2017_model(m1,mb1,c1,m2,mb2,c2,beta,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_WoKo2017(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    mb1 = cube[2]
    c1 = cube[3]
    m2 = cube[4]
    mb2 = cube[5]
    c2 = cube[6]
    theta_r = cube[7]
    kappa_r = 10**cube[8]
    zp = cube[9]

    tmag, lbol, mag = WoKo2017_model(m1,mb1,c1,m2,mb2,c2,theta_r,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_SmCh2017(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    mb1 = cube[2]
    c1 = cube[3]
    m2 = cube[4]
    mb2 = cube[5]
    c2 = cube[6]
    slope_r = cube[7]
    kappa_r = 10**cube[8]
    zp = cube[9]

    tmag, lbol, mag = SmCh2017_model(m1,mb1,c1,m2,mb2,c2,slope_r,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_Me2017_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    beta = cube[3]
    kappa_r = 10**cube[4]
    zp = cube[5]

    tmag, lbol, mag = Me2017_model_ejecta(mej,vej,beta,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_WoKo2017_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    theta_r = cube[3]
    kappa_r = 10**cube[4]
    zp = cube[5]

    tmag, lbol, mag = WoKo2017_model_ejecta(mej,vej,theta_r,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_SmCh2017_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    slope_r = cube[3]
    kappa_r = 10**cube[4]
    zp = cube[5]

    tmag, lbol, mag = SmCh2017_model_ejecta(mej,vej,slope_r,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_Me2017_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    beta = cube[5]
    kappa_r = 10**cube[6]
    zp = cube[7]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = Me2017_model(m1,mb1,c1,m2,mb2,c2,beta,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_WoKo2017_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    theta_r = cube[5]
    kappa_r = 10**cube[6]
    zp = cube[7]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = WoKo2017_model(m1,mb1,c1,m2,mb2,c2,theta_r,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_SmCh2017_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    slope_r = cube[5]
    kappa_r = 10**cube[6]
    zp = cube[7]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = SmCh2017_model(m1,mb1,c1,m2,mb2,c2,slope_r,kappa_r)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_DiUj2017(cube, ndim, nparams):

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

    tmag, lbol, mag = DiUj2017_model(m1,mb1,c1,m2,mb2,c2,th,ph)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_DiUj2017_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    th = cube[3]
    ph = cube[4]
    zp = cube[5]

    tmag, lbol, mag = DiUj2017_model_ejecta(mej,vej,th,ph)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_DiUj2017_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    th = cube[5]
    ph = cube[6]
    zp = cube[7]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = DiUj2017_model(m1,mb1,c1,m2,mb2,c2,th,ph)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_BaKa2016(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    mb1 = cube[2]
    c1 = cube[3]
    m2 = cube[4]
    mb2 = cube[5]
    c2 = cube[6]
    zp = cube[7]

    tmag, lbol, mag = BaKa2016_model(m1,mb1,c1,m2,mb2,c2)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_BaKa2016_ejecta(cube, ndim, nparams):
    t0 = cube[0]  
    mej = 10**cube[1]
    vej = cube[2]
    zp = cube[3]

    tmag, lbol, mag = BaKa2016_model_ejecta(mej,vej)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_BaKa2016_EOSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    zp = cube[5]

    mb1 = lightcurve_utils.EOSfit(m1,c1)
    mb2 = lightcurve_utils.EOSfit(m2,c2)

    tmag, lbol, mag = BaKa2016_model(m1,mb1,c1,m2,mb2,c2)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,mb1,c1,m2,mb2,c2)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_KaKy2016(cube, ndim, nparams):
    t0 = cube[0]
    q = cube[1]
    chi_eff = cube[2]
    mns = cube[3]
    mb = cube[4]
    c = cube[5]
    th = cube[6]
    ph = cube[7]
    zp = cube[8]

    tmag, lbol, mag = KaKy2016_model(q, chi_eff, mns, mb, c, th, ph)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_KaKy2016(q,chi_eff,mns,mb,c)
    if prior == 0.0:
        prob = -np.inf

    return prob

def myloglike_KaKy2016_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    th = cube[3]
    ph = cube[4]
    zp = cube[5]

    tmag, lbol, mag = KaKy2016_model_ejecta(mej,vej,th,ph)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_KaKy2016_EOSFit(cube, ndim, nparams):
    t0 = cube[0]
    q = cube[1]
    chi_eff = cube[2]
    mns = cube[3]
    c = cube[4]
    th = cube[5]
    ph = cube[6]
    zp = cube[7]

    mb = lightcurve_utils.EOSfit(mns,c)

    tmag, lbol, mag = KaKy2016_model(q, chi_eff, mns, mb, c, th, ph)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_KaKy2016(q,chi_eff,mns,mb,c)
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

    tmag, lbol, mag = sn_model(z, 0.0 ,x0,x1,c)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def calc_prob(tmag, lbol, mag, t0, zp):

    if Global.doLuminosity:
        if np.sum(lbol) == 0.0:
            prob = -np.inf
            return prob
        tmag = tmag + t0

        count = 0
        chisquare = np.nan

        t = Global.data_out["tt"]
        y = Global.data_out["Lbol"]
        sigma_y = Global.data_out["Lbol_err"]

        idx = np.where(~np.isnan(y))[0]
        t = t[idx]
        y = y[idx]
        sigma_y = sigma_y[idx]

        ii = np.where(~np.isnan(lbol))[0]
        if len(ii) == 0:
            lbolinterp = np.nan*np.ones(t.shape)
        else:
            f = interp.interp1d(tmag[ii], np.log10(lbol[ii]), fill_value='extrapolate')
            lbolinterp = 10**f(t)

        zp_factor = 10**(zp/-2.5)
        lbolinterp = lbolinterp*zp_factor

        sigma_y = np.abs(sigma_y/(y*np.log(10)))
        sigma = np.sqrt((np.log10(1+Global.errorbudget))**2 + sigma_y**2)
        y = np.log10(y)
        lbolinterp = np.log10(lbolinterp)

        chisquarevals = ((y-lbolinterp)/sigma)**2

        chisquaresum = np.sum(chisquarevals)
        if np.isnan(chisquaresum):
            chisquare = np.nan
            return chisquare

        if not float(len(chisquarevals)-1) == 0:
            chisquaresum = (1/float(len(chisquarevals)-1))*chisquaresum

        chisquare = chisquaresum

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

    elif Global.doLightcurves:
        if len(np.isfinite(lbol)) == 0:
            prob = -np.inf
            return prob

        if np.sum(lbol) == 0.0:
            prob = -np.inf
            return prob
        tmag = tmag + t0

        count = 0
        chisquare = np.nan
        gaussprob = np.nan
        for key in Global.data_out:
            samples = Global.data_out[key]
            t = samples[:,0]
            y = samples[:,1]
            sigma_y = samples[:,2]

            idx = np.where(~np.isnan(y))[0]
            t = t[idx]
            y = y[idx]
            sigma_y = sigma_y[idx]
            if not key in Global.filters: continue

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

            sigma = np.sqrt(Global.errorbudget**2 + sigma_y**2)
            chisquarevals = np.zeros(y.shape)
            chisquarevals = ((y-maginterp)/sigma)**2
            idx = np.where(~np.isfinite(sigma))[0]
            if len(idx) > 0:
                gaussprobvals = 1-scipy.stats.norm.cdf(y[idx], maginterp[idx], Global.errorbudget)
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
    else:
        print "Enable doLuminosity or doLightcurves..."
        exit(0)

def findconst(array):
    idx = np.where(~np.isnan(array))[0]
    if len(idx) == 0:
        return np.nan
    else:
        return array[idx[-1]]

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

