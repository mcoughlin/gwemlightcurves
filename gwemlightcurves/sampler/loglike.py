
import numpy as np
import scipy.stats
from scipy.interpolate import interpolate as interp
from gwemlightcurves import lightcurve_utils, Global
from .model import *

def prior_2Component(Xlan1,Xlan2):
    if Xlan1 < Xlan2:
        return 0.0
    else:
        return 1.0

def prior_2ComponentVel(vej_1,vej_2):
    if vej_1 < vej_2:
        return 1.0
    else:
        return 0.0

def prior_3Component(Xlan1,Xlan2,Xlan3):
    if (Xlan1 > Xlan2) and (Xlan3 > Xlan2):
        return 1.0
    else:
        return 0.0

def prior_3ComponentVel(vej_1,vej_2,vej_3):
    if (vej_1 < vej_2) and (vej_1 < vej_3):
        return 1.0
    else:
        return 0.0

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
    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    print(prob)

    return prob

def myloglike_Ka2017inc_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]  
    Xlan = 10**cube[3]
    iota = cube[4]
    zp = cube[5]

    tmag, lbol, mag = Ka2017inc_model_ejecta(mej,vej,Xlan,iota)
    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    print(prob)

    return prob

def myloglike_Ka2017_A_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    Xlan = 10**cube[3]
    A = 10**cube[4]
    zp = cube[5]

    tmag, lbol, mag = Ka2017_A_model(mej,vej,Xlan,A)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_Ka2017x2_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    zp = cube[7]

    prior = prior_2Component(Xlan_1,Xlan_2)
    if prior == 0.0:
        return -np.inf
    prior = prior_2ComponentVel(vej_1,vej_2)
    #prior = prior_2ComponentVel(vej_2,vej_1)
    if prior == 0.0:
        return -np.inf

    tmag, lbol, mag = Ka2017x2_model_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2)
    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    return prob

def myloglike_Ka2017x2inc_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    iota = cube[7]
    zp = cube[8]

    prior = prior_2Component(Xlan_1,Xlan_2)
    if prior == 0.0:
        return -np.inf
    prior = prior_2ComponentVel(vej_1,vej_2)
    #prior = prior_2ComponentVel(vej_2,vej_1)
    if prior == 0.0:
        return -np.inf

    tmag, lbol, mag = Ka2017x2inc_model_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,iota)
    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    print(prob)

    return prob

def myloglike_Ka2017x3inc_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    mej_3 = 10**cube[7]
    vej_3 = cube[8]
    Xlan_3 = 10**cube[9]
    iota = cube[10]
    zp = cube[11]

    prior = prior_3Component(Xlan_1,Xlan_2,Xlan_3)
    if prior == 0.0:
        return -np.inf
    prior = prior_3ComponentVel(vej_1,vej_2,vej_3)
    #prior = prior_2ComponentVel(vej_2,vej_1)
    if prior == 0.0:
        return -np.inf

    tmag, lbol, mag = Ka2017x3inc_model_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,mej_3,vej_3,Xlan_3,iota)
    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    print(prob)

    return prob

def myloglike_Ka2017x2_ejecta_sigma(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    errorbudget = cube[7]
    zp = cube[8]

    prior = prior_2Component(Xlan_1,Xlan_2)
    if prior == 0.0:
        return -np.inf
    prior = prior_2ComponentVel(vej_1,vej_2)
    #prior = prior_2ComponentVel(vej_2,vej_1)
    if prior == 0.0:
        return -np.inf

    tmag, lbol, mag = Ka2017x2_model_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2)
    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget=errorbudget)

    return prob

def myloglike_Ka2017x3_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    mej_3 = 10**cube[7]
    vej_3 = cube[8]
    Xlan_3 = 10**cube[9]
    zp = cube[10]

    tmag, lbol, mag = Ka2017x3_model_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2,mej_3,vej_3,Xlan_3)
    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_Ka2017_BNSFit(cube, ndim, nparams):

    t0 = cube[0]
    m1 = cube[1]
    c1 = cube[2]
    m2 = cube[3]
    c2 = cube[4]
    Xlan = 10**cube[5]
    zp = cube[6]

    tmag, lbol, mag = Ka2017_model_BNS(m1,c1,m2,c2,Xlan)

    prob = calc_prob(tmag, lbol, mag, t0, zp)
    prior = prior_DiUj2017(m1,m1,c1,m2,m2,c2)
    if prior == 0.0:
        prob = -np.inf

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

    prob = calc_prob(tmag, lbol, mag, t0, zp,errorbudget = Global.errorbudget)
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

    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    print(prob)

    return prob

def myloglike_Me2017_A_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    beta = cube[3]
    kappa_r = 10**cube[4]
    A = 10**cube[5]
    zp = cube[6]

    tmag, lbol, mag = Me2017_A_model(mej,vej,beta,kappa_r,A)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_Me2017x2_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    beta_1 = cube[3]
    kappa_r_1 = 10**cube[4]
    mej_2 = 10**cube[5]
    vej_2 = cube[6]
    beta_2 = cube[7]
    kappa_r_2 = 10**cube[8]
    zp = cube[9]

    tmag, lbol, mag = Me2017x2_model_ejecta(mej_1,vej_1,beta_1,kappa_r_1,mej_2,vej_2,beta_2,kappa_r_2)

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

def myloglike_boxfit(cube, ndim, nparams):
    t0 = cube[0]
    theta_0 = cube[1]
    E = 10**cube[2]
    n = 10**cube[3]
    theta_obs = cube[4]
    p = cube[5]
    epsilon_B = 10**cube[6]
    epsilon_E = 10**cube[7]
    ksi_N = 10**cube[8]
    zp = cube[9]

    if theta_0 > theta_obs:
       prob = -np.inf
    else:
        tmag, lbol, mag = boxfit_model(theta_0, E, n, theta_obs, p, epsilon_B, epsilon_E, ksi_N)

        prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def myloglike_TrPi2018(cube, ndim, nparams):

    t0 = cube[0]
    theta_v = cube[1]
    E0 = 10**cube[2]
    theta_c = cube[3]
    theta_w = cube[4]
    n = 10**cube[5]
    p = cube[6]
    epsilon_E = 10**cube[7]
    epsilon_B = 10**cube[8]
    zp = cube[9]

    tmag, lbol, mag = TrPi2018_model(theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B)

    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)

    return prob

def myloglike_Ka2017_TrPi2018(cube, ndim, nparams):

    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    Xlan = 10**cube[3]
    theta_v = cube[4]
    E0 = 10**cube[5]
    theta_c = cube[6]
    theta_w = cube[7]
    n = 10**cube[8]
    p = cube[9]
    epsilon_E = 10**cube[10]
    epsilon_B = 10**cube[11]
    zp = cube[12]

    tmag, lbol, mag = Ka2017_TrPi2018_model(mej, vej, Xlan, theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B)

    prob = calc_prob(tmag, lbol, mag, t0, zp, errorbudget = Global.errorbudget)
    print(prob)

    return prob

def myloglike_Ka2017_TrPi2018_A(cube, ndim, nparams):

    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    Xlan = 10**cube[3]
    theta_v = cube[4]
    E0 = 10**cube[5]
    theta_c = cube[6]
    theta_w = cube[7]
    n = 10**cube[8]
    p = cube[9]
    epsilon_E = 10**cube[10]
    epsilon_B = 10**cube[11]
    A = 10**cube[12]
    zp = cube[13]

    tmag, lbol, mag = Ka2017_TrPi2018_A_model(mej, vej, Xlan, theta_v, E0, theta_c, theta_w, n, p, epsilon_E, epsilon_B, A)

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    return prob

def calc_prob(tmag, lbol, mag, t0, zp, errorbudget=Global.errorbudget):

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
        sigma = np.sqrt((np.log10(1+errorbudget))**2 + sigma_y**2)
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
        nsamples = 0

        for key in Global.data_out:
            samples = Global.data_out[key]
            t = samples[:,0]
            y = samples[:,1]
            sigma_y = samples[:,2]

            idx = np.where(~np.isnan(y))[0]
            t = t[idx]
            y = y[idx]
            sigma_y = sigma_y[idx]
            if len(idx) == 0: continue
            if not key in Global.filters: continue

            keyslist = ["u","g","r","i","z","y","J","H","K"]

            if key in keyslist:
                idx = keyslist.index(key)
                ii = np.where(np.isfinite(mag[idx]))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    if Global.doWaveformExtrapolate:
                        f = interp.interp1d(tmag[ii], mag[idx][ii], fill_value='extrapolate')
                    else:
                        f = interp.interp1d(tmag[ii], mag[idx][ii], fill_value=np.nan, bounds_error = False)
                    maginterp = f(t)
            elif key in ["w","c","o","V","B","R","I","F606W","F160W","F814W","U","UVW2","UVW1","UVM2"]:
                magave = lightcurve_utils.get_mag(mag,key)

                ii = np.where(np.isfinite(magave))[0]
                if len(ii) == 0:
                    maginterp = np.nan*np.ones(t.shape)
                else:
                    if Global.doWaveformExtrapolate:                    
                        f = interp.interp1d(tmag[ii], magave[ii], fill_value='extrapolate')
                    else:
                        f = interp.interp1d(tmag[ii], magave[ii], fill_value=np.nan, bounds_error = False)
                    maginterp = f(t)
            else:
                continue

            maginterp = maginterp + zp
            sigma = np.sqrt(errorbudget**2 + sigma_y**2)

            chisquarevals = np.zeros(y.shape)
            chisquarevals = ((y-maginterp)/sigma)**2
            idx = np.where(~np.isfinite(sigma))[0]

            if len(idx) > 0:
                gaussprobvals = 1-scipy.stats.norm.cdf(y[idx], maginterp[idx], errorbudget)
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
            nsamples = nsamples + len(y)

        if np.isnan(chisquare):
            prob = -np.inf
        else:
            #prob = scipy.stats.chi2.logpdf(chisquare, count, loc=0, scale=1)
            #prob = -chisquare/2.0
            #prob = chisquare
            if chisquare == 0:
                chiprob = 0 
            else:
                chiprob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)

            prob = chiprob + gaussprob - (nsamples/2.0)*np.log(2.0*np.pi*errorbudget**2)
        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print t0, zp, prob
        return prob
    else:
        print("Enable doLuminosity or doLightcurves...")
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

