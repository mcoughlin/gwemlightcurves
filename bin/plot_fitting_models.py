
import os, sys, glob, copy
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats
import h5py

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 36})
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['text.usetex'] = True

from chainconsumer import ChainConsumer
import corner

import scipy.stats as ss

import pymultinest

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
    parser.add_option("-n","--name",default="1087")
    parser.add_option("-m","--model",default="KaKy2016")
    parser.add_option("--mej",default=0.05,type=float)
    parser.add_option("--vej",default=0.25,type=float)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doBNSFit",  action="store_true", default=False)
    parser.add_option("--doSimulation",  action="store_true", default=False)
    parser.add_option("--doFixMChirp",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("--doJoint",  action="store_true", default=False)
    parser.add_option("--doJointLambda",  action="store_true", default=False)
    parser.add_option("--doJointDisk",  action="store_true", default=False)
    parser.add_option("--doJointDiskBulla",  action="store_true", default=False)
    parser.add_option("--doJointGRB",  action="store_true", default=False)
    parser.add_option("--doJointDiskFoucart",  action="store_true", default=False)
    parser.add_option("--doJointBNS",  action="store_true", default=False)
    parser.add_option("--doJointNSBH",  action="store_true", default=False)
    parser.add_option("--doJointSpin",  action="store_true", default=False)
    parser.add_option("--doJointWang",  action="store_true", default=False)
    parser.add_option("--doLoveC",  action="store_true", default=False)
    parser.add_option("--doLightcurves",  action="store_true", default=False)
    parser.add_option("--doLuminosity",  action="store_true", default=False)
    parser.add_option("--doFixedLimit",  action="store_true", default=False)
    parser.add_option("-f","--filters",default="g,r,i,z")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)
    parser.add_option("--lambdamax",default=400.0,type=float)
    parser.add_option("--lambdamin",default=800.0,type=float)

    parser.add_option("--colormodel",default="a2.0")

    opts, args = parser.parse_args()

    return opts

def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2, :]
    den_pts = pts[Npts/2:, :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir

def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2]
    den_pts = pts[Npts/2:]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

def kde_eval_single(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    td = kde(truth)

    return td

def bhns_model(q,chi_eff,mns,mb,c):

    meje = KaKy2016.calc_meje(q,chi_eff,c,mb,mns)
    vave = KaKy2016.calc_vave(q)
  
    return meje, vave

def bns_model(m1,mb1,c1,m2,mb2,c2):

    mej = DiUj2017.calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = DiUj2017.calc_vej(m1,c1,m2,c2)

    return mej, vej

def bns2_model(m1,c1,m2,c2):

    #mej = DiUj2017.calc_meje(m1,m1,c1,m2,m2,c2)
    #vej = DiUj2017.calc_vej(m1,c1,m2,c2)

    mej = Di2018b.calc_meje(m1,c1,m2,c2)
    vej = Di2018b.calc_vej(m1,c1,m2,c2)

    return mej, vej

def myprior_bhns(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        cube[1] = cube[1]*0.75
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.1 + 0.1

def myprior_bns(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*0.16 + 0.08

def myprior_bhns_EOSFit(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        cube[1] = cube[1]*0.75
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.1 + 0.1

def myprior_bns_EOSFit(cube, ndim, nparams):
        cube[0] = cube[0]*2.5 + 0.5
        cube[1] = cube[1]*0.16 + 0.08
        cube[2] = cube[2]*2.5 + 0.5
        cube[3] = cube[3]*0.16 + 0.08

def myprior_bns_BNSFit(cube, ndim, nparams):
        #cube[0] = cube[0]*2.0 + 1.0
        cube[0] = cube[0]*2.5 + 0.5
        cube[1] = cube[1]*0.16 + 0.08
        #cube[2] = cube[2]*2.0 + 1.0
        cube[2] = cube[2]*2.5 + 0.5
        cube[3] = cube[3]*0.16 + 0.08

def myprior_bns_JointFit(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*4900.0 + 10.0
        #cube[2] = cube[2]*5.0
        cube[2] = cube[2]*100.0

def myprior_bns_JointFitLambda(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*4900.0 + 10.0
        #cube[2] = cube[2]*5.0
        cube[2] = cube[2]*100.0
        cube[3] = cube[3]*5.0

def myprior_bns_JointFitDisk(cube, ndim, nparams):
        cube[0] = cube[0]*(mchirpmax-mchirpmin) + mchirpmin
        cube[1] = cube[1]*1.0 + 1.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[2] = cube[2]*5000
        cube[3] = cube[3]*2.0 - 2.0
        cube[4] = cube[4]*0.5

def myprior_nsbh_JointFitDisk_sim(cube, ndim, nparams):
        cube[0] = cube[0]*0.002 + 2.0 - 0.001
        cube[1] = cube[1]*1.0 + 2.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[2] = cube[2]*5000
        cube[3] = cube[3]*1.0
        cube[4] = cube[4]*0.001 + 0.15

def myprior_bns_JointFitDisk_sim(cube, ndim, nparams):
        cube[0] = cube[0]*0.002 + 1.186 - 0.001
        cube[1] = cube[1]*1.0 + 1.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[2] = cube[2]*5000
        cube[3] = cube[3]*0.001 - 1.0
        cube[4] = cube[4]*0.001 + 0.15

def myprior_nsbh_JointFitDisk(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 1.0
        cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*0.5
        cube[4] = cube[4]*2.0 - 1.0
        #cube[4] = cube[4]*4.0

def myprior_bns_Lambda2(cube, ndim, nparams):
        #cube[0] = cube[0]*1.0 + 1.0
        cube[0] = cube[0]*6.2 + 0.9
        cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[2] = cube[2]*2.0 - 2.0
        cube[3] = cube[3]*0.5
        cube[4] = cube[4]*0.17 + 2.0

def myprior_GRB(cube, ndim, nparams):
        cube[0] = cube[0]*20.0 - 20.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[1] = cube[1]*(100+lambdamax-lambdamin) + lambdamin-50
        cube[2] = cube[2]*1.0
        #cube[3] = cube[3]*1.0 + 1.0
        cube[3] = cube[3]*1.1 + 0.9
        #cube[4] = cube[4]*0.17 + 2.0
        cube[4] = cube[4]*0.27 + 1.95
        #cube[5] = cube[5]*0.5
        cube[5] = cube[5]*0.6     

def myprior_SGRB(cube, ndim, nparams):
        cube[0] = cube[0]*20.0 - 20.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[1] = cube[1]*(100+lambdamax-lambdamin) + lambdamin-50
        cube[2] = cube[2]*1.0
        #cube[3] = cube[3]*1.0 + 1.0
        cube[3] = cube[3]*1.1 + 0.9
        #cube[4] = cube[4]*0.17 + 2.0
        cube[4] = cube[4]*0.27 + 1.95
        #cube[4] = cube[4]*4.0
        #cube[5] = cube[5]*0.5
        cube[5] = cube[5]*0.6
        cube[6] = cube[6]*(mchirpmax-mchirpmin) + mchirpmin

def myprior_Wang(cube, ndim, nparams):
        #cube[0] = cube[0]*1.0 + 1.0
        cube[0] = cube[0]*1.1 + 0.9
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[1] = cube[1]*(100+lambdamax-lambdamin) + lambdamin-50
        #cube[2] = cube[2]*0.17 + 2.0
        cube[2] = cube[2]*0.27 + 1.95
        cube[3] = cube[3]*3.0 - 4.0
        #cube[4] = cube[4]*0.5
        cube[4] = cube[4]*0.6

def myprior_BNS_Lambda2GRB(cube, ndim, nparams):
        cube[0] = cube[0]*20.0 - 20.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[1] = cube[1]*(100+lambdamax-lambdamin) + lambdamin-50
        cube[2] = cube[2]*1.0
        #cube[3] = cube[3]*1.0 + 1.0
        #cube[3] = cube[3]*1.1 + 0.9
        cube[3] = cube[3]*6.2 + 0.9        
        #cube[4] = cube[4]*0.17 + 2.0
        cube[4] = cube[4]*0.27 + 1.95
        #cube[5] = cube[5]*0.5
        cube[5] = cube[5]*0.6

def myprior_NSBH(cube, ndim, nparams):
        cube[0] = cube[0]*20.0 - 20.0
        #cube[1] = cube[1]*(lambdamax-lambdamin) + lambdamin
        cube[1] = cube[1]*(100+lambdamax-lambdamin) + lambdamin-50
        cube[2] = cube[2]*1.0
        #cube[3] = cube[3]*1.0 + 1.0
        cube[3] = cube[3]*6.2 + 0.9
        cube[4] = cube[4]*2.0 - 1.0
        #cube[4] = cube[4]*0.27 + 1.95
        #cube[4] = cube[4]*4.0
        #cube[5] = cube[5]*0.5
        cube[5] = cube[5]*0.6

def myprior_combined_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*5.0 - 5.0
        cube[1] = cube[1]*1.0

def myprior_combined_masses_bns(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*2.0 + 0.0

def myprior_combined_masses_bhns(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        cube[1] = cube[1]*10.0 + 0.0

def myprior_combined_q_lambdatilde(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*4900.0 + 10.0

def prior_gw(mc, q, lambdatilde, eos):

        kdeeval_mchirp = kde_eval_single(kdedir_gwmc,[mc])[0]
        kdeeval_q = kde_eval_single(kdedir_gwq,[q])[0]
        #kdeeval_lambdatilde = kde_eval_single(kdedir_gwlambdatilde,[lambdatilde])[0]
        kdeeval_lambdatilde = 1.0
        kdeeval_eos = kdedir_gweos[eos]

        return kdeeval_mchirp*kdeeval_q*kdeeval_lambdatilde*kdeeval_eos

def prior_bns(m1,mb1,c1,m2,mb2,c2):
        if m1 < m2:
            return 0.0
        else:
            return 1.0

def prior_bhns(q,chi_eff,mns,mb,c):
        return 1.0

def myloglike_bns(cube, ndim, nparams):
        m1 = cube[0]
        mb1 = cube[1]
        c1 = cube[2]
        m2 = cube[3]
        mb2 = cube[4]
        c2 = cube[5]

        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob(mej, vej)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_gw(cube, ndim, nparams):
        m1 = cube[0]
        mb1 = cube[1]
        c1 = cube[2]
        m2 = cube[3]
        mb2 = cube[4]
        c2 = cube[5]

        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob_gw(m1,m2)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns2(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mej, vej = bns2_model(m1,c1,m2,c2)

        prob = calc_prob(mej, vej)
        prior = prior_bns(m1,m1,c1,m2,m2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns2_gw(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mej, vej = bns2_model(m1,c1,m2,c2)

        prob = calc_prob_gw(m1,m2)
        prior = prior_bns(m1,m1,c1,m2,m2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_EOSFit(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = lightcurve_utils.EOSfit(m1,c1)
        mb2 = lightcurve_utils.EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob(mej, vej)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_EOSFit_FixMChirp(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = lightcurve_utils.EOSfit(m1,c1)
        mb2 = lightcurve_utils.EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob1 = calc_prob(mej, vej)
        prob2 = calc_prob_mchirp(m1, m2)
        prob = prob1+prob2
        prior = prior_bns(m1,m1,c1,m2,m2,c2)

        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_gw_EOSFit(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = lightcurve_utils.EOSfit(m1,c1)
        mb2 = lightcurve_utils.EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob_gw(m1, m2)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_gw_BNSFit(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mej, vej = bns2_model(m1,c1,m2,c2)

        prob = calc_prob_gw(m1, m2)
        prior = prior_bns(m1,m1,c1,m2,m2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_JointFit(cube, ndim, nparams):
        q = cube[0]
        lambda1 = cube[1]
        A = cube[2]

        lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])

        c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
        c2 = c1/q
        coeff = lambda_coeff[::-1]
        p = np.poly1d(coeff)
        lambda2 = p(c2)

        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
        if (lambdatilde<lambdamin) or (lambdatilde>lambdamax):
            prob = -np.inf
            return prob

        mc = 1.186
        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        mej, vej = bns2_model(m1,c1,m2,c2)
        mej = mej*A

        if mej>(m1+m2):
            prob = -np.inf
            return prob

        prob = calc_prob(mej,vej)
        prior = prior_bns(m1,m1,c1,m2,m2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_JointFitLambda(cube, ndim, nparams):
        q = cube[0]
        lambda1 = cube[1]
        A = cube[2]
        mc = cube[3]

        lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])

        c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)  
        c2 = c1/q
        coeff = lambda_coeff[::-1]
        p = np.poly1d(coeff)
        lambda2 = p(c2)

        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
        if (lambdatilde<lambdamin) or (lambdatilde>lambdamax):
            prob = -np.inf
            return prob

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        if (m1+m2>2.9) or (m1+m2<1.8):
            prob = -np.inf
            return prob

        mej, vej = bns2_model(m1,c1,m2,c2)
        mej = mej*A

        if mej>(m1+m2):
            prob = -np.inf
            return prob

        prob = calc_prob(mej,vej)
        prior = prior_bns(m1,m1,c1,m2,m2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_Lambda2(cube, ndim, nparams):
        q = cube[0]
        lambda2 = cube[1]
        alpha = cube[2]
        zeta = cube[3]
        mTOV = cube[4]
        mc = 1.186

        lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])

        c2 = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)
        c1 = c2*q
        coeff = lambda_coeff[::-1]
        p = np.poly1d(coeff)
        lambda1 = p(c1)

        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        mej1, vej1 = bns2_model(m1,c1,m2,c2)
        mej1 = mej1/(10**alpha)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0):
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        if (mej1+mej2)>(m1+m2):
            prob = -np.inf
            return prob

        #prob = calc_prob(mej1,vej1)
        prob = calc_prob_disk(mej1,vej1,mej2)
        #prob = calc_prob_mej2(mej2)
        prior = prior_bns(m1,m1,c1,m2,m2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej1 == 0.0:
            prob = -np.inf
        if mej2 == 0.0:
            prob = -np.inf

        print(prob)

        return prob

def myloglike_bhns_JointFitDiskFoucart_sim(cube, ndim, nparams):
        mc = cube[0]
        q = cube[1]
        eos = int(np.floor(cube[2])+1)
        chi_eff = cube[3]
        zeta = cube[4]

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        f = eosdata[str(eos)]["f"]
        lambda2 = f(m2)
        c2 = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)

        mej = KrFo2019.calc_meje(q,chi_eff,c2,m2,f=zeta)

        prob = kde_eval_single(kdedir_mej,[mej])[0]

        print(mej, prob)

        return prob

def myloglike_bns_JointFitDiskBulla_sim(cube, ndim, nparams):
        mc = cube[0]
        q = cube[1]
        eos = int(np.floor(cube[2])+1)
        alpha = cube[3]
        zeta = cube[4]

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        f = eosdata[str(eos)]["f"]
        lambda1, lambda2 = f(m1), f(m2)
        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
        mTOV = eosdata[str(eos)]["mTOV"]

        if (mTOV < 1.96) or (mTOV > 2.32):
            prob = -np.inf
            return prob

        c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
        c2 = c1/q

        vej1 = CoDi2019.calc_vej(m1,c1,m2,c2)

        a= -0.0719
        b= 0.2116
        d= -2.42
        n= -2.905

        log10_mej = a*(m1*(1-2*c1)/c1 + m2*(1-2*c2)/c2) + b*(m1*(m2/m1)**n + m2*(m1/m2)**n)+d
        mej1 = 10**log10_mej
        mej1 = mej1/(10**alpha)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0):
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        print(mdisk, mej2)

        a0 = -1.581
        da = -2.439
        b0 = -0.538
        db = -0.406
        c = 0.953
        d = 0.0417
        beta = 3.910
        qtrans = 0.900

        eta = 0.5 * np.tanh(beta*(q-qtrans))
        a = a0 + da * eta
        b = b0 + db * eta

        mdisk = a*(1+b*np.tanh((c-mtot/mth)/d))

        mdisk[mdisk<-3] = -3.0
        mdisk[rat>0.32] = -3.0
        mdisk = 10**mdisk
 
        if (mej1+mej2)>(m1+m2):
            prob = -np.inf
            return prob

        prob = kde_eval_single(kdedir_mej,[mej1+mej2])[0]

        if prob < 1e-100:
            prob = -np.inf

        if prior == 0.0:
            prob = -np.inf
        if mej1 == 0.0:
            prob = -np.inf
        if mej2 == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_JointFitDiskBulla(cube, ndim, nparams):
        mc = cube[0]
        q = cube[1]
        eos = int(np.floor(cube[2])+1)
        alpha = cube[3]
        zeta = cube[4]

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        f = eosdata[str(eos)]["f"]
        lambda1, lambda2 = f(m1), f(m2)
        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
        mTOV = eosdata[str(eos)]["mTOV"]

        if (mTOV < 1.96) or (mTOV > 2.32):
            prob = -np.inf
            return prob

        c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
        c2 = c1/q

        vej1 = CoDi2019.calc_vej(m1,c1,m2,c2)

        a= -0.0719
        b= 0.2116
        d= -2.42
        n= -2.905

        log10_mej = a*(m1*(1-2*c1)/c1 + m2*(1-2*c2)/c2) + b*(m1*(m2/m1)**n + m2*(m1/m2)**n)+d
        mej1 = 10**log10_mej
        mej1 = mej1/(10**alpha)

        print(mej1)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0):
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        if (mej1+mej2)>(m1+m2):
            prob = -np.inf
            return prob

        prob = kde_eval_single(kdedir_mej,[mej1+mej2])[0]
        prior = prior_gw(mc, q, lambdatilde, eos)
        
        prob = np.log(prob) + np.log(prior)

        if prior == 0.0:
            prob = -np.inf
        if mej1 == 0.0:
            prob = -np.inf
        if mej2 == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_JointFitDiskLimit(cube, ndim, nparams):
        mc = cube[0]
        q = cube[1]
        eos = int(np.floor(cube[2])+1)
        alpha = cube[3]
        zeta = cube[4]

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        f = eosdata[str(eos)]["f"]
        lambda1, lambda2 = f(m1), f(m2)
        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
        mTOV = eosdata[str(eos)]["mTOV"]

        #if (mTOV < 1.96) or (mTOV > 2.32):
        #    prob = -np.inf
        #    return prob

        c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
        c2 = c1/q

        vej1 = CoDi2019.calc_vej(m1,c1,m2,c2)

        a= -0.0719
        b= 0.2116
        d= -2.42
        n= -2.905

        log10_mej = a*(m1*(1-2*c1)/c1 + m2*(1-2*c2)/c2) + b*(m1*(m2/m1)**n + m2*(m1/m2)**n)+d
        mej1 = 10**log10_mej
        mej1 = mej1/(10**alpha)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0):
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        if (mej1+mej2)>(m1+m2):
            prob = -np.inf
            return prob

        if mej1*(10**alpha)+mej2+(10**alpha) <= mejsum_95:
            prob = 1.0
        else:
            prob = -np.inf
        #prob = kde_eval_single(kdedir_mej,[mej1+mej2])[0]
        #prob = calc_prob_disk(mej1,vej1,mej2)
        prior = prior_gw(mc, q, lambdatilde, eos)

        prob = prob + np.log(prior)

        if prior == 0.0:
            prob = -np.inf
        if mej1 == 0.0:
            prob = -np.inf
        if mej2 == 0.0:
            prob = -np.inf

        if np.isfinite(prob):
            print(mc, q, eos, prob)

        return prob

def myloglike_bns_JointFitDisk(cube, ndim, nparams):
        mc = cube[0]
        q = cube[1]
        eos = int(np.floor(cube[2])+1)
        alpha = cube[3]
        zeta = cube[4]

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        f = eosdata[str(eos)]["f"]
        lambda1, lambda2 = f(m1), f(m2)
        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)  
        mTOV = eosdata[str(eos)]["mTOV"]

        #if (mTOV < 1.96) or (mTOV > 2.32):
        #    prob = -np.inf
        #    return prob
 
        c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
        c2 = c1/q

        vej1 = CoDi2019.calc_vej(m1,c1,m2,c2)

        a= -0.0719
        b= 0.2116
        d= -2.42
        n= -2.905

        log10_mej = a*(m1*(1-2*c1)/c1 + m2*(1-2*c2)/c2) + b*(m1*(m2/m1)**n + m2*(m1/m2)**n)+d
        mej1 = 10**log10_mej
        mej1 = mej1/(10**alpha)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0):
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV
        
        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2

        a0 = -1.581
        da = -2.439
        b0 = -0.538
        db = -0.406
        c = 0.953
        d = 0.0417
        beta = 3.910
        qtrans = 0.900

        eta = 0.5 * np.tanh(beta*(q-qtrans))
        a = a0 + da * eta
        b = b0 + db * eta

        mdisk = a*(1+b*np.tanh((c-mtot/mth)/d))
        if mdisk<-3:
            mdisk = -3.0
        if rat>0.32:
            mdisk = -3.0
        mdisk = 10**mdisk
        mej2 = zeta*mdisk

        if (mej1+mej2)>(m1+m2):
            prob = -np.inf
            return prob

        prob = kde_eval_single(kdedir_mej,[mej1*(10**alpha)+mej2+(10**alpha)])[0]
        #prob = kde_eval_single(kdedir_mej,[mej1+mej2])[0]
        #prob = calc_prob_disk(mej1,vej1,mej2)
        prior = prior_gw(mc, q, lambdatilde, eos)

        prob = prob + np.log(prior)

        if prior == 0.0:
            prob = -np.inf
        if mej1 == 0.0:
            prob = -np.inf
        if mej2 == 0.0:
            prob = -np.inf

        return prob

def myloglike_nsbh_JointFitDisk(cube, ndim, nparams):
        q = cube[0]
        lambda2 = cube[1]
        alpha = cube[2]
        zeta = cube[3]
        chi_eff = cube[4]
        mc = 1.186

        c = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        if (m2<0.89):
            prob = -np.inf
            return prob

        a, n = 0.8858, 1.2082
        mb = (1+a*c**n)*m2

        mej1, vej1 = bhns_model(q,chi_eff,m2,mb,c) 
        mej1 = mej1*alpha

        Z1 = 1 + ((1-chi_eff**2)**(1.0/3.0)) * ((1+chi_eff)**(1.0/3.0) + (1-chi_eff)**(1.0/3.0))
        Z2 = np.sqrt(3.0*chi_eff**2 + Z1**2)
        Risco = 3+Z2-np.sign(chi_eff)*np.sqrt((3-Z1)*(3+Z1+2*Z2))

        alpha_fit, beta_fit, gamma_fit, delta_fit = 0.406, 0.139, 0.255, 1.761
        term1 = alpha_fit*(1-2*c)/(eta**(1.0/3.0))
        term2 = beta_fit*Risco*c/eta

        mdisk = (np.max([term1-term2+gamma_fit,0]))**delta_fit
        mtot = m1+m2
        mej2 = zeta*mdisk

        if np.isnan(mej2):
            prob = -np.inf
            return prob

        if (mej1+mej2)>(m2):
            prob = -np.inf
            return prob

        #prob = calc_prob(mej1,vej1)
        prob = calc_prob_disk(mej1,vej1,mej2)
        #prob = calc_prob_mej2(mej2)
        if np.isnan(prob):
            prob = -np.inf
            return prob

        print(mej1, mej2, prob)

        if mej1 == 0.0:
            prob = -np.inf
        if mej2 == 0.0:
            prob = -np.inf

        return prob

def myloglike_GRB(cube, ndim, nparams):
        epsilon = 10**cube[0]
        lambdatilde = cube[1]
        E0_unit = cube[2]
        q = cube[3]
        mTOV = cube[4]
        zeta = cube[5]
        mc = 1.186

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0): 
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        E0_mu, E0_std = -0.81, 0.39
        E0 = scipy.stats.norm(E0_mu, E0_std).ppf(E0_unit)
        Eiso = 1e50 * 10**E0 

        menergy = 1.989*1e30*9.0*1e16*1e7*mdisk

        Eiso_estimate = epsilon * menergy * (1-zeta)
        Eiso_sigma = 0.67*Eiso_estimate

        prob = - 0.5 * (((Eiso_estimate - Eiso) ** 2) / Eiso_sigma ** 2)

        prob2 = calc_prob_KN(q, lambdatilde, zeta, mTOV)
        prob = prob + prob2

        return prob

def myloglike_Wang(cube, ndim, nparams):
        q = cube[0]
        lambdatilde = cube[1]
        mTOV = cube[2]
        sigma = cube[3]
        zeta = cube[4]
        mc = 1.186

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        a = 0.537 
        b = -0.185
        c = -0.514

        xi = np.tanh(a*(eta**2)*((m1+m2) + b*lambdatilde)+c)

        if np.abs(xi)>1:
            prob = -np.inf
            return prob

        mdisk_wang = (10**sigma)*((1+np.sqrt(1-xi**2))/xi)**2

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0): 
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk_dietrich = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mdisk_dietrich = mdisk_dietrich*(1-zeta)

        mdisk_sigma = np.max([mdisk_wang,mdisk_dietrich])*0.64

        prob = - 0.5 * (((mdisk_wang - mdisk_dietrich) ** 2) / mdisk_sigma ** 2)

        prob2 = calc_prob_KN(q, lambdatilde, zeta, mTOV)

        prob = prob + prob2

        return prob

def myloglike_SGRB(cube, ndim, nparams):
        epsilon = 10**cube[0]
        lambdatilde = cube[1]
        E0_unit = cube[2]
        q = cube[3]
        mTOV = cube[4]
        zeta = cube[5]
        mc = cube[6]

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0): 
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        E0_mu, E0_std = 50.30, 0.84
        E0 = scipy.stats.norm(E0_mu, E0_std).ppf(E0_unit)
        Eiso = 10**E0

        menergy = 1.989*1e30*9.0*1e16*1e7*mdisk

        Eiso_estimate = epsilon * menergy * (1-zeta)
        Eiso_sigma = 0.67*Eiso_estimate

        prob = - 0.5 * (((Eiso_estimate - Eiso) ** 2) / Eiso_sigma ** 2)

        prob2 = calc_prob_KN(mc, q, lambdatilde, zeta, mTOV)
 
        prob = prob + prob2

        return prob

def myloglike_BNS_Lambda2GRB(cube, ndim, nparams):
        epsilon = 10**cube[0]
        lambda2 = cube[1]
        E0_unit = cube[2]
        q = cube[3]
        mTOV = cube[4]
        zeta = cube[5]
        mc = 1.186

        lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])

        c2 = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)
        c1 = c2*q
        coeff = lambda_coeff[::-1]
        p = np.poly1d(coeff)
        lambda1 = p(c1)

        lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        if (m1)>(mTOV):
            prob = -np.inf
            return prob

        if (lambdatilde<0):
            prob = -np.inf
            return prob

        R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
        rat = mTOV/R16
        if (rat>0.32):
            prob = -np.inf
            return prob
        mth = (2.38 - 3.606*mTOV/R16)*mTOV

        a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

        x = lambdatilde*1.0
        mtot = m1+m2
        mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
        mej2 = zeta*mdisk

        E0_mu, E0_std = 50.30, 0.84
        E0 = scipy.stats.norm(E0_mu, E0_std).ppf(E0_unit)
        Eiso = 10**E0

        menergy = 1.989*1e30*9.0*1e16*1e7*mdisk

        Eiso_estimate = epsilon * menergy * (1-zeta)
        Eiso_sigma = 0.67*Eiso_estimate

        prob = - 0.5 * (((Eiso_estimate - Eiso) ** 2) / Eiso_sigma ** 2)

        prob2 = calc_prob_KN(q, lambdatilde, zeta, mTOV)

        prob = prob + prob2

        return prob

def myloglike_NSBH(cube, ndim, nparams):
        epsilon = 10**cube[0]
        lambda2 = cube[1]
        E0_unit = cube[2]
        q = cube[3]
        chi_eff = cube[4]
        zeta = cube[5]
        mc = 1.186

        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        if (m2<0.89):
            prob = -np.inf
            return prob            

        c = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)

        Z1 = 1 + ((1-chi_eff**2)**(1.0/3.0)) * ((1+chi_eff)**(1.0/3.0) + (1-chi_eff)**(1.0/3.0))
        Z2 = np.sqrt(3.0*chi_eff**2 + Z1**2)
        Risco = 3+Z2-np.sign(chi_eff)*np.sqrt((3-Z1)*(3+Z1+2*Z2))

        alpha_fit, beta_fit, gamma_fit, delta_fit = 0.406, 0.139, 0.255, 1.761
        term1 = alpha_fit*(1-2*c)/(eta**(1.0/3.0))
        term2 = beta_fit*Risco*c/eta

        mdisk = (np.max([term1-term2+gamma_fit,0]))**delta_fit

        E0_mu, E0_std = 50.30, 0.84
        E0 = scipy.stats.norm(E0_mu, E0_std).ppf(E0_unit)
        Eiso = 10**E0

        menergy = 1.989*1e30*9.0*1e16*1e7*mdisk

        Eiso_estimate = epsilon * menergy * (1-zeta)
        Eiso_sigma = 0.67*Eiso_estimate

        prob = - 0.5 * (((Eiso_estimate - Eiso) ** 2) / Eiso_sigma ** 2)

        prob2 = calc_prob_KN_NSBH(q, lambda2, zeta, chi_eff)

        print(zeta, chi_eff, prob, prob2)

        #prob = prob + prob2

        if np.isnan(prob):
            prob = -np.inf
            return prob

        #print(epsilon,lambdatilde,Eiso,q,chi_eff,zeta,prob)

        return prob

def myloglike_bhns(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        mb = cube[3]
        c = cube[4]

        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob = calc_prob(mej, vej)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_EOSFit(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        c = cube[3]

        mb = lightcurve_utils.EOSfit(mns,c)
        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob = calc_prob(mej, vej)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_EOSFit_FixMChirp(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        c = cube[3]

        mb = lightcurve_utils.EOSfit(mns,c)
        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob1 = calc_prob(mej, vej)
        prob2 = calc_prob_mchirp(q*mns, mns)
        prob = prob1+prob2

        prior = prior_bhns(q,chi_eff,mns,mb,c)

        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_gw_EOSFit(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        c = cube[3]

        mb = lightcurve_utils.EOSfit(mns,c)
        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob = calc_prob_gw(q*mns, mns)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
 
        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_combined(cube, ndim, nparams):
        var1 = cube[0]
        var2 = cube[1]
        vals = np.array([var1,var2]).T

        kdeeval_gw = kde_eval(kdedir_gw,vals)[0]
        prob_gw = np.log(kdeeval_gw)
        kdeeval_em = kde_eval(kdedir_em,vals)[0]
        prob_em = np.log(kdeeval_em)
        prob = prob_gw + prob_em

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myloglike_combined_MChirp(cube, ndim, nparams):
        var1 = cube[0]
        var2 = cube[1]
        vals = np.array([var1,var2]).T

        eta = lightcurve_utils.q2eta(var1)
        m1, m2 = lightcurve_utils.mc2ms(var2, eta)
        prob_gw = calc_prob_mchirp(m1, m2)

        kdeeval_em = kde_eval(kdedir_em,vals)[0]
        prob_em = np.log(kdeeval_em)
        prob = prob_gw + prob_em

        if np.isnan(prob):
            prob = -np.inf

        return prob

def calc_prob(mej, vej): 

        if (mej==0.0) or (vej==0.0):
            prob = np.nan
        else:
            vals = np.array([mej,vej]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def calc_prob_mej2(mej2):

        if (mej2==0.0):
            prob = np.nan
        else:
            vals = np.array([mej2]).T
            kdeeval = kde_eval_single(kdedir_pts_mej2,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def calc_prob_disk(mej1, vej1, mej2):

        if (mej1==0.0) or (vej1==0.0) or (mej2==0.0):
            prob = np.nan
        else:
            vals = np.array([mej1,vej1,mej2]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def calc_prob_KN(mc, q, lambdatilde, zeta, mtov):

        if (mc==0.0) or (q==0.0) or (lambdatilde==0.0) or (zeta==0.0) or (mtov==0.0):
            prob = np.nan
        else:
            vals = np.array([mc, q, lambdatilde, zeta, mtov]).T
            kdeeval = kde_eval(kdedir_pts_KN,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def calc_prob_KN_NSBH(q, lambda2, zeta, chi_eff):

        if (q==0.0) or (lambda2==0.0) or (zeta==0.0) or (chi_eff==0.0):
            prob = np.nan
        else:
            vals = np.array([q, lambda2, zeta, chi_eff]).T
            kdeeval = kde_eval(kdedir_pts_KN,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def calc_prob_gw(m1, m2):

        if (m1==0.0) or (m2==0.0):
            prob = np.nan
        else:
            vals = np.array([m1,m2]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def Gaussian(x, mu, sigma):
    return (1.0/np.sqrt(2.0*np.pi*sigma*sigma))*np.exp(-(x-mu)*(x-mu)/2.0/sigma/sigma)

def calc_prob_mchirp(m1, m2):

        if (m1==0.0) or (m2==0.0):
            prob = np.nan
        else:
            mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
            prob = np.log(Gaussian(mchirp, mchirp_mu, mchirp_sigma))

        if (mchirp < mchirp_mu-2*mchirp_sigma) or (mchirp > mchirp_mu+2*mchirp_sigma):
            prob = np.nan 

        if np.isnan(prob):
            prob = -np.inf

        return prob

# Parse command line
opts = parse_commandline()
lambdamin = opts.lambdamin
lambdamax = opts.lambdamax

if not opts.model in ["KaKy2016", "DiUj2017", "Me2017", "SmCh2017","Ka2017","Ka2017x2","Ka2017_TrPi2018","Bu2019inc","Ka2017x2inc","Bu2019lr","Bu2019lf","Bu2019lm","Bu2019lw"]:
   print("Model must be either: KaKy2016, DiUj2017, Me2017, SmCh2017, Ka2017, Ka2017x2, Ka2017_TrPi2018, Bu2019inc, Ka2017x2inc, Bu2019lr, Bu2019lf, Bu2019lm, Bu2019lw")
   exit(0)

filters = opts.filters.split(",")
colormodel = opts.colormodel.split(",")
if len(colormodel) == 1:
    colormodel = colormodel[0]

baseplotDir = opts.plotDir
if opts.doLightcurves:
    if opts.doModels:
        basename = 'fitting_models'
    elif opts.doGoingTheDistance:
        basename = 'fitting_going-the-distance'
    elif opts.doMassGap:
        basename = 'fitting_massgap'
    elif opts.doFixedLimit:
        basename = 'fitting_limits'
    elif opts.doEvent:
        basename = 'fitting_gws'
    elif opts.doSimulation:
        basename = 'fitting'
    else:
        print("Need to enable --doModels, --doEvent, --doSimulation, --doMassGap, or --doGoingTheDistance")
        exit(0)
elif opts.doLuminosity:
    if opts.doModels:
        basename = 'fit_luminosity'
    elif opts.doEvent:
        basename = 'fit_gws_luminosity'
    else:
        print("Need to enable --doModels, --doEvent, --doSimulation, --doMassGap, or --doGoingTheDistance")
        exit(0)
else:
    print("Need to enable --doLightcurves or --doLuminosity")
    exit(0)

plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_EOSFit_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
elif opts.doBNSFit:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_BNSFit_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s_BNSFit'%opts.model)
elif opts.doSimulation:
    if opts.doJointDiskFoucart:
        plotDir = os.path.join(plotDir,'nsbh')
    elif opts.doJointDiskBulla:
        plotDir = os.path.join(plotDir,'bns')
else:
    if opts.doFixZPT0:
        plotDir = os.path.join(plotDir,'%s_FixZPT0'%opts.model)
    else:
        plotDir = os.path.join(plotDir,'%s'%opts.model) 

if opts.model in ["Ka2017inc","Ka2017x2inc","Ka2017x3inc"]:
    plotDir = os.path.join(plotDir,'%s'%("_".join(colormodel)))

if opts.doModels:
    plotDir = os.path.join(plotDir,"_".join(filters))
    plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    plotDir = os.path.join(plotDir,opts.name)
    plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
    dataDir = plotDir.replace("fitting_models","models").replace("_EOSFit","").replace("_BNSFit","")
elif opts.doSimulation:
    plotDir = os.path.join(plotDir,'M%03d'%(opts.mej*1000))
    plotDir = os.path.join(plotDir,"%.3f"%(opts.errorbudget*100.0))
elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.doLightcurves:
        plotDir = os.path.join(plotDir,"_".join(filters))
    plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    elif opts.doJoint:
        plotDir = os.path.join(plotDir,'joint')
    elif opts.doJointLambda:
        plotDir = os.path.join(plotDir,'joinl')
    elif opts.doJointDisk:
        plotDir = os.path.join(plotDir,'joind')
    elif opts.doJointDiskBulla or opts.doJointDiskFoucart:
        plotDir = os.path.join(plotDir,'joinu')
    elif opts.doJointNSBH:
        plotDir = os.path.join(plotDir,'joinn')
    elif opts.doJointBNS:
        plotDir = os.path.join(plotDir,'joinb')
    elif opts.doJointGRB:
        plotDir = os.path.join(plotDir,'joing')
    elif opts.doJointSpin:
        plotDir = os.path.join(plotDir,'joins')
    elif opts.doJointWang:
        plotDir = os.path.join(plotDir,'joinw')
    plotDir = os.path.join(plotDir,opts.name)
    #dataDir = plotDir.replace("fitting_","").replace("_EOSFit","")
    dataDir = plotDir.replace("fitting_","").replace("fit_","")
    if opts.doEjecta:
        dataDir = dataDir.replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJoint:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joint/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joint","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointLambda:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joinl/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joinl","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointDisk:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joind/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            if opts.doFixedLimit:
                dataDir = dataDir.replace("joind","").replace("_EOSFit","").replace("_BNSFit","")
            else:
                dataDir = dataDir.replace("joind","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointDiskBulla or opts.doJointDiskFoucart:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joinu/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joinu","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointNSBH:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joinn/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joinn","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointBNS:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joinb/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joinb","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointGRB:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joing/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joing","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointSpin:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joins/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joins","ejecta").replace("_EOSFit","").replace("_BNSFit","")
    elif opts.doJointWang:
        if opts.model == "Ka2017_TrPi2018":
            dataDir = dataDir.replace("joinw/","").replace("_EOSFit","").replace("_BNSFit","")
        else:
            dataDir = dataDir.replace("joinw","ejecta").replace("_EOSFit","").replace("_BNSFit","")

    if opts.doFixedLimit:
        dataDir = os.path.join(dataDir,"%.2f"%30.0)

    dataDir = os.path.join(dataDir,"%.2f"%opts.errorbudget)
    plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
    if opts.doJoint or opts.doJointLambda or opts.doJointDisk or opts.doJointDiskBulla or opts.doJointDiskFoucart or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:
        plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.lambdamin,opts.lambdamax))    

if opts.model in ["Ka2017inc","Ka2017x2inc","Ka2017x3inc"]:
    plotDir = plotDir.replace("Ka2017x2inc","inc")

if opts.name == "GW190425":
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_AT2017gfo_sorted")
    eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_AT2017gfo_sorted_SEOB")
else:
    eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new_sorted")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new_sorted_noNICER")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_sorted_GW190814")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new")

filenames = glob.glob(os.path.join(eosdir, '*.dat'))
eosdata = {}
for filename in filenames:
    filenameSplit = filename.replace(".dat","").split("/")[-1].split("-")
    ii = str(int(filenameSplit[-1]))
    data_out = np.loadtxt(filename)
    rarray, marray, larray = data_out[:,0], data_out[:,1], data_out[:,2]
    f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
    eosdata[ii] = {}
    eosdata[ii]["radius"] = rarray
    eosdata[ii]["marray"] = marray
    eosdata[ii]["larray"] = larray
    eosdata[ii]["f"] = f
    eosdata[ii]["mTOV"] = np.max(marray)

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

multifile = lightcurve_utils.get_post_file(plotDir)
data = np.loadtxt(multifile)

mchirp_em = data[:,0]
q_em = data[:,1]
eos_em = data[:,2]
alpha_em = data[:,3]
zeta_em = data[:,4]
lambdatilde_em = np.zeros(eos_em.shape)
mtov_em = np.zeros(eos_em.shape)

for ii, (mc, q, eos) in enumerate(zip(mchirp_em, q_em, eos_em)):
    eos = int(np.floor(eos)+1)
    eta = lightcurve_utils.q2eta(q)
    (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

    f = eosdata[str(eos)]["f"]
    lambda1, lambda2 = f(m1), f(m2)
    lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
    lambdatilde_em[ii] = lambdatilde
    mtov_em[ii] = eosdata[str(eos)]["mTOV"]
data[:,2] = lambdatilde_em
data = np.vstack((data[:,:-1].T,mtov_em.T)).T

labels = [r"$\mathcal{M}_c [{\rm M_\odot}]$",r"$q$",r"$\tilde{\Lambda}$",r"$\log_{10} \alpha$",r"$\zeta$",r"$M_{\rm max} [{\rm M_\odot}]$"]
n_params = len(labels)

#loglikelihood = -(1/2.0)*data[:,1]
#idx = np.argmax(loglikelihood)

if n_params >= 6:
    title_fontsize = 36
    label_fontsize = 36
else:
    title_fontsize = 28
    label_fontsize = 28

plotName = "%s/corner.pdf"%(plotDir)
# If you pass in parameter labels and only one chain, you can also get parameter bounds
#c = ChainConsumer().add_chain( data, parameters=labels)
#c.configure(diagonal_tick_labels=False, tick_font_size=label_fontsize, label_font_size=label_fontsize, max_ticks=3, colors="#FF7F50", smooth=0, kde=[0.3,0.3,0.3,0.3,0.3,0.3], linewidths=2, summary=True, bar_shade=True, statistics="max_symmetric",spacing=3.0)
#fig = c.plotter.plot(figsize="column")

ranges = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
kwargs = dict(bins=50, smooth=1, label_kwargs=dict(fontsize=label_fontsize),
              show_titles=True,
              title_kwargs=dict(fontsize=title_fontsize, pad=20),
              range=ranges,
              color='#0072C1',
              truth_color='tab:orange', quantiles=[0.05, 0.5, 0.95],
              labelpad = 0.1,
              #levels=(0.68, 0.95),
              levels=[0.10, 0.32, 0.68, 0.90],
              plot_density=False, plot_datapoints=False, fill_contours=True,
              max_n_ticks=4, min_n_ticks=3)

fig = corner.corner(data, labels=labels, **kwargs)

#                    quantiles=[0.16, 0.5, 0.84],
#
#                    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2),
#                    show_titles=True,
#                    title_kwargs={"fontsize": title_fontsize, "pad": 20},
#                    label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
#                    smooth=3,
#                    color="coral")
if n_params >= 10:
    fig.set_size_inches(40.0,40.0)
elif n_params > 6:
    fig.set_size_inches(24.0,24.0)
else:
    fig.set_size_inches(24.0,24.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()
