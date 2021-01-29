
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
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['text.usetex'] = True

import corner

import scipy.stats as ss
import plotutils.plotutils as pu

import pymultinest

from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global
from gwemlightcurves.KNModels.table import tidal_lambda_from_tilde

from gwemlightcurves.EjectaFits import KaKy2016
from gwemlightcurves.EjectaFits import DiUj2017 
from gwemlightcurves.EjectaFits import Di2018b
from gwemlightcurves.EjectaFits import CoDi2019
from gwemlightcurves.EjectaFits import KrFo2019

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

        #print(mej, prob)

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

        #print(mdisk, mej2)

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

        #print(mej1)

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

        a0 = -1.581
        da = -2.439
        b0 = -0.538
        db = -0.406
        c = 0.953
        d = 0.0417
        beta = 3.910
        qtrans = 0.900

        eta = 0.5 * np.tanh(beta*((1/q)-qtrans))
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

        #if np.isfinite(prob):
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

        eta = 0.5 * np.tanh(beta*((1/q)-qtrans))
        a = a0 + da * eta
        b = b0 + db * eta

        mdisk = a*(1+b*np.tanh((c-mtot/mth)/d))
        if mdisk<-3:
            mdisk = -3.0
        if rat>0.32:
            mdisk = -3.0
        mdisk = 10**mdisk
        mej2 = zeta*mdisk

        #print(mc, q, eos, alpha, zeta)
        #print(mej1, mej2)
        #exit(0)

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
        print "Need to enable --doModels, --doEvent, --doSimulation, --doMassGap, or --doGoingTheDistance"
        exit(0)
elif opts.doLuminosity:
    if opts.doModels:
        basename = 'fit_luminosity'
    elif opts.doEvent:
        basename = 'fit_gws_luminosity'
    else:
        print "Need to enable --doModels, --doEvent, --doSimulation, --doMassGap, or --doGoingTheDistance"
        exit(0)
else:
    print "Need to enable --doLightcurves or --doLuminosity"
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

    #if opts.doFixedLimit:
    #    dataDir = os.path.join(dataDir,"%.2f"%30.0)

    dataDir = os.path.join(dataDir,"%.2f"%opts.errorbudget)
    plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
    if opts.doJoint or opts.doJointLambda or opts.doJointDisk or opts.doJointDiskBulla or opts.doJointDiskFoucart or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:
        plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.lambdamin,opts.lambdamax))    

if opts.model in ["Ka2017inc","Ka2017x2inc","Ka2017x3inc"]:
    plotDir = plotDir.replace("Ka2017x2inc","inc")

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

#eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/nsat")
#eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new_res")
#eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new_sorted")
if opts.name == "GW190425":
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_AT2017gfo_sorted")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_AT2017gfo_sorted_SEOB")
    eosdir = "/home/michael.coughlin/nmma/q_rerun/Pv2NRTv2/sorted_EOS"
else:
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new_sorted")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new_sorted_noNICER")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_sorted_GW190814")
    #eosdir = os.path.join(opts.dataDir, "gw170817-eft-eos/eos_data/EOS_024_new")
    eosdir = "/home/michael.coughlin/NMMA-public/EOS/chiralEFT_MTOV_NICER"
    #eosdir = "/home/michael.coughlin/NMMA-public/EOS/chiralEFT_MTOV"

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

errorbudget = opts.errorbudget
#n_live_points = 500
#n_live_points = 2000
#n_live_points = 4096
n_live_points = 8192
evidence_tolerance = 0.5
#evidence_tolerance = 0.1

seed = 1
np.random.seed(seed=seed)
lambdatilde_true = None

if opts.doSimulation:
    nsamples = 1000
    pts = opts.mej + np.random.randn(nsamples)*opts.mej*opts.errorbudget

    print(np.mean(pts), np.std(pts))

    kdedir_mej = greedy_kde_areas_1d(pts)
    mchirpmin, mchirpmax = 1.3, 2.0
elif opts.doModels:
    multifile = lightcurve_utils.get_post_file(dataDir)
    data = np.loadtxt(multifile) 

    if opts.doEjecta:
        mej = 10**data[:,1]
        vej = data[:,2]
    elif opts.doMasses:
        print "Masses not implemented..."
        exit(0)
    pts = np.vstack((mej,vej)).T

elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.doGoingTheDistance:
        data_out = lightcurve_utils.going_the_distance(opts.dataDir,opts.name)
        eta = lightcurve_utils.q2eta(data_out["q"])
        m1, m2 = lightcurve_utils.mc2ms(data_out["mc"], eta)
    elif opts.doMassGap:
        data_out, truths = lightcurve_utils.massgap(opts.dataDir,opts.name)
        m1, m2 = data_out["m1"], data_out["m2"]
    elif opts.doEvent:
        if opts.doJointDisk or opts.doJointDiskBulla or opts.doJointDiskFoucart or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:
            #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_EOS")
            #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_SourceProperties")
            #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples")
            #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_comoving_rnsat_196_240")
            if opts.name == "GW190425":
                data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_NRTv2")
                #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_SEOBNRv4T")
            else:
                #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_NRTv1")
                #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_NRTv2")
                #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_NRTv2_NoNICER")
                #data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_GW190814")
                data_out = lightcurve_utils.event(opts.dataDir,opts.name + "_posterior_samples_SEOBNRv4T")
            if data_out:
                costheta_jn = data_out["cos_theta_jn"]
                theta_jn = np.rad2deg(np.arccos(costheta_jn)) 
                theta_mu, theta_std = 32.0, 13.0
                
                theta_min = 180.0 - (32.0 + 13.0*2.0)
                theta_max = 180.0 - (32.0 - 13.0*2.0) 

                idx = np.where((theta_jn>=theta_min) & (theta_jn<=theta_max))[0]
                #data_out = data_out[idx]

                mchirp_gw = data_out["mchirp"]
                q_gw = data_out["mass_ratio"]
                eos_gw = np.floor(data_out["EOS"]).astype(int)

                mchirpmin, mchirpmax = np.min(mchirp_gw), np.max(mchirp_gw)

                eta_gw = lightcurve_utils.q2eta(q_gw)
                (m1_gw,m2_gw) = lightcurve_utils.mc2ms(mchirp_gw,eta_gw)

                lambdatilde_gw = data_out["lambda_tilde"]
                delta_lambdatilde_gw = data_out["delta_lambda_tilde"]

                lambda1_gw, lambda2_gw = tidal_lambda_from_tilde(
                                          m1_gw, m2_gw,
                                          lambdatilde_gw,
                                          delta_lambdatilde_gw)
                q_gw = 1/q_gw

                lambda_s = (lambda1_gw + lambda2_gw)/2.0
                lambda_a = (lambda1_gw - lambda2_gw)/2.0
                b11 = -27.7408 
                c11 = -25.5593
                b12 = 8.42358 
                c12 = 5.58527
                b21 = 122.686 
                c21 = 92.0337
                b22 = -19.7551 
                c22 = 26.8586
                b31 = -175.496 
                c31 = -70.247
                b32 = 133.708 
                c32 = -56.3076
                n = 0.743

                Fn = (1-q_gw**(10/(3-n)))/(1+q_gw**(10/(3-n)))
                lambda_a_fit = Fn * lambda_s * (1\
                     + b11*(q_gw**1)*(lambda_s**(-1.0/5))
                     + b12*(q_gw**2)*(lambda_s**(-1.0/5))
                     + b21*(q_gw**1)*(lambda_s**(-2.0/5))
                     + b22*(q_gw**2)*(lambda_s**(-2.0/5))
                     + b31*(q_gw**1)*(lambda_s**(-3.0/5))
                     + b32*(q_gw**2)*(lambda_s**(-3.0/5))) / (1\
                     + c11*(q_gw**1)*(lambda_s**(-1.0/5))
                     + c12*(q_gw**2)*(lambda_s**(-1.0/5))
                     + c21*(q_gw**1)*(lambda_s**(-2.0/5))
                     + c22*(q_gw**2)*(lambda_s**(-2.0/5))
                     + c31*(q_gw**1)*(lambda_s**(-3.0/5))
                     + c32*(q_gw**2)*(lambda_s**(-3.0/5)))

                kdedir_gwmc = greedy_kde_areas_1d(mchirp_gw)
                kdedir_gwq = greedy_kde_areas_1d(q_gw)
                kdedir_gwlambdatilde = greedy_kde_areas_1d(lambdatilde_gw)

                bin_edges = np.arange(-1,5002)              
                kdedir_gweos, _ = np.histogram(eos_gw, bins=bin_edges)
                kdedir_gweos = kdedir_gweos / float(np.sum(kdedir_gweos))

                for eos in eosdata.keys():
                    idx = np.where(int(eos) == eos_gw)[0]
                    pts = np.vstack((mchirp_gw,q_gw)).T
                    kdedir_mchirp_q = greedy_kde_areas_2d(pts)
                    eosdata[eos]["kdedir_mchirp_q"] = kdedir_mchirp_q

                relerr = np.abs(lambda_a-lambda_a_fit)/np.abs(lambda_a_fit)
                idx = np.where(relerr <= 0.1)[0]

                plotName = "%s/lambdatilde.pdf"%(plotDir)
                bins = np.linspace(np.min(lambdatilde_gw),np.max(lambdatilde_gw),100)
                hist1, bins1 = np.histogram(lambdatilde_gw,bins=bins)
                hist2, bins2 = np.histogram(lambdatilde_gw[idx],bins=bins)
                bins = (bins[1:]+bins[:-1])/2.0
                plt.figure()
                plt.plot(bins,hist1)
                plt.plot(bins,hist2)
                plt.savefig(plotName)
                plt.close()

                print(plotDir)
      
        else:
            data_out = lightcurve_utils.event(opts.dataDir,opts.name)
            if data_out:
                m1, m2 = data_out["m1"], data_out["m2"]
                mchirp, q = data_out["mc"], data_out["q"]
    print(dataDir)
    multifile = lightcurve_utils.get_post_file(dataDir)
    data = np.loadtxt(multifile)

    post_equal_weights = os.path.join(dataDir, 'post_equal_weights.dat')
    if os.path.isfile(post_equal_weights):
        data = np.loadtxt(post_equal_weights)
        idx = [0, 2, 3, 4, 5, 1, 6]
        data = data[:, idx] 

    if opts.doJoint or opts.doJointLambda:
        if opts.model == "Ka2017x2":
            mej = 10**data[:,1]
            vej = data[:,2]
        else:
            mej = 10**data[:,1]
            vej = data[:,2]
        pts = np.vstack((mej,vej)).T
    elif opts.doJointDisk or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:
        if opts.model == "Ka2017x2":
            mej1 = 10**data[:,1]
            vej1 = data[:,2]
            mej2 = 10**data[:,4]
            vej2 = data[:,5]
        elif opts.model == "Ka2017x2inc":
            mej1 = 10**data[:,1]
            vej1 = data[:,2]
            mej2 = 10**data[:,4]
            vej2 = data[:,5]
        elif opts.model in ["Bu2019lr","Bu2019lf","Bu2019lm"]:
            mej1 = 10**data[:,1]
            vej1 = np.random.uniform(low=0.0, high=0.3, size=mej1.shape)
            mej2 = 10**data[:,2]
            vej2 = np.random.uniform(low=0.0, high=0.3, size=mej2.shape)
        elif opts.model in ["Bu2019lw"]:
            mej1 = 0.005*np.ones(len(data[:,1]))
            vej1 = np.random.uniform(low=0.0, high=0.3, size=mej1.shape)
            mej2 = 10**data[:,1]
            vej2 = np.random.uniform(low=0.0, high=0.3, size=mej2.shape)
        else:
            print("--doJointDisk only works with Ka2017x2,Ka2017x2inc,Bu2019lr,Bu2019lf,Bu2019lm,Bu2019lw")
            exit(0)
        pts = np.vstack((mej1,vej1,mej2)).T
        #pts = np.vstack((mej1,vej1)).T
        kdedir = greedy_kde_areas_1d(mej2)
        kdedir_pts_mej2 = copy.deepcopy(kdedir)
        kdedir_mej = greedy_kde_areas_1d(mej1+mej2)
        mejsum = np.sort(mej1+mej2)
        mejsum_95_idx = int(float(len(mejsum))*0.95)
        mejsum_95 = mejsum[mejsum_95_idx]

    elif opts.doJointDiskBulla:
        if opts.model in ["Ka2017","Bu2019inc"]:
            mej = 10**data[:,1]
        else:
            print("--doJointDiskBulla only works with Ka2017,Bu2019inc")
            exit(0)
        kdedir_mej = greedy_kde_areas_1d(mej)
    else:
        pts = np.vstack((m1,m2)).T

    filename = os.path.join(dataDir,"truth_mej_vej.dat")
    if os.path.isfile(filename):
        truths_mej_vej = np.loadtxt(filename)
        truths_mej_vej[0] = np.log10(truths_mej_vej[0])
    else:
        truths_mej_vej = [0, 0]

    #filename = os.path.join(dataDir,"truth.dat")
    #truths = np.loadtxt(filename)

    if opts.doEjecta or opts.doJoint or opts.doJointLambda:
        if opts.model == "Ka2017x2":
            mej_em = data[:,1]
            vej_em = data[:,2]
        else:
            mej_em = data[:,1]
            vej_em = data[:,2]

        mej_true = truths_mej_vej[0]
        vej_true = truths_mej_vej[1]
    elif opts.doJointDisk or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:
        if opts.model == "Ka2017x2" or opts.model == "Ka2017x2inc":
            mej1_em = data[:,1]
            vej1_em = data[:,2]
            mej2_em = data[:,4]
            vej2_em = data[:,5]
        elif opts.model in ["Ka2017lr","Ka2017lf"]:
            mej1_em = data[:,1]
            vej1_em = np.random.uniform(low=0.0, high=0.3, size=mej1_em.shape)
            mej2_em = data[:,2]
            vej2_em = np.random.uniform(low=0.0, high=0.3, size=mej2_em.shape)

        mej_true = truths_mej_vej[0]
        vej_true = truths_mej_vej[1]

    elif opts.doJointDiskBulla:
        if opts.model in ["Ka2017","Bu2019inc"]:
            mej_em = data[:,1]
        mej_true = truths_mej_vej[0]
        vej_true = truths_mej_vej[1]

    elif opts.doMasses:
        if opts.model == "DiUj2017":
            if opts.doEOSFit or opts.doBNSFit:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
            else:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,4])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
        elif opts.model == "KaKy2016":
            if opts.doEOSFit or opts.doBNSFit:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1]*data[:,3],data[:,3])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0]*truths[4],truths[4])
            else:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1]*data[:,3],data[:,3])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0]*truths[4],truths[4])
        elif opts.model == "Ka2017":
            if opts.doEOSFit or opts.doBNSFit:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
            else:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,4])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
        elif opts.model in ["Ka2017x2", "Bu2019inc", "Ka2017x2inc", "Bu2019lf", "Bu2019lr","Bu2019lm","Bu2019lw"]:
            if opts.doEOSFit or opts.doBNSFit:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
            else:
                mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,4])
                mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])

        q_em = 1/q_em  
        q_true = 1/q_true

if not opts.doSimulation:
    kdedir = greedy_kde_areas_2d(pts)
    kdedir_pts = copy.deepcopy(kdedir)

if opts.doModels or opts.doSimulation:
    if opts.doJointDiskBulla:
        parameters = ["mchirp","q","EOS","alpha","zeta"]
        labels = ["$\mathcal{M}_c$",r"$q$",r"EOS",r"$\log_{10} \alpha$",r"$\zeta$"]
        n_params = len(parameters)
        pymultinest.run(myloglike_bns_JointFitDiskBulla_sim, myprior_bns_JointFitDisk_sim, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.doJointDiskFoucart:
        parameters = ["mchirp","q","EOS","chi_eff","zeta"]
        labels = ["$\mathcal{M}_c$",r"$q$",r"EOS",r"$\chi$",r"$\zeta$"]
        n_params = len(parameters)
        pymultinest.run(myloglike_bhns_JointFitDiskFoucart_sim, myprior_nsbh_JointFitDisk_sim, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model == "KaKy2016":
        if opts.doEOSFit or opts.doBNSFit:
            parameters = ["q","chi_eff","mns","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$C$"]
            n_params = len(parameters)        
            pymultinest.run(myloglike_bhns_EOSFit, myprior_bhns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["q","chi_eff","mns","mb","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$m_{\rm b}$",r"$C$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model == "DiUj2017":
        if opts.doEOSFit or opts.doBNSFit:
            parameters = ["m1","c1","m2","c2"]
            labels = [r"$m_1$",r"$C_1$",r"$m_2$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["m1","mb1","c1","m2","mb2","c2"]
            labels = [r"$m_1$",r"$m_{\rm b1}$",r"$C_1$",r"$m_2$",r"$m_{\rm b2}$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.model == "KaKy2016":
        if opts.doEOSFit or opts.doBNSFit:
            parameters = ["q","chi_eff","mns","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$C$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns_gw_EOSFit, myprior_bhns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["q","chi_eff","mns","mb","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$m_{\rm b}$",r"$C$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns_gw, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model == "DiUj2017" or opts.model == "Me2017" or opts.model == "SmCh2017":
        if opts.doEOSFit or opts.doBNSFit:
            parameters = ["m1","c1","m2","c2"]
            labels = [r"$m_1$",r"$C_1$",r"$m_2$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_gw_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["m1","mb1","c1","m2","mb2","c2"]
            labels = [r"$m_1$",r"$m_{\rm b1}$",r"$C_1$",r"$m_2$",r"$m_{\rm b2}$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_gw, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model in ["Ka2017", "Ka2017x2", "Bu2019inc", "Ka2017x2inc", "Bu2019lr", "Bu2019lf","Bu2019lm","Bu2019lw"]:
        if opts.doEOSFit:
            parameters = ["m1","c1","m2","c2"]
            labels = [r"$m_1$",r"$C_1$",r"$m_2$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_gw_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doBNSFit:
            parameters = ["m1","c1","m2","c2"]
            labels = [r"$m_1$",r"$C_1$",r"$m_2$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_gw_BNSFit, myprior_bns_BNSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJoint:
            parameters = ["q","lambda1","A"]
            labels = [r"q",r"$\Lambda_1$",r"A"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_JointFit, myprior_bns_JointFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJointLambda:
            parameters = ["q","lambda1","A"]
            labels = [r"q",r"$\Lambda_1$",r"A"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_JointFitLambda, myprior_bns_JointFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJointDisk or opts.doJointGRB or opts.doJointSpin or opts.doJointWang:
            parameters = ["mchirp","q","EOS","alpha","zeta"]
            labels = ["$M_c$",r"q",r"EOS",r"$\log_{10} \alpha$",r"$\zeta$"]
            n_params = len(parameters)
            if opts.doFixedLimit:
                pymultinest.run(myloglike_bns_JointFitDiskLimit, myprior_bns_JointFitDisk, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
            else:
                pymultinest.run(myloglike_bns_JointFitDisk, myprior_bns_JointFitDisk, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJointDiskBulla:
            parameters = ["mchirp","q","EOS","alpha","zeta"]
            labels = ["$\mathcal{M}_c$",r"$q$",r"EOS",r"$\log_{10} \alpha$",r"$\zeta$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_JointFitDiskBulla, myprior_bns_JointFitDisk, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJointBNS:
            parameters = ["q","lambda2","alpha","zeta","mTOV"]
            labels = [r"q",r"$\Lambda_2$",r"$\log_{10} \alpha$",r"$\zeta$",r"$M_{TOV}$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_Lambda2, myprior_bns_Lambda2, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJointNSBH:
            parameters = ["q","lambda2","alpha","zeta","chieff"]
            labels = [r"q",r"$\Lambda_2$",r"$\alpha$",r"$\zeta$",r"$\chi_{\rm BH}$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_nsbh_JointFitDisk, myprior_nsbh_JointFitDisk, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["m1","mb1","c1","m2","mb2","c2"]
            labels = [r"$m_1$",r"$m_{\rm b1}$",r"$C_1$",r"$m_2$",r"$m_{\rm b2}$",r"$C_2$"]
            n_params = len(parameters)   
            pymultinest.run(myloglike_bns_gw, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

    elif opts.model == "Ka2017_TrPi2018":
        if opts.doJoint:
            parameters = ["q","lambda1","A"]
            labels = [r"q",r"$\Lambda_1$",r"A"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_JointFit, myprior_bns_JointFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.doJointLambda:
            parameters = ["q","lambda1","A","mc"]
            labels = [r"q",r"$\Lambda_1$",r"A",r"$M_c$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_JointFitLambda, myprior_bns_JointFitLambda, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

multifile = lightcurve_utils.get_post_file(plotDir)
data = np.loadtxt(multifile)

if (opts.doModels or opts.doSimulation) and opts.model == "KaKy2016" and opts.doEOSFit:
    m1 = data[:,0]*data[:,2]
    m2 = data[:,2]
    mchirp,eta,q = lightcurve_utils.ms2mc(data[:,0],data[:,2])

elif (opts.doModels or opts.doSimulation) and opts.model == "DiUj2017" and opts.doEOSFit:
    data_new = np.zeros(data.shape)
    labels = [r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$"] 
    mchirp,eta,q = lightcurve_utils.ms2mc(data[:,0],data[:,2])
    data_new[:,0] = 1/q
    data_new[:,1] = mchirp
    data_new[:,2] = data[:,1]
    data_new[:,3] = data[:,3]
    data = data_new

    if opts.doLoveC:
        labels = [r"$q$",r"$M_{\rm c}$",r"$\log_{\rm 10} \lambda_{\rm 1}$",r"$\log_{\rm 10} \lambda_{\rm 2}$"]
        data[:,2] = np.log10(lightcurve_utils.LoveC(data[:,2]))
        data[:,3] = np.log10(lightcurve_utils.LoveC(data[:,3]))

    if opts.doFixMChirp:
        mchirp_mu, mchirp_sigma = np.mean(mchirp), 0.001*np.mean(mchirp)

elif opts.doSimulation and (opts.doJointDiskBulla or opts.doJointDiskFoucart):

    if opts.doJointDiskBulla:
        labels = ["$\mathcal{M}_c$",r"$q$",r"$\tilde{\Lambda}$",r"$\log_{10} \alpha$",r"$\zeta$",r"$M_{\rm TOV}$"]
    elif opts.doJointDiskFoucart:
        labels = ["$\mathcal{M}_c$",r"$q$",r"$\Lambda_2$",r"$\chi$",r"$\zeta$",r"$M_{\rm TOV}$"]        
    
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

        if opts.doJointDiskBulla:
            lambdatilde_em[ii] = lambdatilde
        elif opts.doJointDiskFoucart:
            lambdatilde_em[ii] = lambda2
        mtov_em[ii] = eosdata[str(eos)]["mTOV"]
    data[:,2] = lambdatilde_em
    data = np.vstack((data.T,mtov_em.T)).T
    data[:, 5], data[:, 6] = data[:, 6], data[:, 5].copy()

elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.model == "DiUj2017" or opts.model == "Me2017" or opts.model == "SmCh2017":
        m1 = data[:,0]
        m2 = data[:,2]
        mchirp,eta,q = lightcurve_utils.ms2mc(data[:,0],data[:,2])
        if opts.doEOSFit:
            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0],data[:,2])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,c1,m2,c2 in data[:,:-1]:
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
                mej_em[ii], vej_em[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        elif opts.doBNSFit: 
            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0],data[:,2])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,c1,m2,c2 in data[:,:-1]:  
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
                mej_em[ii], vej_em[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        else:
            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0],data[:,3])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,mb1,c1,m2,mb2,c2 in data[:,:-1]:
                mej_em[ii], vej_em[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        q_em = 1/q_em 
        mej_em = np.log10(mej_em)
    elif opts.model in ["Ka2017", "Ka2017x2", "Ka2017_TrPi2018", "Bu2019inc", "Ka2017x2inc", "Bu2019lf", "Bu2019lr","Bu2019lm","Bu2019lw"]:

        if opts.doEOSFit:
            m1 = data[:,0]
            m2 = data[:,2]
            m1_m2_em = np.vstack((m1,m2)).T

            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0],data[:,2])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,c1,m2,c2 in data[:,:-1]:  
                mb1 = lightcurve_utils.EOSfit(m1,c1)
                mb2 = lightcurve_utils.EOSfit(m2,c2)
                mej_em[ii], vej_em[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        elif opts.doBNSFit: 
            m1 = data[:,0]
            m2 = data[:,2]
            m1_m2_em = np.vstack((m1,m2)).T

            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0],data[:,2])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,c1,m2,c2 in data[:,:-1]:
                mej_em[ii], vej_em[ii] = bns2_model(m1,c1,m2,c2)
                ii = ii + 1
        elif opts.doJoint or opts.doJointLambda or opts.doJointDisk or opts.doJointDiskBulla or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:
            mchirp_em = data[:,0]
            q_em = data[:,1]
            eos_em = data[:,2]
            alpha_em = data[:,3]
            if opts.doJoint:
                mchirp_em = 1.186
            elif opts.doJointLambda:
                mchirp_em = data[:,3]
            elif opts.doJointDisk or opts.doJointDiskBulla or opts.doJointGRB or opts.doJointSpin or opts.doJointWang:
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
                data = np.vstack((data.T,mtov_em.T)).T
                data[:, 5], data[:, 6] = data[:, 6], data[:, 5].copy()

                zeta_em_sort = np.sort(zeta_em)

                pts_KN = np.vstack((mchirp_em,q_em,lambdatilde_em,zeta_em,mtov_em)).T
                iis = [1,2,3]
                boundaries = [1,700,0.5]
                pts_KN_all = copy.deepcopy(pts_KN)
                pts_KN_2 = copy.deepcopy(pts_KN)
                for ii,boundary1 in zip(iis,boundaries):
                    pts_KN_2 = copy.deepcopy(pts_KN)
                    pts_KN_2[:,ii] = 2*boundary1 - pts_KN_2[:,ii]
                    pts_KN_all = np.vstack((pts_KN_all,pts_KN_2))

                kdedir = greedy_kde_areas_2d(pts_KN_all)
                kdedir_pts_KN = copy.deepcopy(kdedir)
            elif opts.doJointBNS:
                mchirp_em = 1.186
                lambda2_em = data[:,1]
                zeta_em = data[:,3]
                mtov_em = data[:,4]

                zeta_em_sort = np.sort(zeta_em)

                pts_KN = np.vstack((q_em,lambda2_em,zeta_em,mtov_em)).T
                iis = [0,1,2,3,3]
                boundaries = [1,5000,0.5,2.0,2.17]
                pts_KN_all = copy.deepcopy(pts_KN)
                pts_KN_2 = copy.deepcopy(pts_KN)
                for ii,boundary1 in zip(iis,boundaries):
                    pts_KN_2 = copy.deepcopy(pts_KN)
                    pts_KN_2[:,ii] = 2*boundary1 - pts_KN_2[:,ii]
                    pts_KN_all = np.vstack((pts_KN_all,pts_KN_2))

                kdedir = greedy_kde_areas_2d(pts_KN_all)
                kdedir_pts_KN = copy.deepcopy(kdedir)

            elif opts.doJointNSBH:
                mchirp_em = 1.186
                lambda2_em = data[:,1]
                zeta_em = data[:,3]
                chieff_em = data[:,4]

                zeta_em_sort = np.sort(zeta_em)

                pts_KN = np.vstack((q_em,lambda2_em,zeta_em,chieff_em)).T
      
                iis = [0,1,2,3,3]
                boundaries = [1,5000,0.5,-1.0,1.0]
                pts_KN_all = copy.deepcopy(pts_KN)
                pts_KN_2 = copy.deepcopy(pts_KN)
                for ii,boundary1 in zip(iis,boundaries):
                    pts_KN_2 = copy.deepcopy(pts_KN)
                    pts_KN_2[:,ii] = 2*boundary1 - pts_KN_2[:,ii]
                    pts_KN_all = np.vstack((pts_KN_all,pts_KN_2))

                kdedir = greedy_kde_areas_2d(pts_KN_all)
                kdedir_pts_KN = copy.deepcopy(kdedir)
 
            eta = lightcurve_utils.q2eta(q_em)
            (m1,m2) = lightcurve_utils.mc2ms(mchirp_em,eta)
            m1_m2_em = np.vstack((m1,m2)).T

            lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])

            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            lambdatilde_em = np.zeros(data[:,0].shape)
            mej2_em, vej2_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)

            filename = os.path.join(plotDir,'q_lambdatilde.dat')
            fid = open(filename,'w')
            ii = 0
            if opts.doJoint:
                labels = [r"q",r"$\tilde{\Lambda}$",r"A"]
                for q,lambda1,A in data[:,:-1]:
                    c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
                    c2 = c1/q
                    coeff = lambda_coeff[::-1]
                    p = np.poly1d(coeff)
                    lambda2 = p(c2)
                    lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
                    eta = lightcurve_utils.q2eta(q)
                    (m1,m2) = lightcurve_utils.mc2ms(mchirp_em,eta)
                    mej_em[ii], vej_em[ii] = bns2_model(m1,c1,m2,c2)
                    mej_em[ii] = mej_em[ii]*A
                    lambdatilde_em[ii] = lambdatilde
                    ii = ii + 1
                    fid.write('%.10f %.10f\n'%(q,lambdatilde))

            elif opts.doJointLambda:
                labels = [r"q",r"$\tilde{\Lambda}$",r"A",r"$M_c$"]
                for q,lambda1,A,mchirp_em in data[:,:-1]:
                    c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
                    c2 = c1/q
                    coeff = lambda_coeff[::-1]
                    p = np.poly1d(coeff)
                    lambda2 = p(c2)
                    lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
                    eta = lightcurve_utils.q2eta(q)
                    (m1,m2) = lightcurve_utils.mc2ms(mchirp_em,eta)

                    mej_em[ii], vej_em[ii] = bns2_model(m1,c1,m2,c2)
                    mej_em[ii] = mej_em[ii]*A
                    lambdatilde_em[ii] = lambdatilde
                    ii = ii + 1
                    fid.write('%.10f %.10f\n'%(q,lambdatilde))

            elif opts.doJointDisk or opts.doJointDiskBulla or opts.doJointGRB or opts.doJointSpin or opts.doJointWang:
                labels = ["$\mathcal{M}_c$",r"$q$",r"$\tilde{\Lambda}$",r"$\log_{10} \alpha$",r"$\zeta$",r"$M_{\rm TOV}$"]
                #for q,lambda1,A,zeta_em in data[:,:-1]:
                for mc,q,lambdatilde,alpha,zeta,mTOV in data[:,:-1]:

                    lambda1 = (13.0/16.0) * (lambdatilde/q) * ((1 + q)**5)/(1 + 12*q + 12*(q**3) + q**4)

                    c1 = 0.371 - 0.0391*np.log(lambda1) + 0.001056*(np.log(lambda1)**2)
                    c2 = c1/q
                    #coeff = lambda_coeff[::-1]
                    #p = np.poly1d(coeff)
                    #lambda2 = p(c2)
                    #lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
                    eta = lightcurve_utils.q2eta(q)
                    (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

                    vej1 = CoDi2019.calc_vej(m1,c1,m2,c2)

                    a= -0.0719
                    b= 0.2116
                    d= -2.42
                    n= -2.905

                    log10_mej = a*(m1*(1-2*c1)/c1 + m2*(1-2*c2)/c2) + b*(m1*(m2/m1)**n + m2*(m1/m2)**n)+d
                    mej1 = 10**log10_mej
                    mej1 = mej1/(10**alpha)

                    mej_em[ii], vej_em[ii] = mej1, vej1
                    lambdatilde_em[ii] = lambdatilde

                    R16 = mc * (lambdatilde/0.0042)**(1.0/6.0)
                    mth = (2.38 - 3.606*mTOV/R16)*mTOV

                    a, b, c, d = -31.335, -0.9760, 1.0474, 0.05957

                    x = lambdatilde*1.0
                    mtot = m1+m2
                    mdisk = 10**np.max([-3,a*(1+b*np.tanh((c-mtot/mth)/d))])
                    mej2 = zeta*mdisk
                    mej2_em[ii] = mej2

                    ii = ii + 1
                    fid.write('%.10f %.10f\n'%(q,lambdatilde))

            elif opts.doJointBNS:
                labels = [r"q",r"$\Lambda_2$",r"$\log_{10} \alpha$",r"$\zeta$",r"$M_{TOV}$"]
                #for q,lambda1,A,zeta_em in data[:,:-1]:
                for q,lambda2,alpha,zeta_em,m_tov in data[:,:-1]:

                    lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])

                    c2 = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)
                    c1 = c2*q
                    coeff = lambda_coeff[::-1]
                    p = np.poly1d(coeff)
                    lambda1 = p(c1)

                    lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)

                    eta = lightcurve_utils.q2eta(q)
                    (m1,m2) = lightcurve_utils.mc2ms(mchirp_em,eta)

                    mej_em[ii], vej_em[ii] = bns2_model(m1,c1,m2,c2)
                    mej_em[ii] = mej_em[ii]/(10**alpha)
                    lambdatilde_em[ii] = lambdatilde

                    a = -2.705
                    b = 0.01324
                    c = -4.426
                    d = -9.107
                    e = 12.94

                    x = lambdatilde*1.0
                    y = (m1+m2)/m_tov
                    mdisk = 10**np.max([-3,(a+np.tanh(b*x+c)+np.tanh(d*y+e))])

                    mej2_em[ii] = zeta_em*mdisk

                    ii = ii + 1
                    fid.write('%.10f %.10f\n'%(q,lambdatilde))

            elif opts.doJointNSBH:
                labels = [r"q",r"$\Lambda_2$",r"$\alpha$",r"$\zeta$",r"$\chi_{\rm BH}$"]
                #for q,lambda1,A,zeta_em in data[:,:-1]:
                for q,lambda2,alpha,zeta_em,chi_eff in data[:,:-1]:
                    c = 0.371 - 0.0391*np.log(lambda2) + 0.001056*(np.log(lambda2)**2)
                    lambda1 = 0.0
                    lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)

                    eta = lightcurve_utils.q2eta(q)
                    (m1,m2) = lightcurve_utils.mc2ms(mchirp_em,eta)

                    a, n = 0.8858, 1.2082
                    mb = (1+a*c**n)*m2

                    mej_em[ii], vej_em[ii] = bhns_model(q,chi_eff,m2,mb,c)
                    mej_em[ii] = mej_em[ii]*alpha
                    lambdatilde_em[ii] = lambdatilde

                    Z1 = 1 + ((1-chi_eff**2)**(1.0/3.0)) * ((1+chi_eff)**(1.0/3.0) + (1-chi_eff)**(1.0/3.0))
                    Z2 = np.sqrt(3.0*chi_eff**2 + Z1**2)
                    Risco = 3+Z2-np.sign(chi_eff)*np.sqrt((3-Z1)*(3+Z1+2*Z2))

                    alpha_fit, beta_fit, gamma_fit, delta_fit = 0.406, 0.139, 0.255, 1.761
                    term1 = alpha_fit*(1-2*c)/(eta**(1.0/3.0))
                    term2 = beta_fit*Risco*c/eta

                    mdisk = (np.max(term1-term2+gamma_fit,0))**delta_fit
                    mej2_em[ii] = zeta_em*mdisk

                    ii = ii + 1
                    fid.write('%.10f %.10f\n'%(q,lambdatilde))

            fid.close()

            print "Q bounds: [1,%.2f]"%(np.percentile(q_em,90))

        else:
            m1 = data[:,0]
            m2 = data[:,3]
            m1_m2_em = np.vstack((m1,m2)).T

            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0],data[:,3])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,mb1,c1,m2,mb2,c2 in data[:,:-1]:
                mej_em[ii], vej_em[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1

        mej_em = np.log10(mej_em) 

    elif opts.model == "KaKy2016":
        m1 = data[:,0]*data[:,2]
        m2 = data[:,2]
        mchirp,eta,q = lightcurve_utils.ms2mc(data[:,0],data[:,2])
        if opts.doEOSFit:
            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0]*data[:,2],data[:,2])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for q,chi,mns,c in data[:,:-1]:
                mb = lightcurve_utils.EOSfit(mns,c)
                mej_em[ii], vej_em[ii] = bhns_model(q,chi,mns,mb,c)
                ii = ii + 1
        else:
            mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,0]*data[:,2],data[:,2])
            mej_em, vej_em = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for q,chi,mns,mb,c in data[:,:-1]:
                mej_em[ii], vej_em[ii] = bhns_model(q,chi,mns,mb,c)
                ii = ii + 1
        q_em = 1/q_em
        mej_em = np.log10(mej_em)

    if opts.doFixMChirp:
        #mchirp_mu, mchirp_sigma = np.mean(mchirp_gw), 0.01*np.mean(mchirp_gw)
        mchirp_mu, mchirp_sigma = np.mean(mchirp), np.std(mchirp)

    combinedDir = os.path.join(plotDir,"com")
    if not os.path.isdir(combinedDir):
        os.makedirs(combinedDir)       

    if opts.doEjecta or opts.doJoint or opts.doJointLambda:
        pts_em = np.vstack((mej_em,vej_em)).T
        pts_gw = np.vstack((mej_gw,vej_gw)).T
        kdedir_em = greedy_kde_areas_2d(pts_em)
        kdedir_gw = greedy_kde_areas_2d(pts_gw)

        parameters = ["mej","vej"]
        n_params = len(parameters)
        pymultinest.run(myloglike_combined, myprior_combined_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        labels_combined = [r"log10 ${\rm M}_{\rm ej}$",r"${\rm v}_{\rm ej}$"]
        multifile = lightcurve_utils.get_post_file(combinedDir)
        data_combined = np.loadtxt(multifile)
        mej_combined = data_combined[:,0]
        vej_combined = data_combined[:,1]
        data_combined = np.vstack((mej_combined,vej_combined)).T

        plotName = "%s/corner_combined.pdf"%(plotDir)
        figure = corner.corner(data_combined, labels=labels_combined,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   truths=[mej_true,vej_true],
                   smooth=3)
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

    elif opts.doJointDisk or opts.doJointDiskBulla or opts.doJointGRB or opts.doJointSpin or opts.doJointWang or opts.doJointNSBH or opts.doJointBNS:

        pts_em = np.vstack((q_em,lambdatilde_em)).T
        pts_gw = np.vstack((q_gw,lambdatilde_gw)).T

        kdedir_em = greedy_kde_areas_2d(pts_em)
        kdedir_gw = greedy_kde_areas_2d(pts_gw)

        grbDir = os.path.join(plotDir,"grb")
        if not os.path.isdir(grbDir):
            os.makedirs(grbDir)

        if opts.doJointDisk or opts.doJointDiskBulla:
            parameters = ["epsilon","lambdatilde","E0","q","mTOV","zeta","mc"]
            n_params = len(parameters)

            pymultinest.run(myloglike_SGRB, myprior_SGRB, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%grbDir, evidence_tolerance = evidence_tolerance, multimodal = False)

            multifile = lightcurve_utils.get_post_file(grbDir)
            data_sgrb = np.loadtxt(multifile)

            idx = np.where((data_sgrb[:,1]>=lambdamin) & (data_sgrb[:,1]<=lambdamax))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,3]>=1.0) & (data_sgrb[:,3]<=2.0))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,4]>=2.0) & (data_sgrb[:,4]<=2.17))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,5]>=0.0) & (data_sgrb[:,5]<=0.5))[0]
            data_sgrb = data_sgrb[idx,:]

            epsilon_sgrb = data_sgrb[:,0]
            lambdatilde_sgrb= data_sgrb[:,1]
            E0_sgrb = data_sgrb[:,2]
            q_sgrb = data_sgrb[:,3]
            mTOV_sgrb = data_sgrb[:,4]
            zeta_sgrb = data_sgrb[:,5]
            mc_sgrb = data_sgrb[:,6]

            E0_mu, E0_std = 50.30, 0.84
            E0_sgrb = scipy.stats.norm(E0_mu, E0_std).ppf(E0_sgrb)
            E0_sgrb = 10**E0_sgrb

            #data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,thetaj_sgrb,thetac_sgrb,q_sgrb,mTOV_sgrb)).T

            #labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\tilde{\Lambda}$","$\Theta_{j}$","$\Theta_{c}$",r"$q$",r"$m_{TOV}$"]

            data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,q_sgrb,mTOV_sgrb,zeta_sgrb,mc_sgrb)).T
            labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\tilde{\Lambda}$",r"$q$",r"$M_{TOV}$",r"$\zeta$",r"$M_c$"]
            plotName = "%s/corner_sgrb.pdf"%(plotDir)
            figure = corner.corner(data_sgrb, labels=labels_sgrb,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   color='forestgreen',
                   smooth=3)
            figure.set_size_inches(14.0,14.0)
            plt.savefig(plotName)
            plt.close()

        elif opts.doJointGRB:
            parameters = ["epsilon","lambdatilde","E0","q","mTOV","zeta"]
            n_params = len(parameters)

            pymultinest.run(myloglike_GRB, myprior_GRB, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%grbDir, evidence_tolerance = evidence_tolerance, multimodal = False)

            multifile = lightcurve_utils.get_post_file(grbDir)
            data_sgrb = np.loadtxt(multifile)

            idx = np.where((data_sgrb[:,1]>=lambdamin) & (data_sgrb[:,1]<=lambdamax))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,3]>=1.0) & (data_sgrb[:,3]<=2.0))[0]
            data_sgrb = data_sgrb[idx,:]    
            idx = np.where((data_sgrb[:,4]>=2.0) & (data_sgrb[:,4]<=2.17))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,5]>=0.0) & (data_sgrb[:,5]<=0.5))[0]
            data_sgrb = data_sgrb[idx,:]

            epsilon_sgrb = data_sgrb[:,0]
            lambdatilde_sgrb= data_sgrb[:,1]
            E0_sgrb = data_sgrb[:,2]
            q_sgrb = data_sgrb[:,3]
            mTOV_sgrb = data_sgrb[:,4]
            zeta_sgrb = data_sgrb[:,5]

            E0_mu, E0_std = -0.81, 0.39
            E0 = scipy.stats.norm(E0_mu, E0_std).ppf(E0_sgrb)
            E0_sgrb = 1e50 * 10**E0

            #data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,np.log10(E0_sgrb),thetaobs_sgrb,q_sgrb,mTOV_sgrb)).T
            data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,q_sgrb,mTOV_sgrb,zeta_sgrb)).T

            #labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\tilde{\Lambda}$",r"$\log_{\rm 10} E_0$",r"$\Theta_{\rm obs}$",r"$q$",r"$m_{TOV}$"]
            labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\tilde{\Lambda}$",r"$q$",r"$M_{TOV}$",r"$\zeta$"]

            plotName = "%s/corner_sgrb.pdf"%(plotDir)
            figure = corner.corner(data_sgrb, labels=labels_sgrb,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   color='forestgreen',
                   smooth=3)
            figure.set_size_inches(14.0,14.0)
            plt.savefig(plotName)
            plt.close()

        elif opts.doJointSpin:
            parameters = ["epsilon","q","lambdatilde","E0","thetaObs"]
            n_params = len(parameters)

            pymultinest.run(myloglike_Spin, myprior_Spin, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%grbDir, evidence_tolerance = evidence_tolerance, multimodal = False)

            multifile = lightcurve_utils.get_post_file(grbDir)
            data_sgrb = np.loadtxt(multifile)

            epsilon_sgrb = data_sgrb[:,0]
            q_sgrb = data_sgrb[:,1]
            lambdatilde_sgrb= data_sgrb[:,2]
            E0_sgrb = data_sgrb[:,3]
            thetaobs_sgrb = data_sgrb[:,4]

            E0_mu, E0_std = -0.81, 0.39
            E0 = scipy.stats.norm(E0_mu, E0_std).ppf(E0_sgrb)
            E0_sgrb = 1e50 * 10**E0

            thetaobs_mu, thetaobs_std = 0.47, 0.08
            thetaobs_sgrb = scipy.stats.norm(thetaobs_mu, thetaobs_std).ppf(thetaobs_sgrb)

            data_sgrb = np.vstack((epsilon_sgrb,q_sgrb,lambdatilde_sgrb,np.log10(E0_sgrb),thetaobs_sgrb)).T

            labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$q$",r"$\tilde{\Lambda}$",r"$\log_{\rm 10} E_0$",r"$\Theta_{\rm obs}$"]
            plotName = "%s/corner_sgrb.pdf"%(plotDir)
            figure = corner.corner(data_sgrb, labels=labels_sgrb,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   color='forestgreen',
                   smooth=3)
            figure.set_size_inches(14.0,14.0)
            plt.savefig(plotName)
            plt.close()

        elif opts.doJointWang:
            parameters = ["q","lambdatilde","mTOV","sigma","zeta"]
            n_params = len(parameters)

            pymultinest.run(myloglike_Wang, myprior_Wang, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%grbDir, evidence_tolerance = evidence_tolerance, multimodal = False)

            multifile = lightcurve_utils.get_post_file(grbDir)
            data_sgrb = np.loadtxt(multifile)

            idx = np.where((data_sgrb[:,1]>=lambdamin) & (data_sgrb[:,1]<=lambdamax))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,0]>=1.0) & (data_sgrb[:,0]<=2.0))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,2]>=2.0) & (data_sgrb[:,2]<=2.17))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,4]>=0.0) & (data_sgrb[:,4]<=0.5))[0]
            data_sgrb = data_sgrb[idx,:]

            q_sgrb = data_sgrb[:,0]
            lambdatilde_sgrb= data_sgrb[:,1]
            mTOV_sgrb = data_sgrb[:,2]
            sigma_sgrb = data_sgrb[:,3]
            zeta_sgrb = data_sgrb[:,4]

            #data_sgrb = np.vstack((q_sgrb,lambdatilde_sgrb,mTOV_sgrb)).T
            data_sgrb = np.vstack((sigma_sgrb,lambdatilde_sgrb,q_sgrb,mTOV_sgrb,zeta_sgrb)).T

            #labels_sgrb = [r"$q$",r"$\tilde{\Lambda}$",r"$m_{TOV}$"]
            labels_sgrb = [r"$\sigma$",r"$\tilde{\Lambda}$",r"$q$",r"$M_{TOV}$",r"$\zeta$"]
            plotName = "%s/corner_sgrb.pdf"%(plotDir)
            figure = corner.corner(data_sgrb, labels=labels_sgrb,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   color='forestgreen',
                   smooth=3)
            figure.set_size_inches(14.0,14.0)
            plt.savefig(plotName)
            plt.close()

        elif opts.doJointBNS:
            parameters = ["epsilon","lambda2","E0","q","mTOV","zeta"]
            n_params = len(parameters)

            pymultinest.run(myloglike_BNS_Lambda2GRB, myprior_BNS_Lambda2GRB, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%grbDir, evidence_tolerance = evidence_tolerance, multimodal = False)

            multifile = lightcurve_utils.get_post_file(grbDir)
            data_sgrb = np.loadtxt(multifile)

            idx = np.where((data_sgrb[:,1]>=lambdamin) & (data_sgrb[:,1]<=lambdamax))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,3]>=1.0) & (data_sgrb[:,3]<=2.0))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,4]>=2.0) & (data_sgrb[:,4]<=2.17))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,5]>=0.0) & (data_sgrb[:,5]<=0.5))[0]
            data_sgrb = data_sgrb[idx,:]

            epsilon_sgrb = data_sgrb[:,0]
            lambdatilde_sgrb= data_sgrb[:,1]
            E0_sgrb = data_sgrb[:,2]
            q_sgrb = data_sgrb[:,3]
            mTOV_sgrb = data_sgrb[:,4]
            zeta_sgrb = data_sgrb[:,5]

            E0_mu, E0_std = 50.30, 0.84
            E0_sgrb = scipy.stats.norm(E0_mu, E0_std).ppf(E0_sgrb)
            E0_sgrb = 10**E0_sgrb

            #data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,thetaj_sgrb,thetac_sgrb,q_sgrb,mTOV_sgrb)).T

            #labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\tilde{\Lambda}$","$\Theta_{j}$","$\Theta_{c}$",r"$q$",r"$m_{TOV}$"]

            data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,q_sgrb,mTOV_sgrb,zeta_sgrb)).T
            labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\Lambda_2$",r"$q$",r"$M_{TOV}$",r"$\zeta$"]
            plotName = "%s/corner_sgrb.pdf"%(plotDir)
            figure = corner.corner(data_sgrb, labels=labels_sgrb,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   color='forestgreen',
                   smooth=3)
            figure.set_size_inches(14.0,14.0)
            plt.savefig(plotName)
            plt.close()

        elif opts.doJointNSBH:
            parameters = ["epsilon","lambda2","E0","q","chieff","zeta"]
            n_params = len(parameters)

            pymultinest.run(myloglike_NSBH, myprior_NSBH, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%grbDir, evidence_tolerance = evidence_tolerance, multimodal = False)

            multifile = lightcurve_utils.get_post_file(grbDir)
            data_sgrb = np.loadtxt(multifile)

            idx = np.where((data_sgrb[:,1]>=lambdamin) & (data_sgrb[:,1]<=lambdamax))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,3]>=1.0) & (data_sgrb[:,3]<=7.0))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,4]>=-1.0) & (data_sgrb[:,4]<=1.0))[0]
            data_sgrb = data_sgrb[idx,:]
            idx = np.where((data_sgrb[:,5]>=0.0) & (data_sgrb[:,5]<=0.5))[0]
            data_sgrb = data_sgrb[idx,:]

            epsilon_sgrb = data_sgrb[:,0]
            lambda2_sgrb= data_sgrb[:,1]
            E0_sgrb = data_sgrb[:,2]
            q_sgrb = data_sgrb[:,3]
            chieff_sgrb = data_sgrb[:,4]
            zeta_sgrb = data_sgrb[:,5]

            lambda1 = 0.0
            lambdatilde_sgrb = (16.0/13.0)*(lambda2_sgrb + lambda1*(q_sgrb**5) + 12*lambda1*(q_sgrb**4) + 12*lambda2_sgrb*q_sgrb)/((q_sgrb+1)**5)

            E0_mu, E0_std = 50.30, 0.84
            E0_sgrb = scipy.stats.norm(E0_mu, E0_std).ppf(E0_sgrb)
            E0_sgrb = 10**E0_sgrb

            #data_sgrb = np.vstack((epsilon_sgrb,lambdatilde_sgrb,thetaj_sgrb,thetac_sgrb,q_sgrb,mTOV_sgrb)).T

            #labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\tilde{\Lambda}$","$\Theta_{j}$","$\Theta_{c}$",r"$q$",r"$m_{TOV}$"]

            data_sgrb = np.vstack((epsilon_sgrb,lambda2_sgrb,q_sgrb,chieff_sgrb,zeta_sgrb)).T
            labels_sgrb = [r"$\log_{\rm 10} \epsilon$",r"$\Lambda_2$",r"$q$",r"$\chi_{\rm BH}$",r"$\zeta$"]
            plotName = "%s/corner_sgrb.pdf"%(plotDir)
            figure = corner.corner(data_sgrb, labels=labels_sgrb,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   color='forestgreen',
                   smooth=3)
            figure.set_size_inches(12.0,12.0)
            plt.savefig(plotName)
            plt.close()

        parameters = ["q","lambdatilde"]
        n_params = len(parameters)
            
        pymultinest.run(myloglike_combined, myprior_combined_q_lambdatilde, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        labels_combined = [r"$q$",r"$\tilde{\Lambda}$"]
        multifile = lightcurve_utils.get_post_file(combinedDir)
        data_combined = np.loadtxt(multifile)
        q_combined = data_combined[:,0]
        lambdatilde_combined = data_combined[:,1]
        data_combined = np.vstack((q_combined,lambdatilde_combined)).T

        xedges, yedges = np.linspace(0.97,1.65,21), np.linspace(50,1050,21)
        H, xedges1, yedges1 = np.histogram2d(q_em, lambdatilde_em, bins=(xedges, yedges))
        H_em = H.T
        H, xedges1, yedges1 = np.histogram2d(q_gw, lambdatilde_gw, bins=(xedges, yedges))
        H_gw = H.T
        H, xedges1, yedges1 = np.histogram2d(q_combined,lambdatilde_combined, bins=(xedges, yedges))
        H_combined = H.T

        X, Y = np.meshgrid((xedges[1:]+xedges[:-1])/2.0, (yedges[1:]+yedges[:-1])/2.0)

        plotName = "%s/corner_combined.pdf"%(plotDir)
        figure = corner.corner(data_combined, labels=labels_combined,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   smooth=3)
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

        color1 = 'coral'
        color2 = 'cornflowerblue'
        color3 = 'forestgreen'
        color4 = 'darkmagenta'

        plotName = "%s/contour_combined.pdf"%(plotDir)
        plt.figure()
        plt.contour(X, Y, H_em, colors=color1)
        plt.contour(X, Y, H_gw, colors=color2)
        plt.contour(X, Y, H_combined, colors=color3)
        plt.savefig(plotName)
        plt.close()

        q_em_hist, bin_edges = np.histogram(q_em, bins=xedges, density=True)
        q_gw_hist, bin_edges = np.histogram(q_gw, bins=xedges, density=True)
        q_combined_hist, bin_edges = np.histogram(q_combined, bins=xedges, density=True)
        bins = (bin_edges[1:]+bin_edges[:-1])/2.0

        plotName = "%s/q.pdf"%(plotDir)
        plt.figure(figsize=(20,12))
        plt.step(bins, q_em_hist, color=color1, label="GW")
        plt.step(bins, q_gw_hist, color=color2, label="EM")
        plt.step(bins, q_combined_hist, color=color3, label="Combined")
        plt.legend(loc="best",fontsize=28)
        plt.xlabel(r"$q$",fontsize=28)
        plt.ylabel("Probability Density Function",fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim([1.0,1.6])
        plt.savefig(plotName)
        plt.close()

        lambdatilde_em_hist, bin_edges = np.histogram(lambdatilde_em, bins=yedges, density=True)
        lambdatilde_gw_hist, bin_edges = np.histogram(lambdatilde_gw, bins=yedges, density=True)
        lambdatilde_combined_hist, bin_edges = np.histogram(lambdatilde_combined, bins=yedges, density=True)
        lambdatilde_sgrb_hist, bin_edges = np.histogram(lambdatilde_sgrb, bins=yedges, density=True)
        bins = (bin_edges[1:]+bin_edges[:-1])/2.0

        plotName = "%s/lambdatilde.pdf"%(plotDir)
        fig = plt.figure(figsize=(20,12))
        plt.step(bins, lambdatilde_em_hist, color=color2, label="EM")
        plt.step(bins, lambdatilde_gw_hist, color=color1, label="GW")
        plt.step(bins, lambdatilde_sgrb_hist, color=color4, label="SGRB")
        plt.step(bins, lambdatilde_combined_hist, color=color3, label="Combined")
        plt.legend(loc="best",fontsize=28)
        plt.xlabel(r"$\tilde{\Lambda}$",fontsize=28)
        plt.ylabel("Probability Density Function",fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim([100,1000])
        plt.savefig(plotName)
        plt.close()

    elif opts.doMasses:

        pts_em = np.vstack((q_em,mchirp_em)).T
        pts_gw = np.vstack((q_gw,mchirp_gw)).T

        kdedir_em = greedy_kde_areas_2d(pts_em)
        kdedir_gw = greedy_kde_areas_2d(pts_gw)

        parameters = ["q","mchirp"]
        n_params = len(parameters)
        if opts.model in ["DiUj2017", "Me2017", "SmCh2017", "Ka2017", "Ka2017x2", "Ka2017x2inc", "Bu2019lf", "Bu2019lr", "Bu2019lm","Bu2019lw"]:
            if opts.doFixMChirp:
                pymultinest.run(myloglike_combined_MChirp, myprior_combined_masses_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)
            else:
                pymultinest.run(myloglike_combined, myprior_combined_masses_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.model == "KaKy2016":
            if opts.doFixMChirp:
                pymultinest.run(myloglike_combined_MChirp, myprior_combined_masses_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)
            else:
                pymultinest.run(myloglike_combined, myprior_combined_masses_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        labels_combined = [r"$q$",r"${\rm M}_{\rm c}$"]
        multifile = lightcurve_utils.get_post_file(combinedDir)
        data_combined = np.loadtxt(multifile)
        q_combined = data_combined[:,0]
        mchirp_combined = data_combined[:,1]
        data_combined = np.vstack((q_combined,mchirp_combined)).T

        plotName = "%s/corner_combined.pdf"%(plotDir)
        figure = corner.corner(data_combined, labels=labels_combined,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   truths=[q_true,mchirp_true],
                   color = 'forestgreen',
                   smooth = 3)
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

#loglikelihood = -(1/2.0)*data[:,1]
#idx = np.argmax(loglikelihood)

if n_params >= 6:
    title_fontsize = 30
    label_fontsize = 30
else:
    title_fontsize = 28
    label_fontsize = 28

plotName = "%s/corner.pdf"%(plotDir)
if opts.doGoingTheDistance:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,
                       title_kwargs={"fontsize": title_fontsize, "pad": 15},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       truths=truths,
                       color="coral",
                       smooth = 3)
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,
                       title_kwargs={"fontsize": title_fontsize, "pad": 15},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       color="coral",
                       smooth = 3)
if n_params >= 10:
    figure.set_size_inches(40.0,40.0)
elif n_params >= 6:
    figure.set_size_inches(28.0,28.0)
else:
    figure.set_size_inches(24.0,24.0)
plt.savefig(plotName)
plt.close()

if opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    colors = ['b','g','r','m','c']
    linestyles = ['-', '-.', ':','--']

    if opts.doEjecta or opts.doJoint or opts.doJointLambda:
        if opts.doEvent:
            if opts.model == "KaKy2016":
                bounds = [-3.0,0.0]
                xlims = [-3.0,0.0]
                ylims = [1e-1,10]
            elif opts.model in ["DiUj2017", "Me2017", "SmCh2017", "Ka2017", "Ka2017x2", "Ka2017_TrPi2018", "Ka2017x2inc", "Bu2019lf", "Bu2019lr", "Bu2019lm","Bu2019lw"]:
                bounds = [-3.0,-1.0]
                xlims = [-3.0,-1.0]
                ylims = [1e-1,10]
        else:
            if opts.model == "KaKy2016":
                bounds = [-3.0,0.0]
                xlims = [-3.0,0.0]
                ylims = [1e-1,10]
            elif opts.model == "DiUj2017" or opts.model == "Me2017" or opts.model == "SmCh2017":
                bounds = [-3.0,-1.0]
                xlims = [-3.0,-1.3]
                ylims = [1e-1,10]
            elif opts.model in ["DiUj2017", "Me2017", "SmCh2017", "Ka2017", "Ka2017x2", "Ka2017_TrPi2018", "Ka2017x2inc", "Bu2019lf", "Bu2019lr", "Bu2019lm","Bu2019lw"]:
                bounds = [-3.0,-1.0]
                xlims = [-3.0,-1.3]
                ylims = [1e-1,10]

        plotName = "%s/mej.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        bins, hist1 = lightcurve_utils.hist_results(mej_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,drawstyle='steps-mid',label="GW")
        bins, hist1 = lightcurve_utils.hist_results(mej_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,drawstyle='steps-mid',label="EM")
        bins, hist1 = lightcurve_utils.hist_results(mej_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,drawstyle='steps-mid',label="GW-EM")
        plt.semilogy([mej_true,mej_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()
   
        if opts.model == "KaKy2016":
            bounds = [0.0,0.3]
            xlims = [0.0,0.3]
            ylims = [1e-1,20]
        elif opts.model == "DiUj2017" or opts.model == "Me2017" or opts.model == "SmCh2017":
            bounds = [0.0,0.3]
            xlims = [0.0,0.3]
            ylims = [1e-1,10]
        elif opts.model in ["DiUj2017", "Me2017", "SmCh2017", "Ka2017", "Ka2017x2", "Ka2017_TrPi2018", "Ka2017x2inc", "Bu2019lf", "Bu2019lr", "Bu2019lm","Bu2019lw"]:
            bounds = [0.0,0.3]
            xlims = [0.0,0.3]
            ylims = [1e-1,40] 

        plotName = "%s/vej.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        bins, hist1 = lightcurve_utils.hist_results(vej_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,drawstyle='steps-mid',label="GW")
        bins, hist1 = lightcurve_utils.hist_results(vej_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,drawstyle='steps-mid',label="EM")
        bins, hist1 = lightcurve_utils.hist_results(vej_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,drawstyle='steps-mid',label="GW-EM")
        #plt.semilogy([vej_true,vej_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()
   
        plotName = "%s/combined.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        h = pu.plot_kde_posterior_2d(data_combined,cmap='viridis')
        #pu.plot_greedy_kde_interval_2d(pts_em,np.array([0.5]),colors='b')
        #pu.plot_greedy_kde_interval_2d(pts_gw,np.array([0.5]),colors='g')
        pu.plot_greedy_kde_interval_2d(data_combined,np.array([0.5]),colors='r')
        plt.plot(mej_true,vej_true,'kx',markersize=20)
        plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$")
        plt.ylabel(r"${v}_{\rm ej}$")
        plt.savefig(plotName)
        plt.close('all')

        m1_m2_em = np.vstack((data[:,0],data[:,2])).T

        plotName = "%s/combined_m1_m2.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        h = pu.plot_kde_posterior_2d(m1_m2_em,cmap='viridis')
        pu.plot_greedy_kde_interval_2d(m1_m2_em,np.array([0.5]),colors='r')
        pu.plot_greedy_kde_interval_2d(m1_m2_gw,np.array([0.5]),colors='k') 
        plt.xlabel(r"$m_1$")
        plt.ylabel(r"$m_2$")
        plt.savefig(plotName)
        plt.close('all')

    elif opts.doMasses:
        if opts.model == "KaKy2016":
            bounds = [0.8,10.0]
            xlims = [0.8,10.0]
            ylims = [1e-1,10]
        elif opts.model == "DiUj2017" or opts.model == "Me2017" or opts.model == "SmCh2017":
            bounds = [0.8,2.0]
            xlims = [0.8,2.0]
            ylims = [1e-1,10]
        elif opts.model in ["DiUj2017", "Me2017", "SmCh2017", "Ka2017", "Ka2017x2", "Ka2017_TrPi2018", "Ka2017x2inc", "Bu2019lf", "Bu2019lr", "Bu2019lm", "Bu2019lw"]:
            bounds = [0.8,2.0]
            xlims = [0.8,2.0]
            ylims = [1e-1,40]

        plotName = "%s/mchirp.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        bins, hist1 = lightcurve_utils.hist_results(mchirp_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,drawstyle='steps-mid',label="GW")
        bins, hist1 = lightcurve_utils.hist_results(mchirp_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,drawstyle='steps-mid',label="EM")
        bins, hist1 = lightcurve_utils.hist_results(mchirp_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',drawstyle='steps-mid',linewidth=3,label="GW-EM")
        #plt.semilogy([mchirp_true,mchirp_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"${\rm M}_{\rm c}$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()

        if opts.model == "KaKy2016":
            bounds = [2.9,9.1]
            xlims = [2.9,9.1]
            ylims = [1e-1,10]
        elif opts.model == "DiUj2017" or opts.model == "Me2017" or opts.model == "SmCh2017":
            bounds = [0.0,2.0]
            xlims = [0.9,2.0]
            ylims = [1e-1,10]
        elif opts.model in ["DiUj2017", "Me2017", "SmCh2017", "Ka2017", "Ka2017x2", "Ka2017_TrPi2018", "Ka2017x2inc", "Bu2019lf", "Bu2019lr", "Bu2019lm", "Bu2019lw"]:
            bounds = [0.0,2.0]
            xlims = [0.9,2.0]
            ylims = [1e-1,10]

        plotName = "%s/q.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        bins, hist1 = lightcurve_utils.hist_results(q_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,drawstyle='steps-mid',label="GW")
        bins, hist1 = lightcurve_utils.hist_results(q_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,drawstyle='steps-mid',label="EM")
        bins, hist1 = lightcurve_utils.hist_results(q_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,drawstyle='steps-mid',label="GW-EM")
        #plt.semilogy([q_true,q_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"$q$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()

        plotName = "%s/combined.pdf"%(plotDir)
        plt.figure(figsize=(12,8))
        h = pu.plot_kde_posterior_2d(data_combined,cmap='viridis')
        #pu.plot_greedy_kde_interval_2d(pts_em,np.array([0.5]),colors='b')
        #pu.plot_greedy_kde_interval_2d(pts_gw,np.array([0.5]),colors='g')
        pu.plot_greedy_kde_interval_2d(data_combined,np.array([0.5]),colors='r')
        plt.plot(q_true,mchirp_true,'kx',markersize=20)
        plt.ylabel(r"${\rm M}_{\rm c}$")
        plt.xlabel(r"$q$")
        plt.savefig(plotName)
        plt.close('all')

if opts.model == "KaKy2016":
    q_min = 3.0
    q_max = 9.0
    chi_min = -1.0
    chi_max = 1.0

    qlin = np.linspace(q_min,q_max,50)
    chilin = np.linspace(chi_min,chi_max,51)

    qlin = (qlin[:-1] + qlin[1:])/2.0
    chilin = (chilin[:-1] + chilin[1:])/2.0

    QGRID,CHIGRID = np.meshgrid(qlin,chilin)
    MGRID = np.zeros(QGRID.shape).T
    VGRID = np.zeros(QGRID.shape).T

    c = 0.147
    mb = 1.47
    mns = 1.35
    for ii in range(len(qlin)):
        for jj in range(len(chilin)):
            MGRID[ii,jj] = KaKy2016.calc_meje(qlin[ii],chilin[jj],c,mb,mns)
            VGRID[ii,jj] = KaKy2016.calc_vave(qlin[ii])

    plt.figure(figsize=(12,10))
    plt.pcolormesh(QGRID,CHIGRID,MGRID.T,vmin=np.min(MGRID),vmax=np.max(MGRID))
    plt.xlabel("Mass Ratio")
    plt.ylabel(r"$\chi$")
    plt.xlim([q_min,q_max])
    plt.ylim([chi_min,chi_max])
    cbar = plt.colorbar()
    cbar.set_label(r'log_{10} ($M_{ej})$')
    plotName = os.path.join(plotDir,'mej_pcolor.pdf')
    plt.savefig(plotName)
    plt.close('all')

    plt.figure(figsize=(12,10))
    plt.pcolormesh(QGRID,CHIGRID,VGRID.T,vmin=np.min(VGRID),vmax=np.max(VGRID))
    plt.xlabel("Mass Ratio")
    plt.ylabel(r"$\chi$")
    plt.xlim([q_min,q_max])
    plt.ylim([chi_min,chi_max])
    cbar = plt.colorbar()
    cbar.set_label(r'$v_{ej}$')
    plotName = os.path.join(plotDir,'vej_pcolor.pdf')
    plt.savefig(plotName)
    plt.close('all')
