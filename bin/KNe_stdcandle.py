
# coding: utf-8

# In[49]:


from __future__ import division, print_function # python3 compatibilty
import optparse
import pandas
import numpy as np                  # import numpy
from time import time               # use for timing functions
import pickle
# make the plots look a bit nicer with some defaults
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# functions for plotting posteriors
import scipy
import corner
#------------------------------------------------------------
# Read the data

import scipy.stats as ss

import pymultinest
import os

from gwemlightcurves.KNModels import KNTable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, RationalQuadratic

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")

    parser.add_option("-a","--analysis_type",default="combined", help="measured,inferred,inferred_bulla,inferred_bulla_lanthanide,combined")

    parser.add_option("-f","--fit_type",default="linear", help="linear,gpr")
    parser.add_option("-l","--lanthanide_type",default="lr", help="lr,lf")

    parser.add_option("--nsamples",default=-1,type=int)

    #parser.add_option("--multinest_samples", default="../plots/gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat")
    #parser.add_option("--multinest_samples", default="../plots/gws/Ka2017_old/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat")
    parser.add_option("--multinest_samples", default="../plots/gws/Ka2017/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/0.10/2-post_equal_weights.dat")
    parser.add_option("-m","--model",default="Ka2017", help="Ka2017,Ka2017x2")

    parser.add_option("--posterior_samples", default="../data/event_data/GW170817_SourceProperties_high_spin.dat,../data/event_data/GW170817_SourceProperties_low_spin.dat")

    parser.add_option("-e","--errorbudget",default=0.10,type=float)

    opts, args = parser.parse_args()

    return opts


def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[:int(Npts/2)]
    den_pts = pts[int(Npts/2):]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir

def kde_eval_single(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    td = kde(truth)

    return td

def myprior_combined(cube, ndim, nparams):
        cube[0] = cube[0]*30.0 - 30.0
        cube[1] = cube[1]*10.0 - 5.0
        cube[2] = cube[2]*20.0 - 10.0
        cube[3] = cube[3]*100.0 - 50.0
        cube[4] = cube[4]*50.0 - 25.0
        cube[5] = cube[5]*50.0 - 25.0
	cube[6] = cube[6]*1.0

def myloglike_combined(cube, ndim, nparams):
        const = cube[0]
        alpha = cube[1]
        beta = cube[2]
        gamma = cube[3]
        delta = cube[4]
        zeta = cube[5]
        sigma = cube[6]

        M = const + alpha*(dmdt) + beta*(color) + gamma*(mej) + delta*(vej) + zeta*(Xlan)
        x = Mag - M
        prob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
        prob = np.sum(prob)

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myprior_inferred(cube, ndim, nparams):
        cube[0] = cube[0]*30.0 - 30.0
        cube[1] = cube[1]*100.0 - 50.0
        cube[2] = cube[2]*50.0 - 25.0
        cube[3] = cube[3]*50.0 - 25.0
        cube[4] = cube[4]*1.0

def myloglike_inferred(cube, ndim, nparams):
        tau = cube[0]
        nu = cube[1]
        delta = cube[2]
        zeta = cube[3]
        sigma = cube[4]

        M = tau + nu*(mej) + delta*(vej) + zeta*(Xlan)
        x = Mag - M
        prob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
        prob = np.sum(prob)

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myprior_measured(cube, ndim, nparams):
        cube[0] = cube[0]*30.0 - 30.0
        cube[1] = cube[1]*10.0 - 5.0
        cube[2] = cube[2]*20.0 - 10.0
        cube[3] = cube[3]*100.0 - 50.0
        cube[4] = cube[4]*10.0

def myloglike_measured(cube, ndim, nparams):
        kappa = cube[0]
        alpha = cube[1]
        beta = cube[2]
        gamma = cube[3]
        sigma = cube[4]

        M = kappa + alpha*(dmdt) + beta*(color) + gamma*dmdti
        x = Mag - M
        prob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
        prob = np.sum(prob)

        if np.isnan(prob):
            prob = -np.inf
        return prob

def myprior_H0(cube, ndim, nparams):
        cube[0] = cube[0]*200.0
        cube[1] = cube[1]*100.0
        cube[2] = cube[2]*600.0

def myloglike_H0(cube, ndim, nparams):
        H0 = cube[0]
        d = cube[1]
        vp = cube[2]

        vr_mu, vr_std = 3327.0, 72.0
        vr = scipy.stats.norm.rvs(vr_mu, vr_std)

        vp_mu, vp_std = 310, 150
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr_mu-vp-H0*d)/vr_std)**2)
        pvp = (1/np.sqrt(2*np.pi*vp_std**2))*np.exp((-1/2.0)*((vp_mu-vp)/vp_std)**2)
        prob_dist = kde_eval_single(kdedir_dist,[d])[0]
        #print(H0, d, vp, np.log(pvr), np.log(pvp), np.log(prob_dist))

        prob = np.log(pvr) + np.log(pvp) + np.log(prob_dist)

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myloglike_H0_GW(cube, ndim, nparams):
        H0 = cube[0]
        d = cube[1]
        vp = cube[2]

        vr_mu, vr_std = 3327.0, 72.0
        vr = scipy.stats.norm.rvs(vr_mu, vr_std)

        vp_mu, vp_std = 310, 150
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr_mu-vp-H0*d)/vr_std)**2)
        pvp = (1/np.sqrt(2*np.pi*vp_std**2))*np.exp((-1/2.0)*((vp_mu-vp)/vp_std)**2)
        prob_dist = kde_eval_single(kdedir_dist,[d])[0]
        prob_gwdist = kde_eval_single(kdedir_gwdist,[d])[0]

        prob = np.log(pvr) + np.log(pvp) + np.log(prob_gwdist)

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myloglike_H0_GWEM(cube, ndim, nparams):
        H0 = cube[0]
        d = cube[1]
        vp = cube[2]

        vr_mu, vr_std = 3327.0, 72.0
        vr = scipy.stats.norm.rvs(vr_mu, vr_std)

        vp_mu, vp_std = 310, 150
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr_mu-vp-H0*d)/vr_std)**2)
        pvp = (1/np.sqrt(2*np.pi*vp_std**2))*np.exp((-1/2.0)*((vp_mu-vp)/vp_std)**2)
        prob_dist = kde_eval_single(kdedir_dist,[d])[0]
        prob_gwdist = kde_eval_single(kdedir_gwdist,[d])[0]

        prob = np.log(pvr) + np.log(pvp) + np.log(prob_dist) + np.log(prob_gwdist)

        if np.isnan(prob):
            prob = -np.inf

        return prob

# Parse command line
opts = parse_commandline()

baseplotDir = os.path.join(opts.plotDir,'standard_candles')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

plotDir = os.path.join(baseplotDir,opts.analysis_type)
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

ModelPath = '%s/svdmodels'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'palegreen'

if opts.analysis_type == "inferred_bulla":
    filename = os.path.join(opts.dataDir, 'standard_candles', 'magcolor_bulla.dat')
    data = np.loadtxt(filename)
    mej, phi, theta, color, Mag, dmdti, dmdt = data.T
    mej = np.log10(mej)
    dmdt = np.log10(dmdt)
    dmdti = np.log10(dmdti)
elif opts.analysis_type == "inferred_bulla_lanthanide":
    filename = os.path.join(opts.dataDir, 'standard_candles', 'magcolor_bulla_%s.dat' % opts.lanthanide_type)
    data = np.loadtxt(filename)
    mejdyn, mejwind, phi, theta, color, Mag, dmdti, dmdt = data.T
    mejdyn = np.log10(mejdyn)
    mejwind = np.log10(mejwind)
    dmdt = np.log10(dmdt)
    dmdti = np.log10(dmdti)
else:
    filename = os.path.join(opts.dataDir, 'standard_candles', 'magcolor.dat')
    data = np.loadtxt(filename)
    mej, vej, Xlan, color, Mag, dmdti, dmdt = data.T
    mej, Xlan = np.log10(mej), np.log10(Xlan)
    dmdt = np.log10(dmdt)
    dmdti = np.log10(dmdti)

posterior_samples = opts.posterior_samples.split(",")
samples_all = {}
for posterior_sample in posterior_samples:
    key = posterior_sample.replace(".dat","").split("/")[-1].split("_")[-2]
    samples_all[key] = KNTable.read_samples(posterior_sample)

if opts.fit_type == "gpr":
    #idx = np.arange(len(Mag))
    #np.random.shuffle(idx)
    #idx1, idx2 = np.array_split(idx,2)

    if opts.analysis_type == "measured":
        param_array = np.vstack((color,dmdt,dmdti)).T
    elif opts.analysis_type == "inferred":
        param_array = np.vstack((mej,vej,Xlan)).T
    elif opts.analysis_type == "inferred_bulla":
        param_array = np.vstack((mej,phi,theta)).T
    elif opts.analysis_type == "inferred_bulla_lanthanide":
        param_array = np.vstack((mejdyn,mejwind,phi,theta)).T

    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
    for i in range(len(param_mins)):
        param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    nsvds, nparams = param_array_postprocess.shape
    kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0,alpha=1.0)
    gp.fit(param_array_postprocess, Mag)

    M, sigma2_pred = gp.predict(np.atleast_2d(param_array_postprocess), return_std=True)
    sigma_best = np.median(sigma2_pred)
    sigma = sigma_best*np.ones(M.shape)
    sigma = sigma2_pred

    if opts.analysis_type == "inferred_bulla":
        idy = np.where( (param_array[:,0] >= -3) & (param_array[:,0] <= -1) & (param_array[:,1] >= 15) & (param_array[:,1] <= 30) & (param_array[:,2] >= 0) & (param_array[:,2] <= 30))[0]
        param_array_slice = param_array[idy,:]

        param_array_slice_postprocess = np.array(param_array_slice)
        param_slice_mins, param_slice_maxs = np.min(param_array_slice_postprocess,axis=0),np.max(param_array_slice_postprocess,axis=0)
        for i in range(len(param_slice_mins)):
            param_array_slice_postprocess[:,i] = (param_array_slice_postprocess[:,i]-param_slice_mins[i])/(param_slice_maxs[i]-param_slice_mins[i])
    
        nsvds, nparams = param_array_slice_postprocess.shape
        kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
        gp_slice = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0,alpha=1.0)
        gp_slice.fit(param_array_slice_postprocess, Mag[idy])
    
        M_slice, sigma2_slice_pred = gp_slice.predict(np.atleast_2d(param_array_slice_postprocess), return_std=True)
        sigma_slice_best = np.median(sigma2_slice_pred)
        sigma_slice = sigma_slice_best*np.ones(M.shape)
        sigma_slice = sigma2_pred

elif opts.fit_type == "linear":

    if opts.analysis_type == "combined":
        parameters = ["K","alpha","beta","gamma","delta","zeta","sigma"]
        labels = [r'K', r'$\alpha$', r'$\beta$', r'$\gamma$', r"$\delta$",r"$\zeta$",r'$\sigma$']
        n_params = len(parameters)

        pymultinest.run(myloglike_combined, myprior_combined, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

        multifile = "%s/2-post_equal_weights.dat"%plotDir
        data = np.loadtxt(multifile)

        K, alpha, beta, gamma, delta, zeta, sigma, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
        idx = np.argmax(loglikelihood)
        K_best, alpha_best, beta_best, gamma_best, delta_best, zeta_best, sigma_best = data[idx,0:-1]

        M = K_best + alpha_best*(dmdt) + beta_best*(color) + gamma_best*(mej) + delta_best*(vej) + zeta_best*Xlan
    elif opts.analysis_type == "measured":
        parameters = ["kappa","alpha","beta","gamma","sigma"]
        labels = [r'$\kappa$', r'$\alpha$', r'$\beta$',r'$\gamma$',r'$\sigma$']
        n_params = len(parameters)

        pymultinest.run(myloglike_measured, myprior_measured, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

        multifile = "%s/2-post_equal_weights.dat"%plotDir
        data = np.loadtxt(multifile)

        kappa, alpha, beta, gamma, sigma, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        idx = np.argmax(loglikelihood)
        kappa_best, alpha_best, beta_best, gamma_best, sigma_best = data[idx,0:-1]

        M = kappa_best + alpha_best*(dmdt) + beta_best*(color) + gamma_best*dmdti
    elif opts.analysis_type == "inferred":
        parameters = ["tau","nu","delta","zeta","sigma"]
        labels = [r'$\tau$', r'$\nu$', r"$\delta$",r"$\zeta$",r'$\sigma$']
        n_params = len(parameters)

        pymultinest.run(myloglike_inferred, myprior_inferred, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

        multifile = "%s/2-post_equal_weights.dat"%plotDir
        data = np.loadtxt(multifile)

        tau, nu, delta, zeta, sigma, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
        idx = np.argmax(loglikelihood)
        tau_best, nu_best, delta_best, zeta_best, sigma_best = data[idx,0:-1]

        M = tau_best + nu_best*(mej) + delta_best*(vej) + zeta_best*Xlan

    sigma = sigma_best*np.ones(M.shape)

    plotName = "%s/corner.pdf"%(plotDir)
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
    figure.set_size_inches(18.0,18.0)
    plt.savefig(plotName)
    plt.close()

if opts.analysis_type == "inferred_bulla":

    phi_unique, theta_unique = np.unique(phi), np.unique(theta)
    phi_unique = phi_unique[::-1]

    fig = plt.figure(figsize=(28, 16))
    gs = gridspec.GridSpec(len(phi_unique), len(theta_unique),
                           hspace=0.05, wspace=0.30)
    for ii in range(len(phi_unique)):
        for jj in range(len(theta_unique)):
            ax = fig.add_subplot(gs[ii, jj])
            plt.axes(ax)
            idx = np.where((phi == phi_unique[ii]) & (theta == theta_unique[jj]))[0]

            plt.errorbar(10**mej[idx], M[idx], sigma[idx], fmt='k.')
            plt.plot(10**mej[idx], Mag[idx], 'bo')
            if not ii == len(phi_unique) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel('$\Theta = %.0f^\circ$' % theta_unique[jj], fontsize=24)
            if not jj == 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.ylabel('$\Phi = %.0f^\circ$' % phi_unique[ii], fontsize=24)

            plt.xlim([0.000005,1.5])
            #plt.ylim([-17,-9])
            #plt.ylim([-10,0])
            plt.gca().invert_yaxis()
            ax.set_xscale('log')

    fig.text(0.5, 0.03, 'Ejecta Mass [solar mass]', ha='center', fontsize=30)
    fig.text(0.05, 0.5, 'Absolute Magnitude', va='center', rotation='vertical', fontsize=30)
    plt.show()
    plotName = os.path.join(plotDir,'fitall.pdf')
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()

    M_trials = np.linspace(np.min(Mag[idy]), np.max(Mag[idy]), 100)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
    plt.axes(ax1)
    plt.errorbar(Mag[idy], M[idy], sigma2_pred[idy], fmt='.', color=color1, zorder=5, label='Broad Priors')
    plt.errorbar(Mag[idy], M_slice, sigma2_slice_pred, fmt='o', color=color2, label='Realistic Priors')
    plt.plot(M_trials, M_trials, '--', color=color3, zorder=10, linewidth=3, alpha=0.5)
    plt.ylabel('Magnitude [Fit]')
    plt.legend(loc=2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.gca().invert_yaxis()
    plt.axes(ax2)
    plt.errorbar(Mag[idy], M[idy]-Mag[idy], sigma2_pred[idy], fmt='.', color=color1, zorder=5)
    plt.errorbar(Mag[idy], M_slice-Mag[idy], sigma2_slice_pred, fmt='o', color=color2)
    plt.plot(M_trials,0.0*np.ones(M_trials.shape), '--', color=color3, zorder=10, linewidth=3, alpha=0.5)
    plt.gca().invert_xaxis()
    plt.ylabel('Data - Fit')
    plt.xlabel('Magnitude [Data]')
    plt.show()
    plotName = os.path.join(plotDir,'fit.pdf')
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()

    print(stop)

elif opts.analysis_type == "inferred_bulla_lanthanide":

    phi_unique, theta_unique = np.unique(phi), np.unique(theta)
    phi_unique = phi_unique[::-1]

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(len(phi_unique), len(theta_unique))
    for ii in range(len(phi_unique)):
        for jj in range(len(theta_unique)):
            ax = fig.add_subplot(gs[ii, jj])
            plt.axes(ax)
            idx = np.where((phi == phi_unique[ii]) & (theta == theta_unique[jj]))[0]

            plt.errorbar(10**mejwind[idx], M[idx], sigma[idx], fmt='k.')
            plt.plot(10**mejwind[idx], Mag[idx], 'bo')
            if not ii == len(phi_unique) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel('$\Theta = %.0f\,K$' % theta_unique[jj], fontsize=24)
            if not jj == 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.ylabel('$\Phi = %.0f^\circ$' % phi_unique[ii], fontsize=24)

            plt.xlim([0.001,0.1])
            #plt.ylim([-17,-9])
            #plt.ylim([-10,0])
            plt.gca().invert_yaxis()
            ax.set_xscale('log')

    fig.text(0.5, 0.02, 'Ejecta Mass [solar mass]', ha='center', fontsize=30)
    fig.text(0.02, 0.5, 'Absolute Magnitude', va='center', rotation='vertical', fontsize=30)
    plt.show()
    plotName = os.path.join(plotDir,'fitall.pdf')
    plt.savefig(plotName)
    plt.close()

else:

    vej_unique, Xlan_unique = np.unique(vej), np.unique(Xlan)
    vej_unique = vej_unique[::-1]

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(len(vej_unique), len(Xlan_unique))
    for ii in range(len(vej_unique)):
        for jj in range(len(Xlan_unique)):
            ax = fig.add_subplot(gs[ii, jj])
            plt.axes(ax)
            idx = np.where((vej == vej_unique[ii]) & (Xlan == Xlan_unique[jj]))[0]
            plt.errorbar(10**mej[idx], M[idx], sigma[idx], fmt='k.')
            plt.plot(10**mej[idx], Mag[idx], 'bo')
            if not ii == len(vej_unique) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel('$X_{\mathrm{lan}} = 10^{%.0f}$' % Xlan_unique[jj], fontsize=24)
            if not jj == 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.ylabel('$v_{\mathrm{ej}} = %.2f\,c$' % vej_unique[ii], fontsize=24)

            plt.xlim([0.0005,0.2])
            plt.ylim([-17,-9])
            #plt.ylim([-10,0])
            plt.gca().invert_yaxis()
            ax.set_xscale('log')

    fig.text(0.5, 0.02, 'Ejecta Mass [solar mass]', ha='center', fontsize=30)
    fig.text(0.02, 0.5, 'Absolute Magnitude', va='center', rotation='vertical', fontsize=30)
    plt.show()
    plotName = os.path.join(plotDir,'fitall.pdf')
    plt.savefig(plotName)
    plt.close()

    print(stop)

M_trials = np.linspace(np.min(Mag), np.max(Mag), 100)

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)
plt.errorbar(Mag, M, sigma_best*np.ones(M.shape), fmt='k.')
plt.plot(M_trials, M_trials, 'b--')
plt.ylabel('Magnitude [Fit]')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.gca().invert_yaxis()
plt.axes(ax2)
plt.errorbar(Mag,M-Mag, sigma_best*np.ones(M.shape), fmt='k.')
plt.gca().invert_xaxis()
plt.ylabel('Data - Fit')
plt.xlabel('Magnitude [Data]')
plt.show()
plotName = os.path.join(plotDir,'fit.pdf')
plt.savefig(plotName, bbox_inches='tight')
plt.close()

samples = KNTable.read_multinest_samples(opts.multinest_samples, opts.model)
if not "FixZPT0" in opts.multinest_samples:
    ZPRange = 5.0
    zp_mu, zp_std = 0.0, 5.0
    #samples["zp"] = scipy.stats.norm(zp_mu, zp_std).ppf(samples["zp"])
if opts.nsamples > 0:
    samples = samples.downsample(Nsamples=opts.nsamples)


# restrict ejecta masses
if opts.analysis_type == "inferred_bulla_lanthanide":
    samples = samples[samples["mej_dyn"] < 0.1]
    samples = samples[samples["mej_wind"] < 0.1]
else:
    samples = samples[samples["mej"] < 0.1]

# These are the default values supplied with respect to generating lightcurves
tini = 0.1
tmax = 14.0
dt = 0.1
#add default values from above to table
samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt

kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
pcklFile = os.path.join(plotDir,"data.pkl")
if os.path.isfile(pcklFile):
    f = open(pcklFile, 'r')
    (model_table) = pickle.load(f)
    f.close()
else:
    model_table = KNTable.model(opts.model, samples, **kwargs)
    f = open(pcklFile, 'wb')
    pickle.dump((model_table), f)
    f.close()

print(len(samples), len(model_table))

N = 1000
idx = np.random.randint(0, high=len(samples), size=N)

if opts.fit_type == "linear":
    if opts.analysis_type == "combined":
        K_mean, K_std = np.mean(K), np.std(K)
        alpha_mean, alpha_std = np.mean(alpha), np.std(alpha)
        beta_mean, beta_std = np.mean(beta), np.std(beta)
        gamma_mean, gamma_std = np.mean(gamma), np.std(gamma)
        delta_mean, delta_std = np.mean(delta), np.std(delta)
        zeta_mean, zeta_std = np.mean(zeta), np.std(zeta)
        sigma_mean, sigma_std = np.mean(sigma), np.std(sigma)

        K = np.random.normal(loc=K_mean, scale=K_std, size=N)
        alpha = np.random.normal(loc=alpha_mean, scale=alpha_std, size=N)
        beta = np.random.normal(loc=beta_mean, scale=beta_std, size=N)
        gamma = np.random.normal(loc=gamma_mean, scale=gamma_std, size=N)
        delta = np.random.normal(loc=delta_mean, scale=delta_std, size=N)
        zeta = np.random.normal(loc=zeta_mean, scale=zeta_std, size=N)
        sigma = np.random.normal(loc=0.0, scale=sigma_mean, size=N)
    elif opts.analysis_type == "inferred":
        tau_mean, tau_std = np.mean(tau), np.std(tau)
        nu_mean, nu_std = np.mean(nu), np.std(nu)
        delta_mean, delta_std = np.mean(delta), np.std(delta)
        zeta_mean, zeta_std = np.mean(zeta), np.std(zeta)
        sigma_mean, sigma_std = np.mean(sigma), np.std(sigma)

        tau = np.random.normal(loc=tau_mean, scale=tau_std, size=N)
        nu = np.random.normal(loc=nu_mean, scale=nu_std, size=N)
        delta = np.random.normal(loc=delta_mean, scale=delta_std, size=N)
        zeta = np.random.normal(loc=zeta_mean, scale=zeta_std, size=N)
        sigma = np.random.normal(loc=0.0, scale=sigma_mean, size=N)
    elif opts.analysis_type == "measured":
        kappa_mean, kappa_std = np.mean(kappa), np.std(kappa)
        alpha_mean, alpha_std = np.mean(alpha), np.std(alpha)
        beta_mean, beta_std = np.mean(beta), np.std(beta)
        gamma_mean, gamma_std = np.mean(gamma), np.std(gamma)
        sigma_mean, sigma_std = np.mean(sigma), np.std(sigma)

        kappa = np.random.normal(loc=kappa_mean, scale=kappa_std, size=N)
        alpha = np.random.normal(loc=alpha_mean, scale=alpha_std, size=N)
        beta = np.random.normal(loc=beta_mean, scale=beta_std, size=N)
        gamma = np.random.normal(loc=gamma_mean, scale=gamma_std, size=N)
        sigma = np.random.normal(loc=0.0, scale=sigma_mean, size=N)
elif opts.fit_type == "gpr":
    sigma = np.random.normal(loc=0.0, scale=sigma_best, size=N)

if opts.model in ["Bu2019inc","Bu2019lr","Bu2019lf","Bu2019lm"]:
    incs = np.empty((0,1))

mus = np.empty((0,1))
for ii in range(N):
    row = samples[idx[ii]]
    zp = row["zp"]

    model = model_table[idx[ii]]
    mag = model['mag']
    t = model['t']
    Kband = mag[-1]
    iband = mag[-6]
    gband = mag[-8]

    jj = np.argmin(Kband)
    jj7 = np.argmin(np.abs(t-(t[jj]+7.0)))
    M_K = Kband[jj] + 5*(np.log10(40.7*1e6) - 1)
    col = iband[jj] - Kband[jj]
    m7 = Kband[jj7]-Kband[jj]
    kk = np.argmin(iband)
    kk7 = np.argmin(np.abs(t-(t[kk]+7.0)))
    m7i = iband[kk7]-iband[kk]
    ll = np.argmin(gband)

    M_i = iband[jj] - iband[kk]

    if opts.analysis_type == "inferred_bulla":
        mej, phi, theta = row['mej'], row['phi'], row['theta']
    elif opts.analysis_type == "inferred_bulla_lanthanide":
        mejdyn, mejwind, phi, theta = row['mej_dyn'], row['mej_wind'], row['phi'], row['theta']
    else:
        mej, vej, Xlan = row['mej'], row['vej'], row['Xlan']

    if opts.fit_type == "linear":
        if opts.analysis_type == "combined":
            mu = -( -M_K + K[ii] + m7*alpha[ii] + col*beta[ii] + np.log10(mej)*gamma[ii] + vej*delta[ii] + np.log10(Xlan)*zeta[ii] + sigma[ii])
        elif opts.analysis_type == "inferred":
            mu = -( -M_K + tau[ii] + np.log10(mej)*nu[ii] + vej*delta[ii] + np.log10(Xlan)*zeta[ii] + sigma[ii])
        elif opts.analysis_type == "measured":
            mu = -( -M_K + kappa[ii] + m7*alpha[ii] + col*beta[ii] + M_i*gamma[ii] + sigma[ii])
    elif opts.fit_type == "gpr":
        if opts.analysis_type == "combined":
            mu = -( -M_K + K[ii] + m7*alpha[ii] + col*beta[ii] + np.log10(mej)*gamma[ii] + vej*delta[ii] + np.log10(Xlan)*zeta[ii] + sigma[ii])
        elif opts.analysis_type == "inferred":
            param_list_postprocess = np.array([np.log10(mej),vej,np.log10(Xlan)])
            for i in range(len(param_mins)):
                param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

            M_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            sigma1 = np.random.normal(loc=0.0, scale=sigma2_pred, size=100)
            mu = -( -M_K + M_pred + sigma[ii])
            #mu = -( -M_K + M_pred + sigma1)
        elif opts.analysis_type == "inferred_bulla":
            param_list_postprocess = np.array([np.log10(mej),phi,theta])
            for i in range(len(param_mins)):
                param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

            M_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            mu = -( -M_K + M_pred + sigma[ii])
        elif opts.analysis_type == "inferred_bulla_lanthanide":
            param_list_postprocess = np.array([np.log10(mejdyn),np.log10(mejwind),phi,theta])
            for i in range(len(param_mins)):
                param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

            M_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            mu = -( -M_K + M_pred + sigma[ii])
        elif opts.analysis_type == "measured":
            if m7 == 0:
                m7 = 0.01
            if M_i == 0:
                M_i = 0.01
            param_list_postprocess = np.array([col,np.log10(m7),np.log10(m7i)])
            for i in range(len(param_mins)):
                param_list_postprocess[i] = (param_list_postprocess[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

            M_pred, sigma2_pred = gp.predict(np.atleast_2d(param_list_postprocess), return_std=True)
            mu = -( -M_K + M_pred + sigma[ii])

    mu = mu - 0.9*zp
    #mu = mu + zp

    mus = np.append(mus,mu)

    if opts.model in ["Bu2019inc","Bu2019lr","Bu2019lf","Bu2019lm"]:
        incs = np.append(incs,row["theta"])

mus = np.array(mus)
dist = 10**((mus/5.0) + 1.0) / 1e6
print(mus, dist)
kdedir_dist = greedy_kde_areas_1d(dist)

if opts.model in ["Bu2019inc","Bu2019lr","Bu2019lf","Bu2019lm"]:
    filename = os.path.join(plotDir,'dist_inc.dat')
    fid = open(filename,'w')
    for d, i in zip(dist, incs):
        fid.write('%.5f %.5f\n' % (d, i))
    fid.close()
else:
    filename = os.path.join(plotDir,'dist.dat')
    fid = open(filename,'w')
    for d in dist:
        fid.write('%.5f\n' % (d))
    fid.close()

print(plotDir)

bin_edges = np.arange(5,85,2)

z = 0.009783
c = 3.e5   # speed of light in km/s
H0 = (c/dist)*z

dist_16, dist_50, dist_84 = np.percentile(dist,16), np.percentile(dist,50), np.percentile(dist,84)

hist_1, bin_edges_1 = np.histogram(dist, bin_edges, density=True)
hist_2, bin_edges_2 = np.histogram(H0, 20, density=True)
bins_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2.0
bins_2 = (bin_edges_2[:-1] + bin_edges_2[1:])/2.0

xticks_1 = np.array([10,20,30,40,50,60])
xticks_2 = (c/xticks_1)*z

fig = plt.figure(figsize=(10,7))

#plt.plot([dist_10,dist_10],[0,1],'--',color=color1)
#plt.plot([dist_50,dist_50],[0,1],'--',color=color1)
#plt.plot([dist_90,dist_90],[0,1],'--',color=color1)
plt.step(bins_1, hist_1, color = color1, linestyle='-',label='EM')
plt.xticks(xticks_1)
plt.xlim([0,80])

color_names = [color2, color3]
for ii, key in enumerate(samples_all.keys()):
    samples = samples_all[key]
    label = 'GW (%s)' % key
    gwdist = samples['luminosity_distance_Mpc']
    hist_1, bin_edges_1 = np.histogram(gwdist, bin_edges, density=True)
    bins_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2.0
    if key == "high":
        linestyle='--'
    else:
        linestyle='-.'
    plt.step(bins_1, hist_1, color = color_names[ii], linestyle=linestyle,label=label)

gwdist = samples_all['high']['luminosity_distance_Mpc']
kdedir_gwdist = greedy_kde_areas_1d(gwdist)

plt.legend()
plt.xlabel('Distance [Mpc]')
plt.ylabel('Probability')
plt.ylim([0,0.10])
plt.grid(True)
plt.show()
plotName = os.path.join(plotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

H0Dir = os.path.join(plotDir,'H0')
if not os.path.isdir(H0Dir):
    os.makedirs(H0Dir)

parameters = ["H0","d","vp"]
labels = [r'$H_0$', r'$D$', r"$v_p$"]
n_params = len(parameters)

pymultinest.run(myloglike_H0, myprior_H0, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%H0Dir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%H0Dir
data = np.loadtxt(multifile)

H0_EM, d, vp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
idx = np.argmax(loglikelihood)
H0_best, d_best, vp_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(H0Dir)
figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                   smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

H0GWDir = os.path.join(plotDir,'H0GW')
if not os.path.isdir(H0GWDir):
    os.makedirs(H0GWDir)

parameters = ["H0","d","vp"]
labels = [r'$H_0$', r'$D$', r"$v_p$"]
n_params = len(parameters)

pymultinest.run(myloglike_H0_GW, myprior_H0, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%H0GWDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%H0GWDir
data = np.loadtxt(multifile)

H0_GW, d, vp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
idx = np.argmax(loglikelihood)
H0_best, d_best, vp_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(H0GWDir)
figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                   smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

H0GWEMDir = os.path.join(plotDir,'H0GWEM')
if not os.path.isdir(H0GWEMDir):
    os.makedirs(H0GWEMDir)

parameters = ["H0","d","vp"]
labels = [r'$H_0$', r'$D$', r"$v_p$"]
n_params = len(parameters)

pymultinest.run(myloglike_H0_GWEM, myprior_H0, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%H0GWEMDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%H0GWEMDir
data = np.loadtxt(multifile)

H0_GWEM, d, vp, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
idx = np.argmax(loglikelihood)
H0_best, d_best, vp_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(H0GWEMDir)
figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                   smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

bin_edges = np.arange(5,150,5)
hist_1, bin_edges_1 = np.histogram(H0_EM, bin_edges, density=True)
hist_2, bin_edges_2 = np.histogram(H0_GW, bin_edges, density=True)
hist_3, bin_edges_3 = np.histogram(H0_GWEM, bin_edges, density=True)
bins = (bin_edges[:-1] + bin_edges[1:])/2.0

bins_small = np.arange(5,150,1)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)

plt.step(bins, hist_1, color = color1, linestyle='-',label='EM')
plt.step(bins, hist_2, color = color2, linestyle='--',label='GW')
plt.step(bins, hist_3, color = color3, linestyle='-.',label='GW-EM')
plt.plot(bins_small, ss.norm.pdf(bins_small, loc=68.9, scale=4.6), color='pink', label='Superluminal')

boxes = []
planck_mu, planck_std = 67.74, 0.46
shoes_mu, shoes_std = 74.03, 1.42
plt.plot([planck_mu,planck_mu],[0,1],alpha=0.3, color='g',label='Planck')
rect1 = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.3, color='g')
rect2 = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.1, color='g')
plt.plot([shoes_mu,shoes_mu],[0,1],alpha=0.3, color='r',label='SHoES')
rect3 = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.3, color='r')
rect4 = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.1, color='r')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)

plt.xlabel('H0 [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.xlim([40,150])
plt.ylim([0,0.1])
plt.show()
plotName = os.path.join(plotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
H0_GW_16, H0_GW_50, H0_GW_84 = np.percentile(H0_GW,16), np.percentile(H0_GW,50), np.percentile(H0_GW,84)
H0_GWEM_16, H0_GWEM_50, H0_GWEM_84 = np.percentile(H0_GWEM,16), np.percentile(H0_GWEM,50), np.percentile(H0_GWEM,84)

print('Sigma: %.2f' % sigma_best)
print('Distance: %.0f +%.0f -%.0f' % (dist_50, dist_84-dist_50, dist_50-dist_16))
print('H0 EM: %.0f +%.0f -%.0f' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))
print('H0 GW: %.0f +%.0f -%.0f' % (H0_GW_50, H0_GW_84-H0_GW_50, H0_GW_50-H0_GW_16))
print('H0 GW-EM: %.0f +%.0f -%.0f' % (H0_GWEM_50, H0_GWEM_84-H0_GWEM_50, H0_GWEM_50-H0_GWEM_16))

pcklFile = os.path.join(plotDir,"H0.pkl")
f = open(pcklFile, 'wb')
pickle.dump((dist,samples_all,H0_EM,H0_GW,H0_GWEM, Mag, M, sigma_best), f)
f.close()
