
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

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")

    parser.add_option("-a","--analysis_type",default="combined", help="measured,inferred,combined")  
 
    parser.add_option("--nsamples",default=-1,type=int)

    parser.add_option("--multinest_samples", default="../plots/gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat")
    parser.add_option("-m","--model",default="Ka2017", help="Ka2017,Ka2017x2")

    parser.add_option("--posterior_samples", default="../data/event_data/GW170817_SourceProperties_high_spin.dat,../data/event_data/GW170817_SourceProperties_low_spin.dat")

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
        cube[3] = cube[3]*20.0 - 10.0
        cube[4] = cube[4]*10.0

def myloglike_measured(cube, ndim, nparams):
        kappa = cube[0]
        alpha = cube[1]
        beta = cube[2]
        gamma = cube[3]
        sigma = cube[4]

        M = kappa + alpha*(dmdt) + beta*(color) + gamma*Magi
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
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr-vp-H0*d)/vr_std)**2)
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
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr-vp-H0*d)/vr_std)**2)
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
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr-vp-H0*d)/vr_std)**2)
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

plotDir1 = os.path.join(baseplotDir,'inferred')
if not os.path.isdir(plotDir1):
    os.makedirs(plotDir1)

plotDir2 = os.path.join(baseplotDir,'measured')
if not os.path.isdir(plotDir2):
    os.makedirs(plotDir2)

pcklFile1 = os.path.join(plotDir1, "H0.pkl")
f = open(pcklFile1, 'r')
(dist_1,samples_all,H0_EM_1,H0_GW,H0_GWEM_1,Mag_1,M_1,sigma_best_1) = pickle.load(f)
f.close() 

pcklFile2 = os.path.join(plotDir2, "H0.pkl")
f = open(pcklFile2, 'r')
(dist_2,samples_all,H0_EM_2,H0_GW,H0_GWEM_2,Mag_2,M_2,sigma_best_2) = pickle.load(f)
f.close()

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'darkgreen'

M_trials = np.linspace(-20, 0, 100)

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)
plt.plot(M_trials, M_trials, '--', color=color3, zorder=10, linewidth=3, alpha=0.5)
plt.errorbar(Mag_1, M_1, sigma_best_1*np.ones(M_1.shape), fmt='.', color=color1, label='Inferred', zorder=5)
#plt.errorbar(Mag_2, M_2, sigma_best_2*np.ones(M_2.shape), fmt='o', color=color2, label='Measured')
plt.ylabel('Magnitude [Fit]')
plt.xlim([-17,-9])
plt.ylim([-17,-9])
plt.legend(loc=2)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.gca().invert_yaxis()
plt.axes(ax2)
plt.errorbar(Mag_1,M_1-Mag_1, sigma_best_1*np.ones(M_1.shape), fmt='.', color=color1, zorder=5)
plt.errorbar(Mag_2,M_2-Mag_2, sigma_best_2*np.ones(M_2.shape), fmt='o', color=color2)
plt.plot(M_trials,0.0*np.ones(M_trials.shape), '--', color=color3, zorder=10, linewidth=3, alpha=0.5)
plt.gca().invert_xaxis()
plt.ylim([-2.5,2.5])
plt.gca().invert_yaxis()
plt.ylabel('Data - Fit')
plt.xlabel('Magnitude [Data]')
plt.show()
plotName = os.path.join(baseplotDir,'fit.pdf')
plt.savefig(plotName, bbox_inches='tight')
plt.close()

bins = np.arange(0,120,0.5)
bin_edges = np.arange(-5,115,2)

hist_1, bin_edges_1 = np.histogram(dist_1, bin_edges, density=True)
hist_2, bin_edges_2 = np.histogram(dist_2, bin_edges, density=True)
bins_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2.0
bins_2 = (bin_edges_2[:-1] + bin_edges_2[1:])/2.0

kdedir_dist_1 = greedy_kde_areas_1d(dist_1)
kdedir_dist_2 = greedy_kde_areas_1d(dist_2)

xticks_1 = np.array([10,30,50,70,90,110])

fig = plt.figure(figsize=(9,6))
ax = plt.gca()

#plt.plot([dist_10,dist_10],[0,1],'--',color=color1)
#plt.plot([dist_50,dist_50],[0,1],'--',color=color1)
#plt.plot([dist_90,dist_90],[0,1],'--',color=color1)
plt.plot(bins, [kde_eval_single(kdedir_dist_1,[d])[0] for d in bins], color = color1, linestyle='-',label='EM (inferred)', linewidth=3)
plt.plot(bins, [kde_eval_single(kdedir_dist_2,[d])[0] for d in bins], color = color2, linestyle='-',label='EM (measured)', linewidth=3)

#plt.step(bins_1, hist_1, color = color1, linestyle='-',label='EM (inferred)', linewidth=3)
#plt.step(bins_2, hist_2, color = color2, linestyle='-',label='EM (measured)', linewidth=3)
plt.xticks(xticks_1)
plt.xlim([0,100])

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
        linestyle='-'
    #plt.step(bins_1, hist_1, color = color3, linestyle=linestyle,label=label,linewidth=3)
    kdedir_gwdist = greedy_kde_areas_1d(gwdist)
    plt.plot(bins, [kde_eval_single(kdedir_gwdist,[d])[0] for d in bins], color = color3, linestyle=linestyle,label=label, linewidth=3)

gwdist = samples_all['low']['luminosity_distance_Mpc']
kdedir_gwdist = greedy_kde_areas_1d(gwdist)

fp_mu, fp_std = 44.0, 7.5
sbf_mu, sbf_std = 40.7, np.sqrt(1.4**2 + 1.9**2)
rect1 = Rectangle((fp_mu - fp_std, 0), 2*fp_std, 1, alpha=0.5, color='c')
rect2 = Rectangle((fp_mu - 2*fp_std, 0), 4*fp_std, 1, alpha=0.2, color='c')
rect3 = Rectangle((sbf_mu - sbf_std, 0), 2*sbf_std, 1, alpha=0.5, color='r')
rect4 = Rectangle((sbf_mu - 2*sbf_std, 0), 4*sbf_std, 1, alpha=0.2, color='r')

ax.plot([sbf_mu,sbf_mu],[0,1],alpha=0.3, color='r',label='SBF')
ax.plot([fp_mu,fp_mu],[0,1],alpha=0.3, color='c',label='FP')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)

plt.legend()
plt.xlabel('Distance [Mpc]')
plt.ylabel('Probability')
plt.ylim([0,0.10])
plt.grid(True)
plt.show()
plotName = os.path.join(baseplotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

bin_edges = np.arange(5,180,5)
hist_1, bin_edges_1 = np.histogram(H0_EM_1, bin_edges, density=True)
hist_2, bin_edges_2 = np.histogram(H0_GW, bin_edges, density=True)
hist_3, bin_edges_3 = np.histogram(H0_GWEM_1, bin_edges, density=True)
hist_4, bin_edges_4 = np.histogram(H0_EM_2, bin_edges, density=True)
hist_5, bin_edges_5 = np.histogram(H0_GWEM_2, bin_edges, density=True)
bins = (bin_edges[:-1] + bin_edges[1:])/2.0

kdedir_gw = greedy_kde_areas_1d(H0_GW)
kdedir_em_1 = greedy_kde_areas_1d(H0_EM_1)
kdedir_em_2 = greedy_kde_areas_1d(H0_EM_2)
kdedir_gwem_1 = greedy_kde_areas_1d(H0_GWEM_1)
kdedir_gwem_2 = greedy_kde_areas_1d(H0_GWEM_2)

planck_mu, planck_std = 67.74, 0.46
shoes_mu, shoes_std = 73.24, 1.74
superluminal_mu, superluminal_std = 68.9, 4.6
rect1 = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2 = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
rect3 = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4 = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
rect5 = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.05, alpha=0.3, color='c')
rect6 = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.05, alpha=0.1, color='c')

rect1b = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2b = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
rect3b = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4b = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
rect5b = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.05, alpha=0.3, color='c')
rect6b = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.05, alpha=0.1, color='c')

bins = np.arange(5,170,1)

fig, ax1 = plt.subplots(figsize=(9,6))
# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.15, 0.55, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.plot(bins, [kde_eval_single(kdedir_gw,[d])[0] for d in bins], color = color2, linestyle='-.',label='GW', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_1,[d])[0] for d in bins], color = color1, linestyle='-',label='EM (inferred)', linewidth=3, zorder=10)
#ax1.plot(bins, [kde_eval_single(kdedir_em_2,[d])[0] for d in bins], color = color1, linestyle='--',label='EM (measured)', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_gwem_1,[d])[0] for d in bins], color = color3, linestyle='-',label='GW-EM (inferred)', linewidth=3, zorder=10)
#ax1.plot(bins, [kde_eval_single(kdedir_gwem_2,[d])[0] for d in bins], color = color3, linestyle='--',label='GW-EM (measured)', linewidth=3, zorder=10)

ax1.plot([planck_mu,planck_mu],[0,1],alpha=0.3, color='g',label='Planck')
ax1.plot([shoes_mu,shoes_mu],[0,1],alpha=0.3, color='r',label='SHoES')
ax1.plot([superluminal_mu,superluminal_mu],[0,1],alpha=0.3, color='c',label='Superluminal')

ax1.add_patch(rect1)
ax1.add_patch(rect2)
ax1.add_patch(rect3)
ax1.add_patch(rect4)
ax1.add_patch(rect5)
ax1.add_patch(rect6)

ax1.set_xlabel('H0 [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]')
ax1.set_ylabel('Probability')
ax1.grid(True)
ax1.legend()
ax1.set_xlim([10,170])
ax1.set_ylim([0,0.04])

ax2.plot(bins, [kde_eval_single(kdedir_gw,[d])[0] for d in bins], color = color2, linestyle='-.',label='GW', linewidth=3, zorder=10)
ax2.plot(bins, [kde_eval_single(kdedir_em_1,[d])[0] for d in bins], color = color1, linestyle='-',label='EM (inferred)', linewidth=3, zorder=10)
#ax2.plot(bins, [kde_eval_single(kdedir_em_2,[d])[0] for d in bins], color = color1, linestyle='--',label='EM (measured)', linewidth=3, zorder=10)
ax2.plot(bins, [kde_eval_single(kdedir_gwem_1,[d])[0] for d in bins], color = color3, linestyle='-',label='GW-EM (inferred)', linewidth=3, zorder=10)
#ax2.plot(bins, [kde_eval_single(kdedir_gwem_2,[d])[0] for d in bins], color = color3, linestyle='--',label='GW-EM (measured)', linewidth=3, zorder=10)
#
##ax2.plot([planck_mu,planck_mu],[0,1],alpha=0.3, color='g',label='Planck')
##ax2.plot([shoes_mu,shoes_mu],[0,1],alpha=0.3, color='r',label='SHoES')
##ax2.plot([superluminal_mu,superluminal_mu],[0,1],alpha=0.3, color='c',label='Superluminal')
#
ax2.errorbar(planck_mu, 0.042, xerr=planck_std, fmt='o', color='g',label='Planck',zorder=10)
ax2.errorbar(shoes_mu, 0.045, xerr=shoes_std, fmt='o', color='r',label='SHoES',zorder=10)
ax2.errorbar(superluminal_mu, 0.048, xerr=superluminal_std, fmt='o', color='c',label='Superluminal')
#
##ax2.add_patch(rect1b)
##ax2.add_patch(rect2b)
##ax2.add_patch(rect3b)
##ax2.add_patch(rect4b)
##ax2.add_patch(rect5b)
##ax2.add_patch(rect6b)
#
plt.setp( ax2.get_yticklabels(), visible=False)

ax2.set_xlim([55,95])
ax2.xaxis.grid(True)
ax2.set_ylim([0,0.055])
xticks_1 = np.array([65,75,85])
ax2.set_xticks(xticks_1)

plt.show()
plotName = os.path.join(baseplotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

