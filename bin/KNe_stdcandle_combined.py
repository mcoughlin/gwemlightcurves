
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
(dist_1,samples_all,H0_EM_1,H0_GW,H0_GWEM_1) = pickle.load(f)
f.close() 

pcklFile2 = os.path.join(plotDir2, "H0.pkl")
f = open(pcklFile2, 'r')
(dist_2,samples_all,H0_EM_2,H0_GW,H0_GWEM_2) = pickle.load(f)
f.close()

bin_edges = np.arange(5,85,2)

hist_1, bin_edges_1 = np.histogram(dist_1, bin_edges, density=True)
hist_2, bin_edges_2 = np.histogram(dist_2, bin_edges, density=True)
bins_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2.0
bins_2 = (bin_edges_2[:-1] + bin_edges_2[1:])/2.0

xticks_1 = np.array([10,20,30,40,50,60])

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'palegreen'

fig = plt.figure(figsize=(10,7))

#plt.plot([dist_10,dist_10],[0,1],'--',color=color1)
#plt.plot([dist_50,dist_50],[0,1],'--',color=color1)
#plt.plot([dist_90,dist_90],[0,1],'--',color=color1)
plt.step(bins_1, hist_1, color = color1, linestyle='-',label='EM (inferred)')
plt.step(bins_2, hist_2, color = color2, linestyle='-',label='EM (measured)')
plt.xticks(xticks_1)
plt.xlim([10,80])

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
    plt.step(bins_1, hist_1, color = color3, linestyle=linestyle,label=label)

gwdist = samples_all['low']['luminosity_distance_Mpc']
kdedir_gwdist = greedy_kde_areas_1d(gwdist)

plt.legend()
plt.xlabel('Distance [Mpc]')
plt.ylabel('Probability')
plt.ylim([0,0.10])
plt.grid(True)
plt.show()
plotName = os.path.join(baseplotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

bin_edges = np.arange(5,150,5)
hist_1, bin_edges_1 = np.histogram(H0_EM_1, bin_edges, density=True)
hist_2, bin_edges_2 = np.histogram(H0_GW, bin_edges, density=True)
hist_3, bin_edges_3 = np.histogram(H0_GWEM_1, bin_edges, density=True)
hist_4, bin_edges_4 = np.histogram(H0_EM_2, bin_edges, density=True)
hist_5, bin_edges_5 = np.histogram(H0_GWEM_2, bin_edges, density=True)
bins = (bin_edges[:-1] + bin_edges[1:])/2.0

bins_small = np.arange(5,150,1)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)

plt.step(bins, hist_2, color = color2, linestyle='-.',label='GW')
plt.step(bins, hist_1, color = color1, linestyle='-',label='EM (inferred)')
plt.step(bins, hist_4, color = color1, linestyle='--',label='EM (measured)')
plt.step(bins, hist_3, color = color3, linestyle='-',label='GW-EM (inferred)')
plt.step(bins, hist_5, color = color3, linestyle='--',label='GW-EM (measured)')
plt.plot(bins_small, ss.norm.pdf(bins_small, loc=68.9, scale=4.6), color='pink', label='Superluminal') 

boxes = []
planck_mu, planck_std = 67.74, 0.46 
shoes_mu, shoes_std = 73.24, 1.74
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
plotName = os.path.join(baseplotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

