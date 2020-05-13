
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
from matplotlib import patches
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

from astropy.cosmology import LambdaCDM

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

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

# Parse command line
opts = parse_commandline()

baseplotDir = os.path.join(opts.plotDir,'standard_candles/GRB_GW/GRB060614/inferred/0.10')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

plotDir1 = os.path.join(baseplotDir,'SBF')
plotDir2 = os.path.join(baseplotDir,'posterior')
plotDir3 = os.path.join(baseplotDir,'0.10_0.00/SBF')
plotDir4 = os.path.join(baseplotDir,'0.00_0.50/SBF')

baseplotDir = os.path.join(opts.plotDir,'standard_candles/systematics')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

pcklFile1 = os.path.join(plotDir1, "H0.pkl")
f = open(pcklFile1, 'r')
(dist_1,H0_EM_1,distance_1,redshift_1) = pickle.load(f)
f.close() 

pcklFile2 = os.path.join(plotDir2, "H0.pkl")
f = open(pcklFile2, 'r')
(dist_2,H0_EM_2,distance_2,redshift_2) = pickle.load(f)
f.close()

pcklFile3 = os.path.join(plotDir3, "H0.pkl")
f = open(pcklFile3, 'r')
(dist_3,H0_EM_3,distance_3,redshift_3) = pickle.load(f)
f.close()

pcklFile4 = os.path.join(plotDir4, "H0.pkl")
f = open(pcklFile4, 'r')
(dist_4,H0_EM_4,distance_4,redshift_4) = pickle.load(f)
f.close()

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'darkgreen'
color4 = 'pink'

bin_edges = np.arange(5,180,5)
bins = (bin_edges[:-1] + bin_edges[1:])/2.0

kdedir_em_1 = greedy_kde_areas_1d(H0_EM_1)
kdedir_em_2 = greedy_kde_areas_1d(H0_EM_2)
kdedir_em_3 = greedy_kde_areas_1d(H0_EM_3)
kdedir_em_4 = greedy_kde_areas_1d(H0_EM_4)

planck_mu, planck_std = 67.74, 0.46
shoes_mu, shoes_std = 74.03, 1.42
superluminal_mu, superluminal_std = 68.9, 4.6
rect1 = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2 = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
rect3 = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4 = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
rect5 = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.07, alpha=0.3, color='c')
rect6 = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.07, alpha=0.1, color='c')

rect1b = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2b = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
rect3b = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4b = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
rect5b = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.07, alpha=0.3, color='c')
rect6b = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.07, alpha=0.1, color='c')

bins = np.arange(5,170,1)

fig, ax1 = plt.subplots(figsize=(9,6))

kdedir_em_1 = greedy_kde_areas_1d(H0_EM_1)
kdedir_em_2 = greedy_kde_areas_1d(H0_EM_2)
kdedir_em_3 = greedy_kde_areas_1d(H0_EM_3)
kdedir_em_4 = greedy_kde_areas_1d(H0_EM_4)

H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM_1,16), np.percentile(H0_EM_1,50), np.percentile(H0_EM_1,84)
print('SBF: $%.0f^{+%.0f}_{-%.0f}$ & ' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))
H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM_3,16), np.percentile(H0_EM_3,50), np.percentile(H0_EM_3,84)
print('SBF - $M_{\\rm ej}: $%.0f^{+%.0f}_{-%.0f}$ & ' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))
H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM_4,16), np.percentile(H0_EM_4,50), np.percentile(H0_EM_4,84)
print('SBF - $X_{\\rm lan}: $%.0f^{+%.0f}_{-%.0f}$ & ' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

ax1.plot(bins, [kde_eval_single(kdedir_em_1,[d])[0] for d in bins], color = color1, linestyle='-',label='SBF', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_2,[d])[0] for d in bins], color = color2, linestyle='--',label='GW', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_3,[d])[0] for d in bins], color = color3, linestyle='-.',label=r'SBF - $M_{\rm ej}$', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_4,[d])[0] for d in bins], color = color4, linestyle='-.',label=r'SBF - $X_{\rm lan}$', linewidth=3, zorder=10)

ax1.plot([planck_mu,planck_mu],[0,1],alpha=0.3, color='g',label='Planck')
ax1.plot([shoes_mu,shoes_mu],[0,1],alpha=0.3, color='r',label='SHoES')
ax1.plot([superluminal_mu,superluminal_mu],[0,1],alpha=0.3, color='c',label='Superluminal')

ax1.add_patch(rect1)
ax1.add_patch(rect2)
ax1.add_patch(rect3)
ax1.add_patch(rect4)
ax1.add_patch(rect5)
ax1.add_patch(rect6)

ax1.set_xlabel('$H_0$ [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]')
ax1.set_ylabel('Probability')
ax1.grid(True)
ax1.legend()
ax1.set_xlim([20,140])
ax1.set_ylim([0,0.04])

plt.show()
plotName = os.path.join(baseplotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

