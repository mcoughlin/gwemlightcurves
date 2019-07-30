
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

    parser.add_option("--pickle_samples", default="../plots/standard_candles/GRB/GRB060614/inferred/H0.pkl,../plots/standard_candles/GRB/GRB150101B/inferred/H0.pkl,../plots/standard_candles/GRB/GRB050709/inferred/H0.pkl,../plots/standard_candles/inferred/H0.pkl")

    parser.add_option("-m","--model",default="Ka2017", help="Ka2017,Ka2017x2")

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

def myprior_H0(cube, ndim, nparams):
        cube[0] = cube[0]*200.0

def myloglike_H0(cube, ndim, nparams):
        H0 = cube[0]

        prob = 0
        for name in data_struct.iterkeys():
            kdedir_H0 = data_struct[name]["kdedir_H0"] 
            prob = prob + np.log(kde_eval_single(kdedir_H0,[H0])[0]) 

        if np.isnan(prob):
            prob = -np.inf

        return prob

# Parse command line
opts = parse_commandline()

baseplotDir = os.path.join(opts.plotDir,'standard_candles','all')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'palegreen'
color4 = 'pink'

color_names = [color1, color2, color3, color4]

pickle_samples = opts.pickle_samples.split(",")
data_struct = {}
for pickle_sample in pickle_samples:
    pickle_sample_split = pickle_sample.split("/")
    f = open(pickle_sample, 'r')
    if pickle_sample_split[-4] == "GRB": 
        (dist,H0_EM,distance) = pickle.load(f)
        name = pickle_sample_split[-3]
        data_struct[name] = {}
        data_struct[name]["dist"] = dist
        data_struct[name]["H0"] = H0_EM
    else:
        (dist,samples_all,H0_EM,H0_GW,H0_GWEM,Mag,M,sigma_best) = pickle.load(f)
        name = "GW170817"
        distance = 40.0
        data_struct[name] = {}
        data_struct[name]["dist"] = dist
        data_struct[name]["H0"] = H0_EM
    data_struct[name]["distance"] = (dist-distance)/np.median(dist)
    data_struct[name]["kdedir_dist"] = greedy_kde_areas_1d(data_struct[name]["distance"])
    data_struct[name]["kdedir_H0"] = greedy_kde_areas_1d(H0_EM)
    f.close()

    print('%s: %.2f +%.2f' % (name, np.median(data_struct[name]["distance"]),np.std(data_struct[name]["distance"])))

    H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
    print('%s: %.0f +%.0f -%.0f' % (name,H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

bins = np.arange(-1.5,1.5,0.1)

fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
for ii, name in enumerate(data_struct.keys()):
    color_name = color_names[ii]

    kdedir = data_struct[name]["kdedir_dist"]
    plt.plot(bins, [kde_eval_single(kdedir,[d])[0] for d in bins], color = color_name, linestyle='-.',label=name, linewidth=3, zorder=10)

plt.legend(loc=2)
plt.xlabel('Relative Error in Distance')
plt.ylabel('Probability')
#plt.ylim([0,0.10])
plt.grid(True)
plt.show()
plotName = os.path.join(baseplotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["H0"]
labels = [r'$H_0$']
n_params = len(parameters)

pymultinest.run(myloglike_H0, myprior_H0, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%baseplotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%baseplotDir
data = np.loadtxt(multifile)

H0_EM, loglikelihood = data[:,0], data[:,1]
idx = np.argmax(loglikelihood)
H0_best = data[idx,0:-1]

bins = np.arange(5,150,1)

fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
for ii, name in enumerate(data_struct.keys()):
    color_name = color_names[ii]

    kdedir = data_struct[name]["kdedir_H0"]
    plt.plot(bins, [kde_eval_single(kdedir,[d])[0] for d in bins], color = color_name, linestyle='-.',label=name, linewidth=3, zorder=10)

hist_1, bin_edges_1 = np.histogram(H0_EM, bins, density=True)
kdedir = greedy_kde_areas_1d(H0_EM)
plt.plot(bins, [kde_eval_single(kdedir,[d])[0] for d in bins], color = 'k', linestyle='-',label="Combined", linewidth=3, zorder=10)

boxes = []
planck_mu, planck_std = 67.74, 0.46 
shoes_mu, shoes_std = 73.24, 1.74
superluminal_mu, superluminal_std = 68.9, 4.6
plt.plot([planck_mu,planck_mu],[0,1],alpha=0.3, color='g',label='Planck')
rect1 = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2 = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
plt.plot([shoes_mu,shoes_mu],[0,1],alpha=0.3, color='r',label='SHoES')
rect3 = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4 = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
plt.plot([superluminal_mu,superluminal_mu],[0,1],alpha=0.3, color='c',label='Superluminal')
rect5 = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.05, alpha=0.3, color='c')
rect6 = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.05, alpha=0.1, color='c')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)

plt.xlabel('H0 [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.xlim([20,150])
plt.ylim([0,0.05])
plt.show()
plotName = os.path.join(baseplotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)

print('H0 EM: %.0f +%.0f -%.0f' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))
