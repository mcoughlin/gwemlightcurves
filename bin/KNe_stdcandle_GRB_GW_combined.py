
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

    parser.add_option("--pickle_samples", default="../plots/standard_candles/GRB_GW/GRB060614/inferred/H0.pkl,../plots/standard_candles/GRB_GW/GRB150101B/inferred/H0.pkl,../plots/standard_candles/GRB_GW/GRB050709/inferred/H0.pkl,../plots/standard_candles/inferred/H0.pkl,../plots/standard_candles/GRB_GW/GRB160821B/inferred/H0.pkl")

    parser.add_option("-m","--model",default="Ka2017", help="Ka2017,Ka2017x2")

    parser.add_option("-e","--errorbudget",default=1.0,type=float)

    parser.add_option("--doNoGW170817",  action="store_true", default=False)

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

def myprior_H0_empirical(cube, ndim, nparams):
        cube[0] = cube[0]*200.0
        cube[1] = cube[1]*2.0 - 1.0
        cube[2] = cube[2]*10.0 - 5.0

def myprior_H0(cube, ndim, nparams):
        cube[0] = cube[0]*200.0
        cube[1] = cube[1]*3.0
        cube[2] = cube[2]*3.0

def myloglike_H0_empirical(cube, ndim, nparams):
        H0 = cube[0]
        q0 = cube[1]
        j0 = cube[2]
        
        c = 3e5

        prob = 0
        for name in data_struct.iterkeys():
            z = data_struct[name]["redshift"] 
            dc = (z*c/H0)*(1 + (1/2.0)*(1-q0)*z - (1.0/6.0)*(1-q0-3*q0**2 + j0)*z**2)
            kdedir_dist = data_struct[name]["kdedir_dist"] 
            prob = prob + np.log(kde_eval_single(kdedir_dist,[dc])[0]) 

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myloglike_H0(cube, ndim, nparams):
        H0 = cube[0]
        Om0 = cube[1]
        Ode0 = cube[2]

        cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

        prob = 0
        for name in data_struct.iterkeys():
            dc = cosmo.comoving_distance(data_struct[name]["redshift"]).value
            kdedir_dist = data_struct[name]["kdedir_dist"]
            prob = prob + np.log(kde_eval_single(kdedir_dist,[dc])[0])

            #pvp = (1/np.sqrt(2*np.pi*data_struct[name]["dist_std"]**2))*np.exp((-1/2.0)*((data_struct[name]["dist_median"]-dc)/data_struct[name]["dist_std"])**2)
            #prob = prob + np.log(pvp)

        if np.isnan(prob):
            prob = -np.inf

        return prob

# Parse command line
opts = parse_commandline()

if opts.doNoGW170817:
    baseplotDir = os.path.join(opts.plotDir,'standard_candles','GRB_GW','no')
else:
    baseplotDir = os.path.join(opts.plotDir,'standard_candles','GRB_GW','all')
plotDir = os.path.join(baseplotDir,opts.analysis_type)
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'palegreen'
color4 = 'pink'
color5 = 'cyan'

color_names = [color1, color2, color3, color4, color5]

pickle_samples = opts.pickle_samples.split(",")
data_struct = {}
for pickle_sample in pickle_samples:
    pickle_sample_split = pickle_sample.split("/")
    f = open(pickle_sample, 'r')
    if "GRB_GW" in pickle_sample: 
        (dist,H0_EM,distance,redshift) = pickle.load(f)
        if ("SBF" in pickle_sample) or ("posterior" in pickle_sample):
            name = pickle_sample_split[-5]
        else:
            name = pickle_sample_split[-4]
        data_struct[name] = {}
        data_struct[name]["dist"] = dist
        data_struct[name]["H0"] = H0_EM
        data_struct[name]["redshift"] = redshift
    else:
        (dist,samples_all,H0_EM,H0_GW,H0_GWEM,Mag,M,sigma_best) = pickle.load(f)
        name = "GW170817"
        distance = 40.7
        redshift = 0.009783
        if opts.doNoGW170817:
            continue

        data_struct[name] = {}
        data_struct[name]["dist"] = dist
        data_struct[name]["H0"] = H0_EM
        data_struct[name]["redshift"] = redshift
    data_struct[name]["distance"] = (dist-distance)/np.median(dist)
    data_struct[name]["kdedir_dist"] = greedy_kde_areas_1d(data_struct[name]["dist"])
    data_struct[name]["kdedir_distance"] = greedy_kde_areas_1d(data_struct[name]["distance"])
    data_struct[name]["kdedir_H0"] = greedy_kde_areas_1d(H0_EM)

    data_struct[name]["dist_median"] = np.median(data_struct[name]["dist"])
    data_struct[name]["dist_std"] = data_struct[name]["dist_median"]*0.1

    f.close()

    print('Distance: %s: %.2f +-%.2f' % (name, np.median(data_struct[name]["distance"]),np.std(data_struct[name]["distance"])))

    H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
    print('H0: %s: %.0f +%.0f -%.0f' % (name,H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

bins = np.arange(-1.5,1.5,0.1)

fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
for ii, name in enumerate(data_struct.keys()):
    color_name = color_names[ii]

    kdedir = data_struct[name]["kdedir_distance"]
    plt.plot(bins, [kde_eval_single(kdedir,[d])[0] for d in bins], color = color_name, linestyle='-.',label=name, linewidth=3, zorder=10)

plt.legend(loc=2)
plt.xlabel('Relative Error in Distance')
plt.ylabel('Probability')
#plt.ylim([0,0.10])
plt.grid(True)
plt.show()
plotName = os.path.join(plotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["H0","Om0","0de"]
labels = [r'$H_0$',r'$\Omega_m$',r'$\Omega_\Lambda$']
n_params = len(parameters)

pymultinest.run(myloglike_H0, myprior_H0, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                   smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

H0_EM, Om0, Ode0, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
idx = np.argmax(loglikelihood)
H0_best, Om0_best, Ode0_best = data[idx,0:-1]

parameters = ["H0","q0","j0"]
labels = [r'$H_0$',r'$q_0$',r'$j_0$']
n_params = len(parameters)

empDir = os.path.join(plotDir,"emp")
if not os.path.isdir(empDir):
    os.makedirs(empDir)

pymultinest.run(myloglike_H0_empirical, myprior_H0_empirical, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%empDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%empDir
data = np.loadtxt(multifile)

plotName = "%s/corner_emp.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                   smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

H0_EM, q0, j0, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3]
idx = np.argmax(loglikelihood)
H0_best, q0_best, j0_best = data[idx,0:-1]

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
shoes_mu, shoes_std = 74.03, 1.42
superluminal_mu, superluminal_std = 68.9, 4.6
plt.plot([planck_mu,planck_mu],[0,1],alpha=0.3, color='g',label='Planck')
rect1 = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2 = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
plt.plot([shoes_mu,shoes_mu],[0,1],alpha=0.3, color='r',label='SHoES')
rect3 = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4 = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
plt.plot([superluminal_mu,superluminal_mu],[0,1],alpha=0.3, color='c',label='Superluminal')
rect5 = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.12, alpha=0.3, color='c')
rect6 = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.12, alpha=0.1, color='c')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)

plt.xlabel('$H_0$ [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.xlim([30,120])
#plt.ylim([0,0.035])
#plt.ylim([0,0.08])
plt.ylim([0,0.12])
plt.show()
plotName = os.path.join(plotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
for ii, name in enumerate(data_struct.keys()):
    parts = plt.violinplot(data_struct[name]["dist"],[np.log10(data_struct[name]["redshift"])])
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)

cosmo = LambdaCDM(H0=H0_best, Om0=Om0_best, Ode0=Ode0_best)
print(H0_best, Om0_best, Ode0_best)

redshifts = np.logspace(-3,1,100)
plt.plot(np.log10(redshifts), cosmo.comoving_distance(redshifts).value, '--', color=color2, label=r'$\Lambda_\mathrm{CDM}$')
plt.xlabel(r'$\log_{10}$ Redshift')
plt.ylabel('Distance [Mpc]')
plt.xlim([-2.5,-0.5])
plt.ylim([0,2500])
#plt.xlim([-2.5,1.0])
plt.ylim([0,8000])
plt.grid(True)
plt.show()
plotName = os.path.join(plotDir,'redshift.pdf')
plt.savefig(plotName)
plt.close()

fig = plt.figure(figsize=(9,6))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)

for ii, name in enumerate(data_struct.keys()):
    parts = plt.violinplot(5.0*(np.log10(data_struct[name]["dist"]*1e6)-1.0),[data_struct[name]["redshift"]],widths=0.02)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)

redshifts = np.logspace(-3,1,100)
plt.plot(redshifts, 5.0*(np.log10(cosmo.comoving_distance(redshifts).value*1e6)-1.0), '--', color=color2, label=r'$\Lambda_\mathrm{CDM}$')
#plt.xlabel(r'Redshift')
plt.ylabel('Distance Mod. [mag]')
plt.xlim([-0.01,0.2])
#plt.ylim([0,2500])
#plt.xlim([-2.5,1.0])
#plt.ylim([0,8000])
plt.grid(True)
plt.setp(ax1.get_xticklabels(), visible=False)

plt.axes(ax2)

for ii, name in enumerate(data_struct.keys()):
    modelval = 5.0*(np.log10(cosmo.comoving_distance(data_struct[name]["redshift"]).value*1e6)-1.0)
    parts = plt.violinplot(5.0*(np.log10(data_struct[name]["dist"]*1e6)-1.0)-modelval,[data_struct[name]["redshift"]],widths=0.02)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)

plt.plot(redshifts,0.0*np.ones(redshifts.shape), '--', color=color2, zorder=10, linewidth=3, alpha=0.5)

plt.yticks([-4,-2,0,2,4])
plt.ylim([-4,4])
plt.grid()
plt.xlabel(r'Redshift')
plt.ylabel('Hubble Res. [mag]')
plt.xlim([-0.01,0.2])

plt.show()
plotName = os.path.join(plotDir,'distmod.pdf')
plt.savefig(plotName)
plt.close()

H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)

print('H0 EM: %.1f +%.1f -%.1f' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

pcklFile = os.path.join(plotDir,"H0.pkl")
f = open(pcklFile, 'wb')
pickle.dump((dist,H0_EM,H0_GW,data_struct,H0_best,Om0_best,Ode0_best), f)
f.close()
