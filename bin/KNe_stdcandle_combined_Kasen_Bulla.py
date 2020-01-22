
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

# Parse command line
opts = parse_commandline()

baseplotDir = os.path.join(opts.plotDir,'standard_candles/GRB_GW/all')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

plotDir1 = os.path.join(baseplotDir,'inferred/0.25')
plotDir2 = os.path.join(baseplotDir,'inferred_bulla/0.25')
plotDir3 = os.path.join(baseplotDir,'inferred/0.10')
plotDir4 = os.path.join(baseplotDir,'inferred_bulla/0.10')

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'darkgreen'
color4 = 'pink'
color5 = 'cyan'

baseplotDir = os.path.join(opts.plotDir,'standard_candles/Kasen_Bulla')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

pcklFile1 = os.path.join(plotDir1, "H0.pkl")
f = open(pcklFile1, 'r')
(dist_1,H0_EM_1,H0_GW,data_struct_1,H0_best_1,Om0_best_1,Ode0_best_1) = pickle.load(f)
f.close() 

pcklFile2 = os.path.join(plotDir2, "H0.pkl")
f = open(pcklFile2, 'r')
(dist_2,H0_EM_2,H0_GW,data_struct_2,H0_best_2,Om0_best_2,Ode0_best_2) = pickle.load(f)
f.close()

pcklFile3 = os.path.join(plotDir3, "H0.pkl")
f = open(pcklFile3, 'r')
(dist_3,H0_EM_3,H0_GW,data_struct_3,H0_best_3,Om0_best_3,Ode0_best_3) = pickle.load(f)
f.close()

pcklFile4 = os.path.join(plotDir4, "H0.pkl")
f = open(pcklFile4, 'r')
(dist_4,H0_EM_4,H0_GW,data_struct_4,H0_best_4,Om0_best_4,Ode0_best_4) = pickle.load(f)
f.close()

fileName = os.path.join(baseplotDir,'table.txt')
fid = open(fileName, 'w')

fid.write('\\hline \n')

for ii, name in enumerate(data_struct_1.keys()):
    print(name)

    if not name in data_struct_4: continue

    fid.write('%s & ' % name)

    H0_EM = data_struct_1[name]["H0"]
    H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
    fid.write('$%.0f^{+%.0f}_{-%.0f}$ & ' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

    H0_EM = data_struct_3[name]["H0"]
    H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
    fid.write('$%.0f^{+%.0f}_{-%.0f}$ & ' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

    H0_EM = data_struct_2[name]["H0"]
    H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
    fid.write('$%.0f^{+%.0f}_{-%.0f}$ & ' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

    H0_EM = data_struct_4[name]["H0"]
    H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)
    fid.write('$%.0f^{+%.0f}_{-%.0f}$' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

    fid.write(' \\\\ \n')
    fid.write('\\hline \n')

fid.close()

cosmo = LambdaCDM(H0=H0_best_1, Om0=Om0_best_1, Ode0=Ode0_best_1)

fig = plt.figure(figsize=(9,6))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)

labels = []
for ii, name in enumerate(data_struct_1.keys()):
    parts = plt.violinplot(5.0*(np.log10(data_struct_1[name]["dist"]*1e6)-1.0),[data_struct_1[name]["redshift"]],widths=0.02)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)

    if ii == 0:
        labels.append((matplotlib.patches.Patch(color=color1), "Kasen et al."))

    parts = plt.violinplot(5.0*(np.log10(data_struct_2[name]["dist"]*1e6)-1.0),[data_struct_1[name]["redshift"]],widths=0.02)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color2)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color2)
        pc.set_edgecolor(color2)

    if ii == 0:
        labels.append((matplotlib.patches.Patch(color=color2), "Bulla"))

redshifts = np.logspace(-3,1,100)
line = plt.plot(redshifts, 5.0*(np.log10(cosmo.comoving_distance(redshifts).value*1e6)-1.0), '--', color=color3, label=r'$\Lambda_\mathrm{CDM}$')
#labels.append(line)
labels.append((matplotlib.lines.Line2D([-100,100],[200,200],color=color3,linestyle="--"), r'$\Lambda_\mathrm{CDM}$'))

#plt.xlabel(r'Redshift')
plt.ylabel('Distance Mod. [mag]')
plt.xlim([-0.01,0.2])
plt.legend(*zip(*labels),loc=2)
#plt.ylim([0,2500])
#plt.xlim([-2.5,1.0])
#plt.ylim([0,8000])
plt.grid(True)
plt.yticks([28.0,31.0,34.0,37.0,40.0,43.0])
plt.setp(ax1.get_xticklabels(), visible=False)

plt.axes(ax2)

for ii, name in enumerate(data_struct_1.keys()):
    modelval = 5.0*(np.log10(cosmo.comoving_distance(data_struct_1[name]["redshift"]).value*1e6)-1.0)
    parts = plt.violinplot(5.0*(np.log10(data_struct_1[name]["dist"]*1e6)-1.0)-modelval,[data_struct_1[name]["redshift"]],widths=0.02)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color1)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color1)
        pc.set_edgecolor(color1)

    parts = plt.violinplot(5.0*(np.log10(data_struct_2[name]["dist"]*1e6)-1.0)-modelval,[data_struct_2[name]["redshift"]],widths=0.02)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(color2)
        vp.set_linewidth(1)
    for pc in parts['bodies']:
        pc.set_facecolor(color2)
        pc.set_edgecolor(color2)

plt.plot(redshifts,0.0*np.ones(redshifts.shape), '--', color=color3, zorder=10, linewidth=3, alpha=0.5)

plt.yticks([-4,-2,0,2,4])
plt.ylim([-4,4])
plt.grid()
plt.xlabel(r'Redshift')
plt.ylabel('Hubble Res. [mag]')
plt.xlim([-0.01,0.2])

plotName = os.path.join(baseplotDir,'distmod.pdf')
plt.savefig(plotName)
plt.close()

bin_edges = np.arange(5,180,5)
bins = (bin_edges[:-1] + bin_edges[1:])/2.0

kdedir_gw = greedy_kde_areas_1d(H0_GW)
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
rect5 = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.12, alpha=0.3, color='c')
rect6 = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.12, alpha=0.1, color='c')

rect1b = Rectangle((planck_mu - planck_std, 0), 2*planck_std, 1, alpha=0.8, color='g')
rect2b = Rectangle((planck_mu - 2*planck_std, 0), 4*planck_std, 1, alpha=0.5, color='g')
rect3b = Rectangle((shoes_mu - shoes_std, 0), 2*shoes_std, 1, alpha=0.8, color='r')
rect4b = Rectangle((shoes_mu - 2*shoes_std, 0), 4*shoes_std, 1, alpha=0.5, color='r')
rect5b = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.12, alpha=0.3, color='c')
rect6b = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.12, alpha=0.1, color='c')

bins = np.arange(5,170,1)

fig, ax1 = plt.subplots(figsize=(9,6))
# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.15, 0.55, 0.25, 0.30]
#ax2 = fig.add_axes([left, bottom, width, height])

ax1.plot(bins, [kde_eval_single(kdedir_gw,[d])[0] for d in bins], color = color2, linestyle='-.',label='GW', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_1,[d])[0] for d in bins], color = color1, linestyle='-',label='Kasen et al. - 0.25', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_2,[d])[0] for d in bins], color = color1, linestyle='--',label='Bulla - 0.25', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_3,[d])[0] for d in bins], color = color3, linestyle='-',label='Kasen et al. - 0.10', linewidth=3, zorder=10)
ax1.plot(bins, [kde_eval_single(kdedir_em_4,[d])[0] for d in bins], color = color3, linestyle='--',label='Bulla - 0.10', linewidth=3, zorder=10)

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
ax1.set_xlim([50,100])
ax1.set_ylim([0,0.12])

#ax2.plot(bins, [kde_eval_single(kdedir_gw,[d])[0] for d in bins], color = color2, linestyle='-.',label='GW - GW170817', linewidth=3, zorder=10)
#ax2.plot(bins, [kde_eval_single(kdedir_em_1,[d])[0] for d in bins], color = color1, linestyle='-',label='Kasen et al. - 0.25', linewidth=3, zorder=10)
#ax2.plot(bins, [kde_eval_single(kdedir_em_2,[d])[0] for d in bins], color = color1, linestyle='--',label='Bulla - 0.25', linewidth=3, zorder=10)
#ax2.plot(bins, [kde_eval_single(kdedir_em_3,[d])[0] for d in bins], color = color3, linestyle='-',label='Kasen et al. - 0.10', linewidth=3, zorder=10)
#ax2.plot(bins, [kde_eval_single(kdedir_em_4,[d])[0] for d in bins], color = color3, linestyle='--',label='Bulla - 0.10', linewidth=3, zorder=10)
#
#ax2.errorbar(planck_mu, 0.135, xerr=planck_std, fmt='o', color='g',label='Planck',zorder=10)
#ax2.errorbar(shoes_mu, 0.13, xerr=shoes_std, fmt='o', color='r',label='SHoES',zorder=10)
#ax2.errorbar(superluminal_mu, 0.125, xerr=superluminal_std, fmt='o', color='c',label='Superluminal')
#
##ax2.add_patch(rect1b)
##ax2.add_patch(rect2b)
##ax2.add_patch(rect3b)
##ax2.add_patch(rect4b)
##ax2.add_patch(rect5b)
##ax2.add_patch(rect6b)
#
#plt.setp( ax2.get_yticklabels(), visible=False)

#ax2.set_xlim([55,95])
#ax2.xaxis.grid(True)
#ax2.set_ylim([0,0.14])
#xticks_1 = np.array([65,75,85])
#ax2.set_xticks(xticks_1)

plt.show()
plotName = os.path.join(baseplotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

