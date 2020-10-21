
import os, sys, glob, pickle
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

from astropy.table import Table, Column

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

import corner
import scipy.stats as ss

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

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

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

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

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*4.0 - 5.0
        cube[1] = cube[1]*0.3
        cube[2] = cube[2]*5.0 - 5.0
        cube[3] = cube[3]*4.0 - 5.0
        cube[4] = cube[4]*0.3
        cube[5] = cube[5]*5.0 - 5.0

def myloglike(cube, ndim, nparams):
        mej1 = cube[0]
        vej1 = cube[1]
        xlan1 = cube[2]
        mej2 = cube[3]
        vej2 = cube[4]
        xlan2 = cube[5]

        prob1 = calc_prob(mej1, vej1, xlan1, mej2, vej2, xlan2, kdedir_pts_1)
        prob2 = calc_prob(mej1, vej1, xlan1, mej2, vej2, xlan2, kdedir_pts_2)

        prior = prior_2Component(xlan1,xlan2)
        if prior == 0.0:
            return -np.inf
        prior = prior_2ComponentVel(vej1,vej2)
        #prior = prior_2ComponentVel(vej2,vej1)
        if prior == 0.0:
            return -np.inf
        #prior = vej2 > 0.1
        #if prior == 0.0:
        #    return -np.inf

        prob = prob1+prob2

        return prob

def calc_prob(mej1, vej1, xlan1, mej2, vej2, xlan2, kdedir_pts):

        if (mej1==0.0) or (vej1==0.0) or (xlan1==0.0) or (mej2==0.0) or (vej2==0.0) or (xlan2==0.0):
            prob = np.nan
        else:
            vals = np.array([mej1,vej1,xlan1,mej2,vej2,xlan2]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

plotDir = '../plots/gws/Ka2017x2_lightcurve_spectra'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

errorbudget = 1.00

plotDir1 = '../plots/gws_spec/Ka2017x2_FixZPT0/5000_25000/GW170817/2.00/'
pcklFile = os.path.join(plotDir1,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data1, t_best1, lambdas_best1, spec_best1, t0_best1, zp_best1, n_params1, labels1, truths1) = pickle.load(f)
f.close()

plotDir2 = '../plots/gws/Ka2017x2_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/'
pcklFile = os.path.join(plotDir2,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data2, tmag2, lbol2, mag2, t0_best2, zp_best2, n_params2, labels2, best2, truths2) = pickle.load(f)
f.close()

kdedir_pts_1 = greedy_kde_areas_2d(data1[:,1:7])
kdedir_pts_2 = greedy_kde_areas_2d(data2[:,1:7])

n_live_points = 500
evidence_tolerance = 0.5

parameters = ["mej1","vej1","xlan1","mej2","vej2","xlan2"]
labels = [r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (X_{\rm lan 1})$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (X_{\rm lan 2})$"]
n_params = len(parameters)
pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

multifile = lightcurve_utils.get_post_file(plotDir)
data = np.loadtxt(multifile)

tmag2 = tmag2 + t0_best2

outputDir = "../output"
ModelPath = '%s/svdmodels'%(outputDir)

modelfile = os.path.join(ModelPath,'Ka2017_mag.pkl')
with open(modelfile, 'rb') as handle:
    svd_mag_model = pickle.load(handle)
Global.svd_mag_model = svd_mag_model

modelfile = os.path.join(ModelPath,'Ka2017_lbol.pkl')
with open(modelfile, 'rb') as handle:
    svd_lbol_model = pickle.load(handle)
Global.svd_lbol_model = svd_lbol_model

mej_1, vej_1, Xlan_1, mej_2, vej_2, Xlan_2, loglikelihood = 10**data[:,0], data[:,1], 10**data[:,2], 10**data[:,3], data[:,4], 10**data[:,5], data[:,6],
idx = np.argmax(loglikelihood)
mej_1_best, vej_1_best, Xlan_1_best, mej_2_best, vej_2_best, Xlan_2_best = 10**data[idx,0], data[idx,1], 10**data[idx,2], 10**data[idx,3], data[idx,4], 10**data[idx,5]
tmag1, lbol1, mag1 = Ka2017x2_model_ejecta(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best)

if n_params >= 6:
    title_fontsize = 36
    label_fontsize = 36
else:
    title_fontsize = 30
    label_fontsize = 30

truths = []
for ii in range(n_params):
    #truths.append(False)
    truths.append(np.nan)

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".2f",
                       truths=truths, smooth=3,
                       color="coral")
if n_params >= 10:
    figure.set_size_inches(40.0,40.0)
elif n_params >= 6:
    figure.set_size_inches(22.0,22.0)
else:
    figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

title_fontsize = 30
label_fontsize = 30

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.jet(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]
tini, tmax, dt = 0.0, 21.0, 0.1    
tt = np.arange(tini,tmax,dt)

color2 = 'coral'
color1 = 'cornflowerblue'

plotName = "%s/models_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,18))
plt.figure(figsize=(20,28))

tini, tmax, dt = 0.0, 21.0, 0.1
tt = np.arange(tini,tmax,dt)

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color, markersize=16)

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=16)

    if filt == "w":
        magave1 = (mag1[1]+mag1[2]+mag1[3])/3.0
    elif filt == "c":
        magave1 = (mag1[1]+mag1[2])/2.0
    elif filt == "o":
        magave1 = (mag1[2]+mag1[3])/2.0
    else:
        magave1 = mag1[magidx]

    if filt == "w":
        magave2 = (mag2[1]+mag2[2]+mag2[3])/3.0
    elif filt == "c":
        magave2 = (mag2[1]+mag2[2])/2.0
    elif filt == "o":
        magave2 = (mag2[2]+mag2[3])/2.0
    else:
        magave2 = mag2[magidx]

    ii = np.where(~np.isnan(magave1))[0]
    f = interp.interp1d(tmag1[ii], magave1[ii], fill_value='extrapolate')
    maginterp1 = f(tt)
    plt.plot(tt,maginterp1+zp_best1,'--',c=color1,linewidth=2,label='Lightcurves+Spectra')
    plt.plot(tt,maginterp1+zp_best1-errorbudget,'-',c=color1,linewidth=2)
    plt.plot(tt,maginterp1+zp_best1+errorbudget,'-',c=color1,linewidth=2)
    plt.fill_between(tt,maginterp1+zp_best1-errorbudget,maginterp1+zp_best1+errorbudget,facecolor=color1,alpha=0.2)

    ii = np.where(~np.isnan(magave2))[0]
    f = interp.interp1d(tmag2[ii], magave2[ii], fill_value='extrapolate')
    maginterp2 = f(tt)
    plt.plot(tt,maginterp2+zp_best2,'--',c=color2,linewidth=2,label='Lightcurves Only')
    plt.plot(tt,maginterp2+zp_best2-errorbudget,'-',c=color2,linewidth=2)
    plt.plot(tt,maginterp2+zp_best2+errorbudget,'-',c=color2,linewidth=2)
    plt.fill_between(tt,maginterp2+zp_best2-errorbudget,maginterp2+zp_best2+errorbudget,facecolor=color2,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 14.0])
    plt.ylim([-17.0,-11.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=48)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

plotName = "%s/models_panels_twocomponent.pdf"%(plotDir)
#plt.figure(figsize=(20,18))
plt.figure(figsize=(20,28))

tini, tmax, dt = 0.0, 21.0, 0.1
tt = np.arange(tini,tmax,dt)

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color, markersize=16)

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=16)

    if filt == "w":
        magave1 = (mag1[1]+mag1[2]+mag1[3])/3.0
    elif filt == "c":
        magave1 = (mag1[1]+mag1[2])/2.0
    elif filt == "o":
        magave1 = (mag1[2]+mag1[3])/2.0
    else:
        magave1 = mag1[magidx]

    if filt == "w":
        magave2 = (mag2[1]+mag2[2]+mag2[3])/3.0
    elif filt == "c":
        magave2 = (mag2[1]+mag2[2])/2.0
    elif filt == "o":
        magave2 = (mag2[2]+mag2[3])/2.0
    else:
        magave2 = mag2[magidx]

    #ii = np.where(~np.isnan(magave1))[0]
    #f = interp.interp1d(tmag1[ii], magave1[ii], fill_value='extrapolate')
    #maginterp1 = f(tt)
    #plt.plot(tt,maginterp1+zp_best1,'--',c=color1,linewidth=2,label='1 Component')
    #plt.plot(tt,maginterp1+zp_best1-errorbudget,'-',c=color1,linewidth=2)
    #plt.plot(tt,maginterp1+zp_best1+errorbudget,'-',c=color1,linewidth=2)
    #plt.fill_between(tt,maginterp1+zp_best1-errorbudget,maginterp1+zp_best1+errorbudget,facecolor=color1,alpha=0.2)

    ii = np.where(~np.isnan(magave2))[0]
    f = interp.interp1d(tmag2[ii], magave2[ii], fill_value='extrapolate')
    maginterp2 = f(tt)
    plt.plot(tt,maginterp2+zp_best2,'--',c=color2,linewidth=2,label='2 Component')
    plt.plot(tt,maginterp2+zp_best2-errorbudget,'-',c=color2,linewidth=2)
    plt.plot(tt,maginterp2+zp_best2+errorbudget,'-',c=color2,linewidth=2)
    plt.fill_between(tt,maginterp2+zp_best2-errorbudget,maginterp2+zp_best2+errorbudget,facecolor=color2,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 14.0])
    plt.ylim([-17.0,-11.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=48)
plt.savefig(plotName, bbox_inches='tight')
plt.close()
