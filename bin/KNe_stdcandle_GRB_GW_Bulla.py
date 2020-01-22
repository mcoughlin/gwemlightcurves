
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

    parser.add_option("-a","--analysis_type",default="inferred", help="measured,inferred,combined") 

    parser.add_option("-f","--fit_type",default="linear", help="linear,gpr")
 
    parser.add_option("-g","--grb_name",default="GRB060614") 

    parser.add_option("--nsamples",default=-1,type=int)

    parser.add_option("--multinest_samples", default="../plots/gws/Ka2017_old/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat")
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
        cube[1] = cube[1]*2000.0

def myloglike_H0(cube, ndim, nparams):
        H0 = cube[0]
        d = cube[1]

        c = 299792458.0*1e-3
        vr_mean, vr_std = redshift*c, redshift_error*c
        pvr = (1/np.sqrt(2*np.pi*vr_std**2))*np.exp((-1/2.0)*((vr_mean-H0*d)/vr_std)**2)
        prob_dist = kde_eval_single(kdedir_dist,[d])[0]
        #print(H0, d, vp, np.log(pvr), np.log(pvp), np.log(prob_dist))

        prob = np.log(pvr) + np.log(prob_dist)

        if np.isnan(prob):
            prob = -np.inf

        return prob

# Parse command line
opts = parse_commandline()
grbname = opts.grb_name

if grbname == "GRB060614":
    redshift, redshift_error = 0.125, 0.0010
    distance = 584.93149
    multinest_samples = "../plots/gws/Ka2017/V_R_I/0_10/ejecta/GRB060614/0.10/2-post_equal_weights.dat"
    #multinest_samples = "../plots/gws/Ka2017/V_R_F606W_I_F814W/0_10/ejecta/GRB060614/1.00/2-post_equal_weights.dat"
    #multinest_samples = "../plots/gws/Ka2017_FixZPT0/V_R_I/0_10/ejecta/GRB060614/1.00/2-post_equal_weights.dat"
elif grbname == "GRB150101B":
    redshift, redshift_error = 0.1343, 0.0030
    distance = 632.22111
    multinest_samples = "../plots/gws/Ka2017/r_J_H_K/0_10/ejecta/GRB150101B/0.10/2-post_equal_weights.dat"
    #multinest_samples = "../plots/gws/Ka2017/r_J_H_K/0_10/ejecta/GRB150101B/1.00/2-post_equal_weights.dat"
    #multinest_samples = "../plots/gws/Ka2017_FixZPT0/r_J_H_K/0_10/ejecta/GRB150101B/1.00/2-post_equal_weights.dat"
elif grbname == "GRB050709":
    redshift, redshift_error = 0.1606, 0.0002
    distance = 765.45608
    #multinest_samples = "../plots/gws/Ka2017_FixZPT0/V_R_F814W/0_10/ejecta/GRB050709/1.00/2-post_equal_weights.dat"
    #multinest_samples = "../plots/gws/Ka2017/V_R_F606W_I_F814W/0_10/ejecta/GRB060614/1.00/2-post_equal_weights.dat"
    #multinest_samples = "../plots/gws/Ka2017/V_R_I_F814W_K/0_10/ejecta/GRB050709/0.10/2-post_equal_weights.dat"
    multinest_samples = "../plots/gws/Ka2017_TrPi2018/V_R_I_F814W_K/0_10/GRB050709/0.10/2-post_equal_weights.dat"
elif grbname == "GRB160821B":
    redshift, redshift_error = 0.162, 0.0002
    distance = 765.45608
    multinest_samples = "../plots/gws/Ka2017_TrPi2018/g_F606W_r_i_z_H_F160W_K/0_10/GRB160821B/0.10/2-post_equal_weights.dat"

baseplotDir = os.path.join(opts.plotDir,'standard_candles','GRB_GW',grbname)
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

plotDir = os.path.join(baseplotDir,opts.analysis_type)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

ModelPath = '%s/svdmodels'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

filename = os.path.join(opts.dataDir, 'standard_candles', 'magcolor_rband.dat')
data = np.loadtxt(filename)

mej, vej, Xlan, color, Mag, dmdti, dmdt = data.T

mej, Xlan = np.log10(mej), np.log10(Xlan)
#dmdt = 1.0/(dmdt**2)
dmdt = np.log10(dmdt)
dmdti = np.log10(dmdti)

#idx = np.where(Xlan == -1)[0]
#mej, vej, Xlan, color, Mag, Magi = mej[idx], vej[idx], Xlan[idx], color[idx], Mag[idx], Magi[idx]
#dmdt = dmdt[idx]

n_live_points = 10000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

if opts.fit_type == "gpr":
    #idx = np.arange(len(Mag))
    #np.random.shuffle(idx)
    #idx1, idx2 = np.array_split(idx,2)

    if opts.analysis_type == "measured":
        param_array = np.vstack((color,dmdt,dmdti)).T
    elif opts.analysis_type == "inferred":
        param_array = np.vstack((mej,vej,Xlan)).T
    param_array_postprocess = np.array(param_array)
    param_mins, param_maxs = np.min(param_array_postprocess,axis=0),np.max(param_array_postprocess,axis=0)
    for i in range(len(param_mins)):
        param_array_postprocess[:,i] = (param_array_postprocess[:,i]-param_mins[i])/(param_maxs[i]-param_mins[i])

    nsvds, nparams = param_array_postprocess.shape
    kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0,alpha=1.0)
    gp.fit(param_array_postprocess, Mag)

    M, sigma2_pred = gp.predict(np.atleast_2d(param_array_postprocess), return_std=True)
    sigma_best = np.median(np.sqrt(sigma2_pred))
    sigma = sigma_best*np.ones(M.shape)

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
     
        M = kappa_best + alpha_best*(dmdt) + beta_best*(color) + gamma_best*Magi
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
    
    plotName = "%s/corner.pdf"%(plotDir)
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
    figure.set_size_inches(18.0,18.0)
    plt.savefig(plotName)
    plt.close()

vej_unique, Xlan_unique = np.unique(vej), np.unique(Xlan)
vej_unique = vej_unique[::-1]

fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(len(vej_unique), len(Xlan_unique))
for ii in range(len(vej_unique)):
    for jj in range(len(Xlan_unique)):
        ax = fig.add_subplot(gs[ii, jj])
        plt.axes(ax)
        idx = np.where((vej == vej_unique[ii]) & (Xlan == Xlan_unique[jj]))[0]
        plt.errorbar(10**mej[idx], M[idx], sigma_best*np.ones(M[idx].shape), fmt='k.')
        plt.plot(10**mej[idx], Mag[idx], 'bo')
        if not ii == len(vej_unique) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel('$X_{\mathrm{lan}} = 10^{%.0f}$' % Xlan_unique[jj], fontsize=24)
        if not jj == 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel('$v_{\mathrm{ej}} = %.2f\,c$' % vej_unique[ii], fontsize=24)

        plt.xlim([0.001,0.1])
        plt.ylim([-17,-9])
        plt.gca().invert_yaxis()
        ax.set_xscale('log')

fig.text(0.5, 0.02, 'Ejecta Mass [solar mass]', ha='center', fontsize=30)
fig.text(0.02, 0.5, 'Absolute Magnitude', va='center', rotation='vertical', fontsize=30)
plt.show()
plotName = os.path.join(plotDir,'fitall.pdf')
plt.savefig(plotName)
plt.close()

fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(4, 1)
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0], sharex = ax1)
plt.axes(ax1)
plt.errorbar(10**mej, M, sigma_best*np.ones(M.shape), fmt='k.')
plt.plot(10**mej, Mag, 'bo')

plt.ylabel('Magnitude')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.gca().invert_yaxis()
plt.axes(ax2)
plt.errorbar(10**mej,M-Mag, sigma_best*np.ones(M.shape), fmt='k.')
plt.ylabel('Model - Data')
plt.xlabel('Ejecta mass [solar masses]')
plt.show()
plotName = os.path.join(plotDir,'fit.pdf')
plt.savefig(plotName)
plt.close()

samples = KNTable.read_multinest_samples(multinest_samples, opts.model)
if opts.nsamples > 0:
    samples = samples.downsample(Nsamples=opts.nsamples)
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
    if opts.model == "Ka2017_TrPi2018":
        model_table = KNTable.model("Ka2017", samples, **kwargs)
    else:
        model_table = KNTable.model(opts.model, samples, **kwargs)
    
    f = open(pcklFile, 'wb')
    pickle.dump((model_table), f)
    f.close()

if opts.model == "Ka2017_TrPi2018":
    samples_gw = KNTable.read_multinest_samples(opts.multinest_samples, "Ka2017")    
else:
    samples_gw = KNTable.read_multinest_samples(opts.multinest_samples, opts.model)

if opts.nsamples > 0:
    samples_gw = samples.downsample(Nsamples=opts.nsamples)
#add default values from above to table
samples_gw['tini'] = tini
samples_gw['tmax'] = tmax
samples_gw['dt'] = dt

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
pcklFile = os.path.join(plotDir,"data_gw.pkl")
if os.path.isfile(pcklFile):
    f = open(pcklFile, 'r')
    (model_table_gw) = pickle.load(f)
    f.close()
else:
    model_table_gw = KNTable.model(opts.model, samples_gw, **kwargs)
    f = open(pcklFile, 'wb')
    pickle.dump((model_table_gw), f)
    f.close()

N = 1000
idx = np.random.randint(0, high=len(samples), size=N)
idy = np.random.randint(0, high=len(samples_gw), size=N)

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

mus = []
for ii in range(N):
    row = samples[idx[ii]]

    model = model_table[idx[ii]]
    mag = model['mag']
    t = model['t']
    #Kband = mag[-1]
    Kband = mag[-7]
    iband = mag[-6]
    gband = mag[-8]

    jj = np.argmin(Kband)
    jj7 = np.argmin(np.abs(t-(t[jj]+7.0)))
    M_K = Kband[jj] + 5*(np.log10(distance*1e6) - 1)
    col = iband[jj] - Kband[jj]
    m7 = Kband[jj7]-Kband[jj]
    kk = np.argmin(iband)
    kk7 = np.argmin(np.abs(t-(t[kk]+7.0)))
    m7i = iband[kk7]-iband[kk]
    ll = np.argmin(gband)

    M_i = iband[jj] - iband[kk]

    mej, vej, Xlan = row['mej'], row['vej'], row['Xlan']

    row_gw = samples_gw[idy[ii]]

    model_gw = model_table_gw[idy[ii]]
    mag_gw = model_gw['mag']
    t_gw = model_gw['t']
    #Kband = mag_gw[-1]
    Kband_gw = mag_gw[-7]
    iband_gw = mag_gw[-6]
    gband_gw = mag_gw[-8]

    jj_gw = np.argmin(Kband_gw)
    jj7_gw = np.argmin(np.abs(t_gw-(t_gw[jj]+7.0)))
    M_K_gw = Kband_gw[jj] + 5*(np.log10(40.0*1e6) - 1)
    col_gw = iband_gw[jj] - Kband_gw[jj]
    m7_gw = Kband_gw[jj7]-Kband_gw[jj]
    kk_gw = np.argmin(iband_gw)
    kk7_gw = np.argmin(np.abs(t_gw-(t_gw[kk]+7.0)))
    m7i_gw = iband[kk7]-iband_gw[kk]
    ll_gw = np.argmin(gband_gw)

    M_i_gw = iband_gw[jj] - iband_gw[kk]

    mej_gw, vej_gw, Xlan_gw = row_gw['mej'], row_gw['vej'], row_gw['Xlan']

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

            param_list_postprocess_gw = np.array([np.log10(mej_gw),vej_gw,np.log10(Xlan_gw)])
            for i in range(len(param_mins)):
                param_list_postprocess_gw[i] = (param_list_postprocess_gw[i]-param_mins[i])/(param_maxs[i]-param_mins[i])

            M_pred_gw, sigma2_pred_gw = gp.predict(np.atleast_2d(param_list_postprocess_gw), return_std=True)

            mu1_mu2 = M_K - M_K_gw - M_pred + M_pred_gw
            mu = mu1_mu2 + 5*(np.log10(40.7*1e6) - 1)

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

    mus.append(mu)

mus = np.array(mus)
dist = 10**((mus/5.0) + 1.0) / 1e6
kdedir_dist = greedy_kde_areas_1d(dist)

bin_edges = np.arange(0,2000,20)

dist_16, dist_50, dist_84 = np.percentile(dist,16), np.percentile(dist,50), np.percentile(dist,84)

hist_1, bin_edges_1 = np.histogram(dist, bin_edges, density=True)
bins_1 = (bin_edges_1[:-1] + bin_edges_1[1:])/2.0

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'palegreen'

fig = plt.figure(figsize=(12,7))

#plt.plot([dist_10,dist_10],[0,1],'--',color=color1)
#plt.plot([dist_50,dist_50],[0,1],'--',color=color1)
#plt.plot([dist_90,dist_90],[0,1],'--',color=color1)
plt.step(bins_1, hist_1, color = color1, linestyle='-',label='EM')
#plt.xlim([10,80])
plt.xlabel('Distance [Mpc]')
plt.ylabel('Probability')
#plt.ylim([0,0.10])
plt.grid(True)
plt.show()
plotName = os.path.join(plotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

H0Dir = os.path.join(plotDir,'H0')
if not os.path.isdir(H0Dir):
    os.makedirs(H0Dir)

parameters = ["H0","d"]
labels = [r'$H_0$', r'$D$']
n_params = len(parameters)

pymultinest.run(myloglike_H0, myprior_H0, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%H0Dir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%H0Dir
data = np.loadtxt(multifile)

H0_EM, d, loglikelihood = data[:,0], data[:,1], data[:,2]
idx = np.argmax(loglikelihood)
H0_best, d_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(H0Dir)
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
bins = (bin_edges[:-1] + bin_edges[1:])/2.0

bins_small = np.arange(5,150,1)

fig = plt.figure(figsize=(10,7))
ax = plt.subplot(111)

plt.step(bins, hist_1, color = color1, linestyle='-',label='EM')
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
plt.xlim([20,150])
plt.ylim([0,0.1])
plt.show()
plotName = os.path.join(plotDir,'H0.pdf')
plt.savefig(plotName)
plt.close()

H0_EM_16, H0_EM_50, H0_EM_84 = np.percentile(H0_EM,16), np.percentile(H0_EM,50), np.percentile(H0_EM,84)

print('Distance: %.0f +%.0f -%.0f' % (dist_50, dist_84-dist_50, dist_50-dist_16))
print('H0 EM: %.0f +%.0f -%.0f' % (H0_EM_50, H0_EM_84-H0_EM_50, H0_EM_50-H0_EM_16))

pcklFile = os.path.join(plotDir,"H0.pkl")
f = open(pcklFile, 'wb')
pickle.dump((dist,H0_EM,distance,redshift), f)
f.close()
