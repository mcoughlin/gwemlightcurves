
# coding: utf-8

# In[49]:


from __future__ import division, print_function # python3 compatibilty
import optparse
import pandas
import numpy as np                  # import numpy
from time import time               # use for timing functions
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
    parser.add_option("--nsamples",default=-1,type=int)

    parser.add_option("--multinest_samples", default="../plots/gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat")
    parser.add_option("-m","--model",default="Ka2017", help="Ka2017,Ka2017x2")

    opts, args = parser.parse_args()

    return opts

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*30.0 - 30.0
        cube[1] = cube[1]*10.0 - 5.0
        cube[2] = cube[2]*20.0 - 10.0
        cube[3] = cube[3]*100.0 - 50.0
        cube[4] = cube[4]*50.0 - 25.0
        cube[5] = cube[5]*50.0 - 25.0
	cube[6] = cube[6]*1.0
         
def myloglike(cube, ndim, nparams):
        const = cube[0]
        alpha = cube[1]
        beta = cube[2]
        gamma = cube[3]
        delta = cube[4]
        zeta = cube[5]
        sigma = cube[6]

        M = const + alpha*(dmdt) + beta*(color) + gamma*(mej) + delta*(vej) + zeta*(Xlan)
        #sigma = np.sqrt(sigma**2 + (alpha*dmdt*0.1)**2 + (beta*color*0.1)**2 + np.abs(gamma*mej*0.1)**2 + (delta*vej*0.1)**2 + np.abs(zeta*Xlan*0.1)**2)

        x = Mag - M
        prob = ss.norm.logpdf(x, loc=0.0, scale=sigma)
        prob = np.sum(prob)

        if np.isnan(prob):
            prob = -np.inf

        return prob

# Parse command line
opts = parse_commandline()

baseplotDir = os.path.join(opts.plotDir,'standard_candles')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

plotDir = baseplotDir
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

ModelPath = '%s/svdmodels'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

filename1 = os.path.join(opts.dataDir, 'standard_candles', 'magcolor.dat')
filename2 = os.path.join(opts.dataDir, 'standard_candles', 'magdmdt.dat')

data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)

mej, vej, Xlan, color, Mag = data1.T
dmdt = data2[:,3]

mej, Xlan = np.log10(mej),  np.log10(Xlan)

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["K","alpha","beta","gamma","delta","zeta","sigma"]
labels = [r'K', r'$\alpha$', r'$\beta$', r'$\gamma$', r"$\delta$",r"$\zeta$",r'$\sigma$']
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

K, alpha, beta, gamma, delta, zeta, sigma, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
idx = np.argmax(loglikelihood)
K_best, alpha_best, beta_best, gamma_best, delta_best, zeta_best, sigma_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": title_fontsize},
                   label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                   smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()

M = K_best + alpha_best*(dmdt) + beta_best*(color) + gamma_best*(mej) + delta_best*(vej) + zeta_best*Xlan

vej_unique, Xlan_unique = np.unique(vej), np.unique(Xlan)
vej_unique = vej_unique[::-1]

fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(len(vej_unique), len(Xlan_unique))
for ii in range(len(vej_unique)):
    for jj in range(len(Xlan_unique)):
        ax = fig.add_subplot(gs[ii, jj])
        plt.axes(ax)
        idx = np.where((vej == vej_unique[ii]) & (Xlan == Xlan_unique[ii]))[0]
        plt.errorbar(10**mej[idx], M[idx], sigma_best*np.ones(M[idx].shape), fmt='k.')
        plt.plot(10**mej[idx], Mag[idx], 'bo')
        if not ii == len(vej_unique) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.xlabel('$X_{\mathrm{lan}} = 10^{%.0f}$' % Xlan_unique[jj])
        if not jj == 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel('$v_{\mathrm{ej}} = %.2f\,c$' % vej_unique[ii])

        plt.xlim([0.001,0.1])
        plt.ylim([-17,-9])
        plt.gca().invert_yaxis()
        ax.set_xscale('log')

fig.text(0.5, 0.04, 'Ejecta Mass [solar mass]', ha='center')
fig.text(0.04, 0.5, 'Ejecta Velocity', va='center', rotation='vertical')
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

samples = KNTable.read_multinest_samples(opts.multinest_samples, opts.model)
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
model_table = KNTable.model(opts.model, samples, **kwargs)

K_mean, K_std = np.mean(K), np.std(K)
alpha_mean, alpha_std = np.mean(alpha), np.std(alpha)
beta_mean, beta_std = np.mean(beta), np.std(beta)
gamma_mean, gamma_std = np.mean(gamma), np.std(gamma)
delta_mean, delta_std = np.mean(delta), np.std(delta)
zeta_mean, zeta_std = np.mean(zeta), np.std(zeta)
sigma_mean, sigma_std = np.mean(sigma), np.std(sigma)

N = 10000
idx = np.random.randint(0, high=len(samples), size=N)

K = np.random.normal(loc=K_mean, scale=K_std, size=N)
alpha = np.random.normal(loc=alpha_mean, scale=alpha_std, size=N)
beta = np.random.normal(loc=beta_mean, scale=beta_std, size=N)
gamma = np.random.normal(loc=gamma_mean, scale=gamma_std, size=N)
delta = np.random.normal(loc=delta_mean, scale=delta_std, size=N)
zeta = np.random.normal(loc=zeta_mean, scale=zeta_std, size=N)
sigma = np.random.normal(loc=0.0, scale=sigma_mean, size=N)

mus = []
for ii in range(N):
    row = samples[idx[ii]]
    model = model_table[idx[ii]]
    mag = model['mag']
    t = model['t']
    Kband = mag[-1]
    iband = mag[-6]

    jj = np.argmin(Kband)
    jj7 = np.argmin(np.abs(t-(t[jj]+7.0)))
    M = Kband[jj] + 5*(np.log10(40.0*1e6) - 1)
    col = Kband[jj] - iband[jj]
    m7 = Kband[jj7]-Kband[jj]

    mej, vej, Xlan = row['mej'], row['vej'], row['Xlan']
    mu = -( -M + K[ii] + m7*alpha[ii] + col*beta[ii] + np.log10(mej)*gamma[ii] + vej*delta[ii] + np.log10(Xlan)*zeta[ii] + sigma[ii])
    mus.append(mu)
mus = np.array(mus)
dist = 10**((mus/5.0) + 1.0) / 1e6

plt.figure(figsize=(10,12))
n, bins, patches = plt.hist(dist, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Distance [Mpc]')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
plotName = os.path.join(plotDir,'dist.pdf')
plt.savefig(plotName)
plt.close()

print('Distance: %.1f +- %.1f' % (np.mean(dist), np.std(dist)))
