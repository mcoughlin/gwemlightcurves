
# ---- Import standard modules to the python path.

import os, sys, copy
import glob
import numpy as np
import argparse
import pickle
import pandas as pd

import h5py
from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec

from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.KNModels.table import CLove
from gwemlightcurves import __version__

from gwemlightcurves.EjectaFits.CoDi2019 import calc_meje, calc_vej

np.random.seed(0)

plotDir = '../plots/fits/'

m1 = np.linspace(1.0, 1.7, 100)
m2 = np.linspace(1.0, 1.7, 100)

m1 = np.linspace(1.0, 2.0, 200)
m2 = np.linspace(1.0, 2.0, 200)

nsamples = 100

filenames = glob.glob("/home/philippe.landry/gw170817eos/gp/macro/MACROdraw-*-0.csv")
idxs = []
for filename in filenames:
    filenameSplit = filename.replace(".csv","").split("/")[-1].split("-")
    idxs.append(int(filenameSplit[1]))
idxs = np.array(idxs)
indices = np.random.randint(0, 2395, size=nsamples)

bins = np.linspace(-3, -1, 50)
hists = {}

for jj in range(nsamples):
    index = indices[jj]
    eospath = "/home/philippe.landry/gw170817eos/spec/macro/macro-spec_%dcr.csv" % index
    data_out = np.genfromtxt(eospath, names=True, delimiter=",")
    marray, larray = data_out["M"], data_out["Lambda"]
    f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
    lambda1, lambda2 = f(m1), f(m2)
    c1, c2 = CLove(lambda1), CLove(lambda2)

    m1_grid, m2_grid = np.meshgrid(m1, m2)
    c1_grid, c2_grid = np.meshgrid(c1, c2)

    mej = calc_meje(m1_grid,c1_grid,m2_grid,c2_grid)
    mej_log10 = np.log10(mej)
    idx = np.where(m1_grid > m2_grid)[0]

    hists[jj], bin_edges = np.histogram(mej_log10[idx], bins=bins)
    hists[jj] = hists[jj]/float(np.sum(hists[jj]))
    continue

    plt.figure()

    plt.figure(figsize=(8,6))
    plt.pcolormesh(m1_grid,m2_grid,mej_log10,vmin=-2.5,vmax=-1)
    plt.xlabel(r"$M_1 (M_\odot)$")
    plt.ylabel(r"$M_2 (M_\odot)$")
    plt.xlim([1.0,1.7])
    plt.ylim([1.0,1.7])
    cbar = plt.colorbar()
    cbar.set_label(r'$log_{10} (M_{ej}/M_\odot)$')
    plotName = os.path.join(plotDir,'mej_pcolor.pdf')
    plt.savefig(plotName,bbox_inches='tight')
    plt.close('all')

# GW170817
gw170817_samples = '../data/event_data/GW170817_SourceProperties_low_spin.dat'
samples = KNTable.read_samples(gw170817_samples)
# Calc lambdas
samples = samples.calc_tidal_lambda(remove_negative_lambda=True)
# Calc compactness
samples = samples.calc_compactness(fit=True)
# calc the mass of ejecta
mej1 = calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'])
mej_gw170817 = np.log10(mej1)
mej_gw170817_hist, bin_edges = np.histogram(mej_gw170817, bins=bins)
mej_gw170817_hist = mej_gw170817_hist / float(np.sum(mej_gw170817_hist))

# GW190425
gw190425_samples = '../data/event_data/GW190425_posterior_samples.dat'
samples = KNTable.read_samples(gw190425_samples)
# Calc lambdas
samples = samples.calc_tidal_lambda(remove_negative_lambda=True)
# Calc compactness
samples = samples.calc_compactness(fit=True)
# calc the mass of ejecta
mej1 = calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'])
mej_gw190425 = np.log10(mej1)
mej_gw190425_hist, bin_edges = np.histogram(mej_gw190425, bins=bins)
mej_gw190425_hist = mej_gw190425_hist / float(np.sum(mej_gw190425_hist))

bins_mid = (bins[1:] + bins[:-1])/2.0

flat_prior = np.ones(bins_mid.shape)
flat_prior = flat_prior / np.sum(flat_prior)

color2 = 'coral'
color1 = 'cornflowerblue'
color3 = 'palegreen'
color4 = 'darkmagenta'

plt.figure(figsize=(8,6))
ax = plt.gca()
plt.step(bins_mid, mej_gw170817_hist, drawstyle='steps-pre', linewidth=3, color=color1, alpha=0.5, label='GW170817', zorder=100)
plt.step(bins_mid, mej_gw190425_hist, drawstyle='steps-pre', linewidth=3, color=color2, alpha=0.5, label='GW190425', zorder=100)
plt.step(bins_mid, flat_prior, drawstyle='steps-pre', linewidth=3, color=color3, alpha=0.5, label='Flat prior', zorder=100)
for ii, key in enumerate(hists.keys()):
    if ii ==0:
        plt.step(bins_mid, hists[key], drawstyle='steps-pre', linewidth=3, color='gray', alpha=0.05, label = r"Flat $M_1$, $M_2$")
    else:
        plt.step(bins_mid, hists[key], drawstyle='steps-pre', linewidth=3, color='gray', alpha=0.05)
plt.xlabel(r"$\log_{10} (M_{ej}/M_\odot)$")
plt.ylabel("Probability Density Function")
plt.xlim([-3,-1])
plt.ylim([1e-3,1e0])
plt.legend(loc=2,ncol=2)
ax.set_yscale("log")
plotName = os.path.join(plotDir,'mej_prior.pdf')
plt.savefig(plotName,bbox_inches='tight')
plt.close('all')
 
