
import os, sys, glob, pickle
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats
from scipy.stats import kde

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

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

plotDir = '../plots/gws/masses_GRB'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

grb = "GRB150101B"
dataDirKN = "../plots/gws/Ka2017_FixZPT0/r_J_H_K/0_10/ejecta/GRB150101B/1.00/"

nsamples = 10
data_out = {}
if not grb in data_out:
    data_out[grb] = {}
multifile = lightcurve_utils.get_post_file(dataDirKN)
data = np.loadtxt(multifile)
data_out[grb]["KN"] = data
data_out[grb]["KN_samples"] = KNTable.read_multinest_samples(multifile,'Ka2017')
#data_out[grb]["KN_samples"] = data_out[grb]["KN_samples"].downsample(Nsamples=nsamples)

ModelPath = '%s/svdmodels'%('../output')
kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False

filts = ["u","g","r","i","z","y","J","H","K"]
errorbudget = 0.0
tini, tmax, dt = 0.1, 14.0, 0.1

samples = data_out[grb]["KN_samples"]
samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt
data_out[grb]["KN_model"] = KNTable.model('Ka2017', samples, **kwargs)
data_out[grb]["KN_peak"] = lightcurve_utils.get_peak(data_out[grb]["KN_model"])

peak_time_H, peak_mag_H = data_out[grb]["KN_peak"]["H"][:,0], data_out[grb]["KN_peak"]["H"][:,1]
peak_time_g, peak_mag_g = data_out[grb]["KN_peak"]["g"][:,0], data_out[grb]["KN_peak"]["g"][:,1]

print(peak_time_g)
print(peak_time_H)
print(peak_mag_g)
print(peak_mag_H)

plotName = "%s/peak.pdf"%(plotDir)

alpha_par = 0.6
nbins = 200

xH, yH = np.array(peak_time_H), np.array(peak_mag_H)
kH = kde.gaussian_kde([peak_time_H, peak_mag_H])
x = np.linspace(0,4, 100)
y =np.linspace(-17., -15., 100)
xiH, yiH = np.mgrid[xH.min():xH.max():nbins*1j, yH.min():yH.max():nbins*1j]
ziH = kH(np.vstack([xiH.flatten(), yiH.flatten()]))
xg, yg = np.array(peak_time_g), np.array(peak_mag_g)
kg = kde.gaussian_kde([peak_time_g, peak_mag_g])
xig, yig = np.mgrid[xg.min():xg.max():nbins*1j, yg.min():yg.max():nbins*1j]
zig = kg(np.vstack([xig.flatten(), yig.flatten()]))

cmapg = plt.get_cmap("Blues")
cmapH = plt.get_cmap('Reds')

plt.figure(figsize=(20,20))
    
axdensity = plt.axes([0.1, 0.1, 0.65, 0.65])
axhistx = plt.axes([0.1, 0.77, 0.65, 0.2])
axhisty = plt.axes([0.77, 0.1, 0.2, 0.65])

axdensity.pcolormesh(xig, yig, zig.reshape(xig.shape), cmap =cmapg, alpha = alpha_par)
axdensity.pcolormesh(xiH, yiH, ziH.reshape(xiH.shape), cmap =cmapH, alpha = alpha_par)
axdensity.set_xlabel("Peak Time [days]",fontsize=30)
axdensity.set_ylabel("Peak Magnitude [mag]",fontsize=30)
axdensity.set_xlim(0, 5.0)
axdensity.set_ylim(-17.5, -13.5)

binnumberx = 50
binnumbery = 50

binx = np.linspace(0.0, 3.5, binnumberx)
biny = np.linspace(-17.5, -13.5, binnumbery)
alpha_par = 0.6

axhistx.hist(peak_time_H, bins = binx, color ="Red", alpha = alpha_par, label = "H")
axhistx.hist(peak_time_g, bins = binx, color ="Blue", alpha = alpha_par, label = "g")
axhistx.tick_params(which = "both",
                    bottom = False,
                    labelbottom = False)
axhistx.set_xlim(0, 5.0)
#axhistx.legend(bbox_to_anchor = (0.3, -2.2))

axhisty.hist(peak_mag_H, bins = biny, orientation ="horizontal", color = "Red", alpha = alpha_par)
axhisty.hist(peak_mag_g, bins = biny, orientation ="horizontal", color = "Blue", alpha = alpha_par)
axhisty.set_ylim(-17.5, -13.5)
axhisty.tick_params(which = "both",
                    left = False,
                    labelleft = False)
plt.show()
plt.savefig(plotName)
plt.close()


