from __future__ import division
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('font', **font)
import numpy as np
import astropy.units as u
from astropy.table import (Table, Column, vstack)
import pickle
import scipy.stats as ss
from scipy.stats import rv_continuous
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import math
import pandas as pd 



### non-standard libraries

from gwemlightcurves.KNModels import KNTable
#from gwemlightcurves import __version__
#from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope
#from twixie import kde
#from gwemlightcurves import lightcurve_utils
#from mass_grid import run_EOS
#

fig, ax = plt.subplots(figsize=(16, 12))

mej_theta_data=np.loadtxt('mej_theta_data_BNS_alsing.txt')


mej_data, thetas = mej_theta_data[:,0], mej_theta_data[:,1]
#
 
l= len(mej_data)

phis = 30+30*np.random.rand(l)
samples = Table((mej_data, phis, thetas), names=('mej', 'phi', 'theta'))

tini = 0.1
tmax = 50.0
dt = 0.1

vmin = 0.02
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 1

beta = 3.0
kappa_r = 0.1
slope_r = -1.2
theta_r = 0.0
Ye = 0.3

samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt
samples['vmin'] = vmin
samples['th'] = th
samples['ph'] = ph
samples['kappa'] = kappa
samples['eps'] = eps
samples['alp'] = alp
samples['eth'] = eth
samples['flgbct'] = flgbct
samples['beta'] = beta
samples['kappa_r'] = kappa_r
samples['slope_r'] = slope_r
samples['theta_r'] = theta_r
samples['Ye'] = Ye

ModelPath = "/home/cosmin.stachie/gwemlightcurves/output/svdmodels"
kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False

#model = "Ka2017"
model = "Bu2019inc"
model_tables = {}
model_tables[model] = KNTable.model(model, samples, **kwargs)

idx = np.where(model_tables[model]['mej'] <= 1e-3)[0]
#print("idx")
#print(idx)
model_tables[model]['mag'][idx] = 10.
model_tables[model]['lbol'][idx] = 1e30
 
mags = model_tables[model]['mag']
t = model_tables[model]['t'][0]

#bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']


t_list, u_list, g_list, r_list, i_list = [], [], [], [], []
z_list, y_list, J_list, H_list, K_list = [], [], [], [], []

data_lists = [u_list, g_list, r_list, i_list, z_list, y_list, J_list, H_list, K_list]
 
mej_samples = samples['mej']
theta_samples = samples['theta']
phi_samples = samples['phi']

u_band_max = []
Type = 'BNS_alsing'
for n, sample in enumerate(mags):
    sample_name = 'lc_'+str(Type)+'_mej_'+str(mej_samples[n])+'_theta_'+str(theta_samples[n])+'_phi_'+str(phi_samples[n])+'.pickle' 
    data_lists = [u_list, g_list, r_list, i_list, z_list, y_list, J_list, H_list, K_list]
    for i, band in enumerate(sample):
        #data_lists[i].append(band)
        data_lists[i] = np.concatenate((data_lists[i], band))
    lightcurve_data = np.column_stack((t, data_lists[0], data_lists[1], data_lists[2], data_lists[3], data_lists[4], data_lists[5], data_lists[6], data_lists[7], data_lists[8]))
    u_band_data = lightcurve_data[:,1]
    u_band_max.append(np.max(u_band_data))



plt.hist2d(mej_data, u_band_max)

plt.title("mej vs peak mag")
plt.xlabel('mej')
plt.ylabel('peak u mag')
plt.savefig("histogram_2d_01.png", bbox_inches='tight')

plt.close() 


 
