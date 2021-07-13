import numpy as np
from astropy.table import (Table, Column, vstack)
import sys
import os
import scipy.stats as ss
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle

### non-standard libraries
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves import __version__
#from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope



#Types = ['BNS_alsing','BNS_farrow','BNS_equal_alsing','BNS_equal_farrow','BNS_uniform','NSBH_uniform','NSBH_zhu','BNS_chirp_q']

Types = ['BNS_alsing', 'BNS_farrow']
    
for Type in Types:
    fig, ax = plt.subplots(figsize=(16, 12))
    print(f'Initializing {Type}')
    #mej_theta_data=np.loadtxt('./mej_theta_data/N_50/mej_theta_data_BNS_alsing.txt')
    mej_theta_data=np.loadtxt(f'./mej_theta_data/mej_theta_data_{Type}.txt')
    mej_theta_data = mej_theta_data[:100]


    mej_data, thetas = mej_theta_data[:,0], mej_theta_data[:,1]

 
    l = len(mej_data)
    print(f'{l} samples loaded')
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

    sample_split = []

    N_parallel = 16
    N_per_core = int(l/N_parallel)
    for k in range(N_per_core, l, N_per_core):
        sample_split.append(samples[(k-N_per_core):k])
    if k < (l):
        sample_split.append(samples[k:l])
    print(f'Running on {N_parallel} cores for ~{N_per_core} samples each')
    mag_data = []
    t_data = [] 
    parallel_data = (Parallel(n_jobs=N_parallel)(delayed(KNTable.model)(model, s, **kwargs) for s in sample_split))
    
    t_list, u_list, g_list, r_list, i_list = [], [], [], [], []
    z_list, y_list, J_list, H_list, K_list = [], [], [], [], []

    #Type = 'BNS_alsing'
    u_band_max = []
    mags = []
    for data in parallel_data:
        for sample in data:
            mag = sample['mag']
            mej = sample['mej']
            t = sample['t']
            phi = sample['phi']
            theta = sample['theta']
        
            data_lists = [u_list, g_list, r_list, i_list, z_list, y_list, J_list, H_list, K_list]
            for i, band in enumerate(mag):
                #data_lists[i].append(band)
                data_lists[i] = np.concatenate((data_lists[i], band))
            lightcurve_data = np.column_stack((t, data_lists[0], data_lists[1], data_lists[2], data_lists[3], data_lists[4], data_lists[5], data_lists[6], data_lists[7], data_lists[8]))

            u_band_data = lightcurve_data[:,1]
            u_band_max.append(np.min(u_band_data))

    mej_bins = np.linspace(np.min(mej_data), np.max(mej_data), 50)
    mag_bins = np.linspace(np.min(u_band_max), np.max(u_band_max), 50)
    
    print(np.shape(mej_data), np.shape(u_band_max))

    plt.hist2d(mej_data, u_band_max, bins = (50, 50))
    #plt.title("mej vs peak mag")
    plt.xlabel('mej')
    plt.ylabel('Peak u mag')
    plt.gca().invert_yaxis()
    plt.savefig(f'mej_mag_hist2d_{Type}.pdf', bbox_inches='tight')
    #fig.colorbar(h)
    plt.close()

