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



fig, ax = plt.subplots(figsize=(16, 12))
#Types = ['BNS_alsing','BNS_farrow','BNS_equal_alsing','BNS_equal_farrow','BNS_uniform','NSBH_uniform','NSBH_zhu','BNS_chirp_q']

#Types = ['BNS_alsing','BNS_farrow']
#Types = ['BNS_uniform', 'NSBH_uniform'] 
#Types = ['NSBH_zhu', 'NSBH_LRR']
#Types = ['NSBH_zhu_edited', 'NSBH_LRR_edited'] 
Types = ['NSBH_zhu']
#Types = ['NSBH_q_range']
#Types = ['NSBH_LRR'] 

 
for Type in Types:
    print(f'Initializing {Type}')
    #mej_theta_data=np.loadtxt('./mej_theta_data/N_50/mej_theta_data_BNS_alsing.txt')
    mej_theta_data=np.loadtxt(f'./mej_theta_data/NSBH_test/mej_theta_data_{Type}.txt')
    mej_data, thetas = mej_theta_data[:,0], mej_theta_data[:,1]
    
    #mej_data, thetas = mej_data[:20], thetas[:20]
 
    l = len(mej_data)
    print(f'{l} samples loaded')
    #phis = 30+30*np.random.rand(l)
    phis = 45 * np.ones(l)
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

    #samples = samples[:20]
 
    ModelPath = "/home/cosmin.stachie/gwemlightcurves/output/svdmodels"
    kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
    kwargs["doAB"] = True
    kwargs["doSpec"] = False

    #model = "Ka2017"
    model = "Bu2019inc"
    model_tables = {}

    sample_split = []

    N_parallel = 16
    #N_parallel = 4
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
    mags = []
    print('saving to pickle files')
    N_pickle = 0
    for data in parallel_data:
        for sample in data:
            N_pickle += 1
            mag = sample['mag']
            t = sample['t']
            sample_length = len(mag[0])
            
            mej = sample['mej'] * np.ones(sample_length)
            phi = sample['phi'] * np.ones(sample_length)
            theta = sample['theta'] * np.ones(sample_length)
            
            sample_name = f'./lightcurves_parallel/phi45_updated/{Type}/lc_{Type}_mej_{mej[0]}_theta_{theta[0]}_phi_{phi[0]}_{N_pickle}.pickle'
            data_lists = [u_list, g_list, r_list, i_list, z_list, y_list, J_list, H_list, K_list]
             
            #for i, band in enumerate(mag):
                #data_lists[i].append(band)
                #data_lists[i] = np.concatenate((data_lists[i], band))

            #lightcurve_data = np.column_stack((t, data_lists[0], data_lists[1], data_lists[2], data_lists[3], data_lists[4], data_lists[5], data_lists[6], data_lists[7], data_lists[8], mej, theta, phi))
            
            lightcurve_data = np.column_stack((t, mag[0], mag[1], mag[2], mag[3], mag[4], mag[5], mag[6], mag[7], mag[8], mej, theta, phi))
            with open(sample_name, 'wb') as filename:
                pickle.dump(lightcurve_data,filename, protocol=pickle.HIGHEST_PROTOCOL)
                #pickle.dump(lightcurve_data,filename)


