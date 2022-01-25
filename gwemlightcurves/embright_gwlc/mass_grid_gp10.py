#!/usr/bin/env python

# ---- Import standard modules to the python path.

import os, sys, copy
import glob
import numpy as np
import argparse
import pickle
#import pandas as pd

#import h5py
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
from gwemlightcurves import __version__

#from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope


# setting seed
np.random.seed(0)


# These are the default values supplied with respect to generating lightcurves
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


def run_EOS(EOS, m1, m2, thetas, type_set = 'None', N_EOS = 100, model_set = 'Bu2019inc', chirp_q = False):
    chi = 0
    N_masses = len(m1) 
    if type_set == 'None':
        sys.exit('Type is not defined')
    
    if type_set == 'BNS_chirp_q':
        type_set = 'BNS' 
    if not chirp_q:
        q = m2/m1
        mchirp = np.power((m1*m2), 3/5) / np.power((m1+m2), 1/5)
        eta = m1*m2/( (m1+m2)*(m1+m2) )
    if chirp_q:
        print('running chirp_q')
        m1 = np.random.normal(m1, .05, N_masses)
        m2 = np.random.normal(m2, .05, N_masses)
        #m1 = np.ones(100) * m1
        #m2 = np.ones(100) * m2
        q = m2 
        mchirp = m1
        eta = lightcurve_utils.q2eta(q) 
        m1, m2 = lightcurve_utils.mc2ms(mchirp, eta)

    #chi_eff = np.random.uniform(-1,1,100)
    
    chi_eff = np.ones(N_masses)*chi
    Xlan_val = 1e-3
    Xlan =  Xlan_val
    
    #if lan_override:
    #    Xlan_val = lan_override_val 
    
    #Xlans = np.ones(1)*Xlan_val 
    c1 = np.ones(N_masses)
    c2 = np.ones(N_masses)
    mb1 = np.ones(N_masses)
    mb2 = np.ones(N_masses)
   
    data = np.vstack((m1,m2,chi_eff,mchirp,eta,q)).T
    samples = KNTable((data), names = ('m1','m2','chi_eff','mchirp','eta','q'))

    #data = np.vstack((m1s,m2s,dists,lambda1s,lambda2s,chi_effs,Xlans,c1,c2,mb1,mb2,mchirps,etas,qs,mej,vej, dyn_mej, wind_mej, mbnss)).T
    #samples = KNTable((data), names = ('m1','m2','dist','lambda1','lambda2','chi_eff','Xlan','c1','c2','mb1','mb2','mchirp','eta','q','mej','vej', 'dyn_mej', 'wind_mej', 'mbns'))    


    lambda1s=[]
    lambda2s=[]
    m1s=[]
    m2s=[]
    #dists=[]
    chi_effs=[]
    Xlans=[]
    qs=[]
    etas=[]
    mchirps=[]
    mbnss=[]
    
    nsamples = N_EOS

    m1s, m2s, dists_mbta = [], [], []
    lambda1s, lambda2s, chi_effs = [], [], []
    mbnss = []
    # EOS_gp10 = os.listdir('~/em-bright/ligo/em_bright/EOS_samples_unit_test/')
    # some EOS indices for unit test
    gp10_idx = [137, 138, 421, 422, 423, 424, 425, 426, 427, 428]
    home_dir = os.getenv('HOME')
    print(home_dir)
    print(os.path.isdir(home_dir))
    if EOS == "gp":
        # read Phil + Reed's EOS files
        # eospostdat = np.genfromtxt("/home/philippe.landry/nseos/eos_post_PSRs+GW170817+J0030.csv",names=True,dtype=None,delimiter=",")
        path_post = "em-bright/ligo/em_bright/EOS_samples_unit_test/eos_post_PSRs+GW170817+J0030.csv"
        path_post = os.path.join(home_dir, path_post)
        print(path_post, '-----')
        eospostdat = np.genfromtxt(path_post, names=True, dtype=None, delimiter=",")
        #eospostdat = np.genfromtxt(f"{home_dir}/em-bright/ligo/em_bright/EOS_samples_unit_test/eos_post_PSRs+GW170817+J0030.csv",names=True,dtype=None,delimiter=",")
        idxs = np.array(eospostdat["eos"])
        weights = np.array([np.exp(weight) for weight in eospostdat["logweight_total"]])
    elif EOS == "Sly":
        eosname = "SLy"
        eos = EOS4ParameterPiecewisePolytrope(eosname)

    Xlan_min, Xlan_max = -9, -1 
 
    for ii, row in enumerate(samples): 
        # m1, m2, dist_mbta, chi_eff = row["m1"], row["m2"], row["dist_mbta"], row["chi_eff"]
        m1, m2, chi_eff = row["m1"], row["m2"], row["chi_eff"]
        if EOS == "spec":
            indices = np.random.randint(0, 2396, size=nsamples)
        elif EOS == "gp":
            indices = np.random.choice(gp10_idx, size=nsamples, replace=True)
            # indices = np.random.choice(np.arange(0,len(idxs)), size=nsamples,replace=True,p=weights/np.sum(weights))
        for jj in range(nsamples):
            if (EOS == "spec") or (EOS == "gp"):
                #index = gp10_idx[jj]
                index = indices[jj] 
                lambda1, lambda2 = -1, -1
                mbns = -1
            # samples lambda's from Phil + Reed's files
            if EOS == "spec":
                while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                    eospath = "/home/philippe.landry/nseos/eos/spec/macro/macro-spec_%dcr.csv" % index
                    data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                    marray, larray = data_out["M"], data_out["Lambda"]
                    f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                    if float(f(m1)) > lambda1: lambda1 = f(m1)
                    if float(f(m2)) > lambda2: lambda2 = f(m2)
                    if np.max(marray) > mbns: mbns = np.max(marray)

                    if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                        index = int(np.random.randint(0, 2396, size=1)) # pick a different EOS if it returns negative Lambda or Mmax
                        lambda1, lambda2 = -1, -1
                        mbns = -1

            elif EOS == "gp":
                while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                    phasetr = 0
                    eospath = "em-bright/ligo/em_bright/EOS_samples_unit_test/MACROdraw-1151%d-%d.csv" % (index, phasetr)
                    eospath = os.path.join(home_dir, eospath)
                    # eospath = f"{home_dir}/em-bright/ligo/em_bright/EOS_samples_unit_test/MACROdraw-1151%d-%d.csv" % (index, phasetr)
                    # eospath = "/home/philippe.landry/nseos/eos/gp/mrgagn/DRAWmod1000-%06d/MACROdraw-%06d/MACROdraw-%06d-%d.csv" % (idxs[index]/1000, idxs[index], idxs[index], phasetr)
                    while os.path.isfile(eospath):
                        data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                        marray, larray = data_out["M"], data_out["Lambda"]
                        f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                        if float(f(m1)) > lambda1: lambda1 = f(m1) # pick lambda from least compact stable branch
                        if float(f(m2)) > lambda2: lambda2 = f(m2)
                        if np.max(marray) > mbns: mbns = np.max(marray) # get global maximum mass
                    
                        phasetr += 1 # check all stable branches
                        eospath = "/home/philippe.landry/nseos/eos/gp/mrgagn/DRAWmod1000-%06d/MACROdraw-%06d/MACROdraw-%06d-%d.csv" % (idxs[index]/1000, idxs[index], idxs[index], phasetr)
                    if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                        index = int(np.random.choice(np.arange(0,len(idxs)), size=1,replace=True,p=weights/np.sum(weights))) # pick a different EOS if it returns negative Lambda or Mmax
                        lambda1, lambda2 = -1, -1
                        mbns = -1
                    
            elif EOS == "Sly":
                lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)
                mbns = eos.maxmass()

            m1s.append(m1)
            m2s.append(m2)
            #dists_mbta.append(dist_mbta)
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)
            chi_effs.append(chi_eff)
            #Xlans.append(10**np.random.uniform(Xlan_min, Xlan_max))
            Xlans.append(Xlan)
            mbnss.append(mbns)
    
    #thetas = 180. * np.arccos(np.random.uniform(-1., 1., len(samples) * nsamples)) / np.pi
    idx_thetas = np.where(thetas > 90.)[0]
    thetas[idx_thetas] = 180. - thetas[idx_thetas]
    Xlans = np.ones(np.array(m1s).shape) * Xlan_val
    
    # make final arrays of masses, distances, lambdas, spins, and lanthanide fractions
    data = np.vstack((m1s,m2s,lambda1s,lambda2s,Xlans,chi_effs,thetas,mbnss)).T
    samples = KNTable(data, names=('m1', 'm2', 'lambda1', 'lambda2','Xlan','chi_eff','theta', 'mbns'))       
 
    # limit masses
    #samples = samples.mass_cut(mass1=3.0,mass2=3.0)
     
    print("m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"])))
    print("m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"])))
       
    
    # Downsample 
    #samples = samples.downsample(Nsamples=100)
    samples = samples.calc_tidal_lambda(remove_negative_lambda=True)
    
    # Calc compactness
    samples = samples.calc_compactness(fit=True)
    
    # Calc baryonic mass 
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
    
       
    #----------------------------------------------------------------------------------
    if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
        #mbns = 2.1
        #idx1 = np.where((samples['m1'] < mbns) & (samples['m2'] < mbns))[0]
        #idx2 = np.where((samples['m1'] > mbns) | (samples['m2'] > mbns))[0]
        
        #1 BNS, 2 NSBH, 3 BBH    
        idx1 = np.where((samples['m1'] <= samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
        idx2 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
        idx3 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] > samples['mbns']))[0]
    
         
    
           
    
        mej, vej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)
        wind_mej, dyn_mej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)   
 
        #from gwemlightcurves.EjectaFits.CoDi2019 import calc_meje, calc_vej
        from gwemlightcurves.EjectaFits.PaDi2019 import calc_meje, calc_vej
        # calc the mass of ejecta
        mej1, dyn_mej1, wind_mej1 = calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'], split_mej=True)
        # calc the velocity of ejecta
        vej1 = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])
    
        samples['mchirp'], samples['eta'], samples['q'] = lightcurve_utils.ms2mc(samples['m1'], samples['m2'])
    
        #samples['q'] = 1.0 / samples['q']
    
        from gwemlightcurves.EjectaFits.KrFo2019 import calc_meje, calc_vave
        # calc the mass of ejecta
           
            
        mej2, dyn_mej2, wind_mej2 = calc_meje(samples['q'],samples['chi_eff'],samples['c2'], samples['m2'], split_mej=True)
        # calc the velocity of ejecta
        vej2 = calc_vave(samples['q'])
           
    
        # calc the mass of ejecta
        mej3 = np.zeros(samples['m1'].shape)

        dyn_mej3 = np.zeros(samples['m1'].shape)
        wind_mej3 = np.zeros(samples['m1'].shape)
        # calc the velocity of ejecta
        vej3 = np.zeros(samples['m1'].shape) + 0.2
            
        mej[idx1], vej[idx1] = mej1[idx1], vej1[idx1]
        mej[idx2], vej[idx2] = mej2[idx2], vej2[idx2]
        mej[idx3], vej[idx3] = mej3[idx3], vej3[idx3]
   
        wind_mej[idx1], dyn_mej[idx1] = wind_mej1[idx1], dyn_mej1[idx1]
        wind_mej[idx2], dyn_mej[idx2] = wind_mej2[idx2], dyn_mej2[idx2]
        wind_mej[idx3], dyn_mej[idx3] = wind_mej3[idx3], dyn_mej3[idx3]   
 
        samples['mej'] = mej
        samples['vej'] = vej
        samples['dyn_mej'] = dyn_mej
        samples['wind_mej'] = wind_mej
         
    
        # Add draw from a gaussian in the log of ejecta mass with 1-sigma size of 70%
        erroropt = 'none'
        if erroropt == 'none':
            print("Not applying an error to mass ejecta")
        elif erroropt == 'log':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.236))
        elif erroropt == 'lin':
            samples['mej'] = np.random.normal(samples['mej'],0.72*samples['mej'])
        elif erroropt == 'loggauss':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.312))
        #idx = np.where(samples['mej'] > 0)[0]
        #samples = samples[idx]
    
        idx = np.where(samples['mej'] <= 0)[0]
        samples['mej'][idx] = 1e-11
            
           
        if (model_set == "Bu2019inc"):  
                idx = np.where(samples['mej'] <= 1e-6)[0]
                samples['mej'][idx] = 1e-11
        elif (model_set == "Ka2017"):
                idx = np.where(samples['mej'] <= 1e-3)[0]
                samples['mej'][idx] = 1e-11
               
            
        print("Probability of having ejecta")
        print(100 * (len(samples) - len(idx)) /len(samples))
        return samples



