#!/usr/bin/env python

# ---- Import standard modules to the python path.
import scipy as sp
import healpy as hp
import itertools
from ligo.skymap.io import fits
from ligo.skymap.distance import parameters_to_marginal_moments, parameters_to_moments
import os, sys, copy
import glob
import numpy as np
import argparse
import pickle
import pandas as pd
from astropy.table import Table

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
from gwemlightcurves.KNModels.table import marginalize_eos_spec
from gwemlightcurves import __version__

from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.add_argument("-o","--outputDir",default="../output")
    parser.add_argument("-p","--plotDir",default="../plots")
    parser.add_argument("-d","--dataDir",default="../data")
    parser.add_argument("-i","--inputDir",default="../input")
    parser.add_argument("--posterior_samples", default="../data/event_data/GW170817_SourceProperties_low_spin.dat")

    parser.add_argument("--cbc_list", default="../data/3G_Lists/list_BNS_detected_3G_median_12.txt")
    parser.add_argument("--cbc_type", default="BNS")

    parser.add_argument("--mindistance",default=1.0,type=float)
    parser.add_argument("--maxdistance",default=1000.0,type=float)

    parser.add_argument("--mchirp_samples", default="../data/chirp_mass/m1m2-s190425z.dat")

    parser.add_argument("-s","--spectraDir",default="../spectra")
    parser.add_argument("-l","--lightcurvesDir",default="../lightcurves")

    parser.add_argument("-a","--analysisType",default="mchirp")

    parser.add_argument("--multinest_samples", default="../plots/limits/Ka2017_FixZPT0/g_r/0_3/ejecta/GW170817/1.00/2-post_equal_weights.dat")
    parser.add_argument("-m","--model",default="Ka2017", help="Ka2017,Ka2017x2")

    parser.add_argument("--doEvent",  action="store_true", default=False)
    parser.add_argument("-e","--event",default="GW190425")
    parser.add_argument("--distance",default=125.0,type=float)
    parser.add_argument("--T0",default=57982.5285236896,type=float)
    parser.add_argument("--errorbudget",default=1.0,type=float)
    parser.add_argument("--nsamples",default=-1,type=int)
    parser.add_argument("--ndownsamples",default=-1,type=int)

    parser.add_argument("--doFixedLimit",  action="store_true", default=False)
    parser.add_argument("--limits",default="20.4,20.4")

    parser.add_argument("-f","--filters",default="u,g,r,i,z,y,J,H,K")
    parser.add_argument("--tmax",default=7.0,type=float)
    parser.add_argument("--tmin",default=0.05,type=float)
    parser.add_argument("--dt",default=0.05,type=float)

    parser.add_argument("--doAddPosteriors",  action="store_true", default=False)
    parser.add_argument("--eostype",default="spec")
    parser.add_argument("--phi_fixed", type=float, default=45)
    parser.add_argument("--Xlan_fixed", type=str, default="uniform")
    parser.add_argument("--skymap_distance", type=str) 
    parser.add_argument("--sigma_ra", type=float, default=6.8)
    parser.add_argument("--sigma_dec", type=float, default=6.8)
    parser.add_argument("--waveform", type=str)
    parser.add_argument("--twixie_flag", default = False, action='store_true') 
    parser.add_argument("--doParallel", default = False, action='store_true')
    parser.add_argument("--Ncore", type = int, default = 8) 

    args = parser.parse_args()
 
    return args


#we introduce skymaps
opts = parse_commandline()

if (opts.skymap_distance):
        skymap, metadata = fits.read_sky_map(opts.skymap_distance, nest=False, distances=True)
        nside = hp.npix2nside(len(skymap[0]))
        npix = len(skymap[0])

        map_struct = {}
        map_struct["prob"] = skymap[0]
        map_struct["distmu"] = skymap[1]
        map_struct["distsigma"] = skymap[2]

        
        #ipix_best = np.argmax(skymap[0])
        #theta_best, phi_best = hp.pix2ang(nside, ipix_best)
        #ra_best = np.rad2deg(phi_best)
        #dec_best = np.rad2deg(0.5 * np.pi - theta_best)
      
        

        #ra_vector = np.linspace(ra_best - opts.sigma_ra, ra_best + opts.sigma_ra, 400)
        #dec_vector = np.linspace(dec_best - opts.sigma_dec, dec_best + opts.sigma_dec, 400)

        #theta_vector = 0.5 * np.pi - np.deg2rad(dec_vector)
        #phi_vector = np.deg2rad(ra_vector)
        #theta_phi_array = list(itertools.product(theta_vector, phi_vector))
        #ipix_vector = [0] * (len(theta_phi_array))

        #for i in range(len(theta_phi_array)):
        #        ipix_vector[i] = hp.ang2pix(nside, theta_phi_array[i][0], theta_phi_array[i][1])
        #ipix_vector = list(set(ipix_vector))

        #all_index = [index for index, value in enumerate(map_struct["prob"])]
        #bad_index = list( set(all_index) - set(ipix_vector))

        #map_struct["prob"][bad_index] = 0
        #map_struct["prob"] = map_struct["prob"] / np.sum(map_struct["prob"])

        distmean, diststd = parameters_to_marginal_moments(map_struct["prob"], map_struct["distmu"], map_struct["distsigma"])




def get_legend(model):

    if model == "DiUj2017":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "KaKy2016":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Me2017":
        legend_name = "Metzger (2017)"
    elif model == "SmCh2017":
        legend_name = "Smartt et al. (2017)"
    elif model == "WoKo2017":
        legend_name = "Wollaeger et al. (2017)"
    elif model == "BaKa2016":
        legend_name = "Barnes et al. (2016)"
    elif model == "Ka2017":
        legend_name = "1 Component"
    elif model == "Ka2017x2":
        legend_name = "2 Component"
    elif model == "RoFe2017":
        legend_name = "Rosswog et al. (2017)"
    elif model == "Bu2019inc":
        legend_name = "je ne suis pas sûr"

    return legend_name

# setting seed
np.random.seed(0)

# Parse command line
#opts = parse_commandline()

mint = opts.tmin
maxt = opts.tmax
dt = opts.dt
tt = np.arange(mint,maxt,dt)

filters = opts.filters.split(",")
limits = [float(x) for x in opts.limits.split(",")]

models = opts.model.split(",")
for model in models:
    if not model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017","BaKa2016","Ka2017","RoFe2017", "Bu2019inc"]:
        print("Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017,BaKa2016,Ka2017,RoFe2017", "Bu2019inc")
        exit(0)

lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir
ModelPath = '%s/svdmodels'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)


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

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,"_".join(models))
plotDir = os.path.join(plotDir,"event")
plotDir = os.path.join(plotDir,opts.event)
plotDir = os.path.join(plotDir,"_".join(filters))
plotDir = os.path.join(plotDir,opts.analysisType)
plotDir = os.path.join(plotDir,opts.waveform)
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
plotDir = os.path.join(plotDir,opts.eostype)
if (opts.model == "Ka2017"):
         if (opts.Xlan_fixed != "uniform"):
                plotDir = os.path.join(plotDir,"Xlan = "+ "%.0f"%(float(opts.Xlan_fixed)))
         else:
                plotDir = os.path.join(plotDir,"Xlan = uniform")
if (opts.model == "Bu2019inc"):
        plotDir = os.path.join(plotDir,"phi = "+ "%.0f"%(opts.phi_fixed))

if opts.analysisType == "cbclist":
    plotDir = os.path.join(plotDir,opts.cbc_type)
    plotDir = os.path.join(plotDir,"%d_%d"%(opts.mindistance,opts.maxdistance))

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
datDir = os.path.join(plotDir,"dat")

if not os.path.isdir(datDir):
    os.makedirs(datDir)

if (opts.analysisType == "posterior") or (opts.analysisType == "mchirp"):
    # read in samples
    if opts.analysisType == "posterior":
        samples = KNTable.read_samples(opts.posterior_samples, Nsamples=opts.nsamples)
        if (opts.eostype == "spec"):
                if opts.doParallel:
                        from joblib import Parallel, delayed
                        samples_array = Parallel(n_jobs=opts.Ncore)(delayed(marginalize_eos_spec)(row, low_latency_flag=False) for row in samples)
                else:
                        samples_array = []
                        for row in samples:
                                samples_array.append(marginalize_eos_spec(row, low_latency_flag=False)) 
                m1s, m2s, chi_effs, lambda1s, lambda2s, r1s, r2s, mb1s, mb2s, mbnss = [], [], [], [], [], [], [], [], [], []
                for temp_samples in samples_array:
                        m1s = m1s + list(temp_samples['m1'])
                        m2s = m2s + list(temp_samples['m2'])
                        chi_effs = chi_effs + list(temp_samples['chi_eff'])
                        lambda1s = lambda1s + list(temp_samples['lambda1'])
                        lambda2s = lambda2s + list(temp_samples['lambda2'])
                        r1s = r1s + list(temp_samples['r1'])
                        r2s = r2s + list(temp_samples['r2'])
                        mb1s = mb1s + list(temp_samples['mb1'])
                        mb2s = mb2s + list(temp_samples['mb2'])
                        mbnss = mbnss + list(temp_samples['mbns'])
        
        else:
                m1s, m2s, mbnss, r1s, r2s, chi_effs, lambda1s, lambda2s, mb1s, mb2s = [], [], [], [], [], [], [], [], [], []

                if opts.eostype == "gp":
                        # read Phil + Reed's EOS files
                        eospostdat = np.genfromtxt("/home/philippe.landry/nseos/eos_post_PSRs+GW170817+J0030.csv",names=True,dtype=None,delimiter=",")
                        idxs = np.array(eospostdat["eos"])
                        weights = np.array([np.exp(weight) for weight in eospostdat["logweight_total"]])
                elif opts.eostype == "Sly":
                        eosname = "SLy"
                        eos = EOS4ParameterPiecewisePolytrope(eosname)
        
                for ii, row in enumerate(samples):
                        print(ii)
                        m1, m2, chi_eff = row["m1"], row["m2"], row["chi_eff"] 
                        nsamples = 100
                        if opts.eostype == "gp":
                                indices = np.random.choice(np.arange(0,len(idxs)), size=nsamples,replace=True,p=weights/np.sum(weights))
                        for jj in range(nsamples):
                                lambda1, lambda2 = -1, -1
                                radius1, radius2 = -1, -1
                                mbaryon1, mbaryon2 = -1, -1
                                mbns = -1
                                index = indices[jj]
                                # samples lambda's from Phil + Reed's files
            
                    	
                                if opts.eostype == "gp":
                                        while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.) or (radius1 < 0) or (radius2 < 0.):
                                                phasetr = 0
                                                eospath = "/home/philippe.landry/nseos/eos/gp/mrgagn/DRAWmod1000-%06d/MACROdraw-%06d/MACROdraw-%06d-%d.csv" % (idxs[index]/1000, idxs[index], idxs[index], phasetr)
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
                    	
                                elif opts.eostype == "Sly":
                                        lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)
                                        mbns = eos.maxmass()
              
  
                                m1s.append(m1)
                                m2s.append(m2)
                                chi_effs.append(chi_eff)
                                lambda1s.append(lambda1)
                                lambda2s.append(lambda2)
                                r1s.append(radius1)
                                r2s.append(radius2)
                                mb1s.append(mbaryon1)
                                mb2s.append(mbaryon2)
                                mbnss.append(mbns)
                                np.random.uniform(0)

        if (opts.Xlan_fixed != "uniform"):
                Xlans = [10**float(opts.Xlan_fixed)] * len(m1s)
        else:
                Xlan_min, Xlan_max = -9, -1
                Xlans = list(10**np.random.uniform(Xlan_min, Xlan_max, len(m1s)))
        phis = [opts.phi_fixed] * len(m1s)
        thetas = 180. * np.arccos(np.random.uniform(-1., 1., len(m1s))) / np.pi
        idx_thetas = np.where(thetas > 90.)[0]
        thetas[idx_thetas] = 180. - thetas[idx_thetas]
        thetas = list(thetas)


        # make final arrays of masses, distances, lambdas, spins, and lanthanide fractions 
        data = np.vstack((m1s,m2s,lambda1s,lambda2s,r1s,r2s,mb1s, mb2s,chi_effs,thetas, phis, mbnss, Xlans)).T
        samples = KNTable(data, names=('m1', 'm2', 'lambda1', 'lambda2', 'r1', 'r2', 'mb1', 'mb2', 'chi_eff','theta', 'phi', 'mbns', "Xlan")) 
        
    else:
        if opts.nsamples < 1:
            print('Please set nsamples >= 1')
            exit(0)
        # read samples from template analysis
        #samples = KNTable.read_mchirp_samples(opts.mchirp_samples, Nsamples=opts.nsamples, twixie_flag = opts.twixie_flag)
        samples = KNTable.read_mchirp_samples(opts.mchirp_samples, twixie_flag = opts.twixie_flag)
        if (opts.eostype == "spec"):
                if opts.doParallel:
                        from joblib import Parallel, delayed
                        samples_array = Parallel(n_jobs=opts.Ncore)(delayed(marginalize_eos_spec)(row, low_latency_flag=True) for row in samples)
                else:
                        samples_array = []
                        for row in samples:
                                samples_array.append(marginalize_eos_spec(row, low_latency_flag=True))
                m1s, m2s, dists_mbta, chi_effs, weights_mbta, lambda1s, lambda2s, r1s, r2s, mb1s, mb2s, mbnss = [], [], [], [], [], [], [], [], [], [], [], []
                for temp_samples in samples_array:
                        m1s = m1s + list(temp_samples['m1'])
                        m2s = m2s + list(temp_samples['m2'])
                        dists_mbta = dists_mbta + list(temp_samples['dist_mbta'])
                        chi_effs = chi_effs + list(temp_samples['chi_eff'])
                        weights_mbta = weights_mbta + list(temp_samples['weight_mbta'])
                        lambda1s = lambda1s + list(temp_samples['lambda1'])
                        lambda2s = lambda2s + list(temp_samples['lambda2'])
                        r1s = r1s + list(temp_samples['r1'])
                        r2s = r2s + list(temp_samples['r2'])
                        mb1s = mb1s + list(temp_samples['mb1'])
                        mb2s = mb2s + list(temp_samples['mb2'])
                        mbnss = mbnss + list(temp_samples['mbns'])
 
        
        else: 
                m1s, m2s, dists_mbta = [], [], []
                lambda1s, lambda2s, chi_effs = [], [], []
                r1s, r2s = [], []
                mb1s, mb2s = [], []
                Xlans = []
                mbnss = []
                weights_mbta = []
                if opts.eostype == "gp":
                        # read Phil + Reed's EOS files
                        eospostdat = np.genfromtxt("/home/philippe.landry/nseos/eos_post_PSRs+GW170817+J0030.csv",names=True,dtype=None,delimiter=",")
                        idxs = np.array(eospostdat["eos"])
                        weights = np.array([np.exp(weight) for weight in eospostdat["logweight_total"]])
                elif opts.eostype == "Sly":
                        eosname = "SLy"
                        eos = EOS4ParameterPiecewisePolytrope(eosname)

       
     
                for ii, row in enumerate(samples): 
                        m1, m2, dist_mbta, chi_eff, weight_mbta = row["m1"], row["m2"], row["dist_mbta"], row["chi_eff"], row["weight_mbta"] 
                        nsamples = 100
                        if opts.eostype == "gp":
                                indices = np.random.choice(np.arange(0,len(idxs)), size=nsamples,replace=True,p=weights/np.sum(weights))
                        for jj in range(nsamples):
                                if (opts.eostype == "gp"):
                                        lambda1, lambda2 = -1, -1
                                        radius1, radius2 = -1, -1
                                        mbns = -1, 
                                        mbaryon1, mbaryon2 = -1, -1
                                        index = indices[jj]
                                # samples lambda's from Phil + Reed's files           

                                if opts.eostype == "gp":
                                        while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                                                phasetr = 0
                                                eospath = "/home/philippe.landry/nseos/eos/gp/mrgagn/DRAWmod1000-%06d/MACROdraw-%06d/MACROdraw-%06d-%d.csv" % (idxs[index]/1000, idxs[index], idxs[index], phasetr)
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
                    	
                                elif opts.eostype == "Sly":
                                        lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)
                                        mbns = eos.maxmass()

                                m1s.append(m1)
                                m2s.append(m2)
                                dists_mbta.append(dist_mbta)
                                lambda1s.append(lambda1)
                                lambda2s.append(lambda2)
                                r1s.append(radius1)
                                r2s.append(radius2)
                                mb1s.append(mbaryon1)
                                mb2s.append(mbaryon2)
                                chi_effs.append(chi_eff)
                                mbnss.append(mbns)
                                weights_mbta.append(weight_mbta)
                                np.random.uniform(0)

 
        if (opts.Xlan_fixed != "uniform"): 
                Xlans = [10**float(opts.Xlan_fixed)] * len(m1s)
        else:
                Xlan_min, Xlan_max = -9, -1
                Xlans = list(10**np.random.uniform(Xlan_min, Xlan_max, len(m1s)))
        phis = [opts.phi_fixed] * len(m1s) 
        thetas = 180. * np.arccos(np.random.uniform(-1., 1., len(m1s))) / np.pi
        idx_thetas = np.where(thetas > 90.)[0]
        thetas[idx_thetas] = 180. - thetas[idx_thetas]
        thetas = list(thetas)

       
        # make final arrays of masses, distances, lambdas, spins, and lanthanide fractions 
        data = np.vstack((m1s,m2s,dists_mbta,lambda1s,lambda2s,r1s,r2s,mb1s,mb2s,chi_effs,thetas, phis, mbnss, weights_mbta, Xlans)).T
        samples = KNTable(data, names=('m1', 'm2', 'dist_mbta', 'lambda1', 'lambda2', 'r1', 'r2', 'mb1', 'mb2', 'chi_eff','theta', 'phi', 'mbns', "weight_mbta", "Xlan"))       
  

    # limit masses
    #samples = samples.mass_cut(mass1=3.0,mass2=3.0)
 
    print("m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"])))
    print("m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"])))
  

    # Downsample 
    #samples = samples.downsample(Nsamples=100)
   
    samples = samples.calc_tidal_lambda(remove_negative_lambda=True)

    # Calc compactness
    #samples = samples.calc_compactness(fit=True)
    samples = samples.calc_compactness(fit=False)
    
    # Calc baryonic mass 
    #samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)

   
    
 
    if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
        #mbns = 2.1
        #idx1 = np.where((samples['m1'] < mbns) & (samples['m2'] < mbns))[0]
        #idx2 = np.where((samples['m1'] > mbns) | (samples['m2'] > mbns))[0]
        
        idx1 = np.where((samples['m1'] <= samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
        idx2 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
        idx3 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] > samples['mbns']))[0]

     

       

        mej, vej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)

        from gwemlightcurves.EjectaFits.PaDi2019 import calc_meje, calc_vej
        # calc the mass of ejecta
        mej1 = calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'])
        # calc the velocity of ejecta
        vej1 = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])

        samples['mchirp'], samples['eta'], samples['q'] = lightcurve_utils.ms2mc(samples['m1'], samples['m2'])

        samples['q'] = 1.0 / samples['q']

        from gwemlightcurves.EjectaFits.KrFo2019 import calc_meje, calc_vave
        # calc the mass of ejecta
       
        
        #mej2 = calc_meje(samples['q'],samples['chi_eff'],samples['c2'], samples['m2'])
        mej2 = calc_meje(samples['q'],samples['chi_eff'],samples['c2'], samples['m2'], samples['mb2'])
        # calc the velocity of ejecta
        vej2 = calc_vave(samples['q'])
       

        # calc the mass of ejecta
        mej3 = np.zeros(samples['m1'].shape)
        # calc the velocity of ejecta
        vej3 = np.zeros(samples['m1'].shape) + 0.2
        
        mej[idx1], vej[idx1] = mej1[idx1], vej1[idx1]
        mej[idx2], vej[idx2] = mej2[idx2], vej2[idx2]
        mej[idx3], vej[idx3] = mej3[idx3], vej3[idx3]


        samples['mej'] = mej
        samples['vej'] = vej
     

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
        
       
        if (opts.model == "Bu2019inc"):  
                idx = np.where(samples['mej'] <= 1e-6)[0]
                samples['mej'][idx] = 1e-11
        elif (opts.model == "Ka2017"):
                idx = np.where(samples['mej'] <= 1e-3)[0]
                samples['mej'][idx] = 1e-11
           
        idx = np.where(samples['mej'] <= 3e-4)[0] 
        print("Probability of having ejecta")
        if opts.analysisType == "mchirp":
                print(100 * (1 - np.sum(samples[idx]['weight_mbta']) / np.sum(samples['weight_mbta'])))
                np.savetxt(os.path.join(plotDir, "HasEjecta.txt"), [100 * (1 - np.sum(samples[idx]['weight_mbta']) / np.sum(samples['weight_mbta']))])
        else:
                print(100 * (len(samples) - len(idx)) / len(samples))
                np.savetxt(os.path.join(plotDir, "HasEjecta.txt"), [100 * (len(samples) - len(idx)) / len(samples)])
       
elif opts.analysisType == "multinest":
    multinest_samples = opts.multinest_samples.split(",")
    samples_all = {}
    for multinest_sample, model in zip(multinest_samples,models):
        # read multinest samples
        samples = KNTable.read_multinest_samples(multinest_sample, model)
        samples["dist"] = opts.distance
        samples_all[model] = samples
elif opts.analysisType == "cbclist":

    tmpfile = opts.cbc_list.replace(".txt",".tmp")
    cbccnt = 0
    lines = [line.rstrip('\n') for line in open(opts.cbc_list)]
    fid = open(tmpfile,'w')
    for line in lines:
        lineSplit = line.split(" ")
        dist = lineSplit[9]
        if (float(dist) > opts.mindistance) and (float(dist) < opts.maxdistance):
            if cbccnt <= opts.nsamples:
                fid.write('%s\n'%line)
            cbccnt = cbccnt + 1
    fid.close()
    cbcratio = float(cbccnt)/float(len(lines))
    
    # read in samples
    samples = KNTable.read_cbc_list(tmpfile)
    # limit masses
    #samples = samples.mass_cut(mass1=3.0,mass2=3.0)

    print("m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"])))
    print("m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"])))

    # Downsample 
    #samples = samples.downsample(Nsamples=1000)

    eosname = "SLy" 
    eos = EOS4ParameterPiecewisePolytrope(eosname)
    lambda1s, lambda2s = [], []
    for row in samples:
        lambda1, lambda2 = eos.lambdaofm(row["m1"]), eos.lambdaofm(row["m2"])
        lambda1s.append(lambda1)
        lambda2s.append(lambda2)
    samples["lambda1"] = lambda1s
    samples["lambda2"] = lambda2s
    samples["Xlan"] = 1e-3

   

    # Calc compactness
    samples = samples.calc_compactness(fit=True)
    # Calc baryonic mass
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)

    if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
        if opts.cbc_type == "BNS":
            from gwemlightcurves.EjectaFits.CoDi2019 import calc_meje, calc_vej
            # calc the mass of ejecta
            samples['mej'] = calc_meje(samples['m1'],samples['c1'], samples['m2'], samples['mc2'])
            # calc the velocity of ejecta
            samples['vej'] = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])
        elif opts.cbc_type == "BHNS":
            samples['chi_eff'] = 1.0
            from gwemlightcurves.EjectaFits.KaKy2016 import calc_meje, calc_vave
            # calc the mass of ejecta
            samples['mej'] = calc_meje(samples['q'],samples['chi_eff'],samples['c1'], samples['mb1'], samples['m1'])
            # calc the velocity of ejecta
            samples['vej'] = calc_vave(samples['q'])

    # HACK: multiply by 10 to get full ejecta
    samples['mej'] = samples['mej'] * 10.0

    idx = np.where(samples['mej']>0.1)[0]
    samples['mej'][idx] = 0.1







if opts.nsamples > 0:
    samples = samples.downsample(Nsamples=opts.ndownsamples)



#add default values from above to table
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



kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False



# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model



pcklFile = os.path.join(plotDir,"data.pkl")
if os.path.isfile(pcklFile):
    f = open(pcklFile, 'rb')
    (model_tables) = pickle.load(f)
    f.close()
else:
    model_tables = {} 
    for model in models:
        if opts.doParallel:
                from joblib import Parallel, delayed
                lightcurve_array = Parallel(n_jobs=opts.Ncore)(delayed(KNTable.model)(model, KNTable(row), **kwargs) for row in samples)
                model_tables[model] = Table([[]], names = (lightcurve_array[0].keys()[0],))
                for keyname in lightcurve_array[0].keys():
                        liste_temp = []
                        for i in range(len(lightcurve_array)):
                                liste_temp.append(lightcurve_array[i][keyname][0]) 
                        model_tables[model][keyname] = liste_temp 
        else:
                model_tables[model] = KNTable.model(model, samples, **kwargs)
        if (opts.model == "Bu2019inc"):
                idx = np.where(model_tables[model]['mej'] <= 1e-6)[0]
                model_tables[model]['mag'][idx] = 10.
                model_tables[model]['lbol'][idx] = 1e30
        elif (opts.model == "Ka2017"):
                idx = np.where(model_tables[model]['mej'] <= 1e-3)[0]
                model_tables[model]['mag'][idx] = 10.
                model_tables[model]['lbol'][idx] = 1e30
      
        
     

    # Now we need to do some interpolation
    for model in models:
        model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model]) 
        #model_tables[model] = lightcurve_utils.interpolate_mags_lbol(model_tables_lbol[model])
    
  

    f = open(pcklFile, 'wb')
    pickle.dump((model_tables), f)
    f.close()

if opts.analysisType == "cbclist":
    fid = open(os.path.join(plotDir,'cbcratio.dat'),'w')
    fid.write('%d %.10f'%(cbccnt,cbcratio))
    fid.close()    

filts = ["u","g","r","i","z","y","J","H","K"]
magidxs = [0,1,2,3,4,5,6,7,8]

idxs = []
for filt in filters:
    idxs.append(filts.index(filt))

filts = [filts[i] for i in idxs]
magidxs = [magidxs[i] for i in idxs]

colors=cm.rainbow(np.linspace(0,1,len(filts)))


mag_all = {}
app_mag_all = {}
if (opts.analysisType == "mchirp"):
        app_mag_all_mbta = {}
        weights_mbta_all = {}
lbol_all = {}

for model in models:
    mag_all[model] = {}
    app_mag_all[model] = {}
    if (opts.analysisType == "mchirp"):
        app_mag_all_mbta[model] = {}
        weights_mbta_all[model] = []
    lbol_all[model] = {}

    lbol_all[model] = np.empty((0,len(tt)), float)
    for filt, color, magidx in zip(filts,colors,magidxs):
        mag_all[model][filt] = np.empty((0,len(tt)))
        app_mag_all[model][filt] = np.empty((0,len(tt)))
        if (opts.analysisType == "mchirp"):
                app_mag_all_mbta[model][filt] = np.empty((0,len(tt)))

peak_mags_all = {}
for model in models:
    model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model])
    for row in model_tables[model]:
        t, lbol, mag = row["t"], row["lbol"], row["mag"]
        if (opts.analysisType == "mchirp"):
            dist_mbta = row['dist_mbta']
            weight_mbta = row['weight_mbta']

        if np.sum(lbol) == 0.0:
            #print "No luminosity..."
            continue

        allfilts = True
        for filt, color, magidx in zip(filts,colors,magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0]
            if len(idx) == 0:
                allfilts = False
                break
        if not allfilts: continue
        for filt, color, magidx in zip(filts,colors,magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0] 
            f = interp.interp1d(t[idx], mag[magidx][idx], fill_value='extrapolate')
            maginterp = f(tt)
            app_maginterp = maginterp + 5*(np.log10((distmean)*1e6) - 1)
            if (opts.analysisType == "mchirp"):
                app_maginterp_mbta = maginterp + 5*(np.log10((dist_mbta)*1e6) - 1)
            mag_all[model][filt] = np.append(mag_all[model][filt],[maginterp],axis=0)
            app_mag_all[model][filt] = np.append(app_mag_all[model][filt],[app_maginterp],axis=0)
            if (opts.analysisType == "mchirp"):
                app_mag_all_mbta[model][filt] = np.append(app_mag_all_mbta[model][filt],[app_maginterp_mbta],axis=0)
        idx = np.where((~np.isnan(np.log10(lbol))) & ~(lbol==0))[0]
        f = interp.interp1d(t[idx], np.log10(lbol[idx]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbol_all[model] = np.append(lbol_all[model],[lbolinterp],axis=0)
        if (opts.analysisType == "mchirp"):
            weights_mbta_all[model] = np.append(weights_mbta_all[model], [weight_mbta])



if opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
    if os.path.isfile(filename):
        data_out = lightcurve_utils.loadEvent(filename)
        data_out_app = lightcurve_utils.loadEvent(filename)
        for ii,key in enumerate(data_out.keys()):
            if key == "t":
                continue
            else:
                data_out[key][:,0] = data_out[key][:,0] - opts.T0
                data_out_app[key][:,0] = data_out_app[key][:,0] - opts.T0
                if (opts.skymap_distance):
                        #data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10((distmean + diststd * np.random.normal(size=len(data_out[key][:,1])))*1e6) - 1)
                        data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10((distmean)*1e6) - 1)
                        data_out[key][:,2] = np.abs(data_out[key][:,2] -5 * diststd /(distmean * np.log(10)))
                else:
                        data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)
    else:
        print('Missing %s... no lightcurves will be plotted' % filename)
        data_out = {}
        data_out_app = {}
elif opts.doFixedLimit:
    data_out = {}
    data_out["t"] = tt
    for filt, limit in zip(filters,limits):
        data_out[filt] = np.vstack((tt,limit*np.ones(tt.shape),np.inf*np.ones(tt.shape))).T
        data_out[filt][:,1] = data_out[filt][:,1] - 5*(np.log10(opts.distance*1e6) - 1)



bounds = [1.0, 15.]
xlims = [1.0, 15.]
ylims = [1e-1,5]

plotName = "%s/mass1.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["m1"], samples["weight_mbta"],Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["m1"], Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["m1"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["m1"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$m_{1}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()


bounds = [1.0, 5.]
xlims = [1.0, 3.]
ylims = [1e-1,5]

plotName = "%s/mass2.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["m2"], samples["weight_mbta"], Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["m2"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["m2"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["m2"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$m_2$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()


bounds = [0, 30.]
xlims = [0, 30.]
ylims = [1e-1,5]

plotName = "%s/radius1.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["r1"] / 1000, samples["weight_mbta"], Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["r1"] / 1000,Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["m1"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["m1"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$R_{1}(km)$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()


bounds = [0, 30.]
xlims = [0, 30.]
ylims = [1e-1,5]

plotName = "%s/radius2.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["r2"]/1000, samples["weight_mbta"], Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["r2"]/1000,Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["m1"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["m1"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$R_{2}(km)$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()



bounds = [-1, 3000.]
xlims = [-1, 3000.]
ylims = [1e-5,1e2]

plotName = "%s/lambda1.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["lambda1"], samples["weight_mbta"], Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["lambda1"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["lambda1"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["lambda1"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$\Lambda_1$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()



bounds = [-1, 3000.]
xlims = [-1, 3000.]
ylims = [1e-5,5]

plotName = "%s/lambda2.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["lambda2"],samples["weight_mbta"],Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["lambda2"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["lambda2"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["lambda2"], 90) 
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$\Lambda_2$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()



bounds = [0., 1.]
xlims = [0., 1.]
ylims = [1e-5,5]

plotName = "%s/c1.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["c1"],samples["weight_mbta"],Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["c1"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["c1"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["c1"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$c_1$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()


bounds = [0., 1.]
xlims = [0., 1.]
ylims = [1e-5,5]

plotName = "%s/c2.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["c2"],samples["weight_mbta"],Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["c2"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["c2"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["c2"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$c_2$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()



bounds = [0., 4.]
xlims = [0., 4.]
ylims = [1e-5,5]

plotName = "%s/mb1.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["mb1"],samples["weight_mbta"],Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["mb1"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["mb1"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["mb1"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$mb_1$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()


bounds = [0., 4.]
xlims = [0., 4.]
ylims = [1e-5,1e2]

plotName = "%s/mb2.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["mb2"],samples["weight_mbta"],Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["mb2"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["mb2"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["mb2"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$mb_2$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()



bounds = [-1., 1.]
xlims = [-1., 1.]
ylims = [1e-5,1e2]
plotName = "%s/chi_eff.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["chi_eff"], samples["weight_mbta"], Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["chi_eff"],Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(samples["chi_eff"], samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(samples["chi_eff"], 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"$\chi_{eff}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()




bounds = [-11.0,-0.5]
xlims = [-11.0,-0.5]
ylims = [1e-5,5]

plotName = "%s/mej.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(np.log10(samples['mej']), samples["weight_mbta"], Nbins=80,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(np.log10(samples['mej']),Nbins=80,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    if (opts.analysisType == "mchirp"):
        lim = lightcurve_utils.weighted_percentile(np.log10(samples['mej']), samples["weight_mbta"], 0.9)
    else:
        lim = np.percentile(np.log10(samples['mej']), 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()




bounds = [0.0,1.0]
xlims = [0.0,1.0]
ylims = [1e-1,20]

plotName = "%s/vej.pdf"%(plotDir)
plt.figure(figsize=(10,8))

for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["vej"],samples["weight_mbta"],Nbins=30,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["vej"],Nbins=30,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name)

plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

bounds = [0.0, 180.]
xlims = [0.0, 180.]
ylims = [1e-4,1e1]
plotName = "%s/theta.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    if (opts.analysisType == "mchirp"):
        bins, hist1 = lightcurve_utils.weighted_hist_results(samples["theta"],samples["weight_mbta"],Nbins=30,bounds=bounds)
    else:
        bins, hist1 = lightcurve_utils.hist_results(samples["theta"],Nbins=30,bounds=bounds) 
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name)

plt.xlabel(r"$\theta$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.yscale("log")
plt.savefig(plotName)
plt.close()

# Can compare low-latency numbers directly to PE runs
if opts.doAddPosteriors:
    samples_posteriors = KNTable.read_samples(opts.posterior_samples)
   

plotName = "%s/mass_parameters.pdf"%(plotDir)
fig = plt.figure(figsize=(14,12))
gs = gridspec.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
plt.axes(ax1)
if (opts.analysisType == "mchirp"):
    bins, hist1 = lightcurve_utils.weighted_hist_results(samples["mchirp"],samples["weight_mbta"],Nbins=25)
else:
    bins, hist1 = lightcurve_utils.hist_results(samples["mchirp"],Nbins=25)
plt.step(bins,hist1,'-',color='k',linewidth=3)
if opts.doAddPosteriors:
    bins, hist1 = lightcurve_utils.hist_results(samples_posteriors["mchirp"],Nbins=25)
    plt.step(bins,hist1,'--',color='r',linewidth=3)    
plt.xlabel(r"Chirp Mass",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.axes(ax2)
if (opts.analysisType == "mchirp"):
    bins, hist1 = lightcurve_utils.weighted_hist_results(samples["q"],samples["weight_mbta"],Nbins=25)
else:
    bins, hist1 = lightcurve_utils.hist_results(samples["q"],Nbins=25)
plt.step(bins,hist1,'-',color='k',linewidth=3, label='Template Bank')
if opts.doAddPosteriors:
    bins, hist1 = lightcurve_utils.hist_results(samples_posteriors["q"],Nbins=25)
    plt.step(bins,hist1,'--',color='r',linewidth=3, label='PE')
plt.xlabel(r"q",fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(plotName)
plt.close()




colors_names=cm.rainbow(np.linspace(0,1,len(models)))
color2 = 'coral'
color1 = 'cornflowerblue'
colors_names=[color1,color2]

linestyles = ['-', '-.', ':','--','-']

plotName = "%s/mag.pdf"%(plotDir)

plt.figure(figsize=(10,8))
cnt = 0
for ii, model in enumerate(models):
    maglen, ttlen = lbol_all[model].shape
    for jj in range(maglen):
        for filt, color, magidx in zip(filts,colors,magidxs):
            if cnt == 0 and ii == 0:
                plt.plot(tt,mag_all[model][filt][jj,:],alpha=0.2,c=color,label=filt,linestyle=linestyles[ii])
            else:
                plt.plot(tt,mag_all[model][filt][jj,:],alpha=0.2,c=color,linestyle=linestyles[ii])
        cnt = cnt + 1

    if opts.doEvent or opts.doFixedLimit:
        for filt, color, magidx in zip(filts,colors,magidxs):
            if filt in data_out:
     
                samples = data_out[filt]
               
                t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
                idx = np.where(~np.isnan(y))[0]
                t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    
                idx = np.where(np.isfinite(sigma_y))[0]
                plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
                idx = np.where(~np.isfinite(sigma_y))[0]
                #plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)
                plt.plot(t[idx],y[idx],'--',c=color)

plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/peaki.pdf"%(plotDir)

plt.figure(figsize=(10,8))
cnt = 0
for model in models:
    plt.scatter(model_tables[model]["peak_tt_i"],model_tables[model]["peak_mag_i"]+np.floor(5*(np.log10(distmean) - 1)),c=np.log10(model_tables[model]["mej"])) 
plt.xlabel(r'$t_{\rm peak}$ [days]')
plt.ylabel(r'Peak i-band magnitude')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label(r'Ejecta mass log10($M_{\odot}$)')
plt.savefig(plotName)
plt.close()

bounds = [15,35]
xlims = [15.0,35.0]
ylims = [1e-2,1]

plotName = "%s/appi.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins, hist1 = lightcurve_utils.hist_results(model_tables[model]["peak_mag_i"]+np.floor(5*(np.log10(distmean) - 1)),Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
plt.xlabel(r"Apparent Magnitude [mag]",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

if opts.analysisType == "cbclist":
    bounds = [15,35]
    xlims = [15.0,35.0]
    ylims = [1,100000]

    plotName = "%s/rates.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,model in enumerate(models):
        legend_name = get_legend(model)
        bins, hist1 = lightcurve_utils.hist_results(model_tables[model]["peak_appmag_i"],Nbins=25,bounds=bounds)
        hist1_cumsum = float(cbccnt)*hist1 / np.sum(hist1)
        hist1_cumsum = np.cumsum(hist1_cumsum)
        plt.semilogy(bins,hist1_cumsum,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    plt.xlabel(r"Apparent Magnitude [mag]",fontsize=24)
    plt.ylabel("Rate of apparent magnitude [per year]",fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    #plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

    for model in models:
        for filt, color, magidx in zip(filts,colors,magidxs):
            fid = open(os.path.join(datDir,'%s_%s_list.dat'%(model,filt)),'w')
            for row in model_tables[model]:
                q = np.max([row["m1"]/row["m2"],row["m2"]/row["m1"]])
                fid.write("%.5f %.5f %.5f %.5f %.5f\n"%(q,row["dist"],row["peak_tt_%s"%filt],row["peak_mag_%s"%filt],row["peak_appmag_%s"%filt]))
            fid.close()

colors_names=cm.rainbow(np.linspace(0,1,len(models)))
colors_names= [color1] * len(models)

for model in models:
    for filt, color, magidx in zip(filts,colors,magidxs):

        fid = open(os.path.join(datDir,'%s_%s.dat'%(model,filt)),'w')
        fid.write("t [days] min(10th percentile) median max(90th percentile)\n")
     
        if (opts.analysisType == "mchirp"):
            magmed, magmax, magmin = np.zeros(mag_all[model][filt].shape[1]), np.zeros(mag_all[model][filt].shape[1]), np.zeros(mag_all[model][filt].shape[1])
            for kk in range(mag_all[model][filt].shape[1]):
                   magmed[kk] = lightcurve_utils.weighted_percentile(mag_all[model][filt][:,kk], weights_mbta_all[model], 0.5)
                   magmax[kk] = lightcurve_utils.weighted_percentile(mag_all[model][filt][:,kk], weights_mbta_all[model], 0.9) + opts.errorbudget
                   magmin[kk] = lightcurve_utils.weighted_percentile(mag_all[model][filt][:,kk], weights_mbta_all[model], 0.1) - opts.errorbudget
        else:
           magmed = np.percentile(mag_all[model][filt], 50, axis=0) 
           magmax = np.percentile(mag_all[model][filt], 90, axis=0) + opts.errorbudget
           magmin = np.percentile(mag_all[model][filt], 10, axis=0) - opts.errorbudget
        for a,b,c,d in zip(tt,magmin,magmed,magmax):
            fid.write("%.5f %.5f %.5f %.5f\n"%(a,b,c,d))
        fid.close()

plotName = "%s/mag_panels.pdf"%(plotDir)
if len(models) < 4:
    plt.figure(figsize=(20,28))
else:
    plt.figure(figsize=(20,28))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    color = color1 
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if opts.doEvent or opts.doFixedLimit:
        if filt in data_out:
            samples = data_out[filt]
        
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    
            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
           
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)
            

    for ii, model in enumerate(models):
        legend_name = get_legend(model)


        
        if (opts.analysisType == "mchirp"):
                magmed, magmax, magmin = np.zeros(mag_all[model][filt].shape[1]), np.zeros(mag_all[model][filt].shape[1]), np.zeros(mag_all[model][filt].shape[1])
                for kk in range(mag_all[model][filt].shape[1]):
                   magmed[kk] = lightcurve_utils.weighted_percentile(mag_all[model][filt][:,kk], weights_mbta_all[model], 0.5)
                   magmax[kk] = lightcurve_utils.weighted_percentile(mag_all[model][filt][:,kk], weights_mbta_all[model], 0.9) + opts.errorbudget
                   magmin[kk] = lightcurve_utils.weighted_percentile(mag_all[model][filt][:,kk], weights_mbta_all[model], 0.1) - opts.errorbudget
        else:
                magmed = np.percentile(mag_all[model][filt], 50, axis=0)
                magmax = np.percentile(mag_all[model][filt], 90, axis=0) + opts.errorbudget
                magmin = np.percentile(mag_all[model][filt], 10, axis=0) - opts.errorbudget

        plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=4,label=legend_name)
        plt.plot(tt,magmin,'-',c=colors_names[ii],linewidth=4)
        plt.plot(tt,magmax,'-',c=colors_names[ii],linewidth=4)
        plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=3)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([mint, maxt])
    if opts.event == "GW190510":
        plt.ylim([-16.0,-8.0])
    else:
        plt.ylim([-24.0, -2.])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    if cnt == 1:
        if opts.event == "GW190510":
            ax1.set_yticks([-16,-14,-12,-10,-8])
        else:
            ax1.set_yticks([-22,-16,-10, -4])
        plt.setp(ax1.get_xticklabels(), visible=False)


        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
    else:
  

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

    if (not cnt == len(filts)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
ax2.set_xlabel('Time [days]',fontsize=48,labelpad=30)
plt.savefig(plotName, bbox_inches='tight')
plt.close()


plotName = "%s/app_mag_panels.pdf"%(plotDir)
if len(models) < 4:
    plt.figure(figsize=(20,28))
else:
    plt.figure(figsize=(20,28))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    color = color1 
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if opts.doEvent or opts.doFixedLimit:
        if filt in data_out_app:
            samples = data_out_app[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    
            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
           
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)
            

    for ii, model in enumerate(models):
        legend_name = get_legend(model)

      
        if (opts.analysisType == "mchirp"):
                app_magmed, app_magmax, app_magmin = np.zeros(app_mag_all[model][filt].shape[1]), np.zeros(app_mag_all[model][filt].shape[1]), np.zeros(app_mag_all[model][filt].shape[1])
                for kk in range(app_mag_all[model][filt].shape[1]):
                        app_magmed[kk] = lightcurve_utils.weighted_percentile(app_mag_all[model][filt][:,kk], weights_mbta_all[model], 0.5)
                        app_magmax[kk] = lightcurve_utils.weighted_percentile(app_mag_all[model][filt][:,kk], weights_mbta_all[model], 0.9) + opts.errorbudget
                        app_magmin[kk] = lightcurve_utils.weighted_percentile(app_mag_all[model][filt][:,kk], weights_mbta_all[model], 0.1) - opts.errorbudget
        else:
                app_magmed = np.percentile(app_mag_all[model][filt], 50, axis=0)
                app_magmax = np.percentile(app_mag_all[model][filt], 90, axis=0) + opts.errorbudget
                app_magmin = np.percentile(app_mag_all[model][filt], 10, axis=0) - opts.errorbudget

        plt.plot(tt,app_magmed,'--',c=colors_names[ii],linewidth=4,label=legend_name)
        plt.plot(tt,app_magmin,'-',c=colors_names[ii],linewidth=4)
        plt.plot(tt,app_magmax,'-',c=colors_names[ii],linewidth=4)
        plt.fill_between(tt,app_magmin,app_magmax,facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=3)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([mint, maxt])
    if opts.event == "GW190510":
        plt.ylim([-16.0,-8.0])
    else:
    
        plt.ylim([11.0, 33.])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    if cnt == 1:

        if opts.event == "GW190510":
            ax1.set_yticks([-16,-14,-12,-10,-8])
        else:
      
            ax1.set_yticks([13,19,25, 31])
        plt.setp(ax1.get_xticklabels(), visible=False)
        



        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
    else:
        

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

    if (not cnt == len(filts)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
ax2.set_xlabel('Time [days]',fontsize=48,labelpad=30)
plt.savefig(plotName, bbox_inches='tight')
plt.close()


if (opts.analysisType == "mchirp"):
    plotName = "%s/app_mag_panels_mbta.pdf"%(plotDir)
    if len(models) < 4:
        plt.figure(figsize=(20,28))
    else:
        plt.figure(figsize=(20,28))

    cnt = 0
    for filt, color, magidx in zip(filts,colors,magidxs):
        color = color1 
        cnt = cnt+1
        vals = "%d%d%d"%(len(filts),1,cnt)
        if cnt == 1:
                ax1 = plt.subplot(eval(vals))
        else:
                ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

        if opts.doEvent or opts.doFixedLimit:
                if filt in data_out_app:
                        samples = data_out_app[filt]
                        t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
                        idx = np.where(~np.isnan(y))[0]
                        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    
                        idx = np.where(np.isfinite(sigma_y))[0]
                        plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
           
                        idx = np.where(~np.isfinite(sigma_y))[0]
                        plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)
            

        for ii, model in enumerate(models):
                legend_name = get_legend(model)

      
        
                app_magmed_mbta, app_magmax_mbta, app_magmin_mbta = np.zeros(app_mag_all_mbta[model][filt].shape[1]), np.zeros(app_mag_all_mbta[model][filt].shape[1]), np.zeros(app_mag_all_mbta[model][filt].shape[1]) 
                for kk in range(app_mag_all_mbta[model][filt].shape[1]):
                        app_magmed_mbta[kk] = lightcurve_utils.weighted_percentile(app_mag_all_mbta[model][filt][:,kk], weights_mbta_all[model], 0.5)
                        app_magmax_mbta[kk] = lightcurve_utils.weighted_percentile(app_mag_all_mbta[model][filt][:,kk], weights_mbta_all[model], 0.9) + opts.errorbudget
                        app_magmin_mbta[kk] = lightcurve_utils.weighted_percentile(app_mag_all_mbta[model][filt][:,kk], weights_mbta_all[model], 0.1) - opts.errorbudget
                

                plt.plot(tt,app_magmed_mbta,'--',c=colors_names[ii],linewidth=4,label=legend_name)
                plt.plot(tt,app_magmin_mbta,'-',c=colors_names[ii],linewidth=4)
                plt.plot(tt,app_magmax_mbta,'-',c=colors_names[ii],linewidth=4)
                plt.fill_between(tt,app_magmin_mbta,app_magmax_mbta,facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=3)

        plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
        plt.xlim([mint, maxt])
        if opts.event == "GW190510":
                plt.ylim([-16.0,-8.0])
        else:
            
                plt.ylim([11.0, 33.])
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

        if cnt == 1:

                if opts.event == "GW190510":
                        ax1.set_yticks([-16,-14,-12,-10,-8])
                else:
              
                        ax1.set_yticks([13,19,25, 31])
                plt.setp(ax1.get_xticklabels(), visible=False)


                plt.xticks(fontsize=36)
                plt.yticks(fontsize=36)
        else:
        

                plt.xticks(fontsize=36)
                plt.yticks(fontsize=36)

        if (not cnt == len(filts)) and (not cnt == 1):
                plt.setp(ax2.get_xticklabels(), visible=False)
 
    ax1.set_zorder(1)
    ax2.set_xlabel('Time [days]',fontsize=48,labelpad=30)
    plt.savefig(plotName, bbox_inches='tight')
    plt.close()



plotName = "%s/gminusr.pdf"%(plotDir)

plt.figure()
cnt = 0
for ii, model in enumerate(models):
    legend_name = get_legend(model)

    if (opts.analysisType == "mchirp"):
        magmed = np.zeros(mag_all[model]["r"].shape[1])
        for kk in range(mag_all[model]["r"].shape[1]):
                magmed[kk] = lightcurve_utils.weighted_percentile(mag_all[model]["g"][:,kk]-mag_all[model]["r"][:,kk],weights_mbta_all[model], 0.5)
    else:          
        magmed = np.median(mag_all[model]["g"]-mag_all[model]["r"],axis=0)
    magmax = np.max(mag_all[model]["g"]-mag_all[model]["r"],axis=0) + opts.errorbudget
    magmin = np.min(mag_all[model]["g"]-mag_all[model]["r"],axis=0) - opts.errorbudget

    plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlim([0.0, 14.0])
plt.ylim([-4., 14.0])
plt.xlabel('Time [days]')
plt.ylabel('Color [g-i]')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/Lbol.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    legend_name = get_legend(model)

    if (opts.analysisType == "mchirp"):
        lbolmed = np.zeros(lbol_all[model].shape[1])
        for kk in range(lbol_all[model].shape[1]):
                lbolmed[kk] = lightcurve_utils.weighted_percentile(lbol_all[model][:,kk], weights_mbta_all[model], 0.5)
    else:
        lbolmed = np.median(lbol_all[model],axis=0)
    lbolmax = np.max(lbol_all[model],axis=0) * (2.5 * opts.errorbudget)
    lbolmin = np.min(lbol_all[model],axis=0) / (2.5 * opts.errorbudget)
    plt.loglog(tt,lbolmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,lbolmin,lbolmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlim([0.0, 50.0])
plt.ylim([1e38, 1e43])
plt.legend(loc="best")
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.savefig(plotName)
plt.close()

