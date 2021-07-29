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
    parser.add_argument("--nsamples",default=10,type=int)

    parser.add_argument("--doFixedLimit",  action="store_true", default=False)
    parser.add_argument("--limits",default="20.4,20.4")

    parser.add_argument("-f","--filters",default="u,g,r,i,z,y,J,H,K")
    parser.add_argument("--tmax",default=7.0,type=float)
    parser.add_argument("--tmin",default=0.05,type=float)
    parser.add_argument("--dt",default=0.05,type=float)

    parser.add_argument("--doAddPosteriors",  action="store_true", default=False)
    parser.add_argument("--eostype",default="spec")

    parser.add_argument("--twixie_flag", default = False, action='store_true')

    args = parser.parse_args()
 
    return args

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
        legend_name = "Bu2019inc"

    return legend_name

# setting seed
np.random.seed(0)


# Parse command line
opts = parse_commandline()

mint = opts.tmin
maxt = opts.tmax
dt = opts.dt
tt = np.arange(mint,maxt,dt)

filters = opts.filters.split(",")
limits = [float(x) for x in opts.limits.split(",")]

models = opts.model.split(",")
for model in models:
    if not model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017","BaKa2016","Ka2017","RoFe2017", "Bu2019inc"]:
        print("Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017,BaKa2016,Ka2017,RoFe2017,Bu2019inc")
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
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
plotDir = os.path.join(plotDir,opts.eostype)
plotDir = os.path.join(plotDir,"Xlan_random")
if opts.analysisType == "cbclist":
    plotDir = os.path.join(plotDir,opts.cbc_type)
    plotDir = os.path.join(plotDir,"%d_%d"%(opts.mindistance,opts.maxdistance))

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
datDir = os.path.join(plotDir,"dat")
if not os.path.isdir(datDir):
    os.makedirs(datDir)

#examples of commands to run code
#python mass_grid.py --analysisType BNS


twixie_tf = opts.twixie_flag
model = opts.model
Type = opts.analysisType
def run_EOS(EOS, m1, m2, chi, type_set=Type, model_set = 'Bu2019inc', twixie = twixie_tf, lan_override=False, lan_override_val=None, chirp_q = False):
    #samples["dist"] = opts.distance
    #samples["phi"] = opts.phi_fixed
    #samples["Xlan"] = 10**opts.Xlan_fixed
    #samples['mbns'] = 0.
    num_samples = 100
    if type_set == 'BNS_chirp_q':
        type_set = 'BNS' 
    if not chirp_q:
        #m1 = np.random.normal(m1, .05, 100)
        #m2 = np.random.normal(m2, .05, 100)
        #m1 = np.random.normal(m1, .001, 100)
        #m2 = np.random.normal(m2, .001, 100)
        q = m2/m1
        mchirp = np.power((m1*m2), 3/5) / np.power((m1+m2), 1/5)
        eta = m1*m2/( (m1+m2)*(m1+m2) )
    if chirp_q:
        print('running chirp_q')
        #m1 = np.random.normal(m1, .05, N_EOS)
        #m2 = np.random.normal(m2, .05, N_EOS)
        m1 = np.random.normal(m1, .05, 1)
        m2 = np.random.normal(m2, .05, 1)
        #m1 = np.ones(100) * m1
        #m2 = np.ones(100) * m2
        q = m2 
        mchirp = m1
        eta = lightcurve_utils.q2eta(q) 
        m1, m2 = lightcurve_utils.mc2ms(mchirp, eta)

    #chi_eff = np.random.uniform(-1,1,100)
    chi_eff = np.ones(1)*chi
    Xlan_val = 1e-3

    
    if lan_override:
        Xlan_val = lan_override_val 
 
    c1 = np.ones(1)
    c2 = np.ones(1)
    mb1 = np.ones(1)
    mb2 = np.ones(1)
    #lambda1s=
    #lambda2s=np.ones(100)
    #m1s=m1
    #m2s=m1
    #dists=dist
    #chi_effs=chi_eff
    #Xlans=Xlan
    #qs=q
    #etas=eta
    #mchirps=mchirp
    #mbnss=np.ones(100)
    #print(str([len(m1), len(m2), len(chi_eff)]))  
    data = np.vstack((m1,m2,chi_eff,mchirp,eta,q)).T
    samples = KNTable((data), names = ('m1','m2','chi_eff','mchirp','eta','q') )

    #data = np.vstack((m1s,m2s,dists,lambda1s,lambda2s,chi_effs,Xlans,c1,c2,mb1,mb2,mchirps,etas,qs,mej,vej, dyn_mej, wind_mej, mbnss)).T
    #samples = KNTable((data), names = ('m1','m2','dist','lambda1','lambda2','chi_eff','Xlan','c1','c2','mb1','mb2','mchirp','eta','q','mej','vej', 'dyn_mej', 'wind_mej', 'mbns'))    



    #if twixie:
        #samples_tmp = KNTable.read_mchirp_samples(opts.mchirp_samples, Nsamples=100, twixie_flag = twixie_tf)
   
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
    
 

     
    print('...running')
    #if (opts.analysisType == "posterior") or (opts.analysisType == "mchirp"):
    
    #need to fix indents
    if True:
        if True: 
            nsamples = num_samples
            if opts.nsamples < 1:
                print('Please set nsamples >= 1')
                exit(0)
            # read samples from template analysis
            #samples = KNTable.read_mchirp_samples(opts.mchirp_samples, Nsamples=opts.nsamples, twixie_flag = twixie_tf) 
           
     
            m1s, m2s, dists_mbta = [], [], []
            lambda1s, lambda2s, chi_effs = [], [], []
            Xlans = []
            mbnss = []
            if EOS == "gp":
                # read Phil + Reed's EOS files
                eospostdat = np.genfromtxt("/home/philippe.landry/nseos/eos_post_PSRs+GW170817+J0030.csv",names=True,dtype=None,delimiter=",")
                idxs = np.array(eospostdat["eos"])
                weights = np.array([np.exp(weight) for weight in eospostdat["logweight_total"]])
            elif EOS == "Sly":
                eosname = "SLy"
                eos = EOS4ParameterPiecewisePolytrope(eosname)
    
            Xlan_min, Xlan_max = -9, -1 
         
            for ii, row in enumerate(samples): 
                #m1, m2, dist_mbta, chi_eff = row["m1"], row["m2"], row["dist_mbta"], row["chi_eff"]
                m1, m2, chi_eff = row["m1"], row["m2"], row["chi_eff"]
                if EOS == "spec":
                    indices = np.random.randint(0, 2396, size=nsamples)
                elif EOS == "gp":
                    indices = np.random.choice(np.arange(0,len(idxs)), size=nsamples,replace=True,p=weights/np.sum(weights))
                for jj in range(nsamples):
                    if (EOS == "spec") or (EOS == "gp"):
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
                    #Xlans.append(Xlan)
                    mbnss.append(mbns)
    
      
            #Xlans = [10**opts.Xlan_fixed] * len(samples) * nsamples
            #phis = [opts.phi_fixed] * len(samples) * nsamples 
            thetas = 180. * np.arccos(np.random.uniform(-1., 1., len(samples) * nsamples)) / np.pi
            idx_thetas = np.where(thetas > 90.)[0]
            thetas[idx_thetas] = 180. - thetas[idx_thetas]
            thetas = list(thetas)
            Xlans = np.ones(np.array(m1s).shape) * Xlan_val
             
            # make final arrays of masses, distances, lambdas, spins, and lanthanide fractions
            #print(str([len(m1s), len(m2s), len(lambda1s), len(lambda2s), len(Xlans), len(chi_effs), len(thetas), len(mbnss)]))
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
    
        #print(samples)
       
        #'''    
        #----------------------------------------------------------------------------------
        if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
            #mbns = 2.1
            #idx1 = np.where((samples['m1'] < mbns) & (samples['m2'] < mbns))[0]
            #idx2 = np.where((samples['m1'] > mbns) | (samples['m2'] > mbns))[0]
            
            idx1 = np.where((samples['m1'] <= samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
            idx2 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
            idx3 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] > samples['mbns']))[0]
    
         
    
           
    
            mej, vej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)
            wind_mej, dyn_mej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)   
 
            from gwemlightcurves.EjectaFits.CoDi2019 import calc_meje, calc_vej
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
            
           
            if (opts.model == "Bu2019inc"):  
                    idx = np.where(samples['mej'] <= 1e-6)[0]
                    samples['mej'][idx] = 1e-11
            elif (opts.model == "Ka2017"):
                    idx = np.where(samples['mej'] <= 1e-3)[0]
                    samples['mej'][idx] = 1e-11
               
            
            print("Probability of having ejecta")
            print(100 * (len(samples) - len(idx)) /len(samples))
            #print(samples) 
            return samples


if __name__ == "__main__":
    chirp_q_tf = True
    chi_list = [-.5, -.25, 0, .25, .5, .75]
    chi_list = [0]
    if twixie_tf:
        chi_list = [0]
    for chi in chi_list: 
        if opts.analysisType == 'NSBH':
            m1 = np.arange(3, 5.8, .1) 
            m2 = np.arange(1, 1.5, .05)
            m1 = np.arange(3, 8, .1)
            m2 = np.arange(1, 2.4, .05)
        #sample 3-10 for NSBH
        if opts.analysisType == 'BNS':
            m1 = np.arange(1, 2.4, .1) 
            m2 = np.arange(1, 2.4, .1)
            m1 = np.arange(1, 3.1, .1)
            m2 = np.arange(1, 3.1, .1)
        if chirp_q_tf:
            chirp_min, xx, yy = lightcurve_utils.ms2mc(1, 1)
            chirp_max, xx, yy = lightcurve_utils.ms2mc(2.5, 2.5)
            #print(chirp_min, chirp_max)
            #m2 becomes q
            #m1 becomes mchirp
            m1 = np.arange(chirp_min, chirp_max, .1) 
            m2 = np.arange(1, 2, .1)
        medians, stds = [], []
        m1_plot, m2_plot=[], []
        lambdatildes=[]
        term1_plot, term2_plot, term3_plot, term4_plot=[],[],[],[]
        for m1m in m1:
            for m2m in m2:
                if m1m >= m2m or chirp_q_tf:
                    print('Initializing '+str(m1m)+' '+str(m2m))
                    runType = 'gp'
                    samples_gp = Test(runType, m1m, m2m, chi, chirp_q=chirp_q_tf)
                    #samples_Sly = Test('Sly', m1m, m2m)
                    #samples_spec = Test('spec', m1m, m2m)
                    samples=samples_gp
                    bounds = [-3.0,-1.0]
                    xlims = [-2.8,-1.0]
                    ylims = [1e-1,2]
                    plotName = "/home/andrew.toivonen/gwemlightcurves/mass_plots/mej_"+str(opts.analysisType)+"_m1_"+str(np.round(m1m, decimals=1))+"_m2_"+str(np.round(m2m, decimals=1))+'_chi_'+str(chi)+".pdf"
                    if twixie_tf:
                        plotName = "/home/andrew.toivonen/gwemlightcurves/mass_plots/mej_"+str(opts.analysisType)+"_m1_"+str(np.round(m1m, decimals=1))+"_m2_"+str(np.round(m2m, decimals=1))+"twixie.pdf"
                    plt.figure(figsize=(15,10))
                    ax = plt.gca()
                    for ii,model in enumerate(models):
                        legend_name = get_legend(model) + ' EOS: '+runType
                        bins, hist1 = lightcurve_utils.hist_results(np.log10(samples_gp["mej"]),Nbins=20,bounds=bounds)
                        plt.step(bins,hist1,'-',color='b',linewidth=3,label=legend_name,where='mid')
                        lim = np.percentile(np.log10(samples_gp["mej"]), 90)
                        plt.plot([lim,lim],ylims,'k--')
                    plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
                    plt.ylabel('Probability Density Function',fontsize=24)
                    #plt.legend(loc="best",prop={'size':24})
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.xlim(xlims)
                    plt.ylim(ylims)
                    #ax.set_yscale('log')
                    plt.savefig(plotName)
                    plt.close()
                  
                    medians.append(np.median(samples_gp['mej']))
                    stds.append(np.std(samples_gp['mej']))
                    m1_plot.append(m1m)
                    m2_plot.append(m2m)
                   
                    lambdatilde = (16.0/13.0)*(samples_gp['lambda2'] + samples_gp['lambda1']*(samples_gp['q']**5) + 12*samples_gp['lambda1']*(samples_gp['q']**4) + 12*samples['lambda2']*samples['q'])/((samples['q']+1)**5)
                    lambdatildes.append(np.mean(np.array(lambdatilde)))
                   
        std_med=[]
        m1_0=[]
        m2_0=[]
          
        for num in range(len(stds)):
            check = 0
            if stds[num] < 1e-6:
                stds[num] = np.nan 
                check = 1
            if medians[num] <= 1e-6:
                medians[num] = np.nan
                check = 1
            std_med.append(stds[num]/medians[num])
            if not twixie_tf:    
                if std_med[num] > 10:
                    std_med[num] = 10
                if std_med[num] < .01:
                    std_med[num] = .01
            if opts.analysisType == 'NSBH':
                if check == 1:
                    m1_0.append(m1_plot[num])
                    m2_0.append(m2_plot[num])               
        plotName = "/home/andrew.toivonen/gwemlightcurves/mass_plots/new/mej_mass_grid_"+str(opts.analysisType)+'_chi_'+str(chi)+".pdf"
        if twixie_tf:
            plotName = "/home/andrew.toivonen/gwemlightcurves/mass_plots/new/mej_mass_grid_"+str(opts.analysisType)+"_twixie.pdf"
        fig2 = plt.figure(figsize=(15,10))
        ax2 = plt.gca()
        if opts.analysisType == 'BNS':
            weight = 800
            if twixie_tf:
                weight = 35
        if opts.analysisType == 'NSBH':
            weight = 50
        plot=plt.scatter(m1_plot, m2_plot, c=np.log10(np.array(medians)), s=np.array(std_med)*weight, cmap='coolwarm')
        plt.scatter(m1_0, m2_0, c='black')
        cbar = fig2.colorbar(mappable=plot)
        cbar.ax.set_ylabel('Log10 Median Ejecta Mass')
        plt.xlabel('m1', fontsize=24)
        plt.ylabel('m2', fontsize=24)
        if chirp_q_tf:
            plt.xlabel('mchirp', fontsize=24)
            plt.ylabel('q', fontsize=24)
        #if opts.analysisType == 'BNS':
            #if not chirp_q_tf:
                #plt.xlim([.9,2.4])
                #plt.ylim([.9,2.4])
        #if opts.analysisType == 'NSBH':
            #plt.xlim([2.9,5.7])
            #plt.ylim([.95,2])
            #plt.xlim([2.9,8])
            #plt.ylim([.95, 2])
        plt.savefig(plotName)
    
    
    
    
        plotName = "/home/andrew.toivonen/gwemlightcurves/mass_plots/new/mej_mass_grid_lambdatilde_"+str(opts.analysisType)+'_chi_'+str(chi)+".pdf"
        if twixie_tf:
            plotName = "/home/andrew.toivonen/gwemlightcurves/mass_plots/new/mej_mass_grid_lambdatilde_"+str(opts.analysisType)+"_twixie.pdf"
        fig2 = plt.figure(figsize=(15,10))
        ax2 = plt.gca()
        if opts.analysisType == 'BNS':
            weight = 7
        if opts.analysisType == 'NSBH':
            weight = 2
        plot=plt.scatter(m1_plot, m2_plot, c=np.log10(np.array(medians)), s=np.array(lambdatildes)*weight, cmap='coolwarm')
        plt.scatter(m1_0, m2_0, c='black')
        cbar = fig2.colorbar(mappable=plot)
        cbar.ax.set_ylabel('Log10 Median Ejecta Mass')
        plt.xlabel('m1', fontsize=24)
        plt.ylabel('m2', fontsize=24)
        if chirp_q_tf:
            plt.xlabel('mchirp', fontsize=24)
            plt.ylabel('q', fontsize=24) 
            
        #if opts.analysisType == 'BNS':
            #if not chirp_q_tf:
                #plt.xlim([.9,2.4])
                #plt.ylim([.9,2.4])
        #if opts.analysisType == 'NSBH':
            #plt.xlim([2.9,5.7])
            #plt.ylim([.95,1.4])
            #plt.xlim([2.9, 5.7])
            #plt.ylim([.95,2])
            #plt.xlim([2.9,8])
            #plt.ylim([.95, 2]) 
        plt.savefig(plotName)


