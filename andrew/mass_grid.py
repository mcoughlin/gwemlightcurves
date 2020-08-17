#!/usr/bin/env python

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
def Test(EOS, m1, m2, chi, type_set=Type, model_set = 'Bu2019inc', twixie = twixie_tf, lan_override=False, lan_override_val=None, chirp_q = False):

    if type_set == 'BNS_chirp_q':
        type_set = 'BNS' 
    if not chirp_q:
        m1 = np.random.normal(m1, .05, 100)
        m2 = np.random.normal(m2, .05, 100)
        q = m2/m1
        mchirp = np.power((m1*m2), 3/5) / np.power((m1+m2), 1/5)
        eta = m1*m2/( (m1+m2)*(m1+m2) )
    if chirp_q:
        print('running chirp_q')
        m1 = np.random.normal(m1, .05, 100)
        m2 = np.random.normal(m2, .05, 100)
        #m1 = np.ones(100) * m1
        #m2 = np.ones(100) * m2
        q = m2 
        mchirp = m1
        eta = lightcurve_utils.q2eta(q) 
        m1, m2 = lightcurve_utils.mc2ms(mchirp, eta)

    dist = np.ones(100)
    #chi_eff = np.random.uniform(-1,1,100)
    chi_eff = np.ones(100)*chi
    Xlan = 1e-3 * np.ones(100)
    
    if lan_override:
        Xlan = lan_override_val * np.ones(100)
 
    c1 = np.ones(1000)
    c2 = np.ones(1000)
    mb1 = np.ones(1000)
    mb2 = np.ones(1000)
    mej = np.ones(1000)
    vej = np.ones(1000)
    

    data = np.vstack((m1,m2,dist,chi_eff,Xlan,mchirp,eta,q)).T
    samples_tmp = KNTable((data), names = ('m1','m2','dist','chi_eff','Xlan','mchirp','eta','q') )

    if twixie:
        samples_tmp = KNTable.read_mchirp_samples(opts.mchirp_samples, Nsamples=100, twixie_flag = twixie_tf)

    lambda1s=[]
    lambda2s=[]
    m1s=[]
    m2s=[]
    dists=[]
    chi_effs=[]
    Xlans=[]
    qs=[]
    etas=[]
    mchirps=[]
    mbnss=[]
    term1_list, term2_list, term3_list, term4_list=[],[],[],[]


    if EOS == "gp":
             # read Phil + Reed's EOS files
            filenames = glob.glob("/home/philippe.landry/gw170817eos/gp/macro/MACROdraw-*-0.csv")
            idxs = []
            for filename in filenames:
                filenameSplit = filename.replace(".csv","").split("/")[-1].split("-")
                idxs.append(int(filenameSplit[1]))
            idxs = np.array(idxs)
    elif EOS == "Sly":
            eosname = "SLy"
            eos = EOS4ParameterPiecewisePolytrope(eosname)


    for ii, row in enumerate(samples_tmp):
            if not twixie:
                m1, m2, dist, chi_eff, q, mchirp, eta, Xlan = row["m1"], row["m2"], row["dist"], row["chi_eff"], row['q'], row['mchirp'], row['eta'], row['Xlan']
            if twixie:
                m1, m2, dist, chi_eff, q, mchirp, eta = row["m1"], row["m2"], row["dist_mbta"], row["chi_eff"], row['q'], row['mchirp'], row['eta']
            nsamples = 10
            if EOS == "spec":
                indices = np.random.randint(0, 2395, size=nsamples)
            elif EOS == "gp":
                indices = np.random.randint(0, len(idxs), size=nsamples)
            for jj in range(nsamples):
                if (EOS == "spec") or (EOS == "gp"):
                    index = indices[jj] 
                
                # samples lambda's from Phil + Reed's files
                if EOS == "spec":
                    eospath = "/home/philippe.landry/gw170817eos/spec/macro/macro-spec_%dcr.csv" % index
                    data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                    marray, larray = data_out["M"], data_out["Lambda"]
                    f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                    lambda1, lambda2 = f(m1), f(m2)
                    mbns = np.max(marray)

                elif EOS == "Sly":
                    lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)
                    mbns = eos.maxmass()
                    #print(mbns)
            
                elif EOS == "gp":
                    lambda1, lambda2 = 0.0, 0.0
                    phasetr = 0
                    while (lambda1==0.0) or (lambda2 == 0.0):
                        eospath = "/home/philippe.landry/gw170817eos/gp/macro/MACROdraw-%06d-%d.csv" % (idxs[index], phasetr)
                        if not os.path.isfile(eospath):
                            break
                        data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                        marray, larray = data_out["M"], data_out["Lambda"]
                        f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                        lambda1_tmp, lambda2_tmp = f(m1), f(m2)
                        if (lambda1_tmp>0) and (lambda1==0.0):
                            lambda1 = lambda1_tmp
                        if (lambda2_tmp>0) and (lambda2 == 0.0):
                            lambda2 = lambda2_tmp
                        phasetr = phasetr + 1
                        mbns = np.max(marray)
    
                lambda1s.append(lambda1)
                lambda2s.append(lambda2)
                m1s.append(m1)
                m2s.append(m2)
                dists.append(dist)
                chi_effs.append(chi_eff)
                Xlans.append(Xlan)
                qs.append(q)
                etas.append(eta)
                mchirps.append(mchirp)
                mbnss.append(mbns)
                  
                
    if twixie:
        Xlans = np.ones(1000)*1e-3           

    data = np.vstack((m1s,m2s,dists,lambda1s,lambda2s,chi_effs,Xlans,c1,c2,mb1,mb2,mchirps,etas,qs,mej,vej, mbnss)).T
    samples = KNTable((data), names = ('m1','m2','dist','lambda1','lambda2','chi_eff','Xlan','c1','c2','mb1','mb2','mchirp','eta','q','mej','vej', 'mbns'))    

    #calc compactness
    samples = samples.calc_compactness(fit=True)

    #clac baryonic mass
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)

    if type_set == 'BNS':

        from gwemlightcurves.EjectaFits.CoDi2019 import calc_meje, calc_vej
        #from gwemlightcurves.EjectaFits.PaDi2019 import calc_meje, calc_vej
        # calc the mass of ejecta
        mej = calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'])
        # calc the velocity of ejecta
        vej = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])

        samples['mchirp'], samples['eta'], samples['q'] = lightcurve_utils.ms2mc(samples['m1'], samples['m2'])

        samples['q'] = 1.0 / samples['q']

        samples['mej'] = mej
        samples['vej'] = vej

        if model_set == 'Bu2019inc':
            idx = np.where(samples['mej']<=1e-6)[0]
            samples['mej'][idx] = 1e-6
            idx2 = np.where(samples['mej']>=1)[0]
            samples['mej'][idx2] = 1e-6
        print('mej = ' +str(samples['mej'][0]))

        


    if type_set == 'NSBH':

        from gwemlightcurves.EjectaFits.KrFo2019 import calc_meje, calc_vave
        # calc the mass of ejecta
        mej = calc_meje(samples['q'],samples['chi_eff'],samples['c2'], samples['m2'])
        # calc the velocity of ejecta
        vej = calc_vave(samples['q'])

        samples['mej'] = mej
        samples['vej'] = vej
        
        if model_set == 'Bu2019inc':
            idx = np.where(samples['mej']<=1e-6)[0]
            samples['mej'][idx] = 1e-6
            idx2 = np.where(samples['mej']>=1)[0]
            samples['mej'][idx2] = 1e-6
        
        print('mej = ' +str(samples['mej'][0]))

    if type_set == 'BNS' or 'NSBH':
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
        idx = np.where(samples['mej'] > 0)[0]
        samples = samples[idx]
    print(EOS + ' calculation finished') 
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
            print(chirp_min, chirp_max)
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


