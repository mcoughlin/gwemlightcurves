
import os, sys, glob
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

import corner

from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

from gwemlightcurves.EjectaFits import KaKy2016
from gwemlightcurves.EjectaFits import DiUj2017

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-n","--name",default="1087")
    parser.add_option("--outputName",default="GWEM")
    parser.add_option("-m","--model",default="KaKy2016")
    parser.add_option("--mej",default=0.005,type=float)
    parser.add_option("--vej",default=0.25,type=float)
    parser.add_option("-e","--errorbudget",default="0.2,1.0")
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doSimulation",  action="store_true", default=False)
    parser.add_option("--doFixMChirp",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-f","--filters",default="g,r,i,z")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)

    opts, args = parser.parse_args()

    return opts

def hist_results(samples,Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-3
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

def bhns_model(q,chi_eff,mns,mb,c):

    meje = KaKy2016.calc_meje(q,chi_eff,c,mb,mns)
    vave = KaKy2016.calc_vave(q)

    return meje, vave

def bns_model(m1,mb1,c1,m2,mb2,c2):

    mej = DiUj2017.calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = DiUj2017.calc_vej(m1,c1,m2,c2)

    return mej, vej

# Parse command line
opts = parse_commandline()

filters = opts.filters.split(",")
names = opts.name.split(",")
errorbudgets = opts.errorbudget.split(",")

if not opts.model in ["KaKy2016", "DiUj2017", "SN"]:
   print "Model must be either: KaKy2016, DiUj2017, SN"
   exit(0)

if not (opts.doEjecta or opts.doMasses):
    print "Enable --doEjecta or --doMasses"
    exit(0)

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'fitting_models'
elif opts.doGoingTheDistance:
    basename = 'fitting_going-the-distance'
elif opts.doMassGap:
    basename = 'fitting_massgap'
elif opts.doSimulation:
    basename = 'fitting'
else:
    print "Need to enable --doModels, --doSimulation, --doMassGap, or --doGoingTheDistance"
    exit(0)
plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
else:
    plotDir = os.path.join(plotDir,'%s'%opts.model)
if opts.doModels:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    plotDir = os.path.join(plotDir,opts.name)
    dataDir = plotDir.replace("fitting_models","models").replace("_EOSFit","")
elif opts.doSimulation:
    plotDir = os.path.join(plotDir,'M%03dV%02d'%(opts.mej*1000,opts.vej*100))
elif opts.doGoingTheDistance or opts.doMassGap:
    plotDir = os.path.join(plotDir,"_".join(filters))
    plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    plotDir = os.path.join(plotDir,opts.name)
    dataDir = plotDir.replace("fitting_","")
    if opts.doEjecta:
        dataDir = dataDir.replace("_EOSFit","")    

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

names = opts.name.split(",")
errorbudgets = opts.errorbudget.split(",")
baseplotDir = plotDir
basedataDir = dataDir

post = {}
for name in names:
    post[name] = {}
    for errorbudget in errorbudgets:
        if opts.doModels:
            plotDir = os.path.join(baseplotDir,"%.2f"%float(errorbudget))
            dataDir = plotDir.replace("fitting_models","models").replace("_EOSFit","")
        elif opts.doSimulation:
            plotDir = os.path.join(baseplotDir,"%.3f"%(float(errorbudget)*100.0))
        elif opts.doGoingTheDistance or opts.doMassGap:
            dataDir = os.path.join(basedataDir,"%.2f"%float(errorbudget))
            plotDir = os.path.join(baseplotDir,"%.2f"%float(errorbudget))

            multifile = lightcurve_utils.get_post_file(dataDir)
            data = np.loadtxt(multifile)

            filename = os.path.join(dataDir,"truth_mej_vej.dat")
            truths_mej_vej = np.loadtxt(filename)
            truths_mej_vej[0] = np.log10(truths_mej_vej[0])

            filename = os.path.join(dataDir,"truth.dat")
            truths = np.loadtxt(filename)

            if opts.doEjecta:
                mej_em = data[:,1]
                vej_em = data[:,2]

                mej_true = truths_mej_vej[0]
                vej_true = truths_mej_vej[1]

            elif opts.doMasses:
                if opts.model == "DiUj2017":
                    if opts.doEOSFit:
                        mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,3])
                        mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
                    else:
                        mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1],data[:,4])
                        mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0],truths[2])
                elif opts.model == "KaKy2016":
                    if opts.doEOSFit:
                        mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1]*data[:,3],data[:,3])
                        mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0]*truths[4],truths[4])
                    else:
                        mchirp_em,eta_em,q_em = lightcurve_utils.ms2mc(data[:,1]*data[:,3],data[:,3])
                        mchirp_true,eta_true,q_true = lightcurve_utils.ms2mc(truths[0]*truths[4],truths[4])
                q_em = 1/q_em
                q_true = 1/q_true

        multifile = lightcurve_utils.get_post_file(plotDir)

        if not multifile: continue
        data = np.loadtxt(multifile)

        post[name][errorbudget] = {}
        if opts.doGoingTheDistance or opts.doMassGap:
            if opts.model == "DiUj2017":
                if opts.doEOSFit:
                    mchirp_gw,eta_gw,q_gw = lightcurve_utils.ms2mc(data[:,0],data[:,2])
                    mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
                    ii = 0
                    for m1,c1,m2,c2 in data[:,:-1]:
                        mb1 = lightcurve_utils.EOSfit(m1,c1)
                        mb2 = lightcurve_utils.EOSfit(m2,c2)
                        mej_gw[ii], vej_gw[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                        ii = ii + 1
                else:
                    mchirp_gw,eta_gw,q_gw = lightcurve_utils.ms2mc(data[:,0],data[:,3])
                    mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
                    ii = 0
                    for m1,mb1,c1,m2,mb2,c2 in data[:,:-1]:
                        mej_gw[ii], vej_gw[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                        ii = ii + 1
                q_gw = 1/q_gw
                mej_gw = np.log10(mej_gw)
            elif opts.model == "KaKy2016":
                if opts.doEOSFit:
                    mchirp_gw,eta_gw,q_gw = lightcurve_utils.ms2mc(data[:,0]*data[:,2],data[:,2])
                    mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
                    ii = 0
                    for q,chi,mns,c in data[:,:-1]:
                        mb = lightcurve_utils.EOSfit(mns,c)
                        mej_gw[ii], vej_gw[ii] = bhns_model(q,chi,mns,mb,c)
                        ii = ii + 1
                else:
                    mchirp_gw,eta_gw,q_gw = lightcurve_utils.ms2mc(data[:,0]*data[:,2],data[:,3])
                    mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
                    ii = 0
                    for q,chi,mns,mb,c in data[:,:-1]:
                        mej_gw[ii], vej_gw[ii] = bhns_model(q,chi,mns,mb,c)
                        ii = ii + 1
                q_gw = 1/q_gw
                mej_gw = np.log10(mej_gw)

            combinedDir = os.path.join(plotDir,"com")
            multifile = lightcurve_utils.get_post_file(combinedDir)
            data_combined = np.loadtxt(multifile)

            if opts.doEjecta:
                mej_combined = data_combined[:,0]
                vej_combined = data_combined[:,1]

                post[name][errorbudget]["mej_em"] = mej_em
                post[name][errorbudget]["vej_em"] = vej_em
                post[name][errorbudget]["mej_gw"] = mej_gw
                post[name][errorbudget]["vej_gw"] = vej_gw
                post[name][errorbudget]["mej_combined"] = mej_combined
                post[name][errorbudget]["vej_combined"] = vej_combined
                post[name][errorbudget]["mej_true"] = mej_true
                post[name][errorbudget]["vej_true"] = vej_true

            elif opts.doMasses:
                q_combined = data_combined[:,0]
                mchirp_combined = data_combined[:,1]

                post[name][errorbudget]["q_em"] = q_em
                post[name][errorbudget]["mchirp_em"] = mchirp_em
                post[name][errorbudget]["q_gw"] = q_gw
                post[name][errorbudget]["mchirp_gw"] = mchirp_gw
                post[name][errorbudget]["q_combined"] = q_combined
                post[name][errorbudget]["mchirp_combined"] = mchirp_combined
                post[name][errorbudget]["q_true"] = q_true
                post[name][errorbudget]["mchirp_true"] = mchirp_true

            post[name][errorbudget]["truths"] = truths

plotDir = os.path.join(baseplotDir,opts.outputName)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

colors = ['b','g','r','m','c']
linestyles = ['-', '-.', ':','--']

if opts.doEjecta:

    if opts.model == "KaKy2016":
        bounds = [-2.5,0.0]
        xlims = [-2.5,0.0]
        ylims = [1e-1,3]
    elif opts.model == "DiUj2017":
        bounds = [-3.0,-1.0]
        xlims = [-3.0,-1.3]
        ylims = [1e-1,10]

    plotName = "%s/mej.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):
 
            color = colors[ii+0]
            colortrue = 'k'
            linestyle = linestyles[jj]   
  
            if jj == 0:
                label = "GW"
                samples = post[name][errorbudget]["mej_gw"]
                bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds) 

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["mej_true"],post[name][errorbudget]["mej_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]

            label = "EM"
            samples = post[name][errorbudget]["mej_em"]
            bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

            if jj == 0:            
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["mej_combined"]
            bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
        maxhist = np.max([maxhist,np.max(hist1)])

    plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

    if opts.model == "KaKy2016":
        bounds = [0.0,1.0]
        xlims = [0.0,1.0]
        ylims = [1e-1,20]
    elif opts.model == "DiUj2017":
        bounds = [0.0,1.0]
        xlims = [0.0,1.0]
        ylims = [1e-1,10]

    plotName = "%s/vej.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):

            color = colors[ii+0]
            colortrue = 'k'
            linestyle = linestyles[jj]

            if jj == 0:
                label = "GW"
                samples = post[name][errorbudget]["vej_gw"]
                bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["vej_true"],post[name][errorbudget]["vej_true"]],[1e-3,20.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]

            label = "EM"
            samples = post[name][errorbudget]["vej_em"]
            bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["vej_combined"]
            bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
        maxhist = np.max([maxhist,np.max(hist1)])

    plt.xlabel(r"$v_{\rm ej}$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

elif opts.doMasses:

    if opts.model == "KaKy2016":
        bounds = [0.8,6.5]
        xlims = [0.8,6.5]
        ylims = [1e-1,5]
    elif opts.model == "DiUj2017":
        bounds = [0.8,2.0]
        xlims = [0.8,2.0]
        ylims = [1e-1,10]

    plotName = "%s/mchirp.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):

            color = colors[ii+0]
            colortrue = 'k'
            linestyle = linestyles[jj]

            if jj == 0:
                label = "GW"
                samples = post[name][errorbudget]["mchirp_gw"]
                bins, hist1 = hist_results(samples,Nbins=25,bounds=bounds)

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["mchirp_true"],post[name][errorbudget]["mchirp_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]
            label = "EM"
            samples = post[name][errorbudget]["mchirp_em"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["mchirp_combined"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
        maxhist = np.max([maxhist,np.max(hist1)])

    plt.xlabel(r"${\rm M}_{\rm c}$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

    if opts.model == "KaKy2016":
        bounds = [2.9,9.1]
        xlims = [2.9,9.1]
        ylims = [1e-1,1]
    elif opts.model == "DiUj2017":
        bounds = [0.0,2.0]
        xlims = [0.9,2.0]
        ylims = [1e-1,10]

    plotName = "%s/q.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):

            color = colors[ii+0]
            colortrue = 'k'
            linestyle = linestyles[jj]

            if jj == 0:
                label = "GW"
                samples = post[name][errorbudget]["q_gw"]
                bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["q_true"],post[name][errorbudget]["q_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]
            label = "EM"
            samples = post[name][errorbudget]["q_em"]
            bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["q_combined"]
            bins, hist1 = hist_results(samples,Nbins=15,bounds=bounds)

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
        maxhist = np.max([maxhist,np.max(hist1)])

    plt.xlabel(r"$q$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()
