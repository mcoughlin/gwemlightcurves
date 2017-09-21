
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

from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../lightcurves")
    #parser.add_option("-n","--name",default="KaKy2016_H4M005V20")
    #parser.add_option("--outputName",default="KaKy2016_error")
    parser.add_option("-n","--name",default="rpft_m05_v2,SED_ns12ns12_kappa10")
    parser.add_option("--outputName",default="Barnes_Rosswog")
    #parser.add_option("-n","--name",default="DiUj2017_H4M005V20")
    #parser.add_option("--outputName",default="DiUj2017_error")
    #parser.add_option("-n","--name",default="APR4-1314_k1,H4-1314_k1,Sly-135_k1")
    #parser.add_option("--outputName",default="DiUj2017_Tanaka")
    #parser.add_option("-n","--name",default="APR4Q3a75_k1,H4Q3a75_k1,MS1Q3a75_k1")
    #parser.add_option("--outputName",default="KaKy2016_Tanaka")

    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False) 
    parser.add_option("-m","--model",default="DiUj2017")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    #parser.add_option("-e","--errorbudget",default="1.0,0.2,0.04")
    parser.add_option("-e","--errorbudget",default="1.0,0.2")
    #parser.add_option("-e","--errorbudget",default="1.0")    
    parser.add_option("-f","--filters",default="g,r,i,z")
    parser.add_option("--tmax",default=7.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)

    #parser.add_option("-l","--labelType",default="errorbar")
    #parser.add_option("-l","--labelType",default="name")
    parser.add_option("-l","--labelType",default="model")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

filters = opts.filters.split(",")

if not opts.model in ["KaKy2016", "DiUj2017", "SN"]:
   print "Model must be either: KaKy2016, DiUj2017, SN"
   exit(0)

if not (opts.doEjecta or opts.doMasses):
    print "Enable --doEjecta or --doMasses"
    exit(0)

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,'models')
if opts.doFixZPT0:
    plotDir = os.path.join(baseplotDir,'models/%s_FixZPT0'%opts.model)
else:
    plotDir = os.path.join(baseplotDir,'models/%s'%opts.model)
plotDir = os.path.join(plotDir,"_".join(filters))
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
if opts.model in ["DiUj2017","KaKy2016"]:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

names = opts.name.split(",")
errorbudgets = opts.errorbudget.split(",")
baseplotDir = plotDir

post = {}
for name in names:
    post[name] = {}
    for errorbudget in errorbudgets:
        if opts.doReduced:
            plotDir = os.path.join(baseplotDir,"%s_reduced"%name)
        else:
            plotDir = os.path.join(baseplotDir,name)
        plotDir = os.path.join(plotDir,"%.2f"%float(errorbudget))

        multifile = lightcurve_utils.get_post_file(plotDir)
 
        if not multifile: continue
        data = np.loadtxt(multifile)

        post[name][errorbudget] = {}
        if opts.model == "KaKy2016":
            if opts.doMasses:
                t0 = data[:,0]
                q = data[:,1]
                chi_eff = data[:,2]
                mns = data[:,3]
                mb = data[:,4]
                c = data[:,5]
                th = data[:,6]
                ph = data[:,7]
                zp = data[:,8]
                loglikelihood = data[:,9]
            elif opts.doEjecta:
                t0 = data[:,0]
                mej = 10**data[:,1]
                vej = data[:,2]
                th = data[:,3]
                ph = data[:,4]
                zp = data[:,5]
                loglikelihood = data[:,6]

                post[name][errorbudget]["mej"] = mej
                post[name][errorbudget]["vej"] = vej

        elif opts.model == "DiUj2017":
            if opts.doMasses:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                th = data[:,7]
                ph = data[:,8]
                zp = data[:,9]
            elif opts.doEjecta:
                t0 = data[:,0]
                mej = 10**data[:,1]
                vej = data[:,2]
                th = data[:,3]
                ph = data[:,4]
                zp = data[:,5]
                loglikelihood = data[:,6]

                post[name][errorbudget]["mej"] = mej
                post[name][errorbudget]["vej"] = vej

        elif opts.model == "SN":
            t0 = data[:,0]
            z = data[:,1]
            x0 = data[:,2]
            x1 = data[:,3]
            c = data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]

        nsamples, n_params = data.shape
        truths = lightcurve_utils.get_truths(name,opts.model,n_params,opts.doEjecta)
        post[name][errorbudget]["truths"] = truths

plotDir = os.path.join(baseplotDir,opts.outputName)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

colors = ['b','g','r','m','c']
linestyles = ['-', '-.', ':','--']

plotName = "%s/mej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
maxhist = -1
for ii,name in enumerate(sorted(post.keys())):
    for jj,errorbudget in enumerate(sorted(post[name].keys())):
        if opts.labelType == "errorbar":
            label = r"$\Delta$m: %.2f"%float(errorbudget)
        elif opts.labelType == "name":
            label = r"%s"%(name.replace("_k1",""))
        elif opts.labelType == "model":
            label = lightcurve_utils.getLegend(opts.outputDir,[name])[0]
        else:
            label = []
        if opts.labelType == "errorbar":
            color = colors[jj]
            colortrue = 'k'
            linestyle = '-'
        elif opts.labelType == "name":
            color = colors[ii]
            colortrue = colors[ii]
            linestyle = linestyles[jj]
        elif opts.labelType == "model":
            color = colors[ii]
            colortrue = colors[ii]
            linestyle = linestyles[jj]
        else:
            color = 'b'
            colortrue = 'k'
            linestyle = '-'

        samples = np.log10(post[name][errorbudget]["mej"])
        if (opts.labelType == "errorbar") and (float(errorbudget) < 1.0):
            bounds=[-2.8,-1.8]
        else:
            bounds=[-3.5,0.0]
        bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds) 
        if (opts.labelType == "name" or opts.labelType == "model") and jj > 0:
            plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
        else:
            plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)

        plt.semilogy([post[name][errorbudget]["truths"][1],post[name][errorbudget]["truths"][1]],[1e-3,100.0],'%s--'%colortrue,linewidth=3)
        maxhist = np.max([maxhist,np.max(hist1)])

plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
if opts.model == "KaKy2016" and opts.labelType == "errorbar":
    plt.xlim([-3.0,-1.0])
elif opts.model == "KaKy2016" and opts.labelType == "name":
    plt.xlim([-3.0,0.5])
elif opts.model == "DiUj2017" and opts.labelType == "name":
    plt.xlim([-3.5,0.0])
elif opts.model == "DiUj2017" and opts.labelType == "errorbar":
    plt.xlim([-3.0,-1.4])
elif opts.model == "DiUj2017" and opts.labelType == "model":
    plt.xlim([-2.5,0.0])

if opts.model == "KaKy2016" and opts.labelType == "errorbar":
    plt.ylim([1e-1,20])
elif opts.model == "KaKy2016" and opts.labelType == "name":
    plt.ylim([1e-1,10])
elif opts.model == "DiUj2017" and opts.labelType == "name":
    plt.ylim([1e-1,10])
elif opts.model == "DiUj2017" and opts.labelType == "errorbar":
    plt.ylim([1e-1,20])
elif opts.model == "DiUj2017" and opts.labelType == "model":
    plt.ylim([1e-1,10])

plt.savefig(plotName)
plt.close()
