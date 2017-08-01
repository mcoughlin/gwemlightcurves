
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

from gwemlightcurves import BHNSKilonovaLightcurve, BNSKilonovaLightcurve, SALT2
from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../lightcurves")
    #parser.add_option("-n","--name",default="BHNS_H4M005V20")
    #parser.add_option("-f","--outputName",default="BHNS_error")
    #parser.add_option("-n","--name",default="BNS_H4M005V20")
    #parser.add_option("-f","--outputName",default="BNS_error")
    #parser.add_option("-n","--name",default="APR4-1215_k1,APR4-1314_k1,H4-1215_k1,H4-1314_k1,Sly-135_k1")
    #parser.add_option("-n","--name",default="APR4-1314_k1,H4-1314_k1,Sly-135_k1")
    #parser.add_option("-f","--outputName",default="BNS_Tanaka")
    #parser.add_option("-n","--name",default="APR4Q3a75_k1,H4Q3a75_k1,MS1Q3a75_k1,MS1Q7a75_k1")
    parser.add_option("-n","--name",default="APR4Q3a75_k1,H4Q3a75_k1,MS1Q3a75_k1")
    parser.add_option("-f","--outputName",default="BHNS_Tanaka")

    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False) 
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    #parser.add_option("-e","--errorbudget",default="1.0,0.2,0.04")
    parser.add_option("-e","--errorbudget",default="1.0,0.2")
    #parser.add_option("-l","--labelType",default="errorbar")
    parser.add_option("-l","--labelType",default="name")

    opts, args = parser.parse_args()

    return opts

def plot_results(samples,label,plotName):

    plt.figure(figsize=(12,10))
    bins1, hist1 = hist_results(samples)
    plt.plot(bins1, hist1)
    plt.xlabel(label)
    plt.ylabel('Probability Density Function')
    plt.show()
    plt.savefig(plotName,dpi=200)
    plt.close('all')

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

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def get_truths(name):
    if name == "BNS_H4M005V20":
        truths = [0,np.log10(0.005),0.2,0.2,3.14,0.0]
    elif name == "BHNS_H4M005V20":
        truths = [0,np.log10(0.005),0.2,0.2,3.14,0.0]
    elif name == "rpft_m005_v2":
        truths = [0,np.log10(0.005),0.2,0.147,1.47,0.2,3.14,0.0]
    elif name == "APR4-1215_k1":
        truths = [0,np.log10(0.009),0.24,False,False,0.0]
    elif name == "APR4-1314_k1":
        truths = [0,np.log10(0.008),0.22,False,False,0.0]
    elif name == "H4-1215_k1":
        truths = [0,np.log10(0.004),0.21,False,False,0.0]
    elif name == "H4-1314_k1":
        truths = [0,np.log10(0.0007),0.17,False,False,0.0]
    elif name == "Sly-135_k1":
        truths = [0,np.log10(0.02),False,False,False,0.0]
    elif name == "APR4Q3a75_k1":
        truths = [0,np.log10(0.01),0.24,False,False,0.0]
    elif name == "H4Q3a75_k1":
        truths = [0,np.log10(0.05),0.21,False,False,0.0]
    elif name == "MS1Q3a75_k1":
        truths = [0,np.log10(0.07),0.25,False,False,0.0]
    elif name == "MS1Q7a75_k1":
        truths = [0,np.log10(0.06),0.25,False,False,0.0]
    elif name == "SED_nsbh1":
        truths = [0,np.log10(0.04),0.2,False,False,0.0]
    else:
        truths = []
        for ii in xrange(n_params):
            truths.append(False)
    return truths

# Parse command line
opts = parse_commandline()

if not opts.model in ["BHNS", "BNS", "SN"]:
   print "Model must be either: BHNS, BNS, SN"
   exit(0)

if not (opts.doEjecta or opts.doMasses):
    print "Enable --doEjecta or --doMasses"
    exit(0)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'models')
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
if opts.doFixZPT0:
    plotDir = os.path.join(baseplotDir,'models/%s_FixZPT0'%opts.model)
else:
    plotDir = os.path.join(baseplotDir,'models/%s'%opts.model)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
if opts.model in ["BNS","BHNS"]:
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)

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

        multifile = get_post_file(plotDir)
        if not multifile: continue
        data = np.loadtxt(multifile)

        post[name][errorbudget] = {}
        if opts.model == "BHNS":
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

        elif opts.model == "BNS":
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

        truths = get_truths(name)
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
            label = r"%s"%(name)
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
        else:
            color = 'b'
            colortrue = 'k'
            linestyle = '-'

        samples = np.log10(post[name][errorbudget]["mej"])
        bins, hist1 = hist_results(samples,Nbins=31,bounds=[-3.5,0.0]) 

        if opts.labelType == "name" and jj > 0:
            plt.semilogy(bins,hist1,'%s%s'%(color,linestyle))
        else:
            plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label)


        plt.semilogy([post[name][errorbudget]["truths"][1],post[name][errorbudget]["truths"][1]],[1e-3,10.0],'%s--'%colortrue)
        maxhist = np.max([maxhist,np.max(hist1)])

plt.xlabel(r"$log_{10} (M_{ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':16})
plt.xlim([-3.5,-1.0])
plt.ylim([1e-1,4])
plt.savefig(plotName)
plt.close()

plotName = "%s/vej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
maxhist = -1
for name in post.keys():
    for errorbudget in post[name].keys():
        if opts.labelType == "errorbar":
            label = r"$\Delta$m: %.2f"%float(errorbudget)
        elif opts.labelType == "name":
            label = r"%s"%(name)
        else:
            label = []
        samples = post[name][errorbudget]["vej"]
        bins, hist1 = hist_results(samples)
        plt.plot(bins,hist1,'-',label=label)
        plt.plot([post[name][errorbudget]["truths"][2],post[name][errorbudget]["truths"][2]],[0,np.max(hist1)],'k--')
        maxhist = np.max([maxhist,np.max(hist1)])
plt.xlabel(r"$v_{ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':16})
plt.ylim([0,maxhist])
plt.savefig(plotName)
plt.close()

