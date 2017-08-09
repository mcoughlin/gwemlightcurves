
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
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-n","--name",default="1087")
    parser.add_option("--outputName",default="GWEM")
    parser.add_option("-m","--model",default="BHNS")
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
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-f","--filters",default="g,r,i,z")

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

    meje = BHNSKilonovaLightcurve.calc_meje(q,chi_eff,c,mb,mns)
    vave = BHNSKilonovaLightcurve.calc_vave(q)

    return meje, vave

def bns_model(m1,mb1,c1,m2,mb2,c2):

    mej = BNSKilonovaLightcurve.calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = BNSKilonovaLightcurve.calc_vej(m1,c1,m2,c2)

    return mej, vej

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def q2eta(q):
    return q/(1+q)**2

def mc2ms(mc,eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    root = np.sqrt(0.25-eta)
    fraction = (0.5+root) / (0.5-root)
    invfraction = 1/fraction

    m2= mc * np.power((1+fraction),0.2) / np.power(fraction,0.6)

    m1= mc* np.power(1+invfraction,0.2) / np.power(invfraction,0.6)
    return (m1,m2)

def ms2mc(m1,m2):
    eta = m1*m2/( (m1+m2)*(m1+m2) )
    mchirp = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))
    q = m2/m1

    return (mchirp,eta,q)

def EOSfit(mns,c):
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb

# Parse command line
opts = parse_commandline()

filters = opts.filters.split(",")
names = opts.name.split(",")
errorbudgets = opts.errorbudget.split(",")

if not opts.model in ["BHNS", "BNS", "SN"]:
   print "Model must be either: BHNS, BNS, SN"
   exit(0)

if not (opts.doEjecta or opts.doMasses):
    print "Enable --doEjecta or --doMasses"
    exit(0)

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'fitting_models'
elif opts.doGoingTheDistance:
    basename = 'fitting_going-the-distance'
elif opts.doSimulation:
    basename = 'fitting'
else:
    print "Need to enable --doModels, --doSimulation, or --doGoingTheDistance"
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
elif opts.doGoingTheDistance:
    plotDir = os.path.join(plotDir,"_".join(filters))
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
        elif opts.doGoingTheDistance:
            dataDir = os.path.join(basedataDir,"%.2f"%float(errorbudget))
            plotDir = os.path.join(baseplotDir,"%.2f"%float(errorbudget))

            multifile = get_post_file(dataDir)
            data = np.loadtxt(multifile)

            if opts.doEjecta:
                mej_em = data[:,1]
                vej_em = data[:,2]

                filename = os.path.join(dataDir,"truth_mej_vej.dat")
                truths_mej_vej = np.loadtxt(filename)
                truths_mej_vej[0] = np.log10(truths_mej_vej[0])
                mej_true = truths_mej_vej[0]
                vej_true = truths_mej_vej[1]

                filename = os.path.join(dataDir,"truth.dat")
                truths = np.loadtxt(filename)

            elif opts.doMasses:
                mchirp_em,eta_em,q_em = ms2mc(data[:,1],data[:,3])
                q_em = 1/q_em

                filename = os.path.join(dataDir,"truth_mej_vej.dat")
                truths_mej_vej = np.loadtxt(filename)
                truths_mej_vej[0] = np.log10(truths_mej_vej[0])

                filename = os.path.join(dataDir,"truth.dat")
                truths = np.loadtxt(filename)
                mchirp_true,eta_true,q_true = ms2mc(truths[0],truths[2])
                q_true = 1/q_true

        multifile = get_post_file(plotDir)
        if not multifile: continue
        data = np.loadtxt(multifile)

        post[name][errorbudget] = {}
        if opts.doGoingTheDistance:
            if opts.model == "BNS":
                if opts.doEOSFit:
                    mchirp_gw,eta_gw,q_gw = ms2mc(data[:,0],data[:,2])
                    mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
                    ii = 0
                    for m1,c1,m2,c2 in data[:,:-1]:
                        mb1 = EOSfit(m1,c1)
                        mb2 = EOSfit(m2,c2)
                        mej_gw[ii], vej_gw[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                        ii = ii + 1
                else:
                    mchirp_gw,eta_gw,q_gw = ms2mc(data[:,0],data[:,3])
                    mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
                    ii = 0
                    for m1,mb1,c1,m2,mb2,c2 in data[:,:-1]:
                        mej_gw[ii], vej_gw[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                        ii = ii + 1
                q_gw = 1/q_gw
                mej_gw = np.log10(mej_gw)

                combinedDir = os.path.join(plotDir,"combined")
                multifile = get_post_file(combinedDir)
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
                bins, hist1 = hist_results(samples,Nbins=25,bounds=[-3.5,0.0]) 

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["mej_true"],post[name][errorbudget]["mej_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]

            label = "EM"
            samples = post[name][errorbudget]["mej_em"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[-3.5,0.0])

            if jj == 0:            
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["mej_combined"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[-3.5,0.0])

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
    plt.xlim([-3.0,-1.4])
    plt.ylim([1e-1,10])
    plt.savefig(plotName)
    plt.close()

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
                bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.0,1.0])

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["vej_true"],post[name][errorbudget]["vej_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]

            label = "EM"
            samples = post[name][errorbudget]["vej_em"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.0,1.0])

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["vej_combined"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.0,1.0])

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
    plt.xlim([0.0,1.0])
    plt.ylim([1e-1,10])
    plt.savefig(plotName)
    plt.close()

elif opts.doMasses:

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
                bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.8,2.0])

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["mchirp_true"],post[name][errorbudget]["mchirp_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]
            label = "EM"
            samples = post[name][errorbudget]["mchirp_em"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.2,2.0])

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["mchirp_combined"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.8,2.0])

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
    plt.xlim([0.8,2.0])
    plt.ylim([1e-1,10])
    plt.savefig(plotName)
    plt.close()

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
                bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.95,2.0])

                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
                plt.semilogy([post[name][errorbudget]["q_true"],post[name][errorbudget]["q_true"]],[1e-3,10.0],'%s--'%colortrue,linewidth=3)

            color = colors[ii+1]
            label = "EM"
            samples = post[name][errorbudget]["q_em"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.95,2.0])

            if jj == 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            color = colors[ii+2]

            label = "GW-EM"
            samples = post[name][errorbudget]["q_combined"]
            bins, hist1 = hist_results(samples,Nbins=25,bounds=[0.95,2.0])

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
    plt.xlim([0.95,2.0])
    plt.ylim([1e-1,10])
    plt.savefig(plotName)
    plt.close()
