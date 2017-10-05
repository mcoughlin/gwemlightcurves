
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

from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../lightcurves")
    #parser.add_option("-n","--name",default="../plots/gws/DiUj2017_FixZPT0/u_g_r_i_z_y_J_H_K/7_14/ejecta/G298048_PS1_GROND_SOFI/1.00,../plots/gws/WoKo2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/G298048_PS1_GROND_SOFI/1.00,../plots/gws/Me2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/G298048_PS1_GROND_SOFI/1.00,../plots/gws/SmCh2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/G298048_PS1_GROND_SOFI/1.00,../plots/gws/WoKo2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/G298048_PS1_GROND_SOFI/1.00")
    parser.add_option("--outputName",default="G298048_PS1_GROND_SOFI")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)

    parser.add_option("-n","--name",default="../plots/gws/DiUj2017_EOSFit_FixZPT0/u_g_r_i_z_y_J_H_K/7_14/masses/G298048_PS1_GROND_SOFI/1.00,../plots/gws/WoKo2017_EOSFit_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/masses/G298048_PS1_GROND_SOFI/1.00,../plots/gws/Me2017_EOSFit_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/masses/G298048_PS1_GROND_SOFI/1.00,../plots/gws/SmCh2017_EOSFit_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/masses/G298048_PS1_GROND_SOFI/1.00,../plots/gws/WoKo2017_EOSFit_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/masses/G298048_PS1_GROND_SOFI/1.00")

    #parser.add_option("-n","--name",default="../plots/gws/Me2017_EOSFit/u_g_r_i_z_y_J_H_K/0_14/masses/G298048_PS1_GROND_SOFI/1.00,../plots/gws/DiUj2017_EOSFit/y_J_H_K/5_14/masses/G298048_PS1_GROND_SOFI/1.00")

    parser.add_option("--mchirp",default=1.1973,type=float)
    parser.add_option("--massratio_min",default=1.0,type=float)
    parser.add_option("--massratio_max",default=1.43,type=float)
    parser.add_option("--mej_min",default=1e-3,type=float)
    parser.add_option("--mej_max",default=1e-2,type=float)

    #parser.add_option("-l","--labelType",default="errorbar")
    parser.add_option("-l","--labelType",default="name")

    opts, args = parser.parse_args()

    return opts

def get_labels(label):
    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","KaKy2016","DiUj2017","SN","tanaka_compactmergers","macronovae-rosswog","Afterglow","metzger_rprocess","WoKo2017","Me2017","SmCh2017"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich and Ujevic (2017)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Van Eerten et al. (2012)","Metzger et al. (2010)","Wollaeger et al. (2017)","Metzger (2017)","Smartt et al. (2017)"]

    idx = models.index(label)
    return models_ref[idx]

# Parse command line
opts = parse_commandline()

if not (opts.doEjecta or opts.doMasses):
    print "Enable --doEjecta or --doMasses"
    exit(0)

names = opts.name.split(",")
post = {}
for plotDir in names:
    plotDirSplit = plotDir.split("/")
    name = plotDirSplit[-6]
    if "EOSFit" in name:
        EOSFit = 1
    else:
        EOSFit = 0

    nameSplit = name.split("_")
    name = nameSplit[0] 
     
    errorbudget = float(plotDirSplit[-1])
    post[name] = {}
        
    multifile = lightcurve_utils.get_post_file(plotDir)
    if not multifile: continue
    data = np.loadtxt(multifile)

    post[name][errorbudget] = {}
    if name == "KaKy2016":
        if opts.doMasses:
            if EOSFit:
                t0 = data[:,0]
                q = data[:,1]
                chi_eff = data[:,2]
                mns = data[:,3]
                c = data[:,4]
                th = data[:,5]
                ph = data[:,6]
                zp = data[:,7]
                loglikelihood = data[:,8]

                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1]*data[:,3],data[:,3])

                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = q
            else:
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

                mchirp,eta,q = lightcurve_utils.ms2mc(data[:,1]*data[:,3],data[:,3])

                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = q

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

    elif name == "DiUj2017": 
        if opts.doMasses:
            if EOSFit:
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                th = data[:,5]
                ph = data[:,6]
                zp = data[:,7]
 
                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
 
                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q
            else:
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
    
                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
    
                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q

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

    elif name == "Me2017":
        if opts.doMasses:
            if EOSFit:
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                beta = data[:,5]
                kappa_r = data[:,6]
                zp = data[:,7]
 
                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
 
                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q
            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                beta = data[:,7]
                kappa_r = data[:,8]
                zp = data[:,9]
    
                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)
    
                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q

        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            beta = data[:,3]
            kappa_r = data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]

            post[name][errorbudget]["mej"] = mej
            post[name][errorbudget]["vej"] = vej

    elif name == "SmCh2017":
        if opts.doMasses:
            if EOSFit:
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                slope_r = data[:,5]
                kappa_r = data[:,6]
                zp = data[:,7]

                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)

                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q
            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                slope_r = data[:,7]
                kappa_r = data[:,8]
                zp = data[:,9]

                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)

                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q
        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            slope_r = data[:,3]
            kappa_r = data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]

            post[name][errorbudget]["mej"] = mej
            post[name][errorbudget]["vej"] = vej

    elif name == "WoKo2017":
        if opts.doMasses:
            if EOSFit:
                t0 = data[:,0]
                m1 = data[:,1]
                c1 = data[:,2]
                m2 = data[:,3]
                c2 = data[:,4]
                theta_r = data[:,5]
                kappa_r = data[:,6]
                zp = data[:,7]

                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)

                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q
            else:
                t0 = data[:,0]
                m1 = data[:,1]
                mb1 = data[:,2]
                c1 = data[:,3]
                m2 = data[:,4]
                mb2 = data[:,5]
                c2 = data[:,6]
                theta_r = data[:,7]
                kappa_r = data[:,8]
                zp = data[:,9]

                mchirp,eta,q = lightcurve_utils.ms2mc(m1,m2)

                post[name][errorbudget]["mchirp"] = mchirp
                post[name][errorbudget]["q"] = 1/q

        elif opts.doEjecta:
            t0 = data[:,0]
            mej = 10**data[:,1]
            vej = data[:,2]
            theta_r = data[:,3]
            kappa_r = data[:,4]
            zp = data[:,5]
            loglikelihood = data[:,6]

            post[name][errorbudget]["mej"] = mej
            post[name][errorbudget]["vej"] = vej

    elif name == "SN":
        t0 = data[:,0]
        z = data[:,1]
        x0 = data[:,2]
        x1 = data[:,3]
        c = data[:,4]
        zp = data[:,5]
        loglikelihood = data[:,6]

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'gws')
plotDir = os.path.join(plotDir,opts.outputName)
if opts.doMasses:
    plotDir = os.path.join(plotDir,'masses')
elif opts.doEjecta:
    plotDir = os.path.join(plotDir,'ejecta')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

colors = ['b','g','r','m','c']
linestyles = ['-', '-.', ':','--']

if opts.doEjecta:

    bounds = [-3.0,0.0]
    xlims = [-3.0,0.0]
    ylims = [1e-1,10]

    plotName = "%s/mej.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):
            if opts.labelType == "errorbar":
                label = r"$\Delta$m: %.2f"%float(errorbudget)
            elif opts.labelType == "name":
                label = get_labels(name)
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
          
            bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds) 
    
            if opts.labelType == "name" and jj > 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
    
            maxhist = np.max([maxhist,np.max(hist1)])
   
    plt.fill_between([np.log10(opts.mej_min),np.log10(opts.mej_max)],[1e-3,1e-3],[1e2,1e2],facecolor='k',alpha=0.2)
 
    plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

    bounds = [0.0,1.0]
    xlims = [0.0,1.0]
    ylims = [1e-1,20]
    
    plotName = "%s/vej.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):
            if opts.labelType == "errorbar":
                label = r"$\Delta$m: %.2f"%float(errorbudget)
            elif opts.labelType == "name":
                label = get_labels(name)
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
    
            samples = post[name][errorbudget]["vej"]
            bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)
    
            if opts.labelType == "name" and jj > 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)
    
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

    bounds = [0.8,4.0]
    xlims = [0.8,4.0]
    ylims = [1e-1,10]

    plotName = "%s/mchirp.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1

    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):

            if opts.labelType == "errorbar":
                label = r"$\Delta$m: %.2f"%float(errorbudget)
            elif opts.labelType == "name":
                label = get_labels(name)
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

            samples = post[name][errorbudget]["mchirp"]
            bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)

            if opts.labelType == "name" and jj > 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)

            maxhist = np.max([maxhist,np.max(hist1)])
    plt.semilogy([opts.mchirp,opts.mchirp],[1e-3,100],'k--')

    plt.xlabel(r"${\rm M}_{\rm c}$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

    bounds = [0.0,2.0]
    xlims = [0.9,2.0]
    ylims = [1e-1,50]

    plotName = "%s/q.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    maxhist = -1
    for ii,name in enumerate(sorted(post.keys())):
        for jj,errorbudget in enumerate(sorted(post[name].keys())):

            if opts.labelType == "errorbar":
                label = r"$\Delta$m: %.2f"%float(errorbudget)
            elif opts.labelType == "name":
                label = get_labels(name)
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

            samples = post[name][errorbudget]["q"]
            bins, hist1 = lightcurve_utils.hist_results(samples,Nbins=25,bounds=bounds)

            if opts.labelType == "name" and jj > 0:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),linewidth=3)
            else:
                plt.semilogy(bins,hist1,'%s%s'%(color,linestyle),label=label,linewidth=3)

            maxhist = np.max([maxhist,np.max(hist1)])

    plt.fill_between([opts.massratio_min,opts.massratio_max],[1e-3,1e-3],[1e2,1e2],facecolor='k',alpha=0.2)

    plt.xlabel(r"$q$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()
