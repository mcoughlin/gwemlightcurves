
import os, sys
import numpy as np
import optparse
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

from gwemlightcurves import BNSKilonovaLightcurve, BHNSKilonovaLightcurve, SALT2
from gwemlightcurves import BHNSKilonovaLightcurveOpt

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
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("-e","--eos",default="H4")
    parser.add_option("--name",default="G298048")

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

def EOSfit(mns,c):
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb

# Parse command line
opts = parse_commandline()

data_out = lightcurve_utils.event(opts.dataDir,opts.name)

if opts.eos == "APR4":
    c = 0.180
    mb = 1.50
elif opts.eos == "ALF2":
    c = 0.161
    mb = 1.49
elif opts.eos == "H4":
    c = 0.147
    mb = 1.47
elif opts.eos == "MS1":
    c = 0.138
    mb = 1.46

tini = 0.1
tmax = 14.0
dt = 0.1

vave = 0.267
vmin = 0.02
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 1

chi_eff = 0.0

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,opts.model)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
plotDir = os.path.join(plotDir,"event")
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
plotDir = os.path.join(plotDir,opts.eos)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

mej, vej = np.zeros(data_out["m1"].shape), np.zeros(data_out["m1"].shape)
ii = 0
for m1, m2 in zip(data_out["m1"],data_out["m2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c)
        mns = m2
        mej[ii], vej[ii] = bhns_model(q,chi,mns,mb,c)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c)
        mb2 = EOSfit(m2,c)
        mej[ii], vej[ii] = bns_model(m1,mb1,c,m2,mb2,c)
    ii = ii + 1
mej = np.log10(mej)

plotName = "%s/mag.pdf"%(plotDir)
plt.figure()
cnt = 0
for m1, m2 in zip(data_out["m1"],data_out["m2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c)
        mns = m2
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c,mb,mns)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c)
        mb2 = EOSfit(m2,c)
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c,m2,mb2,c,flgbct)

    if np.sum(lbol) == 0.0:
        print "No luminosity..."
        continue
    if cnt == 0:
        plt.plot(t,mag[1],'g',alpha=0.2,label="g")
        plt.plot(t,mag[3],'r',alpha=0.2,label="i")
    else:
        plt.plot(t,mag[1],'g',alpha=0.2)
        plt.plot(t,mag[3],'r',alpha=0.2)
    cnt = cnt + 1
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/iminusg.pdf"%(plotDir)
plt.figure()
for m1, m2 in zip(data_out["m1"],data_out["m2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c)
        mns = m2
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c,mb,mns)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c)
        mb2 = EOSfit(m2,c)
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c,m2,mb2,c,flgbct)

    if np.sum(lbol) == 0.0:
        print "No luminosity..."
        continue
    plt.plot(t,mag[3]-mag[1],'k',alpha=0.2)
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/Lbol.pdf"%(plotDir)
plt.figure()
plt.figure()
for m1, m2 in zip(data_out["m1"],data_out["m2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c)
        mns = m2
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c,mb,mns)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c)
        mb2 = EOSfit(m2,c)
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c,m2,mb2,c,flgbct)

    if np.sum(lbol) == 0.0:
        print "No luminosity..."
        continue
    plt.semilogy(t,lbol,'k',alpha=0.2)
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.savefig(plotName)
plt.close()

if opts.model == "BHNS":
    bounds = [-3.0,0.0]
    xlims = [-3.0,0.0]
    ylims = [1e-1,10]
elif opts.model == "BNS":
    bounds = [-3.0,-1.0]
    xlims = [-3.0,-1.0]
    ylims = [1e-1,10]

plotName = "%s/mej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
bins, hist1 = hist_results(mej,Nbins=25,bounds=bounds)
plt.semilogy(bins,hist1,'b-',linewidth=3)
plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

if opts.model == "BHNS":
    bounds = [0.0,1.0]
    xlims = [0.0,1.0]
    ylims = [1e-1,20]
elif opts.model == "BNS":
    bounds = [0.0,1.0]
    xlims = [0.0,1.0]
    ylims = [1e-1,10]

plotName = "%s/vej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
bins, hist1 = hist_results(vej,Nbins=25,bounds=bounds)
plt.semilogy(bins,hist1,'b-',linewidth=3)
plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()
