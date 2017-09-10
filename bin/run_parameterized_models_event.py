
import os, sys
import numpy as np
import optparse
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves import BNSKilonovaLightcurve, BHNSKilonovaLightcurve, BlueKilonovaLightcurve,SALT2
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

def blue_model(m1,mb1,c1,m2,mb2,c2):

    mej = BlueKilonovaLightcurve.calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = BlueKilonovaLightcurve.calc_vej(m1,c1,m2,c2)

    return mej, vej

def arnett_model(m1,mb1,c1,m2,mb2,c2):

    mej = ArnettKilonovaLightcurve.calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = ArnettKilonovaLightcurve.calc_vej(m1,c1,m2,c2)

    return mej, vej

def EOSfit(mns,c):
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb

# Give the compactness-Love and Love-compactness relations
# NKJ-M, 08.2017

def CLove(lmbda):
    """
    Compactness-Love relation for neutron stars from Eq. (78) of Yagi and Yunes, Phys. Rep. 681, 1 (2017), using the YY coefficients and capping the compactness at the Buchdahl limit of 4/9 = 0.44... (since the fit diverges as lambda \to 0). We also cap the compactness at zero, since it becomes negative for large lambda, though these lambdas are so large that they are unlikely to be encountered in practice. In both cases, we raise an error if it runs up against either of the bounds.

    Input: Dimensionless quadrupolar tidal deformability lmbda
    Output: Compactness (mass over radius, in geometrized units, so the result is dimensionless)
    """

    # Give coefficients
    a0 = 0.360
    a1 = -0.0355
    a2 = 0.000705

    # Compute fit
    lmbda = np.atleast_1d(lmbda)
    ll = np.log(lmbda)
    cc = a0 + (a1 + a2*ll)*ll

    for kk in np.arange(len(cc)):
        if cc[kk] > 4./9.:
            print("Warning: Returned a compactness of 4/9 = 0.44... though the fit gives a compactness of %f for the input value of lambda = %f"%(cc[kk], lmbda[kk]))
            cc[kk] = 4./9.
        elif cc[kk] < 0.:
            print("Warning: Returned a compactness of 0. though the fit gives a compactness of %f for the input value of lambda = %f"%(cc[kk], lmbda[kk]))
            cc[kk] = 0.
    return cc

def LoveC(cc):
    """
    Invert the compactness-Love relation given above.
    """
    # Give coefficients
    a0 = 0.360
    a1 = -0.0355
    a2 = 0.000705
    ll = -(a1 + (a1*a1 - 4.*a2*(a0 - cc))**0.5)/(2.*a2)

    return np.exp(ll)

def tidal_lambda_from_tilde(mass1, mass2, lam_til, dlam_til):
    """
    Determine physical lambda parameters from effective parameters.
    """
    mt = mass1 + mass2
    eta = mass1 * mass2 / mt**2
    q = np.sqrt(1 - 4*eta)

    a = (8./13) * (1 + 7*eta - 31*eta**2)
    b = (8./13) * q * (1 + 9*eta - 11*eta**2)
    c = 0.5 * q * (1 - 13272*eta/1319 + 8944*eta**2/1319)
    d = 0.5 * (1 - 15910*eta/1319 + 32850*eta**2/1319 + 3380*eta**3/1319)

    lambda1 = 0.5 * ((c - d) * lam_til - (a - b) * dlam_til)/(b*c - a*d)
    lambda2 = 0.5 * ((c + d) * lam_til - (a + b) * dlam_til)/(a*d - b*c)

    return lambda1, lambda2

# Parse command line
opts = parse_commandline()

data_out = lightcurve_utils.event(opts.dataDir,opts.name)
lambda1, lambda2 = tidal_lambda_from_tilde(data_out["m1"], data_out["m2"], data_out["lambdat"], data_out["dlambdat"])
c1 = CLove(lambda1)
c2 = CLove(lambda2)
data_out["lambda1"], data_out["lambda2"], data_out["c1"], data_out["c2"] = lambda1, lambda2, c1, c2
idx = np.where((~np.isnan(c1)) & (~np.isnan(c2)))[0]
data_out = data_out[idx]
#for key in data_out.keys():
#    data_out[key] = data_out[key][idx]

Nsamples = 100
idx = np.random.permutation(len(data_out["m1"]))
idx = idx[:Nsamples]
data_out = data_out[idx]
#for key in data_out.keys():
#    data_out[key] = data_out[key][idx]

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

beta = 3.0
kappa_r = 10.0

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,opts.model)
plotDir = os.path.join(plotDir,"event")
plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

mej, vej = np.zeros(data_out["m1"].shape), np.zeros(data_out["m1"].shape)
ii = 0
for m1, m2, c1, c2 in zip(data_out["m1"],data_out["m2"],data_out["c1"],data_out["c2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c1)
        mns = m2
        mej[ii], vej[ii] = bhns_model(q,chi_eff,mns,mb,c)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej[ii], vej[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
    elif opts.model == "Blue":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej[ii], vej[ii] = blue_model(m1,mb1,c1,m2,mb2,c2)
    elif opts.model == "Arnett":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej[ii], vej[ii] = arnett_model(m1,mb1,c1,m2,mb2,c2)
    ii = ii + 1
mej = np.log10(mej)

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]

plotName = "%s/mag.pdf"%(plotDir)
plt.figure()
cnt = 0
for m1, m2, c1, c2 in zip(data_out["m1"],data_out["m2"],data_out["c1"],data_out["c2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c1)
        mns = m2
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c,mb,mns)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c1,m2,mb2,c2,flgbct)
    elif opts.model == "Blue":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag, Tobs = BlueKilonovaLightcurve.lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2)
    elif opts.model == "Arnett":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag, Tobs = ArnettKilonovaLightcurve.lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2)

    if np.sum(lbol) == 0.0:
        print "No luminosity..."
        continue

    for filt, color, magidx in zip(filts,colors,magidxs):
        if cnt == 0:
            plt.plot(t,mag[magidx],alpha=0.2,c=color,label=filt)
        else:
            plt.plot(t,mag[magidx],alpha=0.2,c=color)
    cnt = cnt + 1
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/iminusg.pdf"%(plotDir)
plt.figure()
for m1, m2, c1, c2 in zip(data_out["m1"],data_out["m2"],data_out["c1"],data_out["c2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c1)
        mns = m2
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c1,mb,mns)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c1,m2,mb2,c2,flgbct)
    elif opts.model == "Blue":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag = BlueKilonovaLightcurve.lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2)

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
for m1, m2, c1, c2 in zip(data_out["m1"],data_out["m2"],data_out["c1"],data_out["c2"]):
    if opts.model == "BHNS":
        q = m1/m2
        mb = EOSfit(m2,c1)
        mns = m2
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c,mb,mns)
    elif opts.model == "BNS":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c1,m2,mb2,c2,flgbct)
    elif opts.model == "Blue":
        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        t, lbol, mag = BlueKilonovaLightcurve.lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2)

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
elif opts.model == "Blue":
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
elif opts.model == "Blue":
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

if opts.model == "BHNS":
    bounds = [0.0,2.0]
    xlims = [0.0,2.0]
    ylims = [1e-1,10]
elif opts.model == "BNS":
    bounds = [0.0,2.0]
    xlims = [0.0,2.0]
    ylims = [1e-1,10]
elif opts.model == "Blue":
    bounds = [0.0,2.0]
    xlims = [-3.0,-1.0]
    ylims = [1e-1,10]

plotName = "%s/masses.pdf"%(plotDir)
plt.figure(figsize=(10,8))
bins1, hist1 = hist_results(data_out["m1"],Nbins=25,bounds=bounds)
bins2, hist2 = hist_results(data_out["m2"],Nbins=25,bounds=bounds)
plt.semilogy(bins1,hist1,'b-',linewidth=3,label="m1")
plt.semilogy(bins2,hist2,'r--',linewidth=3,label="m2")
plt.xlabel(r"Masses",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()


