
import os, sys, copy
import numpy as np
import optparse

from scipy.interpolate import interpolate as interp
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves import BNSKilonovaLightcurve, BHNSKilonovaLightcurve, BlueKilonovaLightcurve, ArnettKilonovaLightcurve, SALT2
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
    parser.add_option("-m","--model",default="BHNS,BNS,Blue", help="BHNS, BNS, Blue, Arnett")
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

# Equation to relate EOS and neutron star mass to Baryonic mass
# Eq 8: https://arxiv.org/pdf/1708.07714.pdf

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
  
    if (cc > 4./9.).any():
        print("Warning: Returned compactnesses > 4/9 = 0.44 ... setting = 4/9")
        cc[cc > 4./9.] = 4./9.
    if (cc < 0.).any(): 
        print("Warning: Returned compactnesses < 0 ... setting = 0.")
        cc[cc < 0.0] = 0.0

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

# See Eqs. 5 and 6 from
# https://journals.aps.org/prd/pdf/10.1103/PhysRevD.89.103012

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

models = opts.model.split(",")
for model in models:
    if not model in ["BHNS", "BNS", "Blue","Arnett"]:
        print "Model must be either: BHNS, BNS, Blue, Arnett"
        exit(0)

data_out = lightcurve_utils.event(opts.dataDir,opts.name)
data_out["lambda1"], data_out["lambda2"] = tidal_lambda_from_tilde(data_out["m1"], data_out["m2"], data_out["lambdat"], data_out["dlambdat"])
mask = (data_out["lambda1"] < 0) | (data_out["lambda2"] < 0)
data_out = data_out[~mask]
print "Removing %d/%d due to negative lambdas"%(np.sum(~mask),len(mask))
data_out["c1"] = CLove(data_out["lambda1"])
data_out["c2"] = CLove(data_out["lambda2"])
data_out["mb1"] = EOSfit(data_out["m1"],data_out["c1"])
data_out["mb2"] = EOSfit(data_out["m2"],data_out["c2"])

#for key in data_out.keys():
#    data_out[key] = data_out[key][idx]

Nsamples = 1000
idx = np.random.permutation(len(data_out["m1"]))
idx = idx[:Nsamples]
#data_out = data_out[idx]
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
slope_r = -1.2

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,"_".join(models))
plotDir = os.path.join(plotDir,"event")
plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

data_out_all = {}
for model in models:
    data_out_all[model] = copy.copy(data_out)
    ii = 0
    idxs = []

    mej, vej = np.zeros(data_out_all[model]["m1"].shape), np.zeros(data_out_all[model]["m1"].shape)
    for m1, m2, c1, c2, mb1, mb2 in zip(data_out_all[model]["m1"],data_out_all[model]["m2"],data_out_all[model]["c1"],data_out_all[model]["c2"],data_out_all[model]["mb1"],data_out_all[model]["mb2"]):
        if model == "BHNS":
            q = m1/m2
            mns = m2
            mej[ii], vej[ii] = bhns_model(q,chi_eff,mns,mb2,c2)
        elif model == "BNS":
            mej[ii], vej[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
        elif model == "Blue":
            mej[ii], vej[ii] = blue_model(m1,mb1,c1,m2,mb2,c2)
        elif model == "Arnett":
            mej[ii], vej[ii] = arnett_model(m1,mb1,c1,m2,mb2,c2)
        ii = ii + 1

    idx = np.where(mej>0)[0]
    data_out_all[model]["mej"] = mej
    data_out_all[model]["vej"] = vej
    data_out_all[model] = data_out_all[model][idx]
    data_out_all[model]["mej"] = np.log10(data_out_all[model]["mej"])

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]

tini, tmax, dt = 0.1, 14.0, 0.1
tt = np.arange(tini,tmax+dt,dt)

mag_all = {}
lbol_all = {}

for model in models:
    mag_all[model] = {}
    lbol_all[model] = {}

    lbol_all[model] = np.empty((0,len(tt)), float)
    for filt, color, magidx in zip(filts,colors,magidxs):
        mag_all[model][filt] = np.empty((0,len(tt)))

for model in models:
    for m1, m2, c1, c2, mb1, mb2 in zip(data_out_all[model]["m1"],data_out_all[model]["m2"],data_out_all[model]["c1"],data_out_all[model]["c2"],data_out_all[model]["mb1"],data_out_all[model]["mb2"]):
        if model == "BHNS":
            q = m1/m2
            mb = EOSfit(m2,c2)
            mns = m2
            t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi_eff,c1,mb,mns)
        elif model == "BNS":
            t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb1,c1,m2,mb2,c2,flgbct)
        elif model == "Blue":
            t, lbol, mag, Tobs = BlueKilonovaLightcurve.lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2)
        elif model == "Arnett":
            t, lbol, mag, Tobs = ArnettKilonovaLightcurve.lightcurve(tini,tmax,dt,slope_r,kappa_r,m1,mb1,c1,m2,mb2,c2)
    
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
            mag_all[model][filt] = np.append(mag_all[model][filt],[maginterp],axis=0)
        idx = np.where((~np.isnan(np.log10(lbol))) & ~(lbol==0))[0]
        f = interp.interp1d(t[idx], np.log10(lbol[idx]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbol_all[model] = np.append(lbol_all[model],[lbolinterp],axis=0)

linestyles = ['-', '-.', ':','--']

plotName = "%s/mag.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    maglen, ttlen = lbol_all[model].shape
    for jj in xrange(maglen):
        for filt, color, magidx in zip(filts,colors,magidxs):
            if cnt == 0 and ii == 0:
                plt.plot(tt,mag_all[model][filt][jj,:],alpha=0.2,c=color,label=filt,linestyle=linestyles[ii])
            else:
                plt.plot(tt,mag_all[model][filt][jj,:],alpha=0.2,c=color,linestyle=linestyles[ii])
        cnt = cnt + 1
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]
colors_names=cm.rainbow(np.linspace(0,1,len(models)))

plotName = "%s/mag_panels.pdf"%(plotDir)
plt.figure(figsize=(20,18))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    #if opts.doEvent:
    #    if not filt in data_out: continue
    #    samples = data_out[filt]
    #    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    #    idx = np.where(~np.isnan(y))[0]
    #    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    #    plt.errorbar(t,y,sigma_y,fmt='o',c='k')

    for ii, model in enumerate(models):
        if model == "BNS":
            legend_name = "Dietrich and Ujevic (2017)"
        if model == "BHNS":
            legend_name = "Kawaguchi et al. (2016)"
        elif model == "Blue":
            legend_name = "Metzger (2017)"
        elif model == "Arnett":
            legend_name = "Arnett (1982)"

        magmed = np.median(mag_all[model][filt],axis=0)
        magmax = np.max(mag_all[model][filt],axis=0)
        magmin = np.min(mag_all[model][filt],axis=0)

        plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
        plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=24,rotation=0,labelpad=20)
    plt.xlim([0.0, 14.0])
    plt.ylim([-18.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-14,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':24},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=24)
plt.savefig(plotName)
plt.close()

plotName = "%s/iminusg.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    if model == "BNS":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "BHNS":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Blue":
        legend_name = "Metzger (2017)"
    elif model == "Arnett":
        legend_name = "Arnett (1982)"

    magmed = np.median(mag_all[model]["i"]-mag_all[model]["g"],axis=0)
    magmax = np.max(mag_all[model]["i"]-mag_all[model]["g"],axis=0)
    magmin = np.min(mag_all[model]["i"]-mag_all[model]["g"],axis=0)

    plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/Lbol.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    if model == "BNS":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "BHNS":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Blue":
        legend_name = "Metzger (2017)"
    elif model == "Arnett":
        legend_name = "Arnett (1982)"

    lbolmed = np.median(lbol_all[model],axis=0)
    lbolmax = np.max(lbol_all[model],axis=0)
    lbolmin = np.min(lbol_all[model],axis=0)

    plt.loglog(tt,lbolmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,lbolmin,lbolmax,facecolor=colors_names[ii],alpha=0.2)

plt.legend(loc="best")
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.savefig(plotName)
plt.close()

bounds = [-3.0,-1.0]
xlims = [-3.0,-1.0]
ylims = [1e-1,10]

plotName = "%s/mej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    if model == "BNS":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "BHNS":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Blue":
        legend_name = "Metzger (2017)"
    elif model == "Arnett":
        legend_name = "Arnett (1982)"
    bins, hist1 = hist_results(data_out_all[model]["mej"],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
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
for ii,model in enumerate(models):
    if model == "BNS":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "BHNS":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Blue":
        legend_name = "Metzger (2017)"
    elif model == "Arnett":
        legend_name = "Arnett (1982)"
    bins, hist1 = hist_results(data_out_all[model]["vej"],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)

plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

bounds = [0.0,2.0]
xlims = [0.0,2.0]
ylims = [1e-1,10]

plotName = "%s/masses.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    if model == "BNS":
        legend_name = "Dietrich and Ujevic (2017)"
    if model == "BHNS":
        legend_name = "Kawaguchi et al. (2016)"
    elif model == "Blue":
        legend_name = "Metzger (2017)"
    elif model == "Arnett":
        legend_name = "Arnett (1982)"
    bins1, hist1 = hist_results(data_out_all[model]["m1"],Nbins=25,bounds=bounds)
    plt.semilogy(bins1,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    bins2, hist2 = hist_results(data_out_all[model]["m2"],Nbins=25,bounds=bounds)
    plt.semilogy(bins2,hist2,'--',color=colors_names[ii],linewidth=3)
plt.xlabel(r"Masses",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()


