#!/usr/bin/env python

# ---- Import standard modules to the python path.

import os, sys, copy
import numpy as np
import argparse

from scipy.interpolate import interpolate as interp
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

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
    parser.add_argument("--posterior_samples", default="../data/event_data/G298048.dat")

    parser.add_argument("--cbc_list", default="../data/3G_Lists/list_BNS_detected_3G_median_12.txt")
    parser.add_argument("--cbc_type", default="BNS")

    parser.add_argument("--mindistance",default=1.0,type=float)
    parser.add_argument("--maxdistance",default=1000.0,type=float)

    parser.add_argument("-s","--spectraDir",default="../spectra")
    parser.add_argument("-l","--lightcurvesDir",default="../lightcurves")

    parser.add_argument("-a","--analysisType",default="multinest")

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

    parser.add_argument("-f","--filters",default="g,r")
    parser.add_argument("--tmax",default=7.0,type=float)
    parser.add_argument("--tmin",default=0.05,type=float)
    parser.add_argument("--dt",default=0.05,type=float)

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
    if not model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017","BaKa2016","Ka2017","RoFe2017"]:
        print("Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017,BaKa2016,Ka2017,RoFe2017")
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
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
if opts.analysisType == "cbclist":
    plotDir = os.path.join(plotDir,opts.cbc_type)
    plotDir = os.path.join(plotDir,"%d_%d"%(opts.mindistance,opts.maxdistance))

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
datDir = os.path.join(plotDir,"dat")
if not os.path.isdir(datDir):
    os.makedirs(datDir)

if opts.analysisType == "posterior":
    # read in samples
    samples = KNTable.read_samples(opts.posterior_samples)
    # limit masses
    samples = samples.mass_cut(mass1=3.0,mass2=3.0)
    
    samples["dist"] = opts.distance   
 
    print("m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"])))
    print("m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"])))
    
    # Downsample 
    samples = samples.downsample(Nsamples=100)
    # Calc lambdas
    samples = samples.calc_tidal_lambda(remove_negative_lambda=True)
    # Calc compactness
    samples = samples.calc_compactness(fit=True)
    # Calc baryonic mass
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
    
    if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
        from gwemlightcurves.EjectaFits.DiUj2017 import calc_meje, calc_vej
        # calc the mass of ejecta
        samples['mej'] = calc_meje(samples['m1'], samples['mb1'], samples['c1'], samples['m2'], samples['mb2'], samples['c2'])
        # calc the velocity of ejecta
        samples['vej'] = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])
    
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
elif opts.analysisType == "multinest":
    multinest_samples = opts.multinest_samples.split(",")
    samples_all = {}
    for multinest_sample, model in zip(multinest_samples,models):
        # read multinest samples
        samples = KNTable.read_multinest_samples(multinest_sample, model)
        samples["dist"] = opts.distance
        samples_all[model] = samples
elif opts.analysisType == "cbclist":

    tmpfile = opts.cbc_list.replace(".txt",".tmp")
    cbccnt = 0
    lines = [line.rstrip('\n') for line in open(opts.cbc_list)]
    fid = open(tmpfile,'w')
    for line in lines:
        lineSplit = line.split(" ")
        dist = lineSplit[9]
        if (float(dist) > opts.mindistance) and (float(dist) < opts.maxdistance):
            if cbccnt <= opts.nsamples:
                fid.write('%s\n'%line)
            cbccnt = cbccnt + 1
    fid.close()
    cbcratio = float(cbccnt)/float(len(lines))
    
    # read in samples
    samples = KNTable.read_cbc_list(tmpfile)
    # limit masses
    #samples = samples.mass_cut(mass1=3.0,mass2=3.0)

    print("m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"])))
    print("m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"])))

    # Downsample 
    #samples = samples.downsample(Nsamples=1000)

    eosname = "SLy" 
    eos = EOS4ParameterPiecewisePolytrope(eosname)
    lambda1s, lambda2s = [], []
    for row in samples:
        lambda1, lambda2 = eos.lambdaofm(row["m1"]), eos.lambdaofm(row["m2"])
        lambda1s.append(lambda1)
        lambda2s.append(lambda2)
    samples["lambda1"] = lambda1s
    samples["lambda2"] = lambda2s
    samples["Xlan"] = 1e-3

    # Calc compactness
    samples = samples.calc_compactness(fit=True)
    # Calc baryonic mass
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)

    if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
        if opts.cbc_type == "BNS":
            from gwemlightcurves.EjectaFits.Di2018 import calc_meje, calc_vej
            # calc the mass of ejecta
            samples['mej'] = calc_meje(samples['m1'],samples['c1'], samples['m2'], samples['mb2'])
            # calc the velocity of ejecta
            samples['vej'] = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])
        elif opts.cbc_type == "BHNS":
            samples['chi_eff'] = 1.0
            from gwemlightcurves.EjectaFits.KaKy2016 import calc_meje, calc_vave
            # calc the mass of ejecta
            samples['mej'] = calc_meje(samples['q'],samples['chi_eff'],samples['c1'], samples['mb1'], samples['m1'])
            # calc the velocity of ejecta
            samples['vej'] = calc_vave(samples['q'])

    # HACK: multiply by 10 to get full ejecta
    samples['mej'] = samples['mej'] * 10.0

    idx = np.where(samples['mej']>0.1)[0]
    samples['mej'][idx] = 0.1

    print(np.min(samples['mej']),np.max(samples['mej']))

bounds = [-3.0,-1.0]
xlims = [-2.8,-1.0]
ylims = [1e-1,2]

plotName = "%s/mej.pdf"%(plotDir)
plt.figure(figsize=(15,10))
ax = plt.gca()
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins, hist1 = lightcurve_utils.hist_results(np.log10(samples["mej"]),Nbins=20,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name,where='mid')
    lim = np.percentile(np.log10(samples["mej"]), 90)
    plt.plot([lim,lim],ylims,'k--')
plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
#plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
ax.set_yscale('log')
plt.savefig(plotName)
plt.close()

bounds = [0.0,1.0]
xlims = [0.0,1.0]
ylims = [1e-1,20]

plotName = "%s/vej.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins, hist1 = lightcurve_utils.hist_results(samples["vej"],Nbins=25,bounds=bounds)
    plt.step(bins,hist1,'-',color='k',linewidth=3,label=legend_name)

plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

if opts.nsamples > 0:
    samples = samples.downsample(Nsamples=opts.nsamples)

#add default values from above to table
samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt
samples['vmin'] = vmin
samples['th'] = th
samples['ph'] = ph
samples['kappa'] = kappa
samples['eps'] = eps
samples['alp'] = alp
samples['eth'] = eth
samples['flgbct'] = flgbct
samples['beta'] = beta
samples['kappa_r'] = kappa_r
samples['slope_r'] = slope_r
samples['theta_r'] = theta_r
samples['Ye'] = Ye

kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = True
kwargs["doSpec"] = False

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
model_tables = {}
for model in models:
    model_tables[model] = KNTable.model(model, samples, **kwargs)

# Now we need to do some interpolation
for model in models:
    model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model]) 
    #model_tables[model] = lightcurve_utils.interpolate_mags_lbol(model_tables_lbol[model])

if opts.analysisType == "cbclist":
    fid = open(os.path.join(plotDir,'cbcratio.dat'),'w')
    fid.write('%d %.10f'%(cbccnt,cbcratio))
    fid.close()    

filts = ["u","g","r","i","z","y","J","H","K"]
magidxs = [0,1,2,3,4,5,6,7,8]

idxs = []
for filt in filters:
    idxs.append(filts.index(filt))

filts = [filts[i] for i in idxs]
magidxs = [magidxs[i] for i in idxs]

colors=cm.rainbow(np.linspace(0,1,len(filts)))

mag_all = {}
lbol_all = {}

for model in models:
    mag_all[model] = {}
    lbol_all[model] = {}

    lbol_all[model] = np.empty((0,len(tt)), float)
    for filt, color, magidx in zip(filts,colors,magidxs):
        mag_all[model][filt] = np.empty((0,len(tt)))

peak_mags_all = {}
for model in models:
    model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model])
    for row in model_tables[model]:
        t, lbol, mag = row["t"], row["lbol"], row["mag"]

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

if opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
    data_out = lightcurve_utils.loadEvent(filename)
    for ii,key in enumerate(data_out.iterkeys()):
        if key == "t":
            continue
        else:
            data_out[key][:,0] = data_out[key][:,0] - opts.T0
            data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)
elif opts.doFixedLimit:
    data_out = {}
    data_out["t"] = tt
    for filt, limit in zip(filters,limits):
        data_out[filt] = np.vstack((tt,limit*np.ones(tt.shape),np.inf*np.ones(tt.shape))).T
        data_out[filt][:,1] = data_out[filt][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

colors_names=cm.rainbow(np.linspace(0,1,len(models)))
color2 = 'coral'
color1 = 'cornflowerblue'
colors_names=[color1,color2]

linestyles = ['-', '-.', ':','--','-']

plotName = "%s/mag.pdf"%(plotDir)
plt.figure(figsize=(10,8))
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

    if opts.doEvent or opts.doFixedLimit:
        for filt, color, magidx in zip(filts,colors,magidxs):
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
            idx = np.where(~np.isfinite(sigma_y))[0]
            #plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)
            plt.plot(t[idx],y[idx],'--',c=color)

plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/peaki.pdf"%(plotDir)
plt.figure(figsize=(10,8))
cnt = 0
for model in models:
    plt.scatter(model_tables[model]["peak_tt_i"],model_tables[model]["peak_mag_i"]+np.floor(5*(np.log10(opts.distance*1e6) - 1)),c=np.log10(model_tables[model]["mej"]))
plt.xlabel(r'$t_{\rm peak}$ [days]')
plt.ylabel(r'Peak i-band magnitude')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
cbar.set_label(r'Ejecta mass log10($M_{\odot}$)')
plt.savefig(plotName)
plt.close()

bounds = [15,35]
xlims = [15.0,35.0]
ylims = [1e-2,1]

plotName = "%s/appi.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for ii,model in enumerate(models):
    legend_name = get_legend(model)
    bins, hist1 = lightcurve_utils.hist_results(model_tables[model]["peak_appmag_i"],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
plt.xlabel(r"Apparent Magnitude [mag]",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

if opts.analysisType == "cbclist":
    bounds = [15,35]
    xlims = [15.0,35.0]
    ylims = [1,100000]

    plotName = "%s/rates.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,model in enumerate(models):
        legend_name = get_legend(model)
        bins, hist1 = lightcurve_utils.hist_results(model_tables[model]["peak_appmag_i"],Nbins=25,bounds=bounds)
        hist1_cumsum = float(cbccnt)*hist1 / np.sum(hist1)
        hist1_cumsum = np.cumsum(hist1_cumsum)
        plt.semilogy(bins,hist1_cumsum,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    plt.xlabel(r"Apparent Magnitude [mag]",fontsize=24)
    plt.ylabel("Rate of apparent magnitude [per year]",fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(xlims)
    #plt.ylim(ylims)
    plt.savefig(plotName)
    plt.close()

    for model in models:
        for filt, color, magidx in zip(filts,colors,magidxs):
            fid = open(os.path.join(datDir,'%s_%s_list.dat'%(model,filt)),'w')
            for row in model_tables[model]:
                q = np.max([row["m1"]/row["m2"],row["m2"]/row["m1"]])
                fid.write("%.5f %.5f %.5f %.5f %.5f\n"%(q,row["dist"],row["peak_tt_%s"%filt],row["peak_mag_%s"%filt],row["peak_appmag_%s"%filt]))
            fid.close()

colors_names=cm.rainbow(np.linspace(0,1,len(models)))

for model in models:
    for filt, color, magidx in zip(filts,colors,magidxs):
        fid = open(os.path.join(datDir,'%s_%s.dat'%(model,filt)),'w')
        fid.write("t [days] min median max\n")
        #magmed = np.median(mag_all[model][filt],axis=0)
        #magmax = np.max(mag_all[model][filt],axis=0)
        #magmin = np.min(mag_all[model][filt],axis=0)
        magmed = np.percentile(mag_all[model][filt], 50, axis=0) 
        magmax = np.percentile(mag_all[model][filt], 90, axis=0) + opts.errorbudget
        magmin = np.percentile(mag_all[model][filt], 10, axis=0) - opts.errorbudget
        for a,b,c,d in zip(tt,magmin,magmed,magmax):
            fid.write("%.5f %.5f %.5f %.5f\n"%(a,b,c,d))
        fid.close()

plotName = "%s/mag_panels.pdf"%(plotDir)
plt.figure(figsize=(20,28))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if opts.doEvent or opts.doFixedLimit:
        if not filt in data_out: continue
        samples = data_out[filt]
        t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

        idx = np.where(np.isfinite(sigma_y))[0]
        plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
        idx = np.where(~np.isfinite(sigma_y))[0]
        plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)

    for ii, model in enumerate(models):
        legend_name = get_legend(model)

        #magmed = np.median(mag_all[model][filt],axis=0)
        #magmax = np.max(mag_all[model][filt],axis=0)
        #magmin = np.min(mag_all[model][filt],axis=0)

        magmed = np.percentile(mag_all[model][filt], 50, axis=0)
        magmax = np.percentile(mag_all[model][filt], 90, axis=0) + opts.errorbudget
        magmin = np.percentile(mag_all[model][filt], 10, axis=0) - opts.errorbudget

        plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=4,label=legend_name)
        plt.plot(tt,magmin,'-',c=colors_names[ii],linewidth=4)
        plt.plot(tt,magmax,'-',c=colors_names[ii],linewidth=4)
        plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=3)
    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 14.0])
    plt.ylim([-18.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':40},numpoints=1,shadow=True, fancybox=True)

        #ax3 = ax1.twinx()   # mirror them
        #ax3.set_yticks([16,12,8,4,0])
        #app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(opts.distance*1e6) - 1))
        #ax3.set_yticklabels(app.astype(int))

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
    else:
        #ax4 = ax2.twinx()   # mirror them
        #ax4.set_yticks([16,12,8,4,0])
        #app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(opts.distance*1e6) - 1))
        #ax4.set_yticklabels(app.astype(int))

        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)

    if (not cnt == len(filts)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)

ax1.set_zorder(1)
ax2.set_xlabel('Time [days]',fontsize=48,labelpad=30)
plt.savefig(plotName)
plt.close()

plotName = "%s/gminusr.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    legend_name = get_legend(model)

    magmed = np.median(mag_all[model]["g"]-mag_all[model]["r"],axis=0)
    magmax = np.max(mag_all[model]["g"]-mag_all[model]["r"],axis=0) + opts.errorbudget
    magmin = np.min(mag_all[model]["g"]-mag_all[model]["r"],axis=0) - opts.errorbudget

    plt.plot(tt,magmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,magmin,magmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlim([0.0, 14.0])
plt.xlabel('Time [days]')
plt.ylabel('Color [g-i]')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

plotName = "%s/Lbol.pdf"%(plotDir)
plt.figure()
cnt = 0
for ii, model in enumerate(models):
    legend_name = get_legend(model)

    lbolmed = np.median(lbol_all[model],axis=0)
    lbolmax = np.max(lbol_all[model],axis=0) * (2.5 * opts.errorbudget)
    lbolmin = np.min(lbol_all[model],axis=0) / (2.5 * opts.errorbudget)
    plt.loglog(tt,lbolmed,'--',c=colors_names[ii],linewidth=2,label=legend_name)
    plt.fill_between(tt,lbolmin,lbolmax,facecolor=colors_names[ii],alpha=0.2)

plt.xlim([0.0, 50.0])
plt.legend(loc="best")
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.savefig(plotName)
plt.close()


