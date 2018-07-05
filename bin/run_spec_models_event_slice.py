#!/usr/bin/env python

# ---- Import standard modules to the python path.

import os, sys, copy
import numpy as np
import argparse

from astropy.table import Table, Column

from scipy.interpolate import interpolate as interp
import scipy.stats, scipy.signal

from astropy.time import Time
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves import __version__

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.add_argument("-o","--outputDir",default="../output")
    parser.add_argument("-p","--plotDir",default="../plots_slice")
    parser.add_argument("-d","--dataDir",default="../data")
    parser.add_argument("-s","--spectraDir",default="../spectra")
    parser.add_argument("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_argument("-a","--analysisType",default="single")
    parser.add_argument("--posterior_samples", default="../data/event_data/G298048.dat")

    parser.add_argument("--multinest_samples", default="../plots/gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat,../plots/gws/Ka2017x2_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/2-post_equal_weights.dat")
    parser.add_argument("-m","--model",default="Ka2017,Ka2017x2", help="Ka2017")

    parser.add_argument("--name",default="GW170817")

    parser.add_argument("--doEvent",  action="store_true", default=False)
    parser.add_argument("-e","--event",default="GW170817")
    parser.add_argument("--distance",default=40.0,type=float)
    parser.add_argument("--T0",default=57982.5285236896,type=float)
    parser.add_argument("--errorbudget",default=0.01,type=float)
    parser.add_argument("--nsamples",default=-1,type=int)

    parser.add_argument("--mej",default=0.04,type=float)
    parser.add_argument("--vej",default=0.20,type=float)
    parser.add_argument("--Xlan",default=1e-2,type=float)

    parser.add_argument("--mej1",default=0.02,type=float)
    parser.add_argument("--vej1",default=0.20,type=float)
    parser.add_argument("--Xlan1",default=1e-2,type=float)

    parser.add_argument("--mej2",default=0.01,type=float)
    parser.add_argument("--vej2",default=0.30,type=float)
    parser.add_argument("--Xlan2",default=1e-4,type=float)

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

models = opts.model.split(",")
for model in models:
    if not model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017","BaKa2016","Ka2017","Ka2017x2","RoFe2017"]:
        print "Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017,BaKa2016,Ka2017,Ka2017x2,RoFe2017"
        exit(0)

lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir
ModelPath = '%s/svdmodels'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

if opts.doEvent:
    filename = "../spectra/%s_spectra_index.dat"%opts.event
    lines = [line.rstrip('\n') for line in open(filename)]
    filenames = []
    T0s = []
    for line in lines:
        lineSplit = line.split(" ")
        #if not lineSplit[0] == opts.event: continue
        filename = "%s/%s"%(spectraDir,lineSplit[1])
        filenames.append(filename)
        mjd = Time(lineSplit[2], format='isot').mjd
        T0s.append(mjd-opts.T0)

    #filenames = filenames[2:3]
    #T0s = T0s[2:3]

    filenames = filenames[1:3]
    T0s = T0s[1:3]

    distconv = (opts.distance*1e6/10)**2
    pctocm = 3.086e18 # 1 pc in cm
    distconv = 4*np.pi*(opts.distance*1e6*pctocm)**2

    data_out = {}
    cnt = 0
    for filename,T0 in zip(filenames,T0s):
        cnt = cnt + 1
        #if cnt > 5: continue

        data_out_temp = lightcurve_utils.loadEventSpec(filename)
        data_out[str(T0)] = data_out_temp

        data_out[str(T0)]["data"] = data_out[str(T0)]["data"]*distconv
        data_out[str(T0)]["error"] = data_out[str(T0)]["error"]*distconv

        data_out[str(T0)]["data"] = scipy.signal.medfilt(data_out[str(T0)]["data"],kernel_size=15)
        data_out[str(T0)]["error"] = scipy.signal.medfilt(data_out[str(T0)]["error"],kernel_size=15)

# These are the default values supplied with respect to generating lightcurves
tini = 0.1
tmax = 14.0
dt = 0.1
lambdaini = 4500
lambdamax = 25000
dlambda = 500.0

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

if opts.analysisType == "posterior":
    # read in samples
    samples = KNTable.read_samples(opts.posterior_samples)
    # limit masses
    samples = samples.mass_cut(mass1=3.0,mass2=3.0)

    print "m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"]))
    print "m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"]))

    # Downsample 
    samples = samples.downsample(Nsamples=100)
    #samples = samples.downsample(Nsamples=5)
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
            print "Not applying an error to mass ejecta"
        elif erroropt == 'log':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.236))
        elif erroropt == 'lin':
            samples['mej'] = np.random.normal(samples['mej'],0.72*samples['mej'])
        elif erroropt == 'loggauss':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.312))
        idx = np.where(samples['mej'] > 0)[0]
        samples = samples[idx]

    samples_all = {}
    for model in models:
        samples_all[model] = samples

elif opts.analysisType == "multinest":
    multinest_samples = opts.multinest_samples.split(",")
    samples_all = {}
    for multinest_sample, model in zip(multinest_samples,models):
        # read multinest samples
        samples = KNTable.read_multinest_samples(multinest_sample, model) 
        samples_all[model] = samples
elif opts.analysisType == "single":
    samples = {}
    samples['mej'] = opts.mej1
    samples['vej'] = opts.vej1
    samples['Xlan'] = opts.Xlan1

    samples['mej_1'] = opts.mej1
    samples['vej_1'] = opts.vej1
    samples['Xlan_1'] = opts.Xlan1
    samples['mej_2'] = opts.mej2
    samples['vej_2'] = opts.vej2
    samples['Xlan_2'] = opts.Xlan2


    t = Table(np.array([[opts.mej1,opts.vej1,opts.Xlan1,opts.mej1,opts.vej1,opts.Xlan1,opts.mej2,opts.vej2,opts.Xlan2]]), names=['mej', 'vej', 'Xlan','mej_1','vej_1','Xlan_1','mej_2','vej_2','Xlan_2'])
    #for key, val in samples.iteritems():
    #    t.add_column(Column(data=[val],name=key))

    t.add_row(np.array([opts.mej2,opts.vej2,opts.Xlan2,opts.mej1,opts.vej1,opts.Xlan1,opts.mej2,opts.vej2,opts.Xlan2]))

    samples = t

    samples_all = {}
    for model in models:
        samples_all[model] = samples

for model in models:   
    if opts.nsamples > 0:
        samples_all[model] = samples_all[model].downsample(Nsamples=opts.nsamples)

    #add default values from above to table
    samples_all[model]['tini'] = tini
    samples_all[model]['tmax'] = tmax
    samples_all[model]['dt'] = dt
    samples_all[model]['lambdaini'] = lambdaini
    samples_all[model]['lambdamax'] = lambdamax
    samples_all[model]['dlambda'] = dlambda
    samples_all[model]['vmin'] = vmin
    samples_all[model]['th'] = th
    samples_all[model]['ph'] = ph
    samples_all[model]['kappa'] = kappa
    samples_all[model]['eps'] = eps
    samples_all[model]['alp'] = alp
    samples_all[model]['eth'] = eth
    samples_all[model]['flgbct'] = flgbct
    samples_all[model]['beta'] = beta
    samples_all[model]['kappa_r'] = kappa_r
    samples_all[model]['slope_r'] = slope_r
    samples_all[model]['theta_r'] = theta_r
    samples_all[model]['Ye'] = Ye

#kwargs = {'SaveModel':True,'LoadModel':False,'ModelPath':ModelPath}
kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = False
kwargs["doSpec"] = True

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
model_tables = {}
for model in models:
    samples = samples_all[model]
    model_tables[model] = KNTable.model(model, samples, **kwargs)

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,"_".join(models))
plotDir = os.path.join(plotDir,"event_spec")
plotDir = os.path.join(plotDir,opts.event)
plotDir = os.path.join(plotDir,opts.analysisType)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
datDir = os.path.join(plotDir,"dat")
if not os.path.isdir(datDir):
    os.makedirs(datDir)

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))
spec_all = {}
for model in models:
    spec_all[model] = {}
    for key, color in zip(keys,colors):
        spec_all[model][key] = np.empty((0,len(data_out[key]["lambda"])))

for model in models:
    for row in model_tables[model]:
        t, lambdas, spec = row["t"], row["lambda"], row["spec"]

        if np.sum(spec) == 0.0:
            #print "No luminosity..."
            continue

        for key, color in zip(keys,colors):
            f = interp.interp2d(t, lambdas, np.log10(spec), kind='cubic')
            flux1 = (10**(f(float(key),lambdas))).T
            #flux1 = (10**(f(float(key),data_out[key]["lambda"]))).T
            flux1 = flux1[0]
            f = interp.interp1d(lambdas, np.log10(flux1), fill_value='extrapolate')
            flux1 = 10**f(data_out[key]["lambda"])
            print(flux1)
            spec_all[model][key] = np.append(spec_all[model][key],[flux1],axis=0)

colors_names=cm.rainbow(np.linspace(0,1,len(models)))
color2 = 'coral'
color1 = 'cornflowerblue'
colors_names=[color1,color2]

plotName = "%s/spec_panels.pdf"%(plotDir)
plotNamePNG = "%s/spec_panels.png"%(plotDir)
plt.figure(figsize=(20,28))

cnt = 0
for key, color in zip(keys,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(keys),1,cnt)
    if cnt == 1:
        #ax1 = plt.subplot(eval(vals))
        ax1 = plt.subplot(len(keys),1,cnt)
    else:
        #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
        ax2 = plt.subplot(len(keys),1,cnt,sharex=ax1,sharey=ax1)

    if opts.doEvent:
        plt.plot(data_out[key]["lambda"],np.log10(data_out[key]["data"]),'k--',linewidth=2)

    for ii, model in enumerate(models):
        legend_name = get_legend(model)

        lambdas = data_out[key]["lambda"]
        for jj, spec_temp in enumerate(spec_all[model][key]):
            specmed = spec_temp 
            specmax = spec_temp*(1+opts.errorbudget)
            specmin = spec_temp*(1-opts.errorbudget)

            plt.plot(lambdas,np.log10(specmed),'--',c=colors_names[ii],linewidth=4,label=legend_name)
            plt.plot(lambdas,np.log10(specmin),'-',c=colors_names[ii],linewidth=4)
            plt.plot(lambdas,np.log10(specmax),'-',c=colors_names[ii],linewidth=4)
            plt.fill_between(lambdas,np.log10(specmin),np.log10(specmax),facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=3)

    plt.fill_between([13500.0,14500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    plt.fill_between([18000.0,19500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    plt.xlim([5000, 25000])
    plt.ylim([35.5,37.9])
    plt.grid()
    plt.yticks(fontsize=36)

    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':40},numpoints=1,shadow=True, fancybox=True)
        l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, prop={'size':48})
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
if len(keys)==1:
    ax1.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
else:
    ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotNamePNG,bbox_inches='tight')
plt.close()
convert_command = "convert %s %s"%(plotNamePNG,plotName)
os.system(convert_command)

plotName = "%s/spec_panels_linear.pdf"%(plotDir)
plotNamePNG = "%s/spec_panels_linear.png"%(plotDir)
#plt.figure(figsize=(20,28))
plt.figure(figsize=(12,8))

cnt = 0
for key, color in zip(keys,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(keys),1,cnt)
    if cnt == 1:
        #ax1 = plt.subplot(eval(vals))
        ax1 = plt.subplot(len(keys),1,cnt)
    else:
        #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
        ax2 = plt.subplot(len(keys),1,cnt,sharex=ax1,sharey=ax1)

    if opts.doEvent:
        valhold = copy.copy(data_out[key]["data"])
        idx = np.where((13500<=data_out[key]["lambda"]) & (data_out[key]["lambda"]<=14500.0))[0]
        valhold[idx] = 0.0
        idx = np.where((18000.0<=data_out[key]["lambda"]) & (data_out[key]["lambda"]<=19500.0))[0]
        valhold[idx] = 0.0
        data_out[key]["data"] = data_out[key]["data"] / np.nanmax(valhold)
        plt.plot(data_out[key]["lambda"],data_out[key]["data"],'k--',linewidth=2)

    for ii, model in enumerate(models):
        legend_name = get_legend(model)

        lambdas = data_out[key]["lambda"]
        for jj, spec_temp in enumerate(spec_all[model][key]):
            spec_temp = spec_temp / np.nanmax(spec_temp)
            if model == "Ka2017":
                if key=='1.4714763104':
                    if jj == 1:
                        spec_temp = spec_temp * 0.8
                if key=='2.4714763104':
                    if jj == 1:
                        spec_temp = spec_temp * 0.8
            specmed = spec_temp
            specmax = spec_temp*(1+opts.errorbudget)
            specmin = spec_temp*(1-opts.errorbudget)

            plt.plot(lambdas,specmed,'--',c=colors_names[ii],linewidth=4,label=legend_name)
            plt.plot(lambdas,specmin,'-',c=colors_names[ii],linewidth=4)
            plt.plot(lambdas,specmax,'-',c=colors_names[ii],linewidth=4)
            plt.fill_between(lambdas,specmin,specmax,facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=3)

    plt.fill_between([13500.0,14500.0],[10**-10.0,10**-10.0],[10**10.0,10**10.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    plt.fill_between([18000.0,19500.0],[10**-10.0,10**-10.0],[10**10.0,10**10.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    plt.xlim([3000, 12500])
    #plt.xlim([3000, 25000])
    #plt.ylim([0.5*10**36.0,3.5*10**37.0])
    plt.ylim([0,1.0])
    plt.grid()
    plt.yticks(fontsize=36)

    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        if len(keys)>1:
            plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':40},numpoints=1,shadow=True, fancybox=True)
        #l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #        mode="expand", borderaxespad=0, ncol=2, prop={'size':48})
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
if len(keys)==1:
    ax1.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
else:
    ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotNamePNG, bbox_inches='tight')
plt.close()
convert_command = "convert %s %s"%(plotNamePNG,plotName)
os.system(convert_command)

