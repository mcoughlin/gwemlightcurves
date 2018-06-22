
import os, sys, glob, copy
import pickle
from time import time
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats, scipy.signal

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

import corner

from astropy.time import Time

import pymultinest
from gwemlightcurves.sampler import *
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
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")

    parser.add_option("-n","--name",default="GW170817")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--doAbsorption",  action="store_true", default=False)
    parser.add_option("--doSplit",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    #parser.add_option("--T0",default="1,2,3,4,5,6,7")
    parser.add_option("--T0",default="57982.5285236896")
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-m","--model",default="Ka2017")
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--errorbudget",default=2.0,type=float)
    parser.add_option("--lambdamax",default=25000,type=int)
    parser.add_option("--lambdamin",default=5000,type=int)

    opts, args = parser.parse_args()

    return opts

def prior_2Component(Xlan1,Xlan2):
    if Xlan1 < Xlan2:
        return 0.0
    else:
        return 1.0

def prior_2ComponentVel(vej_1,vej_2):
    if vej_1 < vej_2:
        return 1.0
    else:
        return 0.0

def generate_spectra(model,samples):

    #kwargs = {'SaveModel':True,'LoadModel':False,'ModelPath':ModelPath}
    kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
    kwargs["doAB"] = False
    kwargs["doSpec"] = True

    t = Table()
    for key, val in samples.iteritems():
        t.add_column(Column(data=[val],name=key))
    samples = t
    model_table = KNTable.model(model, samples, **kwargs)

    if len(model_table) == 0:
        return [], [], []
    else:
        t, lambdas, spec = model_table["t"][0], model_table["lambda"][0], model_table["spec"][0]
        return t, lambdas, spec

def myloglike_Ka2017x2_spec_ejecta_absorption(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    zp = cube[7]

    prior = prior_2Component(Xlan_1,Xlan_2)
    if prior == 0.0:
        return -np.inf
    prior = prior_2ComponentVel(vej_1,vej_2)
    if prior == 0.0:
        return -np.inf

    t, lambdas, spec = Ka2017x2_model_spec_ejecta_absorption(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2)

    prob = calc_prob_spec(t, lambdas, spec, t0, zp)

    return prob

def myloglike_Ka2017x2_spec_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej_1 = 10**cube[1]
    vej_1 = cube[2]
    Xlan_1 = 10**cube[3]
    mej_2 = 10**cube[4]
    vej_2 = cube[5]
    Xlan_2 = 10**cube[6]
    zp = cube[7]

    prior = prior_2Component(Xlan_1,Xlan_2)
    if prior == 0.0:
        return -np.inf
    prior = prior_2ComponentVel(vej_1,vej_2)
    if prior == 0.0:
        return -np.inf

    t, lambdas, spec = Ka2017x2_model_spec_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2)

    prob = calc_prob_spec(t, lambdas, spec, t0, zp)

    return prob

def myloglike_Ka2017_spec_ejecta(cube, ndim, nparams):
    t0 = cube[0]
    mej = 10**cube[1]
    vej = cube[2]
    Xlan = 10**cube[3]
    zp = cube[4]

    t, lambdas, spec = Ka2017_model_spec_ejecta(mej,vej,Xlan)

    prob = calc_prob_spec(t, lambdas, spec, t0, zp)

    return prob

def Ka2017x2_model_spec_ejecta_absorption(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2):

    tini = 0.1
    tmax = 14.0
    dt = 1.0

    lambdaini = 5000
    lambdamax = 25000
    dlambda = 500.0

    samples = {}
    samples['tini'] = tini
    samples['tmax'] = tmax
    samples['dt'] = dt
    samples['lambdaini'] = lambdaini
    samples['lambdamax'] = lambdamax
    samples['dlambda'] = dlambda

    model = "Ka2017"

    samples['mej'] = mej_1
    samples['vej'] = vej_1
    samples['Xlan'] = Xlan_1
    t, lambdas, spec1 = generate_spectra(model,samples)

    samples['mej'] = mej_2
    samples['vej'] = vej_2
    samples['Xlan'] = Xlan_2
    t, lambdas, spec2 = generate_spectra(model,samples)

    idx1 = np.where(lambdas<=9000.0)[0]
    idx2 = np.where(lambdas>9000.0)[0]

    spec = np.zeros(spec1.shape)
    ii = 0
    for sp1, sp2 in zip(spec1.T,spec2.T):
        #sp = np.max(np.vstack((sp1,sp2)),axis=0)
        spec[idx1,ii] = sp2[idx1]
        spec[idx2,ii] = sp1[idx2]
        ii = ii + 1

    return t, lambdas, spec

def Ka2017x2_model_spec_ejecta(mej_1,vej_1,Xlan_1,mej_2,vej_2,Xlan_2):

    tini = 0.1
    tmax = 14.0
    dt = 1.0

    lambdaini = 5000
    lambdamax = 25000
    dlambda = 500.0

    samples = {}
    samples['tini'] = tini
    samples['tmax'] = tmax
    samples['dt'] = dt
    samples['lambdaini'] = lambdaini
    samples['lambdamax'] = lambdamax
    samples['dlambda'] = dlambda
    samples['mej_1'] = mej_1
    samples['vej_1'] = vej_1
    samples['Xlan_1'] = Xlan_1
    samples['mej_2'] = mej_2
    samples['vej_2'] = vej_2
    samples['Xlan_2'] = Xlan_2

    model = "Ka2017x2"
    t, lambdas, spec = generate_spectra(model,samples)

    return t, lambdas, spec

def Ka2017_model_spec_ejecta(mej,vej,Xlan):

    tini = 0.1
    tmax = 14.0
    dt = 1.0

    lambdaini = 5000
    lambdamax = 25000
    dlambda = 500.0

    samples = {}
    samples['tini'] = tini
    samples['tmax'] = tmax
    samples['dt'] = dt
    samples['lambdaini'] = lambdaini
    samples['lambdamax'] = lambdamax
    samples['dlambda'] = dlambda
    samples['mej'] = mej
    samples['vej'] = vej
    samples['Xlan'] = Xlan

    model = "Ka2017"
    t, lambdas, spec = generate_spectra(model,samples)

    return t, lambdas, spec

def calc_prob_spec(t, lambdas, spec, t0, zp):

    count = 0
    chisquare = np.nan

    for key in data_out:
        wav2, flux2, error = data_out[key]["lambda"], data_out[key]["data"], data_out[key]["error"]
        ii = np.where(np.not_equal(flux2,0))[0]
        wav2, flux2, error = wav2[ii], flux2[ii], error[ii]

        sigma_y = np.abs(error/(flux2*np.log(10)))
        sigma = np.sqrt((np.log10(1+opts.errorbudget))**2 + sigma_y**2)

        f = interp.interp2d(t+t0, lambdas, np.log10(spec), kind='cubic')
        flux1 = (10**(f(float(key),wav2))).T

        zp_factor = 10**(zp/-2.5)
        flux1 = flux1*zp_factor

        if Global.doAbsorption:
            #lambdas_lowpass, spec_lowpass, spec_envelope = lightcurve_utils.get_envelope(wav2,flux1[0])
            #flux1 = spec_lowpass/spec_envelope
            flux1 = flux1 / np.nanmax(flux1)
            chisquarevals = ((flux1-flux2)/opts.errorbudget)**2
            chisquarevals = chisquarevals[0]
        else:
            zp_factor = 10**(zp/-2.5)
            flux1 = flux1*zp_factor
            flux1 = np.log10(np.abs(flux1))
            flux2 = np.log10(np.abs(flux2))
            chisquarevals = ((flux1-flux2)/sigma)**2
            chisquarevals = chisquarevals[0]

        chisquaresum = np.sum(chisquarevals)
        chisquaresum = (1/float(len(chisquarevals)-1))*chisquaresum

        if count == 0:
            chisquare = chisquaresum
        else:
            chisquare = chisquare + chisquaresum
        count = count + 1

    if np.isnan(chisquare):
        prob = -np.inf
    else:
        prob = scipy.stats.chi2.logpdf(chisquare, 1, loc=0, scale=1)

    if np.isnan(prob):
        prob = -np.inf

    if prob == 0.0:
        prob = -np.inf

    #if np.isfinite(prob):
    #    print T, F, prob

    return prob

# Parse command line
opts = parse_commandline()

if not opts.model in ["Ka2017","Ka2017x2"]:
   print "Model must be either: Ka2017, Ka2017x2"
   exit(0)

if opts.doFixZPT0:
    ZPRange = 0.1
    T0Range = 0.1
else:
    ZPRange = 5.0
    T0Range = 14.0

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models_spec'
elif opts.doAbsorption:
    basename = 'abs_spec'
elif opts.doSplit:
    basename = 'spl_spec'
else:
    basename = 'gws_spec'
plotDir = os.path.join(baseplotDir,basename)
if opts.doFixZPT0:
    plotDir = os.path.join(plotDir,'%s_FixZPT0'%opts.model)
else:
    plotDir = os.path.join(plotDir,'%s'%opts.model)
plotDir = os.path.join(plotDir,"%d_%d"%(opts.lambdamin,opts.lambdamax))
if opts.name == "knova_d1_n10_m0.005_vk0.20_fd1.0_Xlan1e-3.0":
    plotDir = os.path.join(plotDir,"KaGrid_H4M005V20X-3")
else:
    plotDir = os.path.join(plotDir,opts.name)
if opts.doModels:
    plotDir = os.path.join(plotDir,"_".join(opts.T0.split(",")))
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

specDir = os.path.join(plotDir,'spec')
if not os.path.isdir(specDir):
    os.makedirs(specDir)

dataDir = opts.dataDir
lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir

if opts.doAbsorption:
    Global.doAbsorption = 1

ModelPath = '%s/svdmodels'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

fileDir = os.path.join(opts.outputDir,opts.model)
filenames = glob.glob('%s/*_spec.dat'%fileDir)
specs, names = lightcurve_utils.read_files_spec(filenames)
speckeys = specs.keys()
for key in speckeys:
    f = interp.interp2d(specs[key]["t"],specs[key]["lambda"],specs[key]["data"].T)
    specs[key]["f"] = f

n_live_points = 100
#n_live_points = 10
evidence_tolerance = 0.5
#evidence_tolerance = 10
max_iter = 0

if opts.doModels:
    data_out_all = lightcurve_utils.loadModelsSpec(opts.outputDir,opts.name)
    keys = data_out_all.keys()
    if not opts.name in data_out_all:
        print "%s not in file..."%opts.name
        exit(0)

    data_out_full = data_out_all[opts.name]
    f = interp.interp2d(data_out_full["t"],data_out_full["lambda"],data_out_full["data"].T)

    T0s = opts.T0.split(",")
    data_out = {}
    for T0 in T0s:

        xnew = float(T0)
        ynew = data_out_full["lambda"]
        znew = f(xnew,ynew)
        data_out[T0] = {}
        data_out[T0]["lambda"] = ynew
        data_out[T0]["data"] = np.squeeze(znew)
        data_out[T0]["error"] = np.zeros(data_out[T0]["data"].shape)

        idx = np.where((data_out[T0]["lambda"] >= opts.lambdamin) & (data_out[T0]["lambda"] <= opts.lambdamax))[0]
        data_out[T0]["lambda"] = data_out[T0]["lambda"][idx]
        data_out[T0]["data"] = data_out[T0]["data"][idx]
        data_out[T0]["error"] = data_out[T0]["error"][idx]

elif opts.doEvent:
    filename = "../spectra/%s_spectra_index.dat"%opts.name
    lines = [line.rstrip('\n') for line in open(filename)]
    filenames = []
    T0s = []
    for line in lines:
        lineSplit = line.split(" ")
        #if not lineSplit[0] == opts.name: continue
        filename = "%s/%s"%(spectraDir,lineSplit[1])
        filenames.append(filename)
        mjd = Time(lineSplit[2], format='isot').mjd
        T0s.append(mjd-float(opts.T0))

    if opts.doSplit:
        filenames = filenames[1:3]
        T0s = T0s[1:3]

    #filenames = filenames[:7]
    #T0s = T0s[:7]

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

        idx = np.where((data_out[str(T0)]["lambda"] >= opts.lambdamin) & (data_out[str(T0)]["lambda"] <= opts.lambdamax))[0]
        data_out[str(T0)]["lambda"] = data_out[str(T0)]["lambda"][idx]
        data_out[str(T0)]["data"] = data_out[str(T0)]["data"][idx]
        data_out[str(T0)]["error"] = data_out[str(T0)]["error"][idx]

        data_out[str(T0)]["data"] = data_out[str(T0)]["data"]*distconv
        data_out[str(T0)]["error"] = data_out[str(T0)]["error"]*distconv

        data_out[str(T0)]["data"] = scipy.signal.medfilt(data_out[str(T0)]["data"],kernel_size=15)
        data_out[str(T0)]["error"] = scipy.signal.medfilt(data_out[str(T0)]["error"],kernel_size=15)

        idx = np.where((data_out[str(T0)]["lambda"] >= 13000) & (data_out[str(T0)]["lambda"] <= 15000))[0]
        data_out[str(T0)]["error"][idx] = np.inf
        idx = np.where((data_out[str(T0)]["lambda"] >= 17900) & (data_out[str(T0)]["lambda"] <= 19700))[0]
        data_out[str(T0)]["error"][idx] = np.inf

elif opts.doAbsorption:
    filename = "../spectra/%s_spectra_index.dat"%opts.name
    lines = [line.rstrip('\n') for line in open(filename)]
    filenames = []
    T0s = []
    for line in lines:
        lineSplit = line.split(" ")
        filename = "%s/%s"%(spectraDir,lineSplit[1])
        filenames.append(filename)
        mjd = Time(lineSplit[2], format='isot').mjd
        T0s.append(mjd-float(opts.T0))

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

        idx = np.where((data_out[str(T0)]["lambda"] >= opts.lambdamin) & (data_out[str(T0)]["lambda"] <= opts.lambdamax))[0]
        data_out[str(T0)]["lambda"] = data_out[str(T0)]["lambda"][idx]
        data_out[str(T0)]["data"] = data_out[str(T0)]["data"][idx]
        data_out[str(T0)]["error"] = data_out[str(T0)]["error"][idx]

        data_out[str(T0)]["data"] = data_out[str(T0)]["data"]*distconv
        data_out[str(T0)]["error"] = data_out[str(T0)]["error"]*distconv

        data_out[str(T0)]["data"] = scipy.signal.medfilt(data_out[str(T0)]["data"],kernel_size=15)
        data_out[str(T0)]["error"] = scipy.signal.medfilt(data_out[str(T0)]["error"],kernel_size=15)

        data_out[str(T0)]["lambda_orig"] = data_out[str(T0)]["lambda"]
        data_out[str(T0)]["data_orig"] = data_out[str(T0)]["data"]
        data_out[str(T0)]["error_orig"] = data_out[str(T0)]["error"]

        valhold = copy.copy(data_out[str(T0)]["data"])
        idx = np.where((13500<=data_out[str(T0)]["lambda"]) & (data_out[str(T0)]["lambda"]<=14500.0))[0]
        valhold[idx] = 0.0
        idx = np.where((18000.0<=data_out[str(T0)]["lambda"]) & (data_out[str(T0)]["lambda"]<=19500.0))[0]
        valhold[idx] = 0.0
        data_out[str(T0)]["data"] = data_out[str(T0)]["data"] / np.nanmax(valhold)
      
        #lambdas_lowpass, spec_lowpass, spec_envelope = lightcurve_utils.get_envelope(data_out[str(T0)]["lambda"],data_out[str(T0)]["data"])
        #f = interp.interp1d(data_out[str(T0)]["lambda"],data_out[str(T0)]["error"], fill_value='extrapolate', kind = 'quadratic')
        #error = f(lambdas_lowpass)
        #data_out[str(T0)]["lambda"] = lambdas_lowpass
        #data_out[str(T0)]["data"] = spec_lowpass/spec_envelope
        #data_out[str(T0)]["error"] = error*data_out[str(T0)]["data"]

        #idx = np.where((data_out[str(T0)]["lambda"] >= 13000) & (data_out[str(T0)]["lambda"] <= 15000))[0]
        #data_out[str(T0)]["error"][idx] = np.inf
        #idx = np.where((data_out[str(T0)]["lambda"] >= 17900) & (data_out[str(T0)]["lambda"] <= 19700))[0]
        #data_out[str(T0)]["error"][idx] = np.inf

else:
    print "Must enable --doModels, --doAbsorption, or --doEvent"
    exit(0)

Global.ZPRange = ZPRange
Global.T0Range = T0Range

if opts.model == "Ka2017":
    parameters = ["t0","mej","vej","xlan","zp"]
    labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej})$",r"$v_{\rm ej}$",r"${\rm log}_{10} (Xlan)$","ZP"]
    n_params = len(parameters)

    pymultinest.run(myloglike_Ka2017_spec_ejecta, myprior_Ka2017_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
elif opts.model == "Ka2017x2":
    parameters = ["t0","mej1","vej1","xlan1","mej2","vej2","xlan2","zp"]
    labels = [r"$T_0$",r"${\rm log}_{10} (M_{\rm ej 1})$",r"$v_{\rm ej 1}$",r"${\rm log}_{10} (Xlan_1)$",r"${\rm log}_{10} (M_{\rm ej 2})$",r"$v_{\rm ej 2}$",r"${\rm log}_{10} (Xlan_2)$","ZP"]
    n_params = len(parameters)
    if opts.doAbsorption:
        pymultinest.run(myloglike_Ka2017x2_spec_ejecta_absorption, myprior_Ka2017x2_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
    elif opts.doSplit:
        pymultinest.run(myloglike_Ka2017x2_spec_ejecta_absorption, myprior_Ka2017x2_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
    else:
        pymultinest.run(myloglike_Ka2017x2_spec_ejecta, myprior_Ka2017x2_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)
else:
    print "Not implemented..."
    exit(0)

#multifile= os.path.join(plotDir,'2-.txt')
multifile = lightcurve_utils.get_post_file(plotDir)
data = np.loadtxt(multifile)

if opts.model == "Ka2017":
    t0, mej, vej, Xlan, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], data[:,4], data[:,5]
    idx = np.argmax(loglikelihood)
    t0_best, mej_best, vej_best, Xlan_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], data[idx,4]
    t_best, lambdas_best, spec_best = Ka2017_model_spec_ejecta(mej_best,vej_best,Xlan_best)
elif opts.model == "Ka2017x2":
    t0, mej_1, vej_1, Xlan_1, mej_2, vej_2, Xlan_2, zp, loglikelihood = data[:,0], 10**data[:,1], data[:,2], 10**data[:,3], 10**data[:,4], data[:,5], 10**data[:,6], data[:,7], data[:,8]
    idx = np.argmax(loglikelihood)
    t0_best, mej_1_best, vej_1_best, Xlan_1_best, mej_2_best, vej_2_best, Xlan_2_best, zp_best = data[idx,0], 10**data[idx,1], data[idx,2], 10**data[idx,3], 10**data[idx,4], data[idx,5], 10**data[idx,6], data[idx,7]

    if opts.doAbsorption:
        t_best, lambdas_best, spec_best = Ka2017x2_model_spec_ejecta_absorption(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best)
    else:
        t_best, lambdas_best, spec_best = Ka2017x2_model_spec_ejecta(mej_1_best,vej_1_best,Xlan_1_best,mej_2_best,vej_2_best,Xlan_2_best) 

truths = lightcurve_utils.get_truths(opts.name,opts.model,n_params,True)

pcklFile = os.path.join(plotDir,"data.pkl")
f = open(pcklFile, 'wb')
pickle.dump((data_out, data, t_best, lambdas_best, spec_best, t0_best, zp_best, n_params, labels, truths), f)
f.close()

if n_params >= 8:
    title_fontsize = 26
    label_fontsize = 30
else:
    title_fontsize = 24
    label_fontsize = 28

plotName = "%s/corner.pdf"%(plotDir)
if opts.doFixZPT0:
    figure = corner.corner(data[:,1:-2], labels=labels[1:-1],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       truths=truths[1:-1], smooth=3)
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       truths=truths, smooth=3)
if n_params >= 8:
    figure.set_size_inches(18.0,18.0)
else:
    figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

spec_best_dic = {}
keys = sorted(data_out.keys())
for ii,key in enumerate(keys):

    f = interp.interp2d(t_best+t0_best, lambdas_best, np.log10(spec_best), kind='cubic')
    flux1 = (10**(f(float(key),data_out[key]["lambda"]))).T
    zp_factor = 10**(zp_best/-2.5)
    flux1 = flux1*zp_factor
    spec_best_dic[key] = {}
    spec_best_dic[key]["lambda"] = data_out[key]["lambda"]
    spec_best_dic[key]["data"] = np.squeeze(flux1)

    filename = os.path.join(specDir,"spec_%s.dat"%key)
    fid = open(filename, 'wb')
    for l,s in zip(spec_best_dic[key]["lambda"],spec_best_dic[key]["data"]):
        fid.write("%.5f %.5e\n"%(l,s))
    fid.close()

if opts.doAbsorption:
    for ii,key in enumerate(keys):
        spec_best_dic[key]["lambda_orig"] = spec_best_dic[key]["lambda"]
        spec_best_dic[key]["data_orig"] = spec_best_dic[key]["data"] 

        spec_best_dic[key]["data"] = spec_best_dic[key]["data"] / np.nanmax(spec_best_dic[key]["data"])

        #lambdas_lowpass, spec_lowpass, spec_envelope = lightcurve_utils.get_envelope(spec_best_dic[key]["lambda"],spec_best_dic[key]["data"])
        #flux1 = spec_lowpass/spec_envelope
        #spec_best_dic[key]["lambda"] = lambdas_lowpass
        #spec_best_dic[key]["data"] = flux1

plotName = "%s/spec.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for key in data_out.keys():
    plt.semilogy(data_out[key]["lambda"],data_out[key]["data"],'r-',linewidth=2)
    plt.semilogy(spec_best_dic[key]["lambda"],spec_best_dic[key]["data"],'k--',linewidth=2)
plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
#plt.legend(loc="best",prop={'size':16},numpoints=1)
plt.grid()
plt.savefig(plotName)
plt.close()

if opts.doAbsorption:
    plotName = "%s/spec_absorption.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for key in data_out.keys():
        plt.semilogy(data_out[key]["lambda_orig"],data_out[key]["data_orig"],'r-',linewidth=2)
        plt.semilogy(spec_best_dic[key]["lambda_orig"],spec_best_dic[key]["data_orig"],'k--',linewidth=2)
    plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
    plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
    #plt.legend(loc="best",prop={'size':16},numpoints=1)
    plt.grid()
    plt.savefig(plotName)
    plt.close()

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotName = "%s/spec_panels.pdf"%(plotDir)
fig = plt.figure(figsize=(22,28))

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

    lambdas = spec_best_dic[key]["lambda"]
    specmed = spec_best_dic[key]["data"]
    specmin = spec_best_dic[key]["data"]/opts.errorbudget
    specmax = spec_best_dic[key]["data"]*opts.errorbudget

    if opts.doAbsorption:
        plt.plot(data_out[key]["lambda"],data_out[key]["data"],'k--',linewidth=2)
        plt.plot(lambdas,specmed,'--',c=color,linewidth=4)

        plt.fill_between([13000.0,15000.0],[0.0,0.0],[1.0,1.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
        plt.fill_between([17900.0,19700.0],[0.0,-100.0],[1.0,1.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    else:
        plt.plot(data_out[key]["lambda"],np.log10(data_out[key]["data"]),'k--',linewidth=2)

        plt.plot(lambdas,np.log10(specmed),'--',c=color,linewidth=4)
        plt.plot(lambdas,np.log10(specmin),'-',c=color,linewidth=4)
        plt.plot(lambdas,np.log10(specmax),'-',c=color,linewidth=4)
        plt.fill_between(lambdas,np.log10(specmin),np.log10(specmax),facecolor=color,edgecolor=color,alpha=0.2,linewidth=3)

        plt.fill_between([13500.0,14500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
        plt.fill_between([18000.0,19500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    plt.xlim([opts.lambdamin,opts.lambdamax])
    if opts.doAbsorption:
        plt.ylim([0.0,1.0])
    else:
        plt.ylim([35.0,39.0])
    plt.grid()
    plt.yticks(fontsize=36)
  
    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotName)
plt.close()

if opts.doAbsorption:
    plotName = "%s/spec_panels_absorption.pdf"%(plotDir)
    fig = plt.figure(figsize=(22,28))
    
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
     
        lambdas = spec_best_dic[key]["lambda_orig"]
        specmed = spec_best_dic[key]["data_orig"]
        specmin = spec_best_dic[key]["data_orig"]/2.0
        specmax = spec_best_dic[key]["data_orig"]*2.0
    
        plt.plot(data_out[key]["lambda_orig"],np.log10(data_out[key]["data_orig"]),'k--',linewidth=2)
     
        plt.plot(lambdas,np.log10(specmed),'--',c=color,linewidth=4)
        plt.plot(lambdas,np.log10(specmin),'-',c=color,linewidth=4)
        plt.plot(lambdas,np.log10(specmax),'-',c=color,linewidth=4)
        plt.fill_between(lambdas,np.log10(specmin),np.log10(specmax),facecolor=color,edgecolor=color,alpha=0.2,linewidth=3)
     
        plt.fill_between([13500.0,14500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
        plt.fill_between([18000.0,19500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    
        plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
        plt.xlim([opts.lambdamin,opts.lambdamax])
        plt.ylim([35.0,39.0])
        plt.grid()
        plt.yticks(fontsize=36)
     
        if (not cnt == len(keys)) and (not cnt == 1):
            plt.setp(ax2.get_xticklabels(), visible=False)
        elif cnt == 1:
            plt.setp(ax1.get_xticklabels(), visible=False)
        else:
            plt.xticks(fontsize=36)
    
    ax1.set_zorder(1)
    ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
    plt.savefig(plotName)
    plt.close()
