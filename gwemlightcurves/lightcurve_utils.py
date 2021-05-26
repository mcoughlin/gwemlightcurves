
import os, sys
import optparse
import numpy as np
import glob
import scipy.stats

from scipy.interpolate import interpolate as interp
from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.signal import argrelextrema

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table

def loadModelsSpec(outputDir,name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog","kasen_kilonova_grid","Ka2017"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Rosswog et al. (2017)","Kasen et al. (2017)","Kasen et al. (2017)"]

    filenames = []
    legend_names = []
    for ii,model in enumerate(models):
        filename = '%s/%s/%s_spec.dat'%(outputDir,model,name)
        if not os.path.isfile(filename):
            continue
        filenames.append(filename)
        legend_names.append(models_ref[ii])
        break
    specs, names = read_files_spec(filenames)

    return specs

def getLegend(outputDir,names):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog","Afterglow","metzger_rprocess","korobkin_kilonova","Blue"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2015)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich and Ujevic (2017)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Van Eerten et al. (2012)","Metzger et al. (2010)","Wollaeger et al. (2017)","Metzger (2017)"]

    filenames = []
    legend_names = []
    for name in names:
        for ii,model in enumerate(models):
            filename = '%s/%s/%s.dat'%(outputDir,model,name)
            if not os.path.isfile(filename):
                continue
            filenames.append(filename)
            legend_names.append(models_ref[ii])
            break

    return legend_names

def loadModels(outputDir,name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog","Blue","Arnett","kasen_kilonova_survey","kasen_kilonova_2D","kasen_kilonova_grid"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich et al. (2016)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Metzger (2017)", "Inserra et al. (2013)", "Kasen (2017)","Kasen (2017)","Kasen (2017)"]

    filenames = []
    legend_names = []
    for ii,model in enumerate(models):
        filename = '%s/%s/%s.dat'%(outputDir,model,name)
        if not os.path.isfile(filename):
            continue
        filenames.append(filename)
        legend_names.append(models_ref[ii])
        break
    mags, names = read_files(filenames)

    return mags

def loadModelsLbol(outputDir,name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog","Blue","Arnett"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich et al. (2016)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Metzger (2017)", "Inserra et al. (2013)"]

    filenames = []
    legend_names = []
    for ii,model in enumerate(models):
        filename = '%s/%s/%s_Lbol.dat'%(outputDir,model,name)
        if not os.path.isfile(filename):
            continue
        filenames.append(filename)
        legend_names.append(models_ref[ii])
        break
    Lbols, names = read_files_lbol(filenames)

    return Lbols

def loadEvent(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = filter(None,lines)

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        lineSplit = list(filter(None,lineSplit))
        mjd = Time(lineSplit[0], format='isot').mjd
        filt = lineSplit[1]
        mag = float(lineSplit[2])
        dmag = float(lineSplit[3])

        if not filt in data:
            data[filt] = np.empty((0,3), float)
        data[filt] = np.append(data[filt],np.array([[mjd,mag,dmag]]),axis=0)

    return data

def loadEventSpec(filename):

    name = filename.split("/")[-1].split(".")[0]
    nameSplit = name.split("_")

    data_out = np.loadtxt(filename)
    spec = {}

    spec["lambda"] = data_out[:,0] # Angstroms
    spec["data"] = np.abs(data_out[:,1]) # ergs/s/cm2./Angs 
    spec["error"] = np.zeros(spec["data"].shape) # ergs/s/cm2./Angs
    spec["error"][:-1] = np.abs(np.diff(spec["data"]))
    spec["error"][-1] = spec["error"][-2]
    idx = np.where(spec["error"] <= 0.5*spec["data"])[0]
    spec["error"][idx] = 0.5*spec["data"][idx]

    return spec

def loadEventPhot(filename):

    cspeed = 2.99792458*10**18 # in AA/sec

    central_filters = {'SWIFT_V':5468,
               'SWIFT_B':4392,
               'SWIFT_U':3465,
               'SWIFT_UVW1':2600,
               'SWIFT_UVM2':2246,
               'SWIFT_UVW2':1928,
               'SDSS_u':3543,
               'PS1_g':4775.6,
               'PS1_r':6129.5,
               'PS1_i':7484.6,
               'PS1_z':8657.8,
               'PS1_y':9603.1,
               'GROND_g':4504.5,
               'GROND_r':6098.0,
               'GROND_i':7604.7,
               'GROND_z':8929.3,
               'GROND_J':12246.5,
               'GROND_H':16330.2,
               'GROND_K':21550.4,
              }

    filters = ['SDSS_u','PS1_g','PS1_r','PS1_i','PS1_z','PS1_y',
               'GROND_J','GROND_H','GROND_K']
    reddening = [0.523,0.39,0.28,0.21,0.16,0.13,0.09,0.06,0.04]
    indexes = [2,4,6,8,10,12,14,16,18]

    data_out = np.loadtxt(filename)
    data = {}
    for row in data_out:
        mjd, phase = row[0], row[1]
        wavelengths, mags, dmags, limits = [], [], [], []
        for filt,red,idx in zip(filters,reddening,indexes):
            if np.isnan(row[idx]): continue
            if row[idx+1] == 9999:
                continue
                dmags.append(np.inf)
                limits.append(1)
            else:
                wavelengths.append(central_filters[filt])
                mags.append(row[idx]-red)
                dmags.append(row[idx+1])  
                limits.append(0)  
        if len(wavelengths) < 2: continue
        wavelengths = np.array(wavelengths)
        mags = np.array(mags)
        dmags = np.array(dmags) 
        limits = np.array(limits)

        data[phase] = {}
        data[phase]["mjd"] = mjd
        data[phase]["phase"] = phase
        data[phase]["wavelengths"] = wavelengths
        data[phase]["mags"] = mags
        data[phase]["dmags"] = dmags
        data[phase]["limits"] = limits
        data[phase]["fnu"] = 10**(-0.4*(data[phase]["mags"]+48.6))
        data[phase]["flam"] = data[phase]["fnu"] * cspeed / data[phase]["wavelengths"]**2
        data[phase]["fnuerr"] = 0.921*data[phase]["fnu"]*data[phase]["dmags"]
        data[phase]["flamerr"] = data[phase]["fnuerr"] * cspeed / data[phase]["wavelengths"]**2

    return data

def loadEventLbol(filename):

    data_out = np.loadtxt(filename)

    data = {}
    data["tt"] = data_out[:,0]
    data["Lbol"] = data_out[:,5]
    data["Lbol_err"] = np.max(np.vstack((data_out[:,6],data_out[:,7])),axis=0)
    data["Lbol_err_up"] = data_out[:,6]
    data["Lbol_err_down"] = data_out[:,7]
    data["T"] = data_out[:,1]
    data["T_err"] = data_out[:,2]
    data["R"] = data_out[:,3]
    data["R_err"] = data_out[:,4]

    return data

def loadLightcurves(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    lines = filter(None,lines)

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        numid = float(lineSplit[0])
        psid = lineSplit[1]
        filt = lineSplit[2]
        mjd = float(lineSplit[3])
        mag = float(lineSplit[4])
        dmag = float(lineSplit[5])

        if not psid in data:
            data[psid] = {}
        if not filt in data[psid]:
            data[psid][filt] = np.empty((0,3), float)
        data[psid][filt] = np.append(data[psid][filt],np.array([[mjd,mag,dmag]]),axis=0)

    return data

def event(dataDir,name):

    filename_samples = '%s/event_data/%s.dat'%(dataDir,name)
    if not os.path.isfile(filename_samples):
        return {}

    data_out = read_posterior_samples(filename_samples)
    return data_out

def going_the_distance(dataDir,name):

    directory = '%s/going-the-distance_data/2015/compare/%s'%(dataDir,name)
    filename_samples = os.path.join(directory,'lalinference_nest/posterior_samples.dat')
    if not os.path.isfile(filename_samples):
        return {}

    data_out = read_posterior_samples(filename_samples)
    return data_out

def massgap(dataDir,name):

    directory = '%s/massgap_data/%s/postblue'%(dataDir,name)
    filename_samples = os.path.join(directory,'posterior_samples.dat')
    if not os.path.isfile(filename_samples):
        return {}

    data_out = read_posterior_samples(filename_samples)

    filename = "%s/massgap_data/injected_values.txt"%(dataDir)
    data = np.loadtxt(filename)
    injnum = data[:,0].astype(int)
    snr = data[:,1]
    m1 = data[:,2]
    m2 = data[:,3]
    q = 1/data[:,4]
    eff_spin = data[:,5]
    totmass = data[:,6]
    a1 = data[:,7]
    a2 = data[:,8]

    idx = int(name)
    truths = {}
    truths["snr"] = snr[idx]
    truths["m1"] = m1[idx]
    truths["m2"] = m2[idx]
    truths["q"] = q[idx]
    truths["a1"] = a1[idx]
    truths["a2"] = a2[idx]

    return data_out, truths

def read_posterior_samples_old(filename_samples):

    data_out = {}
    lines = [line.rstrip('\n') for line in open(filename_samples)]
    data = np.loadtxt(filename_samples,skiprows=1)
    line = lines[0]
    params = line.split("\t")
    params = filter(None, params)

    for ii in range(len(params)):
        param = params[ii]
  
        data_out[param] = data[:,ii]

    return data_out

def read_posterior_samples(filename_samples):

    data_out = Table.read(filename_samples, format='ascii')

    if 'mass_1_source' in list(data_out.columns):
        data_out['m1'] = data_out['mass_1_source']
    if 'mass_2_source' in list(data_out.columns):
        data_out['m2'] = data_out['mass_2_source']

    data_out['mchirp'], data_out['eta'], data_out['q'] = ms2mc(data_out['m1'], data_out['m2'])
    data_out['q'] = 1.0/data_out['q']

    return data_out

def read_files_lbol(files):

    names = []
    Lbols = {}
    for filename in files:
        name = filename.replace("_Lbol.txt","").replace("_Lbol.dat","").split("/")[-1]
        Lbol_d = np.loadtxt(filename)

        Lbols[name] = {}
        Lbols[name]["tt"] = Lbol_d[:,0]
        Lbols[name]["Lbol"] = Lbol_d[:,1]

        names.append(name)

    return Lbols, names

def read_files_spec(files):

    names = []
    specs = {}
    for filename in files:
        name = filename.replace("_spec","").replace(".spec","").replace(".txt","").replace(".dat","").split("/")[-1]
        data_out = np.loadtxt(filename)
        t_d, lambda_d, spec_d = data_out[1:,0], data_out[0,1:], data_out[1:,1:]

        specs[name] = {}
        specs[name]["t"] = t_d
        specs[name]["lambda"] = lambda_d
        specs[name]["data"] = spec_d

        names.append(name)

    return specs, names

def read_files(files):

    names = []
    mags = {}
    for filename in files:
        name = filename.replace(".txt","").replace(".dat","").split("/")[-1]
        mag_d = np.loadtxt(filename)

        t = mag_d[:,0]
        mags[name] = {}
        mags[name]["t"] = mag_d[:,0]
        mags[name]["u"] = mag_d[:,1]
        mags[name]["g"] = mag_d[:,2]
        mags[name]["r"] = mag_d[:,3]
        mags[name]["i"] = mag_d[:,4]
        mags[name]["z"] = mag_d[:,5]
        mags[name]["y"] = mag_d[:,6]
        mags[name]["J"] = mag_d[:,7]
        mags[name]["H"] = mag_d[:,8]
        mags[name]["K"] = mag_d[:,9]

        names.append(name)

    return mags, names

def xcorr_mags(mags1,mags2):
    nmags1 = len(mags1)
    nmags2 = len(mags2)
    xcorrvals = np.zeros((nmags1,nmags2))
    chisquarevals = np.zeros((nmags1,nmags2))
    for ii,name1 in enumerate(mags1.iterkeys()):
        for jj,name2 in enumerate(mags2.iterkeys()):

            t1 = mags1[name1]["t"]
            t2 = mags2[name2]["t"]
            t = np.unique(np.append(t1,t2))
            t = np.arange(-100,100,0.1)      

            mag1 = np.interp(t, t1, mags1[name1]["g"])
            mag2 = np.interp(t, t2, mags2[name2]["g"])

            indexes1 = np.where(~np.isnan(mag1))[0]
            indexes2 = np.where(~np.isnan(mag2))[0]
            indexes = np.intersect1d(indexes1,indexes2)
            mag1 = mag1[indexes1]
            mag2 = mag2[indexes2]

            indexes1 = np.where(~np.isinf(mag1))[0]
            indexes2 = np.where(~np.isinf(mag2))[0]
            indexes = np.intersect1d(indexes1,indexes2)
            mag1 = mag1[indexes1]
            mag2 = mag2[indexes2]

            if len(indexes) == 0:
                xcorrvals[ii,jj] = 0.0
                continue           

            if len(mag1) < len(mag2):
                mag1vals = (mag1 - np.mean(mag1)) / (np.std(mag1) * len(mag1))
                mag2vals = (mag2 - np.mean(mag2)) / (np.std(mag2))
            else:
                mag1vals = (mag1 - np.mean(mag1)) / (np.std(mag1))
                mag2vals = (mag2 - np.mean(mag2)) / (np.std(mag2) * len(mag2))

            xcorr = np.correlate(mag1vals, mag2vals, mode='full')
            xcorr_corr = np.max(np.abs(xcorr))

            #mag1 = mag1 * 100.0 / np.sum(mag1)
            #mag2 = mag2 * 100.0 / np.sum(mag2)

            nslides = len(mag1) - len(mag1)
            if nslides == 0:
                chisquares = scipy.stats.chisquare(mag1, f_exp=mag1)[0]
            elif nslides > 0:
                chisquares = []
                for kk in range(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag1, f_exp=mag2[kk:len(mag1)])[0] 
                    chisquares.append(chisquare)
            elif nslides < 0:
                chisquares = []
                for kk in range(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag2, f_exp=mag1[kk:len(mag2)])[0] 
                    chisquares.append(chisquare)

            xcorrvals[ii,jj] = xcorr_corr
            chisquarevals[ii,jj] = np.min(np.abs(chisquares))

    return xcorrvals, chisquarevals

def norm_sym_ratio(eta):
    # Assume floating point precision issues
    #if np.any(np.isclose(eta, 0.25)):
    #eta[np.isclose(eta, 0.25)] = 0.25

    # Assert phyisicality
    assert np.all(eta <= 0.25)

    return np.sqrt(1 - 4. * eta)

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

def hist_results(samples,Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-10
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

def weighted_hist_results(samples, weights, Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, weights=weights, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-10
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)



def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-pos*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def EOSfit(mns,c):
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb

def get_truths(name,model,n_params,doEjecta):
    truths = []
    for ii in range(n_params):
        #truths.append(False)
        truths.append(np.nan)

    if not model in ["DiUj2017","KaKy2016","Me2017","SmCh2017","WoKo2017"]:
        return truths

    if not doEjecta:
        return truths

    if name == "DiUj2017_H4M005V20":
        truths = [0,np.log10(0.005),0.2,0.2,3.14,0.0]
    elif name == "KaKy2016_H4M005V20":
        truths = [0,np.log10(0.005),0.2,0.2,3.14,0.0]
    elif name == "rpft_m005_v2":
        truths = [0,np.log10(0.005),0.2,False,False,False]
    elif name == "rpft_m05_v2":
        truths = [0,np.log10(0.05),0.2,False,False,False]
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
    elif name == "SED_ns12ns12_kappa10":
        truths = [0,np.log10(0.0079), 0.12,False,False,False]
    elif name == "Ka2017_H4M005V20X-3":
        truths = [0,np.log10(0.005),0.2,-3.0,0.0]
    return truths

def get_macronovae_rosswog(name):
    
    if name == "SED_wind1":
        params = [0.01, 0.05, 0.3]
    elif name == "SED_wind2":
        params = [0.01, 0.05, 0.25]
    elif name == "SED_wind3":
        params = [0.01, 0.05, 0.35]
    elif name == "SED_wind4":
        params = [0.05, 0.05, 0.25]
    elif name == "SED_wind5":
        params = [0.05, 0.05, 0.3]
    elif name == "SED_wind6":
        params = [0.05, 0.05, 0.35]
    elif name == "SED_wind7":
        params = [0.05, 0.1, 0.25]
    elif name == "SED_wind8":
        params = [0.05, 0.1, 0.3]
    elif name == "SED_wind9":
        params = [0.1, 0.1, 0.25]
    elif name == "SED_wind10":
        params = [0.01, 0.1, 0.25]
    elif name == "SED_wind11":
        params = [0.01, 0.25, 0.25]
    elif name == "SED_wind12": 
        params = [0.01, 0.5, 0.25]
    elif name == "SED_wind13":
        params = [0.1, 0.01, 0.35]
    elif name == "SED_wind14":
        params = [0.1, 0.05, 0.3]
    elif name == "SED_wind15":
        params = [0.2, 0.01, 0.35]
    elif name == "SED_wind16":
        params = [0.2, 0.05, 0.3]
    elif name == "SED_wind17":
        params = [0.2, 0.1, 0.25]
    elif name == "SED_wind18":
        params = [0.01, 0.01, 0.35]
    elif name == "SED_wind19":
        params = [0.05, 0.25, 0.25]
    elif name == "SED_wind20":
        params = [0.1, 0.25, 0.25]
    elif name == "SED_wind21":
        params = [0.2, 0.25, 0.25]
    else:
        params = [-1,-1,-1]
    return params

def calc_peak_mags(model_table, filts=["u","g","r","i","z","y","J","H","K"], magidxs=[0,1,2,3,4,5,6,7,8]):
    """
    # Peak magnitudes and times in each band"
    """

    # Initiaize peak mag dictionarts
    model_table_tts = {}
    model_table_mags = {}
    #model_table_appmags = {}
    for filt, magidx in zip(filts, magidxs):
        model_table_tts[filt] = []
        model_table_mags[filt] = []
        #model_table_appmags[filt] = []

    for row in model_table:
        t, lbol, mag = row["t"], row["lbol"], row["mag"]
        for filt, magidx in zip(filts,magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0]
            if len(idx) == 0:
                model_table_tts[filt].append(np.nan)
                model_table_mags[filt].append(np.nan)
                #model_table_appmags[filt].append(np.nan)
            else:
                ii = np.argmin(mag[magidx][idx])
                model_table_tts[filt].append(t[idx][ii])
                model_table_mags[filt].append(mag[magidx][idx][ii])
                #model_table_appmags[filt].append(mag[magidx][idx][ii]+5*(np.log10(row["dist"]*1e6) - 1))

    for filt, magidx in zip(filts, magidxs):
        model_table["peak_tt_%s"%filt] = model_table_tts[filt]
        model_table["peak_mag_%s"%filt] = model_table_mags[filt]        
        #model_table["peak_appmag_%s"%filt] = model_table_appmags[filt]  

    return model_table


def interpolate_mags_lbol(model_table, filts=["u","g","r","i","z","y","J","H","K"], magidxs=[0,1,2,3,4,5,6,7,8]):
    """
    """
    from scipy.interpolate import interpolate as interp
    tt = np.arange(model_table['tini'][0], model_table['tmax'][0] + model_table['dt'][0], model_table['dt'][0])
    mag_all = {}
    lbol_all = np.empty((0, len(tt)), float)

    for filt in filts:
        mag_all[filt] = np.empty((0,len(tt)))

    for row in model_table:
        t, lbol, mag = row["t"], row["lbol"], row["mag"]

        if np.sum(lbol) == 0.0:
            continue

        allfilts = True
        for filt, magidx in zip(filts, magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0]
            if len(idx) == 0:
                allfilts = False
                break

        if not allfilts: continue

        for filt, magidx in zip(filts, magidxs):
            idx = np.where(~np.isnan(mag[magidx]))[0]
            f = interp.interp1d(t[idx], mag[magidx][idx], fill_value='extrapolate')
            maginterp = f(tt)
            mag_all[filt] = np.append(mag_all[filt], [maginterp], axis=0)

        idx = np.where((~np.isnan(np.log10(lbol))) & ~(lbol==0))[0]
        f = interp.interp1d(t[idx], np.log10(lbol[idx]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbol_all = np.append(lbol_all, [lbolinterp], axis=0)

    # Ad to model table
    model_table["lbol"] = lbol_all
    for filt in filts:
        model_table["lbol"] = lbol
        model_table["mag_%s"%filt] = mag_all[filt]

    return model_table


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
        legend_name = "Kasen (2017)"
    elif model == "RoFe2017":
        legend_name = "Rosswog et al. (2017)"

    return legend_name

def get_mag(mag,key):
    if key == "u":
        magave = 1.0*mag[0]
    elif key == "g":
        magave = 1.0*mag[1]
    elif key == "r":
        magave = 1.0*mag[2]
    elif key == "i":
        magave = 1.0*mag[3]
    elif key == "z":
        magave = 1.0*mag[4]
    elif key == "y":
        magave = 1.0*mag[5]
    elif key == "J":
        magave = 1.0*mag[6]
    elif key == "H":
        magave = 1.0*mag[7]
    elif key == "K":
        magave = 1.0*mag[8]
    elif key == "w": 
        magave = (mag[1]+mag[2]+mag[3])/3.0
    elif key in ["U","UVW2","UVW1","UVM2"]:
        magave = 1.0*mag[0]
    elif key == "B":
        magave = 1.0*mag[1]
    elif key in ["c","V","F606W"]:
        magave = (mag[1]+mag[2])/2.0
    elif key == "o":
        magave = (mag[2]+mag[3])/2.0
    elif key == "R":
        magave = 1.0*mag[4]
    elif key in ["I","F814W"]:
        magave = (mag[4]+mag[5])/2.0
    elif key == "F160W":
        magave = 1.0*mag[7]
    return magave

def get_med(magtable, errorbudget = 0.0, filts = ["u","g","r","i","z","y","J","H","K"]):

    mag_all = {}
    med_all = {}
    for ii, filt in enumerate(filts):
        med_all[filt] = {}
        for jj, row in enumerate(magtable):
            t, lbol, mag = row["t"], row["lbol"], row["mag"]
            if (jj==0):
                mag_all[filt] = np.empty((0,len(t)))

            maginterp = get_mag(mag,filt)
            mag_all[filt] = np.append(mag_all[filt],[maginterp],axis=0)
        magmed = np.percentile(mag_all[filt], 50, axis=0)
        magmax = np.percentile(mag_all[filt], 90, axis=0) + errorbudget
        magmin = np.percentile(mag_all[filt], 10, axis=0) - errorbudget
        magmax2 = np.percentile(mag_all[filt], 95, axis=0) + errorbudget
        magmin2 = np.percentile(mag_all[filt], 5, axis=0) - errorbudget

        med_all[filt]["10"] = magmin
        med_all[filt]["50"] = magmed
        med_all[filt]["90"] = magmax
        med_all[filt]["5"] = magmin2
        med_all[filt]["95"] = magmax2

    return med_all

def get_peak(magtable, filts = ["u","g","r","i","z","y","J","H","K"]):

    peaks_all = {}
    for ii, filt in enumerate(filts):
        peaks_all[filt] = {}
        for jj, row in enumerate(magtable):
            t, lbol, mag = row["t"], row["lbol"], row["mag"]
            maginterp = get_mag(mag,filt)
            if (jj==0):
                peaks_all[filt] = np.empty((0,2)) 

            idx = np.argmin(maginterp)
            time_min = t[idx]
            mag_min = maginterp[idx]

            peaks_all[filt] = np.append(peaks_all[filt],[[time_min,mag_min]],axis=0)
    return peaks_all

def get_envelope(lambdas,spec):

    lambdas_all = lambdas*1.0

    idx = np.where(np.isfinite(np.log10(spec)))[0]
    lambdas = lambdas[idx]
    spec = spec[idx]
    
    if len(lambdas) == 0:
        return lambdas_all, np.nan*np.zeros(lambdas_all.shape), np.nan*np.zeros(lambdas_all.shape)

    lambdas_interp = np.arange(lambdas[0],lambdas[-1],1.0)
    f = interp.interp1d(lambdas, np.log10(spec), fill_value='extrapolate')
    spec = 10**f(lambdas_interp)
    lambdas = lambdas_interp*1.0

    spec_lowpass = butter_lowpass_filter(spec, 0.002/3.0, 1.0, order=5)

    idx = np.where((lambdas >= 10200) & (lambdas <= 10100))[0]
    spec_lowpass[idx] = np.nan
    idx = np.where((lambdas >= 13000) & (lambdas <= 15000))[0]
    spec_lowpass[idx] = np.nan
    idx = np.where((lambdas >= 17900) & (lambdas <= 19700))[0]
    spec_lowpass[idx] = np.nan
    idx = np.where(~np.isnan(spec_lowpass))[0]
    lambdas_lowpass = lambdas[idx]
    spec_lowpass = spec_lowpass[idx]

    f = interp.interp1d(lambdas_lowpass, np.log10(spec_lowpass), fill_value='extrapolate')
    spec_lowpass = 10**f(lambdas_all)
    lambdas = lambdas_all*1.0

    idx = argrelextrema(spec_lowpass, np.greater)[0]
    idx = np.hstack((0,idx,len(spec_lowpass)-1))
    spec_envelope = spec_lowpass[idx]
   
    try:
        f = interp.interp1d(lambdas[idx], spec_lowpass[idx], fill_value='extrapolate', kind = 'quadratic')
        spec_envelope = f(lambdas)
        spec_envelope = np.max(np.vstack((spec_lowpass,spec_envelope)),axis=0)
    except:
        spec_envelope = np.nan*np.zeros(lambdas_all.shape)

    return lambdas, spec_lowpass, spec_envelope

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def nanpercentile(arr, q):
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    if type(q) is list:
        qs = []
        qs.extend(q)
    else:
        qs = [q]
    if len(qs) < 2:
        quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
    else:
        quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

    result = []
    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

        result.append(quant_arr)

    return result

def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    _,nC,nR = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nC*nR*ind + nR*np.arange(nR)[:,None] + np.arange(nC)
    return np.take(arr, idx)
