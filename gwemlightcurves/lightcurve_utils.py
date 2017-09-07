
import os, sys
import optparse
import numpy as np
import glob
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

from astropy.time import Time

def loadModelsSpec(outputDir,name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Rosswog et al. (2017)"]

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

def loadModels(outputDir,name):

    models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog","Blue"]
    models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich et al. (2016)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Metzger (2017)"]

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

def loadEvent(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = filter(None,lines)

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        lineSplit = filter(None,lineSplit)
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
    event = nameSplit[0]
    instrument = nameSplit[1]
    specdata = nameSplit[2]    

    data_out = np.loadtxt(filename)
    spec = {}

    spec["lambda"] = data_out[:,0] # Angstroms
    if instrument == "XSH":
        spec["data"] = np.abs(data_out[:,1])*1e-17 # ergs/s/cm2./Angs 
    else:
        spec["data"] = np.abs(data_out[:,1]) # ergs/s/cm2./Angs
    spec["error"] = np.zeros(spec["data"].shape) # ergs/s/cm2./Angs
    spec["error"][:-1] = np.abs(np.diff(spec["data"]))
    spec["error"][-1] = spec["error"][-2]
    idx = np.where(spec["error"] <= 0.5*spec["data"])[0]
    spec["error"][idx] = 0.5*spec["data"][idx]

    return spec

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

def read_posterior_samples(filename_samples):

    data_out = {}
    lines = [line.rstrip('\n') for line in open(filename_samples)]
    data = np.loadtxt(filename_samples,skiprows=1)
    line = lines[0]
    params = line.split("\t")
    params = filter(None, params)

    for ii in xrange(len(params)):
        param = params[ii]
  
        data_out[param] = data[:,ii]

    return data_out

def read_files_lbol(files,tmin=-100.0,tmax=100.0):

    names = []
    Lbols = {}
    for filename in files:
        name = filename.replace(".txt","").replace(".dat","").split("/")[-1]
        Lbol_d = np.loadtxt(filename)
        #Lbol_d = Lbol_d[1:,:]

        t = Lbol_d[:,0]
        Lbol = Lbol_d[:,1]
        try:
            index = np.nanargmin(Lbol)
            index = 0
        except:
            index = 0
        t0 = t[index]

        Lbols[name] = {}
        Lbols[name]["t"] = Lbol_d[:,0]
        indexes1 = np.where(Lbols[name]["t"]>=tmin)[0]
        indexes2 = np.where(Lbols[name]["t"]<=tmax)[0]
        indexes = np.intersect1d(indexes1,indexes2)

        Lbols[name]["t"] = Lbol_d[indexes,0]
        Lbols[name]["Lbol"] = Lbol_d[indexes,1]

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
                for kk in xrange(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag1, f_exp=mag2[kk:len(mag1)])[0] 
                    chisquares.append(chisquare)
            elif nslides < 0:
                chisquares = []
                for kk in xrange(np.abs(nslides)):
                    chisquare = scipy.stats.chisquare(mag2, f_exp=mag1[kk:len(mag2)])[0] 
                    chisquares.append(chisquare)

            print name1, name2, xcorr_corr, np.min(np.abs(chisquares)), len(mag1), len(mag2)
            xcorrvals[ii,jj] = xcorr_corr
            chisquarevals[ii,jj] = np.min(np.abs(chisquares))

    return xcorrvals, chisquarevals
