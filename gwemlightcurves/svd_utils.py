# https://arxiv.org/abs/1705.07084

import os, sys, glob
import numpy as np
import scipy.interpolate
from scipy.interpolate import interpolate as interp
from scipy.interpolate import griddata

from gwemlightcurves import lightcurve_utils, Global

from sklearn import gaussian_process

def calc_svd_lbol(tini,tmax,dt, n_coeff = 100, model = "BaKa2016"):

    if model == "BaKa2016":    
        fileDir = "../output/barnes_kilonova_spectra"
    elif model == "Ka2017":
        fileDir = "../output/kasen_kilonova_survey"
    elif model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"

    filenames = glob.glob('%s/*_Lbol.dat'%fileDir)

    lbols, names = lightcurve_utils.read_files_lbol(filenames)
    lbolkeys = lbols.keys()

    tt = np.arange(tini,tmax+dt,dt)

    for key in lbolkeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            lbols[key]["mej"] = mej0
            lbols[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                del lbols[key]
                continue
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            lbols[key]["mej"] = mej0
            lbols[key]["vej"] = vej0
            lbols[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            lbols[key]["mej"], lbols[key]["vej"], lbols[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        ii = np.where(np.isfinite(lbols[key]["Lbol"]))[0]
        f = interp.interp1d(lbols[key]["tt"][ii], np.log10(lbols[key]["Lbol"][ii]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbols[key]["Lbol"]= np.log10(lbolinterp)

    lbolkeys = lbols.keys()

    lbol_array = []
    param_array = []
    for key in lbolkeys:
        lbol_array.append(lbols[key]["Lbol"])
        if model == "BaKa2016":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["vej"]])
        elif model == "Ka2017":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["vej"],np.log10(lbols[key]["Xlan"])])
        elif model == "RoFe2017":
            param_array.append([np.log10(lbols[key]["mej"]),lbols[key]["vej"],lbols[key]["Ye"]]) 

    param_array = np.array(param_array)
    lbol_array_postprocess = np.array(lbol_array)

    means,stds = np.mean(lbol_array_postprocess,axis=0),np.std(lbol_array_postprocess,axis=0)
    for i in range(len(means)):
        lbol_array_postprocess[:,i] = (lbol_array_postprocess[:,i]-means[i])/stds[i]

    lbol_array_postprocess[np.isnan(lbol_array_postprocess)]=0.0

    UA, sA, VA = np.linalg.svd(lbol_array_postprocess, full_matrices=True)

    n, n = UA.shape
    m, m = VA.shape

    #n_coeff = m
    cAmat = np.zeros((n_coeff,n))
    for i in range(n):
        cAmat[:,i] = np.dot(lbol_array_postprocess[i,:],VA[:,:n_coeff])

    svd_model = {}
    svd_model["n_coeff"] = n_coeff
    svd_model["param_array"] = param_array
    svd_model["cAmat"] = cAmat
    svd_model["VA"] = VA
    svd_model["stds"] = stds
    svd_model["means"] = means

    return svd_model

def calc_svd_mag(tini,tmax,dt, n_coeff = 100, model = "BaKa2016"):

    if model == "BaKa2016":
        fileDir = "../output/barnes_kilonova_spectra"
    elif model == "Ka2017":
        fileDir = "../output/kasen_kilonova_survey"
    elif model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"

    filenames_all = glob.glob('%s/*.dat'%fileDir)
    idxs = []
    for ii,filename in enumerate(filenames_all):
        if "_Lbol.dat" in filename: continue
        if "_spec.dat" in filename: continue
        idxs.append(ii)
    filenames = [filenames_all[idx] for idx in idxs]

    mags, names = lightcurve_utils.read_files(filenames)
    magkeys = mags.keys()

    tt = np.arange(tini,tmax+dt,dt)
    filters = ["u","g","r","i","z","y","J","H","K"]

    for key in magkeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            mags[key]["mej"] = mej0
            mags[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                del mags[key]
                continue
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e","")) 
 
            mags[key]["mej"] = mej0
            mags[key]["vej"] = vej0
            mags[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            mags[key]["mej"], mags[key]["vej"], mags[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        mags[key]["data"] = np.zeros((len(tt),len(filters)))

        for jj,filt in enumerate(filters):
            ii = np.where(np.isfinite(mags[key][filt]))[0]
            f = interp.interp1d(mags[key]["t"][ii], mags[key][filt][ii], fill_value='extrapolate')
            maginterp = f(tt)
            mags[key]["data"][:,jj] = maginterp
        mags[key]["data_vector"] = np.reshape(mags[key]["data"],len(tt)*len(filters),1)

    magkeys = mags.keys()

    mag_array = []
    param_array = []
    for key in magkeys:
        mag_array.append(mags[key]["data_vector"])
        if model == "BaKa2016":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["vej"]])
        elif model == "Ka2017":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["vej"],np.log10(mags[key]["Xlan"])])
        elif model == "RoFe2017":
            param_array.append([np.log10(mags[key]["mej"]),mags[key]["vej"],mags[key]["Ye"]])    

    param_array = np.array(param_array)
    mag_array_postprocess = np.array(mag_array)
    
    means,stds = np.mean(mag_array_postprocess,axis=0),np.std(mag_array_postprocess,axis=0)
    for i in range(len(means)):
        mag_array_postprocess[:,i] = (mag_array_postprocess[:,i]-means[i])/stds[i]
    
    mag_array_postprocess[np.isnan(mag_array_postprocess)]=0.0
    UA, sA, VA = np.linalg.svd(mag_array_postprocess, full_matrices=True)

    n, n = UA.shape
    m, m = VA.shape

    #n_coeff = m
    cAmat = np.zeros((n_coeff,n))
    for i in range(n):
        cAmat[:,i] = np.dot(mag_array_postprocess[i,:],VA[:,:n_coeff])

    svd_model = {}
    svd_model["n_coeff"] = n_coeff
    svd_model["param_array"] = param_array
    svd_model["cAmat"] = cAmat
    svd_model["VA"] = VA
    svd_model["stds"] = stds
    svd_model["means"] = means

    return svd_model

def calc_lc(tini,tmax,dt,param_list,svd_mag_model=None,svd_lbol_model=None, model = "BaKa2016"):

    tt = np.arange(tini,tmax+dt,dt)

    if svd_mag_model == None:
        svd_mag_model = calc_svd_mag(tini,tmax,dt,model=model) 
    if svd_lbol_model == None:
        svd_lbol_model = calc_svd_lbol(tini,tmax,dt,model=model)

    n_coeff = svd_mag_model["n_coeff"]
    param_array = svd_mag_model["param_array"]
    cAmat = svd_mag_model["cAmat"]
    VA = svd_mag_model["VA"]
    stds = svd_mag_model["stds"]
    means = svd_mag_model["means"]

    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    cAproj = np.zeros((n_coeff,))
    for i in range(n_coeff):
        gp.fit(param_array, cAmat[i,:])
        y_pred, sigma2_pred = gp.predict(param_list, eval_MSE=True)
        #grid_z0 = griddata(param_array,cAmat[i,:],param_list, method='nearest')
        #grid_z1 = griddata(param_array,cAmat[i,:],param_list, method='linear')
        cAproj[i] = y_pred

    mag_back = np.dot(VA[:,:n_coeff],cAproj)
    mag_back = mag_back*stds+means

    mAB = np.reshape(mag_back,(9,len(tt)))

    n_coeff = svd_lbol_model["n_coeff"]
    param_array = svd_lbol_model["param_array"]
    cAmat = svd_lbol_model["cAmat"]
    VA = svd_lbol_model["VA"]
    stds = svd_lbol_model["stds"]
    means = svd_lbol_model["means"]

    cAproj = np.zeros((n_coeff,))
    for i in range(n_coeff):
        gp.fit(param_array, cAmat[i,:])
        y_pred, sigma2_pred = gp.predict(param_list, eval_MSE=True)

        #grid_z0 = griddata(param_array,cAmat[i,:],param_list, method='nearest')
        #grid_z1 = griddata(param_array,cAmat[i,:],param_list, method='linear')
        cAproj[i] = y_pred

    lbol_back = np.dot(VA[:,:n_coeff],cAproj)
    lbol_back = lbol_back*stds+means

    lbol = 10**lbol_back

    return np.squeeze(tt), np.squeeze(lbol), mAB

