# https://arxiv.org/abs/1705.07084

import os, sys, glob
import numpy as np
import scipy.interpolate
from scipy.interpolate import interpolate as interp
from scipy.interpolate import griddata

from .model import register_model
from .. import KNTable

from gwemlightcurves import lightcurve_utils, Global
from gwemlightcurves.EjectaFits.DiUj2017 import calc_meje, calc_vej

def get_BaKa2016_model(table, **kwargs):

    if not Global.svd_mag_model == 0:
        svd_mag_model = Global.svd_mag_model
    else:
        print "Generating Lightcurve SVD Model..."
        svd_mag_model = calc_svd_mag(table['tini'][0], table['tmax'][0], table['dt'][0])
        Global.svd_mag_model = svd_mag_model

    if not Global.svd_lbol_model == 0:
        svd_lbol_model = Global.svd_lbol_model
    else:
        print "Generating Lightcurve Lbol Model..."
        svd_lbol_model = calc_svd_lbol(table['tini'][0], table['tmax'][0], table['dt'][0])
        Global.svd_lbol_model = svd_lbol_model

    if not 'mej' in table.colnames:
        # calc the mass of ejecta
        table['mej'] = calc_meje(table['m1'], table['mb1'], table['c1'], table['m2'], table['mb2'], table['c2'])
        # calc the velocity of ejecta
        table['vej'] = calc_vej(table['m1'], table['c1'], table['m2'], table['c2'])

    # Throw out smaples where the mass ejecta is less than zero.
    mask = (table['mej'] > 0)
    table = table[mask]
    if len(table) == 0: return table

    # Log mass ejecta
    table['mej10'] = np.log10(table['mej'])
    # Initialize lightcurve values in table

    timeseries = np.arange(table['tini'][0], table['tmax'][0]+table['dt'][0], table['dt'][0])
    table['t'] = [np.zeros(timeseries.size)]
    table['lbol'] = [np.zeros(timeseries.size)]
    table['mag'] =  [np.zeros([9, timeseries.size])]

    # calc lightcurve for each sample
    for isample in range(len(table)):
        table['t'][isample], table['lbol'][isample], table['mag'][isample] = calc_lc(table['tini'][isample], table['tmax'][isample],
                                                                     table['dt'][isample], table['mej'][isample],
                                                                     table['vej'][isample],svd_mag_model = svd_mag_model, svd_lbol_model = svd_lbol_model)
    return table

def calc_svd_lbol(tini,tmax,dt):

    fileDir = "../output/barnes_kilonova_spectra"
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

        ii = np.where(np.isfinite(lbols[key]["Lbol"]))[0]
        f = interp.interp1d(lbols[key]["tt"][ii], np.log10(lbols[key]["Lbol"][ii]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbols[key]["Lbol"]= np.log10(lbolinterp)

    lbol_array = []
    param_array = []
    for key in lbolkeys:
        lbol_array.append(lbols[key]["Lbol"])
        param_array.append([lbols[key]["mej"],lbols[key]["vej"]])

    param_array = np.array(param_array)
    lbol_array_postprocess = np.array(lbol_array)

    means,stds = np.mean(lbol_array_postprocess,axis=0),np.std(lbol_array_postprocess,axis=0)
    for i in range(len(means)):
        lbol_array_postprocess[:,i] = (lbol_array_postprocess[:,i]-means[i])/stds[i]

    lbol_array_postprocess[np.isnan(lbol_array_postprocess)]=0.0
    UA, sA, VA = np.linalg.svd(lbol_array_postprocess, full_matrices=True)

    n, n = UA.shape
    m, m = VA.shape

    n_coeff = m
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

def calc_svd_mag(tini,tmax,dt):

    fileDir = "../output/barnes_kilonova_spectra"
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
        mags[key]["data"] = np.zeros((len(tt),len(filters)))

        for jj,filt in enumerate(filters):
            ii = np.where(np.isfinite(mags[key][filt]))[0]
            f = interp.interp1d(mags[key]["t"][ii], mags[key][filt][ii], fill_value='extrapolate')
            maginterp = f(tt)
            mags[key]["data"][:,jj] = maginterp
        mags[key]["data_vector"] = np.reshape(mags[key]["data"],len(tt)*len(filters),1)

    mag_array = []
    param_array = []
    for key in magkeys:
        mag_array.append(mags[key]["data_vector"])
        param_array.append([mags[key]["mej"],mags[key]["vej"]])
    
    param_array = np.array(param_array)
    mag_array_postprocess = np.array(mag_array)
    
    means,stds = np.mean(mag_array_postprocess,axis=0),np.std(mag_array_postprocess,axis=0)
    for i in range(len(means)):
        mag_array_postprocess[:,i] = (mag_array_postprocess[:,i]-means[i])/stds[i]
    
    mag_array_postprocess[np.isnan(mag_array_postprocess)]=0.0
    UA, sA, VA = np.linalg.svd(mag_array_postprocess, full_matrices=True)

    n, n = UA.shape
    m, m = VA.shape

    n_coeff = m
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

def calc_lc(tini,tmax,dt,mej,vej,svd_mag_model=None,svd_lbol_model=None):

    tt = np.arange(tini,tmax+dt,dt)

    if svd_mag_model == None:
        svd_mag_model = calc_mag_svd(tini,tmax,dt) 
    if svd_lbol_model == None:
        svd_lbol_model = calc_lbol_svd(tini,tmax,dt)

    n_coeff = svd_mag_model["n_coeff"]
    param_array = svd_mag_model["param_array"]
    cAmat = svd_mag_model["cAmat"]
    VA = svd_mag_model["VA"]
    stds = svd_mag_model["stds"]
    means = svd_mag_model["means"]

    print mej, vej

    cAproj = np.zeros((n_coeff,))
    for i in range(n_coeff):
        cAproj[i] = griddata(param_array,cAmat[i,:],[mej,vej], method='cubic')
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
        cAproj[i] = griddata(param_array,cAmat[i,:],[mej,vej], method='cubic')
    lbol_back = np.dot(VA[:,:n_coeff],cAproj)
    lbol_back = lbol_back*stds+means

    lbol = 10**lbol_back

    return np.squeeze(tt), np.squeeze(lbol), mAB

register_model('BaKa2016', KNTable, get_BaKa2016_model,
                 usage="table")
