# https://arxiv.org/abs/1705.07084

import os, sys, glob, pickle, copy
import numpy as np
import scipy.interpolate
from scipy.interpolate import interpolate as interp
from scipy.interpolate import griddata

from .model import register_model
from .. import KNTable

from gwemlightcurves import lightcurve_utils, Global, svd_utils
from gwemlightcurves.EjectaFits.DiUj2017 import calc_meje, calc_vej

def get_Ka2017inc_model(table, **kwargs):

    if 'LoadModel' in kwargs:
        LoadModel = kwargs['LoadModel']
    else:
        LoadModel = False

    if 'SaveModel' in kwargs:
        SaveModel = kwargs['SaveModel']
    else:
        SaveModel = False

    if 'ModelPath' in kwargs:
        ModelPath = kwargs['ModelPath']

    if 'doAB' in kwargs:
        doAB = kwargs['doAB']
    else:
        doAB = True

    if 'doSpec' in kwargs:
        doSpec = kwargs['doSpec']
    else:
        doSpec = False

    if doSpec:
        print('Spectra not available for Ka2017inc...')
        exit(0)

    timeseries = np.arange(table['tini'][0], table['tmax'][0]+table['dt'][0], table['dt'][0])
    table['t'] = [np.zeros(timeseries.size)]
    if doAB:
        table['lbol'] = [np.zeros(timeseries.size)]
        table['mag'] =  [np.zeros([9, timeseries.size])]
    elif doSpec:
        lambdas = np.arange(table['lambdaini'][0], table['lambdamax'][0]+table['dlambda'][0], table['dlambda'][0])
        table['lambda'] = [np.zeros(lambdas.size)]
        table['spec'] =  [np.zeros([lambdas.size, timeseries.size])]

    if not 'n_coeff' in table.colnames:
        if doAB:
            table['n_coeff'] = 43
        elif doSpec:
            table['n_coeff'] = 21

    if doAB:
        if not Global.svd_mag_color_model == 0:
            svd_mag_color_model = Global.svd_mag_color_model
        else:
            #if False:
            if LoadModel:
                modelfile = os.path.join(ModelPath,'%s.pkl' % table['colormodel'][0])
                with open(modelfile, 'rb') as handle:
                    svd_mag_color_model = pickle.load(handle)
            else:
                svd_mag_color_model = svd_utils.calc_svd_color_model(table['tini'][0], table['tmax'][0], table['dt'][0], model = table['colormodel'][0], n_coeff = table['n_coeff'][0])
                modelfile = os.path.join(ModelPath,'%s.pkl' % table['colormodel'][0])
                with open(modelfile, 'wb') as handle:
                    pickle.dump(svd_mag_color_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            Global.svd_mag_color_model = svd_mag_color_model

    table1 = KNTable.model('Ka2017', table, **kwargs)

    # calc lightcurve for each sample
    for isample in range(len(table)):
        if doAB:
            if Global.svd_mag_color_model == "a1.0":
                table['t'][isample] = table1['t'][isample]
                table['mag'][isample] = table1['mag'][isample]
                table['lbol'][isample] = table1['lbol'][isample]
            else:
                table['t'][isample], table['mag'][isample] = svd_utils.calc_color(table['tini'][isample], table['tmax'][isample],table['dt'][isample], [table['iota'][isample]],svd_mag_color_model = Global.svd_mag_color_model)

                for ii, mag_slice in enumerate(table['mag'][isample]):
                    orig = table1['mag'][isample][ii]
                    dcolor = table['mag'][isample][ii]
                    table['mag'][isample][ii] = orig + dcolor
                table['lbol'][isample] = table1['lbol'][isample] 

    return table

register_model('Ka2017inc', KNTable, get_Ka2017inc_model,
                 usage="table")
