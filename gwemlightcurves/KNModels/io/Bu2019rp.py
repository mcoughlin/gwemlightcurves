# https://arxiv.org/abs/1705.07084

import os, sys, glob, pickle
import numpy as np
import scipy.interpolate
from scipy.interpolate import interpolate as interp
from scipy.interpolate import griddata

from .model import register_model
from .. import KNTable

from gwemlightcurves import lightcurve_utils, Global, svd_utils
from gwemlightcurves.EjectaFits.DiUj2017 import calc_meje, calc_vej

def get_Bu2019rp_model(table, **kwargs):

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

    if not 'n_coeff' in table.colnames:
        if doAB:
            table['n_coeff'] = 43
            #table['n_coeff'] = 2
        elif doSpec:
            table['n_coeff'] = 21

    if not 'gptype' in table.colnames:
        table['gptype'] = sklearn

    if doAB:
        if not Global.svd_mag_model == 0:
            svd_mag_model = Global.svd_mag_model
        else:
            if LoadModel:
            #if True:
                if table['gptype'] == "sklearn":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_mag.pkl')
                elif table['gptype'] == "gpytorch":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_mag_gpy.pkl')
                with open(modelfile, 'rb') as handle:
                    svd_mag_model = pickle.load(handle)
            else:
                svd_mag_model = svd_utils.calc_svd_mag(table['tini'][0], table['tmax'][0], table['dt'][0], model = "Bu2019rp", n_coeff = table['n_coeff'][0], gptype=table['gptype'])
                if table['gptype'] == "sklearn":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_mag.pkl')
                elif table['gptype'] == "gpytorch":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_mag_gpy.pkl')

                with open(modelfile, 'wb') as handle:
                    pickle.dump(svd_mag_model, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

            Global.svd_mag_model = svd_mag_model

        if not Global.svd_lbol_model == 0:
            svd_lbol_model = Global.svd_lbol_model
        else:
            if LoadModel:
            #if True:
                if table['gptype'] == "sklearn":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_lbol.pkl')
                elif table['gptype'] == "gpytorch":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_lbol_gpy.pkl')

                with open(modelfile, 'rb') as handle:
                    svd_lbol_model = pickle.load(handle)
            else:
                svd_lbol_model = svd_utils.calc_svd_lbol(table['tini'][0], table['tmax'][0], table['dt'][0], model = "Bu2019rp", n_coeff = table['n_coeff'][0], gptype=table['gptype'])

                if table['gptype'] == "sklearn":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_lbol.pkl')
                elif table['gptype'] == "gpytorch":
                    modelfile = os.path.join(ModelPath,'Bu2019rp_lbol_gpy.pkl')

                with open(modelfile, 'wb') as handle:
                    pickle.dump(svd_lbol_model, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

            Global.svd_lbol_model = svd_lbol_model
    elif doSpec:
        if not Global.svd_spec_model == 0:
            svd_spec_model = Global.svd_spec_model
        else:
            if LoadModel:
            #if True:
                modelfile = os.path.join(ModelPath,'Bu2019rp_spec.pkl')
                with open(modelfile, 'rb') as handle:
                    svd_spec_model = pickle.load(handle)
            else:
                svd_spec_model = svd_utils.calc_svd_spectra(table['tini'][0], table['tmax'][0], table['dt'][0], table['lambdaini'][0], table['lambdamax'][0], table['dlambda'][0], model = "Bu2019rp", n_coeff = table['n_coeff'][0])
                modelfile = os.path.join(ModelPath,'Bu2019rp_spec.pkl')
                with open(modelfile, 'wb') as handle:
                    pickle.dump(svd_spec_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            Global.svd_spec_model = svd_spec_model

    # Throw out smaples where the mass ejecta is less than zero.
    mask = (table['mej_1'] > 0)
    table = table[mask]
    if len(table) == 0: return table

    timeseries = np.arange(table['tini'][0], table['tmax'][0]+table['dt'][0], table['dt'][0])
    table['t'] = [np.zeros(timeseries.size)]
    if doAB:
        table['lbol'] = [np.zeros(timeseries.size)]
        table['mag'] =  [np.zeros([9, timeseries.size])]
    elif doSpec:
        lambdas = np.arange(table['lambdaini'][0], table['lambdamax'][0]+table['dlambda'][0], table['dlambda'][0])
        table['lambda'] = [np.zeros(lambdas.size)]
        table['spec'] =  [np.zeros([lambdas.size, timeseries.size])]

    # calc lightcurve for each sample
    for isample in range(len(table)):
        print('Generating sample %d/%d' % (isample, len(table)))
        if doAB:
            table['t'][isample], table['lbol'][isample], table['mag'][isample] = svd_utils.calc_lc(table['tini'][isample], table['tmax'][isample],table['dt'][isample], [np.log10(table['mej_1'][isample]),np.log10(table['mej_2'][isample]),table['phi'][isample],table['theta'][isample],table['a'][isample]],svd_mag_model = svd_mag_model, svd_lbol_model = svd_lbol_model, model = "Bu2019rp", gptype=table['gptype'])
        elif doSpec:
            table['t'][isample], table['lambda'][isample], table['spec'][isample] = svd_utils.calc_spectra(table['tini'][isample], table['tmax'][isample],table['dt'][isample], table['lambdaini'][isample], table['lambdamax'][isample]+table['dlambda'][isample], table['dlambda'][isample], [np.log10(table['mej'][isample]),table['phi'][isample],table['theta'][isample]],svd_spec_model = svd_spec_model, model = "Bu2019rp", gptype=table['gptype'])

    return table

register_model('Bu2019rp', KNTable, get_Bu2019rp_model,
                 usage="table")
