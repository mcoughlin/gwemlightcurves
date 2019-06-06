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

def get_Ka2017x2inc_model(table, **kwargs):

    if 'doAB' in kwargs:
        doAB = kwargs['doAB']
    else:
        doAB = True

    if 'doSpec' in kwargs:
        doSpec = kwargs['doSpec']
    else:
        doSpec = False

    if 'ModelPath' in kwargs:
        ModelPath = kwargs['ModelPath']

    timeseries = np.arange(table['tini'][0], table['tmax'][0]+table['dt'][0], table['dt'][0])
    table['t'] = [np.zeros(timeseries.size)]
    if doAB:
        table['lbol'] = [np.zeros(timeseries.size)]
        table['mag'] =  [np.zeros([9, timeseries.size])]
    elif doSpec:
        lambdas = np.arange(table['lambdaini'][0], table['lambdamax'][0]+table['dlambda'][0], table['dlambda'][0])
        table['lambda'] = [np.zeros(lambdas.size)]
        table['spec'] =  [np.zeros([lambdas.size, timeseries.size])]

    table1 = copy.copy(table)
    table1['mej'] = table['mej_1']
    table1['vej'] = table['vej_1']
    table1['Xlan'] = table['Xlan_1']

    table2 = copy.copy(table)
    table2['mej'] = table['mej_2']
    table2['vej'] = table['vej_2']
    table2['Xlan'] = table['Xlan_2']

    if len(Global.svd_mag_color_models) == 0:
        colormodels = table["colormodel"][0]
        colormodel1, colormodel2 = colormodels[0], colormodels[1]

        Global.svd_mag_color_models = []
        for cm in colormodels:
            modelfile = os.path.join(ModelPath,'%s.pkl' % cm)
            with open(modelfile, 'rb') as handle:
                svd_mag_color_model = pickle.load(handle)
            Global.svd_mag_color_models.append(svd_mag_color_model)
        table1['colormodel'] = [colormodel1]
        table2['colormodel'] = [colormodel2]

    Global.svd_mag_color_model = Global.svd_mag_color_models[0]
    print(Global.svd_mag_color_model)
    exit(0)
    table1 = KNTable.model('Ka2017inc', table1, **kwargs)
    Global.svd_mag_color_model = Global.svd_mag_color_models[1]
    print(Global.svd_mag_color_model)
    table2 = KNTable.model('Ka2017inc', table2, **kwargs)
    exit(0)

    # calc lightcurve for each sample
    for isample in range(len(table)):
        if doAB:
            table['t'][isample], table['lbol'][isample], table['mag'][isample] = table1['t'][isample], table1['lbol'][isample] + table2['lbol'][isample], -2.5*np.log10(10**(-table1['mag'][isample]*0.4) + 10**(-table2['mag'][isample]*0.4))
        elif doSpec:
            table['t'][isample], table['lambda'][isample], table['spec'][isample] = table1['t'][isample], table1['lambda'][isample], table1['spec'][isample] + table2['spec'][isample]

    return table

register_model('Ka2017x2inc', KNTable, get_Ka2017x2inc_model,
                 usage="table")
