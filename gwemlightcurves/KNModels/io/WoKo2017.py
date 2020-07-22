# https://arxiv.org/abs/1705.07084

import os, sys
import numpy as np
import scipy.interpolate
from scipy.interpolate import interpolate as interp

from .model import register_model
from .. import KNTable

from gwemlightcurves.EjectaFits.DiUj2017 import calc_meje, calc_vej

def get_WoKo2017_model(table, **kwargs):
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
                                                                     table['vej'][isample], table['theta_r'][isample], table['kappa'][isample])
    return table

def calc_lc(tini,tmax,dt,mej,vej,theta_r,kappa_r,model="DZ2"):

    mejconst = [-1.13,-1.01,-0.94,-0.94,-0.93,-0.93,-0.95,-0.99]
    vejconst = [-1.28,-1.60,-1.52,-1.56,-1.61,-1.61,-1.55,-1.33]
    kappaconst = [2.65,2.27,2.02,1.87,1.76,1.56,1.33,1.13]

    if model == "DZ2":
        mej0 = 0.013+0.005
        vej0 = 0.132+0.08
        kappa0 = 1.0
        modelfile = "../data/macronova_models_wollaeger2017/DZ2_mags_2017-03-20.dat"
    elif model == "gamA2":
        mej0 = 0.013+0.005
        vej0 = 0.132+0.08
        kappa0 = 1.0
        modelfile = "../data/macronova_models_wollaeger2017/gamA2_mags_2017-03-20.dat"
    elif model == "gamB2":
        mej0 = 0.013+0.005
        vej0 = 0.132+0.08
        kappa0 = 1.0
        modelfile = "../data/macronova_models_wollaeger2017/gamB2_mags_2017-03-20.dat"

    data_out = np.loadtxt(modelfile)
    ndata, nslices = data_out.shape
    ints = np.arange(0,ndata,ndata/9)

    tvec_days = np.arange(tini,tmax+dt,dt)
    mAB = np.zeros((len(tvec_days),8))

    for ii in range(len(ints)):
        idx = np.arange(ndata/9) + ii*(ndata/9)
        data_out_slice = data_out[idx,:]

        t = data_out_slice[:,1]
        data = data_out_slice[:,2:]
        #idx = np.where((t >= 0) & (t <= 7))[0]
        #t = t[idx]
        #data = data[idx,:]
        nt, nbins = data.shape

        a_i = (360/(2*np.pi))*np.arccos(1 - np.arange(nbins)*2/float(nbins))
        b_i = (360/(2*np.pi))*np.arccos(1 - (np.arange(nbins)+1)*2/float(nbins))
        bins = (a_i + b_i)/2.0

        idx = np.argsort(np.abs(bins-theta_r*2*np.pi))
        idx1 = idx[0]
        idx2 = idx[1]
        weight1 = 1/np.abs(bins[idx1]-theta_r*2*np.pi)
        weight2 = 1/np.abs(bins[idx1]-theta_r*2*np.pi)
        if not np.isfinite(weight1):
            weight1, weight2 = 1.0, 0.0
        elif not np.isfinite(weight2):
            weight1, weight2 = 0.0, 1.0
        else:
            weight1, weight2 = weight1/(weight1+weight2), weight2/(weight1+weight2)

        if ii == 0:
            #f     = scipy.interpolate.interp2d(bins,t,np.log10(data), kind='cubic')
            f1 = interp.interp1d(t, np.log10(data[:,idx1]), fill_value='extrapolate')
            f2 = interp.interp1d(t, np.log10(data[:,idx2]), fill_value='extrapolate')
        else:
            #f     = scipy.interpolate.interp2d(bins,t,data, kind='cubic')
            f1 = interp.interp1d(t,data[:,idx1], fill_value='extrapolate')
            f2 = interp.interp1d(t,data[:,idx2], fill_value='extrapolate')

        fam1, fam2  = f1(tvec_days), f2(tvec_days)
        fam = weight1*fam1+weight2*fam2

        if ii == 0:
            lbol = 10**fam
        else:
            mAB[:,int(ii-1)] = np.squeeze(fam + mejconst[int(ii-1)]*np.log10(mej/mej0) + vejconst[int(ii-1)]*np.log10(vej/vej0)) #+ kappaconst[int(ii-1)]*np.log10(kappa_r/kappa0))

    tmax = (kappa_r/10)**0.35 * (mej/10**-2)**0.318 * (vej/0.1)**-0.60
    Lmax = 2.8*10**40 * (kappa_r/10)**-0.60 * (mej/10**-2)**0.426 * (vej/0.1)**0.776

    tvec_days = tvec_days*tmax/tvec_days[np.argmax(lbol)]
    lbol = lbol*Lmax/np.max(lbol)

    wavelengths = [4775.6, 6129.5, 7484.6, 8657.8, 9603.1, 12350, 16620, 21590]
    wavelength_interp = 3543

    mAB_y = np.zeros(tvec_days.shape)
    for ii in range(len(tvec_days)):
        mAB_y[ii] = np.interp(wavelength_interp,wavelengths,mAB[ii,:])
    mAB_new = np.zeros((len(tvec_days),9))
    mAB_new[:,0] = np.squeeze(mAB_y)
    mAB_new[:,1:] = mAB

    return np.squeeze(tvec_days), np.squeeze(lbol), mAB_new.T

register_model('WoKo2017', KNTable, get_WoKo2017_model,
                 usage="table")
