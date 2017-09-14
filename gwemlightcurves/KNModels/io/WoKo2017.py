# https://arxiv.org/abs/1705.07084

import os, sys
import numpy as np
import scipy.interpolate

from .model import register_model
from .. import KNTable

def get_WoKo2017_model(table, **kwargs):
    if not 'mej' in table.colnames:
        # calc the mass of ejecta
        table['mej'] = calc_meje(table['q'], table['chi_eff'], table['c'], table['mb'], table['mns'])
        # calc the velocity of ejecta
        table['vej'] = calc_vave(table['q'])

    # Throw out smaples where the mass ejecta is less than zero.
    mask = (table['mej'] > 0)
    table = table[mask]
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
                                                                     table['vej'][isample], table['theta_r'][isample], table['kappa_r'][isample])
    return table

def calc_meje(m1,mb1,c1,m2,mb2,c2):

    a= -1.35695
    b=  6.11252
    c=-49.43355
    d=  16.1144
    n=  -2.5484

    tmp1=((mb1*((m2/m1)**(1.0/3.0))*(1.0-2.0*c1)/c1)+(mb2*((m1/m2)**(1.0/3.0))*(1.0-2.0*c2)/c2))*a
    tmp2=(mb1*((m2/m1)**n)+mb2*((m1/m2)**n))*b
    tmp3=(mb1*(1.0-m1/mb1)+mb2*(1.0-m2/mb2))*c

    meje_fit=np.maximum(tmp1+tmp2+tmp3+d,0)/1000.0

    return meje_fit

def calc_vrho(m1,c1,m2,c2):
    a=-0.219479
    b=0.444836
    c=-2.67385

    return ((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))*a+b

def calc_vz(m1,c1,m2,c2):
    a=-0.315585
    b=0.63808
    c=-1.00757

    return ((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))*a+b

def calc_vej(m1,c1,m2,c2):
    return np.sqrt(calc_vrho(m1,c1,m2,c2)**2.0+calc_vz(m1,c1,m2,c2)**2.0)

def calc_qej(m1,c1,m2,c2):
    vrho=calc_vrho(m1,c1,m2,c2)
    vz=calc_vz(m1,c1,m2,c2)
    vrho2=vrho*vrho
    vz2=vz*vz

    tmp1=3.*vz+np.sqrt(9*vz2+4*vrho2)
    qej=((2.0**(4.0/3.0))*vrho2+(2.*vrho2*tmp1)**(2.0/3.0))/((vrho**5.0)*tmp1)**(1.0/3.0)

    return qej

def calc_phej(m1,c1,m2,c2):
  return 4.0*calc_qej(m1,c1,m2,c2)*np.pi/2.0

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

    for ii in xrange(len(ints)):
        idx = np.arange(ndata/9) + ii*(ndata/9)
        data_out_slice = data_out[idx,:]

        t = data_out_slice[:,1]
        data = data_out_slice[:,2:]
        nt, nbins = data.shape

        a_i = (360/(2*np.pi))*np.arccos(1 - np.arange(nbins)*2/float(nbins))
        b_i = (360/(2*np.pi))*np.arccos(1 - (np.arange(nbins)+1)*2/float(nbins)) 
        bins = (a_i + b_i)/2.0

        if ii == 0:
            f     = scipy.interpolate.interp2d(bins,t,np.log10(data), kind='linear')
        else:
            f     = scipy.interpolate.interp2d(bins,t,data, kind='linear')

        fam    = f(theta_r*2*np.pi,tvec_days)

        if ii == 0:
            lbol = 10**fam
        else:
            mAB[:,int(ii-1)] = np.squeeze(fam + mejconst[int(ii-1)]*np.log10(mej/mej0) + vejconst[int(ii-1)]*np.log10(vej/vej0)) #+ kappaconst[int(ii-1)]*np.log10(kappa_r/kappa0))

    tmax = (kappa_r/10)**0.35 * (mej/10**-2)**0.318 * (vej/0.1)**-0.60
    Lmax = 2.8*10**40 * (kappa_r/10)**-0.60 * (mej/10**-2)**0.426 * (vej/0.1)**0.776
    t = t*tmax/t[np.argmax(lbol)]
    lbol = lbol*Lmax/np.max(lbol)

    wavelengths = [4775.6, 6129.5, 7484.6, 8657.8, 9603.1, 12350, 16620, 21590]
    wavelength_interp = 3543

    mAB_y = np.zeros(tvec_days.shape)
    for ii in xrange(len(tvec_days)):
        mAB_y[ii] = np.interp(wavelength_interp,wavelengths,mAB[ii,:])
    mAB_new = np.zeros((len(tvec_days),9))
    mAB_new[:,0] = np.squeeze(mAB_y)
    mAB_new[:,1:] = mAB

    return np.squeeze(tvec_days), np.squeeze(lbol), mAB_new.T

register_model('WoKo2017', KNTable, get_WoKo2017_model,
                 usage="table")
