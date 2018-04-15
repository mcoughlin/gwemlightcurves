
import os, sys, glob, pickle
import optparse
import numpy as np
import pandas
from scipy.interpolate import interpolate as interp
import scipy.stats

from astropy.table import Table, Column

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

import corner

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global
from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope

plotDir = '../plots/gws/Ka2017_combine'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

filename = '../plots/fitting_gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/joint/GW170817/1.00/0_640/q_lambdatilde.dat' 
data = np.loadtxt(filename)

eosfilename = '../input/lalsim/polytrope_table.dat'
df = pandas.read_table(eosfilename, skiprows=1, delim_whitespace=True, names=['eos','logP1','gamma1','gamma2','gamma3'])

labels = [r"q",r"$\tilde{\Lambda}$"]
plotName = "%s/q_lambda_corner.pdf"%(plotDir)
figure, axes = plt.subplots(2, 2, figsize=(20, 20))
corner.corner(data, labels=labels,fig=figure,smooth=3,color="k",
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 30},
                       label_kwargs={"fontsize": 30}, title_fmt=".2f")
#eosnames = ['AP4','SLy','MPA1','H4','MS1b','MS1']
eosnames = df['eos']
eosnames = ["ALF2","MPA1","AP3","SLy","ALF4","AP4","WFF3","WFF1"]
qs = np.linspace(1,1.5,100)
mc = 1.188
        
for ii,eosname in enumerate(eosnames):
    lambdatildes = []
    eos = EOS4ParameterPiecewisePolytrope(eosname)
    plotline = True
    for q in qs:
        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        try:
            lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)

            lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
            lambdatildes.append(lambdatilde)
        except:
            plotline = False
            lambdatildes.append(0)
    if plotline:
        axes[1][0].plot(qs,lambdatildes,'b--')
        axes[1][0].text(qs[-1],lambdatildes[-1],eosname,fontsize=24)

        axes[1][1].fill_between([np.min(lambdatildes),np.max(lambdatildes)],[-1000,-1000],[1000,1000],facecolor='gray',alpha=0.5)
        axes[1][1].text(np.min(lambdatildes),5+31*ii,eosname,fontsize=24)

axes[1][0].set_xlim([1,1.6])
axes[1][0].set_ylim([0,640])

figure.set_size_inches(14.0,14.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()
