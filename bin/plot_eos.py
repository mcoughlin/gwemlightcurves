
import os, sys, glob, pickle
import optparse
import numpy as np

import pandas

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope

plotDir = '../plots/'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

color1 = 'coral'
color2 = 'cornflowerblue'

eosfilename = '../input/lalsim/polytrope_table.dat'
df = pandas.read_table(eosfilename, skiprows=1, delim_whitespace=True, names=['eos','logP1','gamma1','gamma2','gamma3'])

#eosnames = ['AP4','SLy','MPA1','H4','MS1b','MS1']
eosnames = df['eos']
#eosnames = ["ALF2","MPA1","AP3","SLy","ALF4","AP4","WFF3","WFF1"]

plotName = "%s/mass_radius.pdf"%(plotDir)
masses = np.linspace(1.0,5.0,100)
plt.figure()        
for ii,eosname in enumerate(eosnames):
    print(eosname)
    radii = []
    eos = EOS4ParameterPiecewisePolytrope(eosname)
    for mass in masses:

        lambda1 = eos.lambdaofm(mass)
        radius = eos.radiusofm(mass)

        radii.append(radius)

    plt.plot(masses,radii,'-',label=eos)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

