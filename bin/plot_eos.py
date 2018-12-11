
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
#plt.rcParams['xtick.labelsize']=30
#plt.rcParams['ytick.labelsize']=30

from astropy.table import (Table, Column, vstack)
from distutils.spawn import find_executable

from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope

import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
    
def get_eps(EOS,mass):                   
    MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_mr.dat'), format='ascii')
    energy_density_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'],  np.log10(MassRadiusBaryMassTable['rho_c']))
    eps = 10**(et.values_from_table(mass, MassRadiusBaryMassTable['mass'], np.log10(MassRadiusBaryMassTable['rho_c']), energy_density_of_mass_const))
    return eps

plotDir = '../plots/eos/'
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
masses = np.linspace(0.5,3.0,100)
plt.figure(figsize=(12,8))        
for ii,eosname in enumerate(eosnames):
    print(eosname)
    radii = np.zeros(masses.shape)
    eos = EOS4ParameterPiecewisePolytrope(eosname)

    try:
        eps = get_eps(eosname.lower(),masses)
    except:
        continue

    fid = open(os.path.join(plotDir,'%s.dat'%eosname),'w')
    for jj,mass in enumerate(masses):
        
        try:
            lambda1 = eos.lambdaofm(mass)
            radius = eos.radiusofm(mass)
        except:
            continue

        radii[jj] = radius

        fid.write('%.10f %.10f\n'%(mass,radius))

    plt.plot(radii,masses,'-',label=eosname)
    fid.close()

plt.legend(loc='best')
plt.xlim(xmin=9,xmax=15)
plt.ylim(ymax=2.6)
plt.xlabel('Radius [km]')
plt.ylabel('Mass [solar masses]')
plt.savefig(plotName, bbox_inches='tight')
plt.close()

plotName="%s/love_EOS.pdf"%(plotDir)
plt.figure(figsize=(12,8))
for ii,eosname in enumerate(eosnames):
    print(eosname)
    lovenumbers=np.zeros(masses.shape)
    eos=EOS4ParameterPiecewisePolytrope(eosname)
    for jj,mass in enumerate(masses):
	try:
	    lovenumber=eos.k2ofm(mass)
	except:
	    continue
	lovenumbers[jj]=lovenumber
    plt.plot(masses,lovenumbers,'+',label=eosname)
plt.legend(loc='best')
plt.xlabel('Mass [solar masses]')
plt.ylabel('Love number')
plt.savefig(plotName, bbox_inches='tight')
plt.close()

plotName="%s/lambda_EOS.pdf"%(plotDir)
plt.figure(figsize=(12,8))
for ii,eosname in enumerate(eosnames):
    print(eosname)
    lambdas=np.zeros(masses.shape)
    eos=EOS4ParameterPiecewisePolytrope(eosname)
    for jj,mass in enumerate(masses):
        try:
            lambda1=eos.lambdaofm(mass)
        except:
            continue
        lambdas[jj]=lambda1
    plt.plot(masses,lambdas,'-',label=eosname)
plt.legend(loc='best')
plt.xlabel('Mass [solar masses]')
plt.ylabel('Lambda')
plt.savefig(plotName, bbox_inches='tight')
plt.close()

plotName = "%s/radius_density.pdf"%(plotDir)
plt.figure(figsize=(12,8))
for ii,eosname in enumerate(eosnames):
    print(eosname)
    radii = np.zeros(masses.shape)
    eos = EOS4ParameterPiecewisePolytrope(eosname)

    try:
        eps = get_eps(eosname.lower(),masses)
    except:
        continue

    for jj,mass in enumerate(masses):
	try:
            radius = eos.radiusofm(mass)
        except:
            continue

        radii[jj] = radius

    plt.plot(radii,eps,'-',label=eosname)
plt.legend(loc='best')
plt.xlabel('Radius [km]')
plt.ylabel('$\log_{10}$($\epsilon_0$ (g/cm^3))')
plt.savefig(plotName, bbox_inches='tight')
plt.close()
