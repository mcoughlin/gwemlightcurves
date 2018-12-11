
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

plotDir = '../plots/highenergy/'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

kev = np.array([[2.34,1.8,np.inf],[9.21,6.75,2.55],[15.39,4.3,1.3],[15.94,5.65,1.85],[109.39,22.5,2.5],[158.50,24.5,2.5]]) # erg / s / cm^2 / Hz
kev[:,1:] = kev[:,1:] * 1e-15
wavelength = 1e-9 * 1240.0 / 1000.0
c = 3e8
f = (c/wavelength)
kev_uJy = kev * 1.0
kev_uJy[:,1:] = kev_uJy[:,1:] * 1e23 / f
kev_uJy[:,1:] = kev_uJy[:,1:] * 1e6

data = {}
data['2-4'] = [[216.91,69,15],[256.76,55,12],[272.67,44,11],[288.61,46,11],[115.05,178.1,23.44],[162.89,195.7,28.29],[196.79,157.90,10.95]]
data['4-8'] = [[216.88,39,9],[272.61,36,7],[288.55,35,7],[80.10,37.4,4.2],[112.04,127.4,8.85],[162.89,141.90,14.48],[125.30,82.0,9.3],[149.26,98.9,8.5],[181.64,89.6,13.3]]
data['8-12'] = [[216.85,28,7],[115.05,108.9,14.50],[162.89,89.4,14.07],[125.30,63.7,8.2],[149.26,52.7,6.5],[181.64,57.0,10.9]]
data['12-18'] = [[216.80,21,5],[115.05,131.5,12.00],[162.89,124.1,12.28]]
data['optical'] = [[218.37,0.070,np.inf]]
data['kev'] = np.vstack((kev_uJy,[218.37,1.22e-3,0.25e-3]))

color1 = 'coral'
color2 = 'cornflowerblue'

keys = data.keys()
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotName = "%s/bands.pdf"%(plotDir)
masses = np.linspace(0.5,3.0,100)
plt.figure(figsize=(12,8))
for ii,key in enumerate(keys):
    data_out = np.array(data[key])
    plt.errorbar(data_out[:,0],data_out[:,1],yerr=data_out[:,2],label=key,color=colors[ii],fmt='o')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('Time [days]')
plt.ylabel(r'Flux Density [$\mu$Jy]')
plt.savefig(plotName, bbox_inches='tight')
plt.close()

