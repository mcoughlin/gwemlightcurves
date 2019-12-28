
import os, sys
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
from astropy.time import Time

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from scipy.optimize import curve_fit

from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
 
    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-s","--spectraDir",default="../spectra")
    parser.add_option("-n","--name",default="XSGW0818_smooth,G298048_PESSTO_20170819,XSGW0820_smooth,G298048_PESSTO_20170821,XSGW0822_smooth,XSGW0823_smooth,XSGW0824_smooth,XSGW0825_smooth,XSGW0826_smooth,XSGW0827_smooth")
    parser.add_option("-l","--labels",default="08/18,08/19,08/20,08/21,08/22,08/23,08/24,08/25,08/26,08/27")
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-f","--outputName",default="G298048_spectra")

    opts, args = parser.parse_args()

    return opts


# Parse command line
opts = parse_commandline()

plotDir = opts.plotDir
spectraDir = opts.spectraDir
outputDir = opts.outputDir
baseplotDir = opts.plotDir

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'models_spec'
else:
    basename = 'gws_spec'
plotDir = os.path.join(plotDir,basename)
plotDir = os.path.join(plotDir,opts.outputName)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

data = {}
names = opts.name.split(",")
for name in names:
    filename = "%s/%s.dat"%(spectraDir,name)
    data_out = lightcurve_utils.loadEventSpec(filename)    
    data[name] = data_out

labels = opts.labels.split(",")
maxhist = -1e10
colors=cm.rainbow(np.linspace(0,1,len(names)))
plotName = "%s/models_spec.pdf"%(plotDir)
plt.figure(figsize=(10,8))
for name, color, label in zip(names, colors, labels):
    spec_d = data[name]
    plt.semilogy(spec_d["lambda"],spec_d["data"],'-',c=color,linewidth=2,label=label)

#plt.xlim([3000,30000])
plt.ylim([10.0**-18.0,5 * 10.0**-16.0])
plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
plt.legend(ncol=2)
plt.grid()
plt.savefig(plotName,bbox_inches='tight')
plt.close()

