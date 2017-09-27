
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

from gwemlightcurves import BHNSKilonovaLightcurve, BNSKilonovaLightcurve, SALT2
from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
 
    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")
    parser.add_option("-n","--name",default="G298048_PESSTO_20170818,G298048_PESSTO_20170819,G298048_PESSTO_20170820,G298048_PESSTO_20170821,G298048_XSH_20170819,G298048_XSH_20170821")
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-f","--outputName",default="G298048_spectra")

    opts, args = parser.parse_args()

    return opts


# Parse command line
opts = parse_commandline()

lightcurvesDir = opts.lightcurvesDir
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

    names = opts.name.split(",")
    filenames = []
    legend_names = []
    for name in names:
        for ii,model in enumerate(models):
            filename = '%s/%s/%s_spec.dat'%(outputDir,model,name)
            if not os.path.isfile(filename):
                continue
            filenames.append(filename)
            legend_names.append(models_ref[ii])
            break
    specs, names = lightcurve_utils.read_files_spec(filenames)

    if opts.doEvent:
        filename = "%s/%s.dat"%(spectraDir,opts.event)
        data_out = lightcurve_utils.loadEventSpec(filename)

    maxhist = -1e10
    colors = ["g","r","c","y","m"]
    plotName = "%s/models_spec.pdf"%(plotDir)
    plt.figure(figsize=(12,10))
    for ii,name in enumerate(names):
        spec_d = specs[name]
        spec_d_mean = np.mean(spec_d["data"],axis=0)
        linestyle = "%s-"%colors[ii]
        plt.loglog(spec_d["lambda"],np.abs(spec_d_mean),linestyle,label=legend_names[ii],linewidth=2)
        maxhist = np.max([maxhist,np.max(np.abs(spec_d_mean))])
 
    if opts.doEvent:
        plt.errorbar(data_out["lambda"],np.abs(data_out["data"])*maxhist/np.max(np.abs(data_out["data"])),fmt='--',c='k',label='event')

    plt.xlim([3000,30000])
    #plt.ylim([10.0**39,10.0**43])
    plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
    plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(plotName)
    plt.close()

