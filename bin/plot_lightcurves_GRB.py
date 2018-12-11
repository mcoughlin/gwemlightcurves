
import os, sys, glob, copy
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

import corner

from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("--outputName",default="GRB")

    opts, args = parser.parse_args()

    return opts

# setting seed
np.random.seed(0)

# Parse command line
opts = parse_commandline()

lightcurvesDir = opts.lightcurvesDir

filename = "%s/GRB.dat"%lightcurvesDir
lines = [line.rstrip('\n') for line in open(filename)]

lightcurveFiles = glob.glob(os.path.join(lightcurvesDir,'GRB*.dat')) + [os.path.join(lightcurvesDir,'GW170817.dat')]

data = {}
data_peak = {}
data_peak_app = {}
for lightcurveFile in lightcurveFiles:
    grbname = lightcurveFile.split('/')[-1].replace('.dat','')
    if grbname == "GRB": continue
    data_out = lightcurve_utils.loadEvent(lightcurveFile)
    data[grbname] = data_out

    if grbname == "GW170817":
        grb, filts, mjd, dist = "GW170817", "r", 57982.5285236896, 40.0
    else:
        for line in lines:
            lineSplit = line.split(" ")
            if lineSplit[0] == grbname: 
                grb = lineSplit[0]
                filts = lineSplit[1]
                mjd = float(lineSplit[2])
                dist = float(lineSplit[3])
                break

    print(grbname)

    magmin = np.inf
    for band in data_out.iterkeys():
        banddata = data_out[band]
        idx = np.where(np.isfinite(banddata[:,2]))[0]
        if len(idx) == 0: continue
        thismagmin = np.min(banddata[idx,1])

        if thismagmin > magmin: continue
        thismagmin = magmin
        data_peak[grbname] = copy.deepcopy(banddata)
        data_peak[grbname][:,0] = data_peak[grbname][:,0] - mjd
        data_peak[grbname][:,1] = data_peak[grbname][:,1] - 5*(np.log10(dist*1e6) - 1)

        data_peak_app[grbname] = copy.deepcopy(banddata)
        data_peak_app[grbname][:,0] = data_peak_app[grbname][:,0] - mjd

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,'GRB')
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

colors = ['b','g','r','m','c']
linestyles = ['-', '-.', ':','--']

plotName = "%s/lightcurves.pdf"%(plotDir)
plt.figure(figsize=(10,8))
ax = plt.subplot(111)
maxhist = -1
for ii,name in enumerate(sorted(data_peak.keys())):
    banddata = data_peak[name]
    tt = banddata[:,0]
    mag = banddata[:,1]
    magerr = banddata[:,2]
    plt.plot(tt,mag,'k--')
    plt.errorbar(tt,mag,magerr,fmt='kx')

dist = 200.0
magmin = 20.3 - 5*(np.log10(dist*1e6) - 1)
magmax = 0.0 - 5*(np.log10(dist*1e6) - 1)
plt.fill_between([9.1/24.0,36.0/24.0], [magmin,magmin], [magmax,magmax], facecolor='gray', interpolate=True,alpha=0.3)

plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.xlim([0.001,5.0])
plt.ylim([-30.0,-10.0])
ax.set_xscale("log")
plt.gca().invert_yaxis()
plt.show()
plt.savefig(plotName,dpi=200)
plt.close('all')

plotName = "%s/lightcurves_apparent.pdf"%(plotDir)
plt.figure(figsize=(10,8))
ax = plt.subplot(111)
maxhist = -1
for ii,name in enumerate(sorted(data_peak_app.keys())):
    banddata = data_peak_app[name]
    tt = banddata[:,0]
    mag = banddata[:,1]
    magerr = banddata[:,2]
    plt.plot(tt,mag,'k--')
    plt.errorbar(tt,mag,magerr,fmt='kx')

magmin = 20.3 
magmax = 0.0 
plt.fill_between([9.1/24.0,36.0/24.0], [magmin,magmin], [magmax,magmax], facecolor='gray', interpolate=True,alpha=0.3)

plt.xlabel('Time [days]')
plt.ylabel('Apparent AB Magnitude')
plt.xlim([0.001,5.0])
plt.ylim([10.0,30.0])
ax.set_xscale("log")
plt.gca().invert_yaxis()
plt.show()
plt.savefig(plotName,dpi=200)
plt.close('all')
