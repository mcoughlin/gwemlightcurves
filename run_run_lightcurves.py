
import os, sys
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-d","--dataDir",default="lightcurves")
    parser.add_option("--doGWs",  action="store_true", default=False)

    opts, args = parser.parse_args()

    return opts

def loadLightcurves(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[1:]
    lines = filter(None,lines)
    
    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        numid = float(lineSplit[0])
        psid = lineSplit[1]
        filt = lineSplit[2]
        mjd = float(lineSplit[3])
        mag = float(lineSplit[4])
        dmag = float(lineSplit[5])
    
        if not psid in data:
            data[psid] = {}
        if not filt in data[psid]:
            data[psid][filt] = np.empty((0,3), float)
        data[psid][filt] = np.append(data[psid][filt],np.array([[mjd,mag,dmag]]),axis=0)

    return data

# Parse command line
opts = parse_commandline()
dataDir = opts.dataDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%dataDir
else:
    filename = "%s/lightcurves.tmp"%dataDir
data = loadLightcurves(filename)
for name in data.iterkeys():

    data_out = data[name]
    mags = {}
    shapes = {}
    filters = []
    for ii,key in enumerate(data_out.iterkeys()):
        if ii == 0:
            samples = data_out[key].copy()
        else:
            samples = np.vstack((samples,data_out[key].copy()))
        filters.append(key)

        mags[key] = np.min(data_out[key][:,1])
        F1 = data_out[key][0,1]
        t1 = data_out[key][0,0]
        F2 = data_out[key][-1,1]
        t2 = data_out[key][-1,0]
        shapes[key] = (1/F1)*((F2-F1)/(t2-t1))

    tt = np.sort(samples[:,0])
    ttmin = np.min(tt)
    ttmax = np.max(tt)

    ttdiff = ttmax - ttmin

    if opts.doGWs:
        system_call = "python run_lightcurves.py --name %s --doGWs"%(name)
    else:
        system_call = "python run_lightcurves.py --name %s"%(name)
    os.system(system_call)

