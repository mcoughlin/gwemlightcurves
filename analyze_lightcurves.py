
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

filename = "%s/lightcurves.tmp"%dataDir
data = loadLightcurves(filename)
for name in data.iterkeys():

    gmag, rmag, imag, zmag, ymag, wmag = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    gshape, rshape, ishape, zshape, yshape, wshape = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

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

    cut10 = ttdiff <= 20
    if ("g" in mags) and ("z" in mags):
        cut4 = mags["g"] - mags["z"] > -2.5*np.log10(0.15)
    else:
        cut4 = True
    if ("r" in mags) and ("z" in mags):
        cut5 = mags["r"] - mags["z"] > -2.5*np.log10(0.4)
    else:
        cut5 = True
    if "r" in shapes:
        cut12a = shapes["r"] > 0
    else:
        cut12a = True
    if "i" in shapes:
        cut12b = shapes["i"] > 0
    else:
        cut12b = True
    if "z" in shapes:
        cut12c = shapes["z"] > 0
    else:
        cut12c = True
    if ("i" in mags) and ("z" in mags):
        cut14 = mags["i"] - mags["z"] > 0.5
    else:
        cut14 = True

    cut0a = len(filters) > 1
    if not (("i" in mags) or ("z" in mags)):
        cut0b = False
    else:
        cut0b = True

    if ("r" in mags) and ("w" in mags):
        cut15a = mags["w"] - mags["r"] > 0
    else:
        cut15a = True
    if ("i" in mags) and ("w" in mags):
        cut15b = mags["w"] - mags["i"] > 0
    else:
        cut15b = True
    if ("z" in mags) and ("w" in mags):
        cut15c = mags["w"] - mags["z"] > 0
    else:
        cut15c = True
    if ("r" in mags) and ("i" in mags):
        cut15d = mags["r"] - mags["i"] > 0
    else:
        cut15d = True
    if ("g" in mags) and ("i" in mags):
        cut15e = mags["g"] - mags["i"] > 0
    else:
        cut15e = True
    if ("g" in mags) and ("r" in mags):
        cut15f = mags["g"] - mags["r"] > 0
    else:
        cut15f = True

    if cut10 and cut4 and cut5 and cut12a and cut12b and cut12c and cut14 and cut0a and cut0b and cut15a and cut15b and cut15c and cut15d and cut15e and cut15f:
        print name, ttdiff, filters, mags

