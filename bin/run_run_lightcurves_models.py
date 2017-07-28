
import os, sys, glob
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

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../lightcurves")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-n","--name",default="tanaka_compactmergers")
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)

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

if not opts.model in ["BHNS", "BNS", "SN"]:
   print "Model must be either: BHNS, BNS, SN"
   exit(0)

dataDir = opts.dataDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%dataDir
else:
    filename = "%s/lightcurves.tmp"%dataDir

if opts.doModels:
    filenames = glob.glob("%s/%s/*.dat"%(opts.outputDir,opts.name))
    for filename in filenames:
        if "Lbol" in filename: continue
        name = filename.split("/")[-1].replace(".dat","")
        if opts.doEjecta:
            system_call = "python run_lightcurves_models.py --model %s --name %s --doModels --doEjecta --errorbudget %.2f"%(opts.model,name,opts.errorbudget)
        elif opts.doMasses:
            system_call = "python run_lightcurves_models.py --model %s --name %s --doModels --doMasses --errorbudget %.2f"%(opts.model,name,opts.errorbudget)
        else:
            print "Enable --doEjecta or --doMasses"
            exit(0)
        os.system(system_call)
else:
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

        cut10 = ttdiff <= 20
        if not cut10: continue
        cut0a = len(filters) > 1
        if not cut0a: continue

        plotDir = "plots/lightcurves_BHNS/%s"%name
        filename = os.path.join(plotDir,'samples.dat')
        if os.path.isfile(filename): continue

        if opts.doGWs:
            system_call = "python run_lightcurves_models.py --name %s --doGWs"%(name)
        else:
            system_call = "python run_lightcurves_models.py --name %s"%(name)
        os.system(system_call)

