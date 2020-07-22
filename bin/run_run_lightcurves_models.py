
import os, sys, glob
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("-n","--name",default="tanaka_compactmergers")
    parser.add_option("-m","--model",default="KaKy2016")
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

if not opts.model in ["KaKy2016", "DiUj2017", "SN"]:
   print "Model must be either: KaKy2016, DiUj2017, SN"
   exit(0)

lightcurvesDir = opts.lightcurvesDir

if opts.doGWs:
    filename = "%s/lightcurves_gw.tmp"%lightcurvesDir
else:
    filename = "%s/lightcurves.tmp"%lightcurvesDir

if opts.doEOSFit:
    eosfitFlag = "--doEOSFit"
else:
    eosfitFlag = ""

if opts.doModels:
    filenames = glob.glob("%s/%s/*.dat"%(opts.outputDir,opts.name))
    for filename in filenames:
        if "Lbol" in filename: continue
        name = filename.split("/")[-1].replace(".dat","")
        if opts.doEjecta:
            system_call = "python run_lightcurves_models.py --model %s --name %s --doModels --doEjecta --errorbudget %.2f %s"%(opts.model,name,opts.errorbudget,eosfitFlag)
        elif opts.doMasses:
            system_call = "python run_lightcurves_models.py --model %s --name %s --doModels --doMasses --errorbudget %.2f %s"%(opts.model,name,opts.errorbudget,eosfitFlag)
        else:
            print "Enable --doEjecta or --doMasses"
            exit(0)
        os.system(system_call)
elif opts.doGoingTheDistance:
    folders = glob.glob("%s/going-the-distance_data/2015/compare/*"%(opts.dataDir))
    for folder in folders:
        name = folder.split("/")[-1]

        if opts.doEjecta:
            system_call = "python run_lightcurves_models.py --model %s --name %s --doGoingTheDistance --doEjecta --errorbudget %.2f %s"%(opts.model,name,opts.errorbudget,eosfitFlag)
        elif opts.doMasses:
            system_call = "python run_lightcurves_models.py --model %s --name %s --doModels --doMasses --errorbudget %.2f %s"%(opts.model,name,opts.errorbudget,eosfitFlag)
        else:
            print "Enable --doEjecta or --doMasses"
            exit(0)
        os.system(system_call)

elif opts.doMassGap:
    filename = "%s/massgap_data/injected_values.txt"%(opts.dataDir)
    data_out = np.loadtxt(filename)
    injnum = data_out[:,0].astype(int)
    snr = data_out[:,1]
    m1 = data_out[:,2]
    m2 = data_out[:,3]
    q = 1/data_out[:,4]
    eff_spin = data_out[:,5]
    totmass = data_out[:,6]
    a1 = data_out[:,7]
    a2 = data_out[:,8]

    for ii in range(len(injnum)):
        if not ((q[ii] >= 3) and (q[ii] <=9)): continue
        if not ((m2[ii] >= 1) and (m2[ii] <=3)): continue
        #if not ((q[ii] >= 3) and (q[ii] <=5)): continue

        c1, c2 = 0.147, 0.147
        mb1, mb2 = lightcurve_utils.EOSfit(m1[ii],c1), lightcurve_utils.EOSfit(m2[ii],c2)
        th = 0.2
        ph = 3.14

        t, lbol, mag = KaKy2016_model(q[ii],a1[ii],m2[ii],mb2,c2,th,ph)
        #print q[ii],a1[ii],m2[ii],mb2,c2,th,ph
        if len(mag) == 0: continue
        idx = np.where(np.isfinite(mag[2]))[0]

        if opts.doEOSFit:
            eosfitFlag = "--doEOSFit"
        else:
            eosfitFlag = ""

        if opts.doEjecta:
            system_call = "python run_lightcurves_models.py --model %s --name %d --doMassGap --doEjecta --errorbudget %.2f %s"%(opts.model,injnum[ii],opts.errorbudget,eosfitFlag)
        elif opts.doMasses:
            system_call = "python run_lightcurves_models.py --model %s --name %d --doModels --doMasses --errorbudget %.2f %s"%(opts.model,injnum[ii],opts.errorbudget,eosfitFlag)
        else:
            print "Enable --doEjecta or --doMasses"
            exit(0)
        #os.system(system_call)

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

        plotDir = "plots/lightcurves_KaKy2016/%s"%name
        filename = os.path.join(plotDir,'samples.dat')
        if os.path.isfile(filename): continue

        if opts.doGWs:
            system_call = "python run_lightcurves_models.py --name %s --doGWs"%(name)
        else:
            system_call = "python run_lightcurves_models.py --name %s"%(name)
        os.system(system_call)

