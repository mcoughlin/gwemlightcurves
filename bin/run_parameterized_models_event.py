
import os, sys, copy
import numpy as np
import optparse

from scipy.interpolate import interpolate as interp
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves.KNModels import KNTable


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("--posterior_samples", default="../data/event_data/G298048.dat")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-m","--model",default="BHNSKilonovaLightcurve,BNSKilonovaLightcurve,BlueKilonovaLightcurve,ArnettKilonovaLightcurve", help="BHNSKilonovaLightcurve, BNSKilonovaLightcurve, BlueKilonovaLightcurve, ArnettKilonovaLightcurve")
    parser.add_option("--name",default="G298048")

    opts, args = parser.parse_args()
 
    return opts

def hist_results(samples,Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-3
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

# Parse command line
opts = parse_commandline()

models = opts.model.split(",")
for model in models:
    if not model in ["BHNSKilonovaLightcurve", "BNSKilonovaLightcurve", "BlueKilonovaLightcurve", "ArnettKilonovaLightcurve"]:
        print "Model must be either: BHNSKilonovaLightcurve, BNSKilonovaLightcurve, BlueKilonovaLightcurve, ArnettKilonovaLightcurve"
        exit(0)

# These are the default values supplied with respect to generating lightcurves
tini = 0.1
tmax = 50.0
dt = 0.1

vmin = 0.02
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 1

beta = 3.0
kappa_r = 10.0
slope_r = -1.2

# read in samples
samples = KNTable.read_samples('data/event_data/G298048.dat')
# Calc lambdas
samples = samples.calc_tidal_lambda(remove_negative_lambda=True)
# Calc compactness
samples = samples.calc_compactness()
# Calc baryonic mass
samples = samples.calc_baryonic_mass()
#samples = samples.downsample()

#add default values from above to table
samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt
samples['vmin'] = vmin
samples['th'] = th
samples['ph'] = ph
samples['kappa'] = kappa
samples['eps'] = eps
samples['alp'] = alp
samples['eth'] = eth
samples['flgbct'] = flgbct
samples['beta'] = beta
samples['kappa_r'] = kappa_r
samples['slope_r'] = slope_r

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
model_tables = {}
for model in models:
    model_tables[model] = KNTable.model(model, samples)

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,"_".join(models))
plotDir = os.path.join(plotDir,"event")
plotDir = os.path.join(plotDir,opts.name)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]

tini, tmax, dt = 0.1, 50.0, 0.1
tt = np.arange(tini,tmax+dt,dt)

mag_all = {}
lbol_all = {}

for model in models:
    mag_all[model] = {}
    lbol_all[model] = {}

    lbol_all[model] = np.empty((0,len(tt)), float)
    for filt, color, magidx in zip(filts,colors,magidxs):
        mag_all[model][filt] = np.empty((0,len(tt)))
