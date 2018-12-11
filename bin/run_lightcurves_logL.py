
import os, sys
import numpy as np
import optparse
import glob

from astropy.table import Table, Column
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
from matplotlib.pyplot import cm

from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots") 
    parser.add_option("-m","--model",default="kasen_kilonova_grid")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-n","--name",default="GW170817")

    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doWaveformExtrapolate",  action="store_true", default=False)

    parser.add_option("--doGWs",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)

    parser.add_option("--doLightcurves",  action="store_true", default=False)
    parser.add_option("--doLuminosity",  action="store_true", default=False)
    parser.add_option("--doSpectra",  action="store_true", default=False)

    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("-f","--filters",default="u,g,r,i,z,J,H,K")
    parser.add_option("--tmax",default=14.0,type=float)
    parser.add_option("--tmin",default=0.05,type=float)
    parser.add_option("--dt",default=0.05,type=float)

    opts, args = parser.parse_args()
 
    return opts

# Parse command line
opts = parse_commandline()

if opts.doFixZPT0:
    ZPRange = 0.1
    T0Range = 0.1
else:
    ZPRange = 5.0
    T0Range = 14.0

filters = opts.filters.split(",")
errorbudget = opts.errorbudget
mint = opts.tmin
maxt = opts.tmax
dt = opts.dt
tt = np.arange(mint,maxt,dt)

baseplotDir = opts.plotDir
basename = 'logL'
plotDir = os.path.join(baseplotDir,basename)
if opts.doFixZPT0:
    plotDir = os.path.join(plotDir,'%s_FixZPT0'%opts.model)
else:
    plotDir = os.path.join(plotDir,'%s'%opts.model)
plotDir = os.path.join(plotDir,"_".join(filters))
plotDir = os.path.join(plotDir,"%.0f_%.0f"%(opts.tmin,opts.tmax))
plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

lightcurveFilesAll = glob.glob(os.path.join(opts.outputDir,opts.model,'*.dat'))
lightcurveFiles = []
for lightcurveFile in lightcurveFilesAll:
    if opts.doLuminosity or opts.doLightcurves:
        if (not "Lbol" in lightcurveFile) and (not "spec" in lightcurveFile):
            lightcurveFiles.append(lightcurveFile)
    elif opts.doSpectra:
        if "spec" in lightcurveFile:
            lightcurveFiles.append(lightcurveFile)
    else:
        print("Add --doLuminosity, --doSpectra, or --doLightcurves")
        exit(0)

mags = {}
lbols = {}
for filename in lightcurveFiles:
    name = filename.replace(".txt","").replace(".dat","").split("/")[-1]
    mag_d = np.loadtxt(filename)
    mags[name] = mag_d

    filename_lbol = filename.replace(".dat","_Lbol.dat")
    lbol_d = np.loadtxt(filename)
    lbols[name] = lbol_d

lightcurvesDir = opts.lightcurvesDir

if opts.doEvent:
    filename = "%s/%s.dat"%(lightcurvesDir,opts.name)
    data_out = lightcurve_utils.loadEvent(filename)    
else:
    print("Add --doEvent")
    exit(0)

for ii,key in enumerate(data_out.iterkeys()):
    if key == "t":
        continue
    else:
        data_out[key][:,0] = data_out[key][:,0] - opts.T0
        data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

for ii,key in enumerate(data_out.iterkeys()):
    if key == "t":
        continue
    else:
        idxs = np.intersect1d(np.where(data_out[key][:,0]>=mint)[0],np.where(data_out[key][:,0]<=maxt)[0])
        data_out[key] = data_out[key][idxs,:]

for ii,key in enumerate(data_out.iterkeys()):
    idxs = np.where(~np.isnan(data_out[key][:,2]))[0]
    if key == "t":
        continue
    else:
        data_out[key] = data_out[key][idxs,:]

for ii,key in enumerate(data_out.iterkeys()):
    if ii == 0:
        samples = data_out[key].copy()
    else:
        samples = np.vstack((samples,data_out[key].copy()))

Global.data_out = data_out
Global.errorbudget = errorbudget
Global.ZPRange = ZPRange
Global.T0Range = T0Range
Global.doLightcurves = 1
Global.filters = filters
Global.doWaveformExtrapolate = opts.doWaveformExtrapolate

loglikes = {}
maxloglike = -np.inf
for name in mags.keys():
    mag_d = mags[name]
    tmag = mag_d[:,0]
    mag = mag_d[:,1:].T

    lbol_d = lbols[name]
    lbol = lbol_d[:,1]
    t0, zp = 0, 0

    prob = calc_prob(tmag, lbol, mag, t0, zp)

    loglikes[name] = prob

    if prob > maxloglike:
        maxloglike = prob
        maxname = name

mag_d = mags[maxname]
tmag = mag_d[:,0]
mag = mag_d[:,1:].T

filename = os.path.join(plotDir,'loglike.dat')
fid = open(filename,'w')
for name in mags.keys():
    keySplit = name.split("_")
    mej0 = float(keySplit[3].replace("m",""))
    vej0 = float(keySplit[4].replace("vk",""))
    if len(keySplit) == 6:
        Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
    elif len(keySplit) == 7:
        if "Xlan1e" in keySplit[6]:
            Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
        elif "Xlan1e" in keySplit[5]:
            Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
    fid.write('%.5f %.5f %.5e %.5f\n'%(mej0,vej0,Xlan0,loglikes[name]))
fid.close()

plotName = "%s/models_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,18))
plt.figure(figsize=(20,28))

zp_best = 0
colors=cm.rainbow(np.linspace(0,1,len(filters)))
tini, tmax, dt = 0.0, 21.0, 0.1
tt = np.arange(tini,tmax,dt)

cnt = 0
for filt, color in zip(filters,colors):
    cnt = cnt+1
    if cnt == 1:
        ax1 = plt.subplot(len(filters),1,cnt)
    else:
        ax2 = plt.subplot(len(filters),1,cnt,sharex=ax1,sharey=ax1)

    if not filt in data_out: continue
    samples = data_out[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,label='%s-band'%filt)

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=10)

    magave = lightcurve_utils.get_mag(mag,filt)
    ii = np.where(~np.isnan(magave))[0]
    f = interp.interp1d(tmag[ii], magave[ii], fill_value='extrapolate')
    maginterp = f(tt)
    plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2)
    plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    if opts.name == "GW170817":
        plt.xlim([0.0, 18.0])
        plt.ylim([-18.0,-10.0])
    else:
        plt.xlim([0.0, 7.0])
        plt.ylim([-22.0,-10.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        #l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filters):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=48)
plt.savefig(plotName)
plt.close()

