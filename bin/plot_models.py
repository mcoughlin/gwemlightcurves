
import os, sys
import optparse
import numpy as np

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import gwemlightcurves.lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
 
    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")

    #parser.add_option("-n","--name",default="a80_leak_HR,ALF2Q3a30,bp_CaFN_hv_h,Ia_1994D")
    #parser.add_option("-n","--name",default="rpft_m001_v1,ALF2Q3a30,a80_leak_HR,bp_CaFN_hv_h,neutron_precursor")
    #parser.add_option("-f","--outputName",default="combined")
    parser.add_option("-n","--name",default="rpft_m05_v2,BHNS_H4M050V02,BNS_H4M050V02,neutron_precursor3,SED_nsbh1")
    parser.add_option("-f","--outputName",default="fiducial")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

outputDir = opts.outputDir
baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,opts.outputName)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog"]
models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich and Ujevic (2017)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)"]

names = opts.name.split(",")
filenames = []
legend_names = []
for name in names:
    for ii,model in enumerate(models):
        filename = '%s/%s/%s.dat'%(outputDir,model,name)
        if not os.path.isfile(filename):
            continue
        filenames.append(filename)
        legend_names.append(models_ref[ii])
        break
mags, names = gwemlightcurves.lightcurve_utils.read_files(filenames)

colors = ["g","r","c","y","m"]
plotName = "%s/models.pdf"%(plotDir)
plt.figure()
for ii,name in enumerate(names):
    mag_d = mags[name]
    indexes = np.where(~np.isnan(mag_d["g"]))[0]
    index1 = indexes[0]
    index2 = int(len(indexes)/2)
    offset = -mag_d["g"][index2] + ii*3
    offset = 0.0
    t = mag_d["t"]
    #offset = -mag_d["g"][index1]
    linestyle = "%s-"%colors[ii]
    plt.semilogx(t,mag_d["i"]+offset,linestyle,label=legend_names[ii])
    linestyle = "%s--"%colors[ii]
    plt.semilogx(t,mag_d["g"]+offset,linestyle)
    #textstr = textstrings[ii]
    #textstr = name
    #plt.text(2, ii*3 - 1, textstr)

plt.xlim([10**-2,50])
plt.xlabel('Time [days]')
plt.ylabel('AB Magnitude')
plt.legend(loc="best")
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
plt.ylim([-20,10])
plt.grid()
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

filenames = []
legend_names = []
for name in names:
    for ii,model in enumerate(models):
        filename = '%s/%s/%s_Lbol.dat'%(outputDir,model,name)
        if not os.path.isfile(filename):
            continue
        filenames.append(filename)
        legend_names.append(models_ref[ii])
        break
Lbols, names = gwemlightcurves.lightcurve_utils.read_files_lbol(filenames)

colors = ["g","r","c","y","m"]
plotName = "%s/models_Lbol.pdf"%(plotDir)
plt.figure()
for ii,name in enumerate(names):
    Lbol_d = Lbols[name]
    indexes = np.where(~np.isnan(Lbol_d["Lbol"]))[0]
    index1 = indexes[0]
    index2 = int(len(indexes)/2)
    offset = 0.0
    t = Lbol_d["t"]
    #offset = -mag_d["g"][index1]
    linestyle = "%s-"%colors[ii]
    plt.loglog(t,Lbol_d["Lbol"]+offset,linestyle,label=legend_names[ii])
    #textstr = textstrings[ii]
    #textstr = name
    #plt.text(2, ii*3 - 1, textstr)

plt.xlim([10**-2,50])
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.legend(loc="best")
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
#plt.ylim([-5,20])
plt.grid()
plt.savefig(plotName)
plt.close()
