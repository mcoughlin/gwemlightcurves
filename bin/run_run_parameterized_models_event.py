
import os, sys
import numpy as np

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

from gwemlightcurves import lightcurve_utils, Global

distance_sets = [[1,300],[300,1000],[1000,10000],[10000,100000]]
filts = ["u","g","r","i","z","y","J","H","K"]

cbc_type = "BNS"
cbc_type = "BHNS"
cbc_list = "../data/3G_Lists/list_%s_detected_3G_median_12.txt"%cbc_type

baseoutputDir = "../plots/Me2017/event/3G/%s"%cbc_type
lines = [line.rstrip('\n') for line in open(cbc_list)]

data = {}
for distance_set in distance_sets:
    
    distance_underscore = "%d_%d"%(distance_set[0],distance_set[1])
    data[distance_underscore] = {}

    outputDir = os.path.join(baseoutputDir,distance_underscore,"dat")
    filename = os.path.join(outputDir,"Me2017_u_list.dat")
    #if not os.path.isfile(filename):
    if True:
        system_call = "python run_parameterized_models_event.py --nsamples 1000 --mindistance %d --maxdistance %d --cbc_type %s --cbc_list %s"%(distance_set[0],distance_set[1],cbc_type,cbc_list)
        print(system_call)
        print(stop)
        os.system(system_call)

    for filt in filts:
        filename = os.path.join(outputDir,"Me2017_%s_list.dat"%filt)
        if not os.path.isfile(filename): continue
        data_out = np.loadtxt(filename)
        data[distance_underscore][filt] = data_out

    outputDir = os.path.join(baseoutputDir,distance_underscore)
    filename = os.path.join(outputDir,"cbcratio.dat")
    if not os.path.isfile(filename): continue
    data_out = np.loadtxt(filename)
    data[distance_underscore]["rate"] = data_out

color2 = 'coral'
color1 = 'cornflowerblue'
color3 = 'palegreen'
color4 = 'darkmagenta'
colors_names=[color1,color2,color3,color4]

bounds = [16,34]
xlims = [15.0,35.0]
ylims = [1e-2,1]

plotName = "%s/appi.pdf"%(baseoutputDir)
plt.figure(figsize=(12,8))
for ii,distance_set in enumerate(distance_sets):
    distance_underscore = "%d_%d"%(distance_set[0],distance_set[1])
    legend_name = "%d-%d Mpc"%(distance_set[0],distance_set[1])
    bins, hist1 = lightcurve_utils.hist_results(data[distance_underscore]["K"][:,4],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    bins, hist1 = lightcurve_utils.hist_results(data[distance_underscore]["g"][:,4],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'--',color=colors_names[ii],linewidth=3)
plt.xlabel(r"Apparent Magnitude [mag]",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

ylims = [1,1e6]
plotName = "%s/rates.pdf"%(baseoutputDir)
plt.figure(figsize=(10,8))
for ii,distance_set in enumerate(distance_sets):
    distance_underscore = "%d_%d"%(distance_set[0],distance_set[1])
    legend_name = "%d-%d Mpc"%(distance_set[0],distance_set[1])
    bins, hist1 = lightcurve_utils.hist_results(data[distance_underscore]["K"][:,4],Nbins=25,bounds=bounds)
    hist1_cumsum = float(data[distance_underscore]["rate"][0])*hist1 / np.sum(hist1)
    hist1_cumsum = np.cumsum(hist1_cumsum)
    plt.semilogy(bins,hist1_cumsum,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    bins, hist1 = lightcurve_utils.hist_results(data[distance_underscore]["g"][:,4],Nbins=25,bounds=bounds)
    hist1_cumsum = float(data[distance_underscore]["rate"][0])*hist1 / np.sum(hist1)
    hist1_cumsum = np.cumsum(hist1_cumsum)
    plt.semilogy(bins,hist1_cumsum,'--',color=colors_names[ii],linewidth=3)
plt.xlabel(r"Apparent Magnitude [mag]",fontsize=24)
plt.ylabel("Rate of apparent magnitude [per year]",fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()

bounds = [0.9,3.1]
xlims = [1.0,3.0]
ylims = [1e-2,10]

plotName = "%s/q.pdf"%(baseoutputDir)
plt.figure(figsize=(12,8))
for ii,distance_set in enumerate(distance_sets):
    distance_underscore = "%d_%d"%(distance_set[0],distance_set[1])
    legend_name = "%d-%d Mpc"%(distance_set[0],distance_set[1])
    bins, hist1 = lightcurve_utils.hist_results(data[distance_underscore]["K"][:,0],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'-',color=colors_names[ii],linewidth=3,label=legend_name)
    bins, hist1 = lightcurve_utils.hist_results(data[distance_underscore]["g"][:,0],Nbins=25,bounds=bounds)
    plt.semilogy(bins,hist1,'--',color=colors_names[ii],linewidth=3)
plt.xlabel(r"Mass Ratio",fontsize=24)
plt.ylabel('Probability Density Function',fontsize=24)
plt.legend(loc="best",prop={'size':24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig(plotName)
plt.close()
