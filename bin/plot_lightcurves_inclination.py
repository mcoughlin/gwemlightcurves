
# coding: utf-8

# In[49]:


from __future__ import division, print_function # python3 compatibilty
import optparse
import pandas
import numpy as np                  # import numpy
from time import time               # use for timing functions
# make the plots look a bit nicer with some defaults
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# functions for plotting posteriors
import scipy
import corner
#------------------------------------------------------------
# Read the data

import scipy.stats as ss

import os

from gwemlightcurves.KNModels import KNTable

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")

    parser.add_option("--multinest_samples", default="../plots/gws/Ka2017x2inc_FixZPT0/a2.0_t45/u_g_r_i_z_y_J_H_K/0_7/ejecta/GW170817/1.00/2-post_equal_we,../plots/gws/Ka2017x2inc_FixZPT0/a4.0_t45/u_g_r_i_z_y_J_H_K/0_7/ejecta/GW170817/1.00/2-post_equal_we,../plots/gws/Ka2017x2inc_FixZPT0/DZ2_t45/u_g_r_i_z_y_J_H_K/0_7/ejecta/GW170817/1.00/2-post_equal_wei")
    parser.add_option("-m","--model",default="Ka2017x2inc", help="Ka2017,Ka2017x2")

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

baseplotDir = os.path.join(opts.plotDir,'angles')
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)

plotDir = baseplotDir
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

multinest_samples = opts.multinest_samples.split(",")
samples_all = {}
for multinest_sample in multinest_samples:
    samples = KNTable.read_multinest_samples(multinest_sample, opts.model)
    key = multinest_sample.split("/")[-7]
    samples_all[key] = samples

color1 = 'cornflowerblue'
color2 = 'coral'
color3 = 'forestgreen'

color_names = [color1, color2, color3]
linestyles = ['-','--','-.']
bins_hist = np.arange(-95,95,10)
bins_hist = np.arange(-10,190,10)

plt.figure(figsize=(10,7))
ax = plt.subplot(111)

for ii, key in enumerate(list(samples_all.keys())):
    label = key.split("_")
    if label[0] == "a2.0":
        label = "Kasen et al: a=2"
    elif label[0] == "a4.0":
        label = "Kasen et al: a=4"
    elif label[0] == "DZ2":
        label = r"Wollaeger et al: $\mathrm{DZ}_2$"
 
    color = color_names[1]
    linestyle = linestyles[ii]
    inc = samples_all[key]["inclination"]
    #inc[inc > 90] = inc[inc > 90] - 180.0
    hist, bin_edges = np.histogram(inc, bins_hist, density=True)
    perc_10, perc_50, perc_90 = np.percentile(inc,10), np.percentile(inc,50), np.percentile(inc,90)
    #plt.plot([perc_10,perc_10],[0,1],'--',color=color)
    #plt.plot([perc_50,perc_50],[0,1],'-',color=color)
    #plt.plot([perc_90,perc_90],[0,1],'--',color=color)
    #n, bins, patches = plt.hist(inc, bins_hist, normed=1, facecolor=color, alpha=0.75)
    bins = (bin_edges[:-1] + bin_edges[1:])/2.0
    plt.step(bins, hist, color = color, linestyle=linestyle, label=label,
             linewidth=3)

    print(key, perc_10, perc_50, perc_90)

superluminal_mu, superluminal_std = 0.30, 0.02
superluminal_mu = 180-superluminal_mu*360.0/(2*np.pi)
superluminal_std = superluminal_std*360.0/(2*np.pi)
plt.plot([superluminal_mu,superluminal_mu],[0,1],alpha=0.3, color='g',label='Superluminal')
rect1 = Rectangle((superluminal_mu - superluminal_std, 0), 2*superluminal_std, 0.05, alpha=0.8, color='g')
rect2 = Rectangle((superluminal_mu - 2*superluminal_std, 0), 4*superluminal_std, 0.05, alpha=0.5, color='g')

gws_mu, gws_std = 152, 14
plt.plot([gws_mu,gws_mu],[0,1],alpha=0.3, color='r',label='GW')
rect3 = Rectangle((gws_mu - gws_std, 0), 2*gws_std, 0.05, alpha=0.3, color='r')
rect4 = Rectangle((gws_mu - 2*gws_std, 0), 4*gws_std, 0.05, alpha=0.1, color='r')

afterglow_mu, afterglow_std = 0.52, 0.16
afterglow_mu = 180-afterglow_mu*360.0/(2*np.pi)
afterglow_std = afterglow_std*360.0/(2*np.pi)
plt.plot([afterglow_mu,afterglow_mu],[0,1],alpha=0.3, color='b',label='Afterglow')
rect5 = Rectangle((afterglow_mu - afterglow_std, 0), 2*afterglow_std, 0.05, alpha=0.3, color='b')
rect6 = Rectangle((afterglow_mu - 2*afterglow_std, 0), 4*afterglow_std, 0.05, alpha=0.1, color='b')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)

plt.xlabel('Inclination [Degrees]')
plt.ylabel('Probability')
plt.grid(True)
plt.xlim([0,180])
plt.ylim([0,0.02])
plt.legend(loc=2)
plt.show()
plotName = os.path.join(plotDir,'inc.pdf')
plt.savefig(plotName)
plt.close()

