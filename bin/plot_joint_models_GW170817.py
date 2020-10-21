
import os, sys, glob, pickle
import optparse
import numpy as np
import pandas
from scipy.interpolate import interpolate as interp
import scipy.stats as ss

from astropy.table import Table, Column

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
from matplotlib.colors import ListedColormap

import corner
import seaborn as sns
sns.set(style="ticks", color_codes=True)

#import seaborn as sns
#sns.set()
import pandas as pd

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global
from gwemlightcurves.EOS.EOS4ParameterPiecewisePolytrope import EOS4ParameterPiecewisePolytrope

from pycbc.io.inference_hdf import InferenceFile
from pycbc import conversions

def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2, :]
    den_pts = pts[Npts/2:, :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

def myloglike_combined(cube, ndim, nparams):
        var1 = cube[0]
        var2 = cube[1]
        vals = np.array([var1,var2]).T

        kdeeval_gw = kde_eval(kdedir_gw,vals)[0]
        prob_gw = np.log(kdeeval_gw)
        kdeeval_em = kde_eval(kdedir_em,vals)[0]
        prob_em = np.log(kdeeval_em)
        prob = prob_gw + prob_em

        if np.isnan(prob):
            prob = -np.inf

        return prob

def myprior_combined(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*640.0

plotDir = '../plots/gws/Ka2017_combine/EOS'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

n_live_points = 5000
evidence_tolerance = 0.5

filename = '../plots/fitting_gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/joint/GW170817/1.00/0_640/q_lambdatilde.dat' 
data = np.loadtxt(filename)
q_em = data[:,0]
lambdatilde_em = data[:,1]

filename = '/home/mcoughlin/gw170817-common-eos/uniform_mass_prior_common_eos_posteriors.hdf'
lambda1='lambdasym*((mass2/mass1)**3)'
lambda2='lambdasym*((mass1/mass2)**3)'
# read samples
params = [lambda1, lambda2]
with InferenceFile(filename, "r") as fp:
    samples = fp.read_samples(params)
mass1 = samples['mass1'][:]
mass2 = samples['mass2'][:]
lambdasym = samples['lambdasym'][:]
lambda1=lambdasym*((mass2/mass1)**3)
lambda2=lambdasym*((mass1/mass2)**3)
lambdatilde_gw = conversions.lambda_tilde(mass1, mass2, lambda1, lambda2)
q_gw = mass1/mass2

pts_em = np.vstack((q_em,lambdatilde_em)).T
pts_gw = np.vstack((q_gw,lambdatilde_gw)).T
kdedir_em = greedy_kde_areas_2d(pts_em)
kdedir_gw = greedy_kde_areas_2d(pts_gw)

parameters = ["q","lambdatilde"]
n_params = len(parameters)
pymultinest.run(myloglike_combined, myprior_combined, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

labels = [r"q",r"$\tilde{\Lambda}$"]
multifile = lightcurve_utils.get_post_file(plotDir)
data_combined = np.loadtxt(multifile)
q_combined = data_combined[:,0]
lambdatilde_combined = data_combined[:,1]
data_combined = np.vstack((q_combined,lambdatilde_combined)).T

plotName = "%s/corner_combined.pdf"%(plotDir)
figure = corner.corner(data_combined, labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True, title_kwargs={"fontsize": 24},
    label_kwargs={"fontsize": 28}, title_fmt=".2f")
figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

print("GW: %.5f EM: %.5f Combined: %.5f"%(np.percentile(q_gw,90),np.percentile(q_em,90),np.percentile(q_combined,90)))
print("GW: %.5f EM: %.5f Combined: %.5f"%(np.percentile(lambdatilde_gw,10),np.percentile(lambdatilde_em,10),np.percentile(lambdatilde_combined,10)))

q_percentiles = np.percentile(q_em,[10,50,90])
lambdatilde_percentiles = np.percentile(lambdatilde_em,[10,50,90]) 

eosfilename = '../input/lalsim/polytrope_table.dat'
df = pandas.read_table(eosfilename, skiprows=1, delim_whitespace=True, names=['eos','logP1','gamma1','gamma2','gamma3'])

color1 = 'coral'
color2 = 'cornflowerblue'

labels = [r"$q$",r"$\tilde{\Lambda}$"]
plotName = "%s/q_lambda_corner.pdf"%(plotDir)
figure, axes = plt.subplots(2, 2, figsize=(20, 20))
corner.corner(data, labels=labels,fig=figure,smooth=3,color=color2,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 30},
                       label_kwargs={"fontsize": 30}, title_fmt=".2f")
#eosnames = ['AP4','SLy','MPA1','H4','MS1b','MS1']
eosnames = df['eos']
eosnames = ["ALF2","MPA1","AP3","SLy","ALF4","AP4","WFF3","WFF1"]
eosnames = ["MPA1","SLy","AP4","WFF1"]
qs = np.linspace(1,1.5,100)
mc = 1.188
        
for ii,eosname in enumerate(eosnames):
    lambdatildes = []
    eos = EOS4ParameterPiecewisePolytrope(eosname)
    plotline = True
    for q in qs:
        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        try:
            lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)

            lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
            lambdatildes.append(lambdatilde)
        except:
            plotline = False
            lambdatildes.append(0)
    if plotline:
        axes[1][0].plot(qs,lambdatildes,'--',c=color1)
        axes[1][0].text(qs[-1],lambdatildes[-1],eosname,fontsize=24)

        axes[1][1].fill_between([np.min(lambdatildes),np.max(lambdatildes)],[-1000,-1000],[1000,1000],facecolor=color1,alpha=0.5)
        axes[1][1].text(np.min(lambdatildes),5+31*ii,eosname,fontsize=24)

axes[1][0].set_xlim([1,1.6])
axes[1][0].set_ylim([0,640])

figure.set_size_inches(14.0,14.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

plotName = "%s/q_lambda_corner_seaborn.pdf"%(plotDir)
fig_width_pt = 400.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'pdf',
           'axes.labelsize': 17,
           'font.size': 17,
           'legend.fontsize': 17,
           'xtick.labelsize': 17,
           'ytick.labelsize': 17,
           'text.usetex': True,
           'figure.figsize': fig_size}
plt.rcParams.update(params)

# read data
filename = 'plot.out'
data = np.loadtxt(filename)
ql = pd.DataFrame(data, columns=["x", "y"])
cm = ListedColormap(['cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue'])
my_cmap = cm(np.arange(cm.N))
my_cmap[:,-1] = np.linspace(0, 1, cm.N)
my_cmap = ListedColormap(my_cmap)

#plot data
g = sns.JointGrid(x="x", y="y", data=ql,space=0.2,xlim=(1.0,1.7),ylim=(0,640))
g = g.plot_joint(sns.kdeplot, cmap=my_cmap,
    shade=True,shade_lowest=False,kernel='gau')
g = g.plot_marginals(sns.distplot, kde=False, color=color2)
g.set_axis_labels("$q$", "$\\tilde{\\Lambda}$");
g.ax_marg_x.plot([1.38,1.38], [0.,4800.], color=color2,linestyle='--', linewidth = 2)
g.ax_marg_y.plot([0.,1500], [197.,197.], color=color2,linestyle='--', linewidth = 2)
g.ax_marg_x.text(1.42,3500,'$q < 1.38$',color=color2,fontsize=14)
g.ax_marg_y.text(1100,150,'$\\tilde{\\Lambda} > 197$',color=color2,fontsize=14,rotation=-90)

bounds = [1.0,1.7]
bins, hist1 = lightcurve_utils.hist_results(q_gw,Nbins=15,bounds=bounds)
hist1 = 4800.0*hist1/np.max(hist1)
for ii in range(len(bins)-1):
    bin_start, bin_end = bins[ii], bins[ii+1]
    val = hist1[ii]
    #g.ax_marg_x.fill_between([bin_start, bin_end],[0,0],[val,val],facecolor=color1,alpha=1.0)
    #g.ax_marg_x.plot([bin_start, bin_end],[val,val],color=color1,alpha=1.0)

bounds = [0,600]
bins, hist1 = lightcurve_utils.hist_results(lambdatilde_gw,Nbins=25,bounds=bounds)
hist1 = 1500.0*hist1/np.max(hist1)
for ii in range(len(bins)-1):
    bin_start, bin_end = bins[ii], bins[ii+1]
    val = hist1[ii]
    #g.ax_marg_y.plot([val,val],[bin_start, bin_end],color=color1,alpha=1.0)

for ii,eosname in enumerate(eosnames):
    lambdatildes = []
    eos = EOS4ParameterPiecewisePolytrope(eosname)
    plotline = True
    for q in qs:
        eta = lightcurve_utils.q2eta(q)
        (m1,m2) = lightcurve_utils.mc2ms(mc,eta)

        try:
            lambda1, lambda2 = eos.lambdaofm(m1), eos.lambdaofm(m2)

            lambdatilde = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
            lambdatildes.append(lambdatilde)
        except:
            plotline = False
            lambdatildes.append(0)

    lambdatildes = np.array(lambdatildes)
    idx = np.where((qs>=1) & (qs<=1.38))[0]

    if plotline:
        g.ax_joint.plot(qs,lambdatildes,'--',c=color1,alpha=0.85)
        g.ax_marg_y.fill_between([0.,1500],[np.min(lambdatildes[idx]),np.min(lambdatildes[idx])],[np.max(lambdatildes[idx]),np.max(lambdatildes[idx])],facecolor=color1,alpha=1.0)
        if eosname == "AP4":
            g.ax_joint.text(qs[-1]+0.05,lambdatildes[-1],"APR4",fontsize=17,color=color1,alpha=0.85)
            g.ax_marg_y.text(1550,np.max(lambdatildes[idx])+20,"APR4",fontsize=14,color=color1,rotation=-90)
        else:
            g.ax_joint.text(qs[-1]+0.05,lambdatildes[-1],eosname,fontsize=17,color=color1,alpha=0.85)
            g.ax_marg_y.text(1550,np.max(lambdatildes[idx])+20,eosname,fontsize=14,color=color1,rotation=-90)

plt.savefig(plotName, bbox_inches='tight')
plt.close()

#plotName = "%s/q_lambda_corner_sns.pdf"%(plotDir)
#figure = plt.figure(figsize=(20,20))
#with sns.axes_style('white'):
#  sns.jointplot("x", "y", data.T, kind='hex')
#figure.set_size_inches(14.0,14.0)
#plt.savefig(plotName, bbox_inches='tight')
#plt.close()
