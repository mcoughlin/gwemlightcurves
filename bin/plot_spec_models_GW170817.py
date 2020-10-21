
import os, sys, glob, pickle
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

from astropy.table import Table, Column

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30

import corner

import pymultinest
from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

def easyint(x,y,xref):
    ir = (xref>=min(x))&(xref<=max(x))
    yint = interp.interp1d(x[np.argsort(x)],y[np.argsort(x)])(xref[ir])
    #yout = np.zeros(len(xref),dmodel=float)
    yout = np.zeros(len(xref),)
    yup = y[-1]
    ylow = y[0]
    yout[ir] = yint
    yout[xref<np.min(x)] = ylow
    yout[xref>np.max(x)] = yup
    return yout

def spec2mag(lam,Llam,band):

    S = 0.1089/band[:,0]**2
    S1 = S*band[:,1]
    ZP = np.trapz(S1,x=band[:,0])
   
    c    = 2.99e10
    nu  = np.flipud(c/lam*1e8)

    D_cm = 10*3.0857e16*100 # 10 pc in cm

    spec = np.array(zip(lam,Llam/(4*np.pi*D_cm**2)))
    spec1 = easyint(spec[:,0],spec[:,1],band[:,0])

    conv = spec1*band[:,1]
    flux = np.trapz(conv,x=band[:,0])
    mag = -2.5*np.log10(flux/ZP)

    return mag

plotDir = '../plots/gws/Ka2017_combine'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

errorbudgetmag = 1.00
errorbudget = 2.00

plotDir1 = '../plots/gws_spec/Ka2017_FixZPT0/5000_25000/GW170817/2.00/'
pcklFile = os.path.join(plotDir1,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data1, t_best1, lambdas_best1, spec_best1, t0_best1, zp_best1, n_params1, labels1, truths1) = pickle.load(f)
f.close()

plotDir2 = '../plots/gws_spec/Ka2017x2_FixZPT0/5000_25000/GW170817/2.00/'
pcklFile = os.path.join(plotDir2,"data.pkl")
f = open(pcklFile, 'r')
(data_out, data2, t_best2, lambdas_best2, spec_best2, t0_best2, zp_best2, n_params2, labels2, truths2) = pickle.load(f)
f.close()

plotDir3 = '../plots/gws/Ka2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/'
pcklFile = os.path.join(plotDir3,"data.pkl")
f = open(pcklFile, 'r')
(data_out3, data3, tmag3, lbol3, mag3, t0_best3, zp_best3, n_params3, labels3, best3, truths3) = pickle.load(f)
f.close()

spec_best_dic1 = {}
for key in data_out:
    f = interp.interp2d(t_best1+t0_best1, lambdas_best1, np.log10(spec_best1), kind='cubic')
    flux1 = (10**(f(float(key),data_out[key]["lambda"]))).T
    zp_factor = 10**(zp_best1/-2.5)
    flux1 = flux1*zp_factor
    spec_best_dic1[key] = {}
    spec_best_dic1[key]["lambda"] = data_out[key]["lambda"]
    spec_best_dic1[key]["data"] = np.squeeze(flux1)

spec_best_dic2 = {}
for key in data_out:
    f = interp.interp2d(t_best2+t0_best2, lambdas_best2, np.log10(spec_best2), kind='cubic')
    flux1 = (10**(f(float(key),data_out[key]["lambda"]))).T
    zp_factor = 10**(zp_best2/-2.5)
    flux1 = flux1*zp_factor
    spec_best_dic2[key] = {}
    spec_best_dic2[key]["lambda"] = data_out[key]["lambda"]
    spec_best_dic2[key]["data"] = np.squeeze(flux1)

filts = np.genfromtxt('../input/filters.dat')
filtnames = ["u","g","r","i","z","y","J","H","K"]

mag1, mag2 = {}, {}
for ii in range(9):
    mag1[ii], mag2[ii] = [], []

tmag = []
for key in data_out:
    tmag.append(float(key))
    for ii in range(9):
        band = np.array(zip(filts[:,0]*10,filts[:,ii+1]))
        mag1[ii].append(spec2mag(spec_best_dic1[key]["lambda"],spec_best_dic1[key]["data"],band))
        mag2[ii].append(spec2mag(spec_best_dic2[key]["lambda"],spec_best_dic2[key]["data"],band))

tmag = np.array(tmag)
for ii in range(9):
    mag1[ii], mag2[ii] = np.array(mag1[ii]), np.array(mag2[ii])

title_fontsize = 30
label_fontsize = 30

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.jet(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]
tini, tmax, dt = 0.0, 10.0, 0.1
tt = np.arange(tini,tmax,dt)

color2 = 'coral'
color1 = 'cornflowerblue'

plotName = "%s/models_spec_panels.pdf"%(plotDir)
#plt.figure(figsize=(20,18))
plt.figure(figsize=(20,28))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    if not filt in data_out3: continue
    samples = data_out3[filt]
    t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
    idx = np.where(~np.isnan(y))[0]
    t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
    if len(t) == 0: continue

    idx = np.where(np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color, markersize=16)

    idx = np.where(~np.isfinite(sigma_y))[0]
    plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color, markersize=16)

    magave1 = mag1[magidx]
    magave2 = mag2[magidx]

    ii = np.where(~np.isnan(magave1))[0]
    f = interp.interp1d(tmag[ii], magave1[ii], fill_value='extrapolate')
    if filt == 'u':
       tt1 = tt[tt<=3.0]
    elif filt == 'g':
       tt1 = tt[tt<=6.5]
    else:
       tt1 = tt[tt<=21.0]

    maginterp1 = f(tt1)
    plt.plot(tt1,maginterp1,'--',c=color1,linewidth=2,label='1 Component')
    plt.plot(tt1,maginterp1-errorbudgetmag,'-',c=color1,linewidth=2)
    plt.plot(tt1,maginterp1+errorbudgetmag,'-',c=color1,linewidth=2)
    plt.fill_between(tt1,maginterp1-errorbudgetmag,maginterp1+errorbudgetmag,facecolor=color1,alpha=0.2)

    ii = np.where(~np.isnan(magave2))[0]
    f = interp.interp1d(tmag[ii], magave2[ii], fill_value='extrapolate')
    maginterp2 = f(tt1)
    plt.plot(tt1,maginterp2,'--',c=color2,linewidth=2,label='2 Component')
    plt.plot(tt1,maginterp2-errorbudgetmag,'-',c=color2,linewidth=2)
    plt.plot(tt1,maginterp2+errorbudgetmag,'-',c=color2,linewidth=2)
    plt.fill_between(tt1,maginterp2-errorbudgetmag,maginterp2+errorbudgetmag,facecolor=color2,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 10.0])
    plt.ylim([-17.0,-11.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=48)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

keys = sorted(data_out.keys())
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotName = "%s/spec_panels_fit.pdf"%(plotDir)
plotNamePNG = "%s/spec_panels_fit.png"%(plotDir)
fig = plt.figure(figsize=(22,28))

cnt = 0
for key, color in zip(keys,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(keys),1,cnt)
    if cnt == 1:
        #ax1 = plt.subplot(eval(vals))
        ax1 = plt.subplot(len(keys),1,cnt)
    else:
        #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
        ax2 = plt.subplot(len(keys),1,cnt,sharex=ax1,sharey=ax1)

    plt.plot(data_out[key]["lambda"],np.log10(data_out[key]["data"]),'--',c='k',linewidth=4,zorder=99)

    lambdas = spec_best_dic1[key]["lambda"]
    specmed = spec_best_dic1[key]["data"]
    specmin = spec_best_dic1[key]["data"]/errorbudget
    specmax = spec_best_dic1[key]["data"]*errorbudget

    plt.plot(lambdas,np.log10(specmed),'--',c=color1,linewidth=2,label="1 Component")
    plt.plot(lambdas,np.log10(specmin),'-',c=color1,linewidth=2)
    plt.plot(lambdas,np.log10(specmax),'-',c=color1,linewidth=2)
    plt.fill_between(lambdas,np.log10(specmin),np.log10(specmax),facecolor=color1,edgecolor=color1,alpha=0.2,linewidth=3)

    lambdas = spec_best_dic2[key]["lambda"]
    specmed = spec_best_dic2[key]["data"]
    specmin = spec_best_dic2[key]["data"]/errorbudget
    specmax = spec_best_dic2[key]["data"]*errorbudget

    plt.plot(lambdas,np.log10(specmed),'--',c=color2,linewidth=2,label="2 Component")
    plt.plot(lambdas,np.log10(specmin),'-',c=color2,linewidth=2)
    plt.plot(lambdas,np.log10(specmax),'-',c=color2,linewidth=2)
    plt.fill_between(lambdas,np.log10(specmin),np.log10(specmax),facecolor=color2,edgecolor=color2,alpha=0.2,linewidth=3)

    plt.fill_between([13500.0,14500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    plt.fill_between([18000.0,19500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    plt.xlim([5000, 25000])
    plt.ylim([35.5,37.9])
    plt.grid()

    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, prop={'size':48})
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotNamePNG, bbox_inches='tight')
plt.close()
convert_command = "convert %s %s"%(plotNamePNG,plotName)
os.system(convert_command)

keys_tmp = sorted(data_out.keys())
keys_float = np.array(keys_tmp,dtype=np.float64)
idx = np.where(keys_float <= 5.0)[0]
keys = [keys_tmp[ii] for ii in idx]
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotName = "%s/spec_panels_fit_early.pdf"%(plotDir)
plotNamePNG = "%s/spec_panels_fit_early.png"%(plotDir)
fig = plt.figure(figsize=(22,28))

cnt = 0
for key, color in zip(keys,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(keys),1,cnt)
    if cnt == 1:
        #ax1 = plt.subplot(eval(vals))
        ax1 = plt.subplot(len(keys),1,cnt)
    else:
        #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
        ax2 = plt.subplot(len(keys),1,cnt,sharex=ax1,sharey=ax1)

    plt.plot(data_out[key]["lambda"],data_out[key]["data"],'--',c='k',linewidth=4,zorder=99)

    lambdas = spec_best_dic1[key]["lambda"]
    specmed = spec_best_dic1[key]["data"]
    specmin = spec_best_dic1[key]["data"]/errorbudget
    specmax = spec_best_dic1[key]["data"]*errorbudget

    plt.plot(lambdas,specmed,'--',c=color1,linewidth=2,label="1 Component")
    plt.plot(lambdas,specmin,'-',c=color1,linewidth=2)
    plt.plot(lambdas,specmax,'-',c=color1,linewidth=2)
    plt.fill_between(lambdas,specmin,specmax,facecolor=color1,edgecolor=color1,alpha=0.2,linewidth=3)

    lambdas = spec_best_dic2[key]["lambda"]
    specmed = spec_best_dic2[key]["data"]
    specmin = spec_best_dic2[key]["data"]/errorbudget
    specmax = spec_best_dic2[key]["data"]*errorbudget

    plt.plot(lambdas,specmed,'--',c=color2,linewidth=2,label="2 Component")
    plt.plot(lambdas,specmin,'-',c=color2,linewidth=2)
    plt.plot(lambdas,specmax,'-',c=color2,linewidth=2)
    plt.fill_between(lambdas,specmin,specmax,facecolor=color2,edgecolor=color2,alpha=0.2,linewidth=3)

    plt.fill_between([13500.0,14500.0],[10**-100.0,10**-100.0],[10**100.0,10**100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    plt.fill_between([18000.0,19500.0],[10**-100.0,10**-100.0],[10**100.0,10**100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    plt.xlim([5000, 25000])
    plt.ylim([10**35.5,10**37.9])
    plt.grid()

    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, prop={'size':48})
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotNamePNG, bbox_inches='tight')
plt.close()
convert_command = "convert %s %s"%(plotNamePNG,plotName)
os.system(convert_command)

keys_tmp = sorted(data_out.keys())
keys_float = np.array(keys_tmp,dtype=np.float64)
idx = np.where(keys_float >= 5.0)[0]
keys = [keys_tmp[ii] for ii in idx]
colors=cm.rainbow(np.linspace(0,1,len(keys)))

plotName = "%s/spec_panels_fit_late.pdf"%(plotDir)
plotNamePNG = "%s/spec_panels_fit_late.png"%(plotDir)
fig = plt.figure(figsize=(22,28))

cnt = 0
for key, color in zip(keys,colors):
    cnt = cnt+1
    vals = "%d%d%d"%(len(keys),1,cnt)
    if cnt == 1:
        #ax1 = plt.subplot(eval(vals))
        ax1 = plt.subplot(len(keys),1,cnt)
    else:
        #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
        ax2 = plt.subplot(len(keys),1,cnt,sharex=ax1,sharey=ax1)

    plt.plot(data_out[key]["lambda"],data_out[key]["data"],'--',c='k',linewidth=4,zorder=99)

    lambdas = spec_best_dic1[key]["lambda"]
    specmed = spec_best_dic1[key]["data"]
    specmin = spec_best_dic1[key]["data"]/errorbudget
    specmax = spec_best_dic1[key]["data"]*errorbudget

    plt.plot(lambdas,specmed,'--',c=color1,linewidth=2,label="1 Component")
    plt.plot(lambdas,specmin,'-',c=color1,linewidth=2)
    plt.plot(lambdas,specmax,'-',c=color1,linewidth=2)
    plt.fill_between(lambdas,specmin,specmax,facecolor=color1,edgecolor=color1,alpha=0.2,linewidth=3)

    lambdas = spec_best_dic2[key]["lambda"]
    specmed = spec_best_dic2[key]["data"]
    specmin = spec_best_dic2[key]["data"]/errorbudget
    specmax = spec_best_dic2[key]["data"]*errorbudget

    plt.plot(lambdas,specmed,'--',c=color2,linewidth=2,label="2 Component")
    plt.plot(lambdas,specmin,'-',c=color2,linewidth=2)
    plt.plot(lambdas,specmax,'-',c=color2,linewidth=2)
    plt.fill_between(lambdas,specmin,specmax,facecolor=color2,edgecolor=color2,alpha=0.2,linewidth=3)

    plt.fill_between([13500.0,14500.0],[10**-100.0,10**-100.0],[10**100.0,10**100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    plt.fill_between([18000.0,19500.0],[10**-100.0,10**-100.0],[10**100.0,10**100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

    plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
    plt.xlim([5000, 25000])
    plt.ylim([10**35.5,10**36.9])
    plt.grid()

    if (not cnt == len(keys)) and (not cnt == 1):
        plt.setp(ax2.get_xticklabels(), visible=False)
    elif cnt == 1:
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, prop={'size':48})
    else:
        plt.xticks(fontsize=36)

ax1.set_zorder(1)
ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
plt.savefig(plotNamePNG, bbox_inches='tight')
plt.close()
convert_command = "convert %s %s"%(plotNamePNG,plotName)
os.system(convert_command)
