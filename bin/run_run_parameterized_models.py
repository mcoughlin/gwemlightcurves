
import os, sys, pickle
import glob
import numpy as np

from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves import lightcurve_utils

mej1 = 5e-3
vej1  = 0.45
Xlan1 = 1e-3
mej2s = [5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1]
vej2 = 0.15
Xlan2 = 1e-2

outputDir = "../output/Ka2017x2"
plotDir = "../plots/ASsNova"
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

data = {}
legend_names = {}
for mej2 in mej2s:
    name = "Ka2017x2_M%03dV%02dX%d_M%03dV%02dX%d"%(mej1*1000,vej1*100,np.log10(Xlan1),mej2*1000,vej2*100,np.log10(Xlan2))
    filename = os.path.join(outputDir,"%s.dat"%name)

    if not os.path.isfile(filename):
        system_call = "python run_parameterized_models.py --doAB --doEjecta --model Ka2017x2 --mej1 %.5f --vej1 %.5f --Xlan1 %.5f --mej2 %.5f --vej2 %.5f --Xlan2 %.5f"%(mej1,vej1,Xlan1,mej2,vej2,Xlan2)
        os.system(system_call)

    mag_d = lightcurve_utils.read_files([filename])
    key = mag_d[0].keys()[0]
    mag_d = mag_d[0][key]

    data_out = np.loadtxt(filename)
    data[name] = mag_d
    legend_names[name] = "%.3f"%mej2  

pcklFile = "%s/data.pkl"%(plotDir)
f = open(pcklFile, 'wb')
pickle.dump((data), f)
f.close()

filts = ["u","g","r","i","z","y","J","H","K"]
colors=cm.rainbow(np.linspace(0,1,len(filts)))
magidxs = [0,1,2,3,4,5,6,7,8]

keys = data.keys()
idx = np.argsort(np.array(legend_names.keys()))
keys = [keys[ii] for ii in idx]
color_names=cm.rainbow(np.linspace(0,1,len(keys)))

zp_best=0.0
errorbudget=1.0

plotName = "%s/panels.pdf"%(plotDir)
plotNamePNG = "%s/panels.png"%(plotDir)
plt.figure(figsize=(20,28))

cnt = 0
for filt, color, magidx in zip(filts,colors,magidxs):
    cnt = cnt+1
    vals = "%d%d%d"%(len(filts),1,cnt)
    if cnt == 1:
        ax1 = plt.subplot(eval(vals))
    else:
        ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

    for key,color in zip(keys,color_names):
        legend_name = legend_names[key]
        magave = data[key][filt] 
        tt = data[key]["t"]

        ii = np.where(~np.isnan(magave))[0]
        f = interp.interp1d(tt[ii], magave[ii], fill_value='extrapolate')
        maginterp = f(tt)
        plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2,label=legend_name)
        plt.plot(tt,maginterp+zp_best-errorbudget,'-',c=color,linewidth=2)
        plt.plot(tt,maginterp+zp_best+errorbudget,'-',c=color,linewidth=2)
        plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)

    plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
    plt.xlim([0.0, 14.0])
    plt.ylim([-18.0,-8.0])
    plt.gca().invert_yaxis()
    plt.grid()

    if cnt == 1:
        ax1.set_yticks([-18,-16,-14,-12,-10,-8])
        plt.setp(ax1.get_xticklabels(), visible=False)
        l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
    elif not cnt == len(filts):
        plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

ax1.set_zorder(1)
plt.xlabel('Time [days]',fontsize=48)
plt.savefig(plotNamePNG, bbox_inches='tight')
plt.close()
convert_command = "convert %s %s"%(plotNamePNG,plotName)
os.system(convert_command)
