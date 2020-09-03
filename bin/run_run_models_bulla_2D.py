
import os, sys
import glob
import numpy as np

from scipy.interpolate import interpolate as interp

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from shutil import copy
from shutil import move
#from statsmodels.nonparametric.smoothers_lowess import lowess
baseplotDir = "../plots/angles"

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","tanaka_compactmergers","macronovae-rosswog","kasen_kilonova_survey","kasen_kilonova_grid"]
models = ["kasen_kilonova_grid"]
models = ["bulla_1D_dz31"]
models = ["bulla_2Component_lnsbh"]
models = ["bulla_2D"]
models = ["bulla_reprocess"]
#models = ["RosswogLike_2D"]

ntheta = 11
costhetas = np.linspace(0,1,ntheta)
thetas = np.rad2deg(np.arccos(costhetas))

filts = ["u","g","r","i","z","y","J","H","K"]
onedfilename = "/home/michael.coughlin/gwemlightcurves/output/kasen_kilonova_grid/knova_d1_n10_m0.040_vk0.10_fd1.0_Xlan1e-2.0.dat"
oned = np.loadtxt(onedfilename)

for model in models:
    
    files = glob.glob("../data/%s/*"%model)
    print(files)
    #for file in files:
        #name = file.split("/")[-1]
        #copy(file,"../data/bulla_2D/"+name)
    #model = "bulla_2D"
    #files = files[20:25]

    plotDir = os.path.join(baseplotDir,model)
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)
    files.sort()
    for file in files:
        name = file.split("/")[-1].replace(".mod","").replace(".spec","").replace("_AB","").replace(".h5","").replace(".dat","").replace(".txt","")
        success = True
        magall = {}
        for theta in thetas:
            filename = "../output/%s/%s_%.1f.dat"%(model,name,theta)
            if not os.path.isfile(filename):
                system_call = "python2 run_models.py --doAB --model %s --name %s --theta %.1f"%(model,name,theta)
                print(system_call)
                os.system(system_call)
            else:
                system_call = "python2 run_models.py --doAB --model %s --name %s --theta %.1f"%(model,name,theta)
                print(system_call)
                print("Overwriting files...")
                os.system(system_call)
            if not os.path.isfile(filename):
                success = False
                break
            mag_ds = np.loadtxt(filename) 

            ndata, nfilt = mag_ds.shape
            for jj in range(nfilt):
                if jj == 0: continue
                ii = np.where(np.isfinite(mag_ds[:,jj]))[0]
                f = interp.interp1d(mag_ds[ii,0], mag_ds[ii,jj], fill_value='extrapolate')
                maginterp = f(mag_ds[:,0])
                mag_ds[:,jj] = maginterp

            magall[theta] = mag_ds
        if not success:
            print(file+" failed")
            continue
        colors=cm.rainbow(np.linspace(0,1,len(thetas)))
    
        plotName = "%s/%s.pdf"%(plotDir,name)
        fig = plt.figure(figsize=(14,12))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Absolute AB Magnitude')
        ax.set_xlim([0,10])
        ax.set_ylim([-20,0])
        for theta, color in zip(thetas,colors):
            t, mag = magall[theta][:,0], magall[theta][:,-1]
            ax.plot(t,mag,alpha=1.0,c=color)
            #line1.set_label("K band")
            #ax.legend()
            t, mag = magall[theta][:,0], magall[theta][:,-4]
            ax.plot(t,mag,'--',alpha=1.0,c=color)
            #line2.set_label("y band")
            #ax.legend()
            t, mag = magall[theta][:,0], magall[theta][:,2]
            ax.plot(t,mag,'.-',alpha=1.0,c=color)
            #line3.set_label("g band")
            #ax.legend()
       
        #color = 'k'
        #oned = [[np.nan],[np.nan]]
        #t, mag = oned[:,0], oned[:,-1]
        #dumL = plt.plot(t,mag,alpha=1.0,c=color)
        #dumL.set_label("%s band"%(theta, "K"))
        #t, mag = oned[:,0], oned[:,-4]
        #dumL = plt.plot(t,mag,'--',alpha=1.0,c=color)
        #dumL.set_label("%s band"%(theta, "y"))
        #t, mag = oned[:,0], oned[:,2]
        #dumL = plt.plot(t,mag,'.-',alpha=1.0,c=color)
        #dumL.set_label("%s band"%(theta, "g"))
        custom_lines = [Line2D([0], [0], ls = "-", color="k"), Line2D([0], [0], ls = "--", color="k"), Line2D([0], [0], ls = "-.", color="k")]
        ax.legend(custom_lines,("K band","y band","g band"),loc="upper right",ncol=1)
        plt.xlabel('Time [days]')
        plt.ylabel('Absolute AB Magnitude')
        plt.xlim([0,10])
        plt.ylim([-20,0])
        plt.gca().invert_yaxis()
        #dumx, dumy = np.mgrid[:N, :N]
        colorbar_ax = fig.add_axes([0.93, 0.11, 0.025, 0.77])
        dummyCM = cm.ScalarMappable(cmap = "rainbow")
        cbar = plt.colorbar(dummyCM,cax=colorbar_ax)
        cbar.ax.set_ylabel('cos $\\theta_{obs}$',rotation=90)
        plt.savefig(plotName,bbox_inches='tight')
        plt.close()
    
        plotName = "%s/%s_color.pdf"%(plotDir,name)
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Color [AB Magnitude]')
        ax.set_xlim([0,10])
        ax.set_ylim([-10,5])
        for theta, color in zip(thetas,colors):
            t, mag1 = magall[theta][:,0], magall[theta][:,-1]
            t, mag2 = magall[theta][:,0], magall[theta][:,-4]
            t, mag3 = magall[theta][:,0], magall[theta][:,2]
    
            ax.plot(t,mag1-mag2,alpha=1.0,c=color)
            ax.plot(t,mag1-mag3,'--',alpha=1.0,c=color)
    
        #t, mag1 = oned[:,0], oned[:,-1]
        #t, mag2 = oned[:,0], oned[:,-4]
        #t, mag3 = oned[:,0], oned[:,2]
        #color = 'k'
        #plt.plot(t,mag1-mag2,alpha=1.0,c=color)
        #plt.plot(t,mag1-mag3,'--',alpha=1.0,c=color)
    
        custom_lines = [Line2D([0], [0], ls = "-", color="k"), Line2D([0], [0], ls = "--", color="k")]
        ax.legend(custom_lines,("K-y","K-g"),loc="upper right",ncol=1)
        plt.xlabel('Time [days]')
        plt.ylabel('Color [AB Magnitude]')
        plt.xlim([0,10])
        plt.ylim([-10,5])
        plt.gca().invert_yaxis()
        colorbar_ax = fig.add_axes([0.93, 0.11, 0.025, 0.77])
        dummyCM = cm.ScalarMappable(cmap = "rainbow")
        cbar = plt.colorbar(dummyCM,cax=colorbar_ax)
        cbar.ax.set_ylabel('cos $\\theta_{obs}$',rotation=90)
        plt.savefig(plotName,bbox_inches="tight")
        plt.close()
    
        tts = [1.0, 3.0, 5.0]
        linestyles = ['-','--','-.']
    
        colors=cm.rainbow(np.linspace(0,1,len(filts)-1))
        plotName = "%s/%s_inc.pdf"%(plotDir,name)
        plt.figure(figsize=(20,32))
        cnt = 0
        for jj, filt in enumerate(filts):
            if jj == len(filts)-1: continue
            if jj == 0: continue
            cnt = cnt+1
            if cnt == 1:
                ax1 = plt.subplot(len(filts)-2,1,cnt)
            else:
                ax2 = plt.subplot(len(filts)-2,1,cnt,sharex=ax1)
            color = colors[jj]
    
            for tt, linestyle in zip(tts,linestyles):
                vals = []
                for ii,theta in enumerate(thetas):
                    t, mag1 = magall[theta][:,0], magall[theta][:,jj+1]
                    t, mag2 = magall[theta][:,0], magall[theta][:,len(filts)]
    
                    color1 = mag1-mag2
                    idx = np.argmin(np.abs(t-tt))
    
                    vals.append(color1[idx])
   
                #vals = lowess(vals, thetas, frac=0.3)
                #vals = vals[:,1]
                vals = vals - np.median(vals)
  
                plt.plot(thetas,vals,linestyle=linestyle,alpha=1.0,color=color,
                        linewidth=5.0)
    
            plt.ylabel('%s-%s' % (filt,filts[-1]),fontsize=48,rotation=0,labelpad=40)
            #plt.xlim([0.0, np.max(thetas)])
            #plt.ylim([-5.0, 5.0])
            plt.grid()
            if cnt == len(filts)-2:
                pass
                #ax2.set_xticks([0,30,60,90,120,150])
                #ax2.set_xticks([0,45,90,135,180])
                #plt.setp(ax2.get_xticklabels(), visible=False)
            elif cnt == 1:
                plt.setp(ax1.get_xticklabels(), visible=False)
            else:
                plt.setp(ax2.get_xticklabels(), visible=False)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.gca().invert_yaxis()
    
        ax1.set_zorder(1)
        plt.xlabel('Inclination [degrees]', fontsize=48)
        plt.tight_layout()
        plt.savefig(plotName)
        plt.close()

#for modell in models:
#    if not os.path.exists("../plots/%s"%(modell)):
#        os.mkdir("../plots/%s"%(modell))
#    if not os.path.exists("../output/%s"%(modell)):
#        os.mkdir("../output/%s"%(modell))
#    move("../plots/bulla_2D/*","../plots/%s/"%(modell))
#    move("../output/bulla_2D/*","../output/%s/"%(modell))
print("stop")

#print stop

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog","kasen_kilonova_survey","kasen_kilonova_grid"]
models = ["kasen_kilonova_grid"]

for model in models:
    files = glob.glob("../data/%s/*"%model)
    for file in files:
        name = file.split("/")[-1].replace(".mod","").replace(".spec","").replace("_AB","").replace(".h5","").replace(".dat","").replace(".txt","")

        filename = "../output/%s/%s_spec.dat"%(model,name)
        if os.path.isfile(filename): continue
        system_call = "python2 run_models.py --doSpec --model %s --name %s"%(model,name)
        os.system(system_call)

