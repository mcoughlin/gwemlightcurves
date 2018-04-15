
import os, sys, glob
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
from scipy.interpolate import InterpolatedUnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from astropy.time import Time

import matplotlib
matplotlib.rc('pdf', fonttype=42)
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit

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
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")

    parser.add_option("-m","--model",default="Ka2017")

    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("-e","--event",default="G298048_PS1_GROND_SOFI")
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)

    parser.add_option("--doAB",  action="store_true", default=False)
    parser.add_option("--doSpec",  action="store_true", default=False)
    parser.add_option("--doLuminosity",  action="store_true", default=False)

    parser.add_option("--errorbudget",default=1.0,type=float)
    parser.add_option("--filters",default="u,g,r,i,z,y,J,H,K")

    parser.add_option("--mej",default=0.001,type=float)

    opts, args = parser.parse_args()

    return opts

# Parse command line
opts = parse_commandline()

filters = opts.filters.split(",")

lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir

outputDir = opts.outputDir
baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,opts.model,'all','%.3f'%opts.mej)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

tini = 0.1
tmax = 7.0
tmax = 21.0
#dt = 0.1
#dt = 0.25
dt = 0.5
lambdaini = 3700
lambdamax = 28000
#dlambda = 50.0 
dlambda = 500.0

if opts.doAB:

    if opts.model == "BaKa2016":
        fileDir = "../output/barnes_kilonova_spectra"
    elif opts.model == "Ka2017":
        fileDir = "../output/kasen_kilonova_grid"
    elif opts.model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"

    filenames_all = glob.glob('%s/*.dat'%fileDir)
    idxs = []
    for ii,filename in enumerate(filenames_all):
        if "_Lbol.dat" in filename: continue
        if "_spec.dat" in filename: continue
        idxs.append(ii)
    filenames = [filenames_all[idx] for idx in idxs]

    mags, names = lightcurve_utils.read_files(filenames)
    magkeys = mags.keys()

    tt = np.arange(tini,tmax+dt,dt)
    filters = ["u","g","r","i","z","y","J","H","K"]

    for key in magkeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            mags[key]["mej"] = mej0
            mags[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                #del mags[key]
                #continue
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            mags[key]["mej"] = mej0
            mags[key]["vej"] = vej0
            mags[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            mags[key]["mej"], mags[key]["vej"], mags[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        mags[key]["data"] = np.zeros((len(tt),len(filters)))

        for jj,filt in enumerate(filters):
            ii = np.where(np.isfinite(mags[key][filt]))[0]
            f = interp.interp1d(mags[key]["t"][ii], mags[key][filt][ii], fill_value='extrapolate')
            maginterp = f(tt)
            mags[key]["data"][:,jj] = maginterp
        mags[key]["data_vector"] = np.reshape(mags[key]["data"],len(tt)*len(filters),1)

    filenames = glob.glob('%s/*_Lbol.dat'%fileDir)

    lbols, names = lightcurve_utils.read_files_lbol(filenames)
    lbolkeys = lbols.keys()

    tt = np.arange(tini,tmax+dt,dt)

    for key in lbolkeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            lbols[key]["mej"] = mej0
            lbols[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            lbols[key]["mej"] = mej0
            lbols[key]["vej"] = vej0
            lbols[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            lbols[key]["mej"], lbols[key]["vej"], lbols[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        ii = np.where(np.isfinite(lbols[key]["Lbol"]))[0]
        f = interp.interp1d(lbols[key]["tt"][ii], np.log10(lbols[key]["Lbol"][ii]), fill_value='extrapolate')
        lbolinterp = 10**f(tt)
        lbols[key]["Lbol"]= np.log10(lbolinterp)

    param_array = []
    for key in lbolkeys:
        if opts.model == "BaKa2016":
            param_array.append([lbols[key]["mej"],lbols[key]["vej"]])
        elif opts.model == "Ka2017":
            param_array.append([lbols[key]["mej"],lbols[key]["vej"],lbols[key]["Xlan"]])
        elif opts.model == "RoFe2017":
            param_array.append([lbols[key]["mej"],lbols[key]["vej"],lbols[key]["Ye"]])
    param_array = np.array(param_array)
    idx = np.where(param_array[:,0] == opts.mej)[0]
    param_array = param_array[idx,:]
    unique_rows = np.unique(param_array[:,:2],axis=0)

    if opts.doEvent:
        filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
        data_out = lightcurve_utils.loadEvent(filename)
        for ii,key in enumerate(data_out.iterkeys()):
            if key == "t":
                continue
            else:
                data_out[key][:,0] = data_out[key][:,0] - opts.T0
                data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)
    
    Xlans = np.unique(param_array[:,2])
    colors=cm.rainbow(np.linspace(0,1,len(Xlans)))
    cmap = cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    plotName = "%s/models_panels.pdf"%(plotDir)
    fig = plt.figure(figsize=(20,18))
    #plt.figure(figsize=(20,48))

    cnt = 0
    for unique_row in unique_rows:
        cnt = cnt+1
        if cnt == 1:
            ax1 = plt.subplot(len(unique_rows),1,cnt)
        else:
            ax2 = plt.subplot(len(unique_rows),1,cnt,sharex=ax1,sharey=ax1)

        if opts.doEvent:
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c='k',markersize=15)
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c='k',markersize=15)

        idxs = np.where((unique_row[0] == param_array[:,0]) & (unique_row[1] == param_array[:,1]))[0]
        idx = np.argsort(param_array[idxs,2])
        idxs = idxs[idx]

        for idx in idxs:
            name = magkeys[idx]
            mag_d = mags[name]
            params = param_array[idx,:]
            ii = np.where(params[2] == Xlans)[0]

            plt.plot(tt,mag_d["data"][:,1],'--',c=colors[ii][0],linewidth=4,label="$10^{%.0f}$"%np.log10(params[2]))
            plt.plot(tt,mag_d["data"][:,8],'-',c=colors[ii][0],linewidth=4)   

        lab = "%.3f %.3f"%(unique_row[0],unique_row[1])
        lab = "%.2f"%unique_row[1]

        plt.ylabel('%s'%lab,fontsize=48,rotation=0,labelpad=40)
        plt.xlim([0.0, 18.0])
        plt.ylim([-18.0,-10.0])
        plt.gca().invert_yaxis()
        plt.grid()
 
        if cnt == 1:
            #ax1.set_yticks([-18,-16,-14,-12,-10])
            plt.setp(ax1.get_xticklabels(), visible=False)
            l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(unique_rows):
            plt.setp(ax2.get_xticklabels(), visible=False)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

    ax1.set_zorder(1)
    plt.xlabel('Time [days]',fontsize=48)

    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    #sm._A = []
    #cbar = plt.colorbar(sm)

    #divider = make_axes_locatable(ax1)
    #cax = divider.new_vertical(size="50%", pad=0.0, pack_start=True)
    #fig.add_axes(cax)
    #cbar = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
    #                            norm=norm,
    #                            orientation='vertical')
    #cbar.set_label('X_{lan}')
    plt.savefig(plotName)
    plt.close()

    plotName = "%s/luminosity_panels.pdf"%(plotDir)
    plt.figure(figsize=(20,18))
    #plt.figure(figsize=(20,48))

    cnt = 0
    for unique_row in unique_rows:
        cnt = cnt+1
        if cnt == 1:
            ax1 = plt.subplot(len(unique_rows),1,cnt)
        else:
            ax2 = plt.subplot(len(unique_rows),1,cnt,sharex=ax1,sharey=ax1)

        if opts.doEvent:
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c='k',markersize=15)
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c='k',markersize=15)

        idxs = np.where((unique_row[0] == param_array[:,0]) & (unique_row[1] == param_array[:,1]))[0]
        idx = np.argsort(param_array[idxs,2])
        idxs = idxs[idx]

        for idx in idxs:
            name = magkeys[idx]
            lbol_d = lbols[name]
            params = param_array[idx,:]
            ii = np.where(params[2] == Xlans)[0]

            plt.plot(tt,lbol_d["Lbol"],'--',c=colors[ii][0],linewidth=4,label="$10^{%.0f}$"%np.log10(params[2]))
            plt.plot(tt,lbol_d["Lbol"],'-',c=colors[ii][0],linewidth=4)

        lab = "%.3f %.3f"%(unique_row[0],unique_row[1])
        lab = "%.2f"%unique_row[1]

        plt.ylabel('%s'%lab,fontsize=48,rotation=0,labelpad=40)
        plt.xlim([0.0, 18.0])
        plt.ylim([38.0,42.0])
        plt.grid()

        if cnt == 1:
            ax1.set_yticks([38,39,40,41,42])
            plt.setp(ax1.get_xticklabels(), visible=False)
            l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(unique_rows):
            plt.setp(ax2.get_xticklabels(), visible=False)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

    ax1.set_zorder(1)
    plt.xlabel('Time [days]',fontsize=48)
    plt.savefig(plotName)
    plt.close()
    
elif opts.doSpec:

    if opts.model == "BaKa2016":
        fileDir = "../output/barnes_kilonova_spectra"
    elif opts.model == "Ka2017":
        fileDir = "../output/kasen_kilonova_grid"
    elif opts.model == "RoFe2017":
        fileDir = "../output/macronovae-rosswog_wind"

    filenames = glob.glob('%s/*_spec.dat'%fileDir)

    specs, names = lightcurve_utils.read_files_spec(filenames)
    speckeys = specs.keys()

    tt = np.arange(tini,tmax+dt,dt)
    lambdas = np.arange(lambdaini,lambdamax+dlambda,dlambda)

    for key in speckeys:
        keySplit = key.split("_")
        if keySplit[0] == "rpft":
            mej0 = float("0." + keySplit[1].replace("m",""))
            vej0 = float("0." + keySplit[2].replace("v",""))
            specs[key]["mej"] = mej0
            specs[key]["vej"] = vej0
        elif keySplit[0] == "knova":
            mej0 = float(keySplit[3].replace("m",""))
            vej0 = float(keySplit[4].replace("vk",""))
            if len(keySplit) == 6:
                Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))
            elif len(keySplit) == 7:
                #del specs[key]
                #continue
                if "Xlan1e" in keySplit[6]:
                    Xlan0 = 10**float(keySplit[6].replace("Xlan1e",""))
                elif "Xlan1e" in keySplit[5]:
                    Xlan0 = 10**float(keySplit[5].replace("Xlan1e",""))

            specs[key]["mej"] = mej0
            specs[key]["vej"] = vej0
            specs[key]["Xlan"] = Xlan0
        elif keySplit[0] == "SED":
            specs[key]["mej"], specs[key]["vej"], specs[key]["Ye"] = lightcurve_utils.get_macronovae_rosswog(key)

        data = specs[key]["data"].T
        data[data==0.0] = 1e-20
        f = interp.interp2d(specs[key]["t"], specs[key]["lambda"], np.log10(data), kind='cubic')
        #specs[key]["data"] = (10**(f(tt,lambdas))).T
        specs[key]["data"] = f(tt,lambdas).T

    speckeys = specs.keys()
    param_array = []
    for key in speckeys:
        if opts.model == "BaKa2016":
            param_array.append([specs[key]["mej"],specs[key]["vej"]])
        elif opts.model == "Ka2017":
            param_array.append([specs[key]["mej"],specs[key]["vej"],specs[key]["Xlan"]])
        elif opts.model == "RoFe2017":
            param_array.append([specs[key]["mej"],specs[key]["vej"],specs[key]["Ye"]])

    param_array = np.array(param_array)
    idx = np.where(param_array[:,0] == opts.mej)[0]
    param_array = param_array[idx,:]
    unique_rows = np.unique(param_array[:,:2],axis=0)

    if opts.doEvent:
        events = opts.event.split(",")
        eventdata = {}
        for event in events:
            filename = "%s/%s.dat"%(spectraDir,event)
            data_out = lightcurve_utils.loadEventSpec(filename)
            eventdata[event] = data_out

    Xlans = np.unique(param_array[:,2])
    colors=cm.rainbow(np.linspace(0,1,len(Xlans)))
    cmap = cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    plotName = "%s/spec_panels.pdf"%(plotDir)
    plt.figure(figsize=(20,18))
    #plt.figure(figsize=(20,48))

    cnt = 0
    for unique_row in unique_rows:
        cnt = cnt+1
        if cnt == 1:
            ax1 = plt.subplot(len(unique_rows),1,cnt)
        else:
            ax2 = plt.subplot(len(unique_rows),1,cnt,sharex=ax1,sharey=ax1)

        if opts.doEvent:
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c='k',markersize=15)
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c='k',markersize=15)
 
        idxs = np.where((unique_row[0] == param_array[:,0]) & (unique_row[1] == param_array[:,1]))[0]
        idx = np.argsort(param_array[idxs,2])
        idxs = idxs[idx]

        for idx in idxs:
            name = speckeys[idx]
            spec_d = specs[name]
            params = param_array[idx,:]
            ii = np.where(params[2] == Xlans)[0]

            f = interp.interp2d(tt, lambdas, spec_d["data"].T, kind='cubic')
            spec1 = np.squeeze(f(1.5,lambdas).T)
            spec2 = np.squeeze(f(4.5,lambdas).T)

            plt.plot(lambdas,spec1,'--',c=colors[ii][0],linewidth=4,label="$10^{%.0f}$"%np.log10(params[2]))
            plt.plot(lambdas,spec2,'-',c=colors[ii][0],linewidth=4)

        lab = "%.3f %.3f"%(unique_row[0],unique_row[1])
        lab = "%.2f"%unique_row[1]

        plt.ylabel('%s'%lab,fontsize=48,rotation=0,labelpad=40)
        plt.ylim([-50.0,50.0])
        plt.grid()

        if cnt == 1:
            #ax1.set_yticks([38,39,40,41,42])
            plt.setp(ax1.get_xticklabels(), visible=False)
            l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(unique_rows):
            plt.setp(ax2.get_xticklabels(), visible=False)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

    ax1.set_zorder(1)
    plt.xlim([3000,30000])
    #plt.ylim([10.0**39,10.0**43])
    plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
    plt.savefig(plotName)
    plt.close()

