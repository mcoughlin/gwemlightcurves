
import os, sys
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

from scipy.optimize import curve_fit

from gwemlightcurves.sampler import *
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.sampler import run
from gwemlightcurves import __version__
from gwemlightcurves import lightcurve_utils, Global

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
 
    parser.add_option("-o","--outputDir",default="../output_hst")
    parser.add_option("-p","--plotDir",default="../plots_hst")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")
    parser.add_option("-f","--outputName",default="G298048_all")

    parser.add_option("--doCadence",  action="store_true", default=False)
    parser.add_option("-c","--cadence",default="0.25,0.75,1.25,1.75,2.5,4.5,6.5,8.5")

    parser.add_option("--doEvent",  action="store_true", default=False)
    #parser.add_option("-e","--event",default="G298048_PS1_GROND_SOFI")
    #parser.add_option("-e","--event",default="G298048_XSH_PESSTO")
    #parser.add_option("-e","--event",default="G298048_20170822")
    #parser.add_option("-e","--event",default="G298048_PESSTO_20170818,G298048_PESSTO_20170819,G298048_PESSTO_20170820,G298048_PESSTO_20170821,G298048_XSH_20170819,G298048_XSH_20170821")
    parser.add_option("-e","--event",default="GW170817")
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)

    parser.add_option("--doAB",  action="store_true", default=False)
    parser.add_option("--doSpec",  action="store_true", default=False)
    parser.add_option("--doLuminosity",  action="store_true", default=False)

    parser.add_option("--errorbudget",default=1.0,type=float)
    parser.add_option("--filters",default="u,g,r,i,z,y,J,H,K")

    opts, args = parser.parse_args()

    return opts


def bns_model_ejecta(mej,vej,th,ph):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    vave = 0.267
    vmin = 0.00
    kappa = 10.0
    eps = 1.58*(10**10)
    alp = 1.2
    eth = 0.5

    flgbct = 1

    t, lbol, mag = BNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth,flgbct)

    return t, lbol, mag

def blue_model_ejecta(mej,vej,beta,kappa_r):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    t, lbol, mag, Tobs = BlueKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,beta,kappa_r)

    return t, lbol, mag, Tobs

def arnett_model_ejecta(mej,vej,slope_r,kappa_r):

    tini = 0.1
    tmax = 50.0
    dt = 0.1

    t, lbol, mag, Tobs = ArnettKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,slope_r,kappa_r)

    return t, lbol, mag, Tobs

# Parse command line
opts = parse_commandline()

filters = opts.filters.split(",")

lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir

outputDir = opts.outputDir
baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,opts.outputName)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog","Afterglow","metzger_rprocess","korobkin_kilonova","Blue"]
models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2015)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich and Ujevic (2017)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Van Eerten et al. (2012)","Metzger et al. (2010)","Wollaeger et al. (2017)","Metzger (2017)"]

if opts.doAB:

    if opts.doEvent:
        filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
        data_out = lightcurve_utils.loadEvent(filename)
        for ii,key in enumerate(data_out.iterkeys()):
            if key == "t":
                continue
            else:
                data_out[key][:,0] = data_out[key][:,0] - opts.T0
                data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)

    if opts.doCadence:
        cadence = [float(x) for x in opts.cadence.split(",")]
   
    filts = ["g","r","i","z","y","J","H","K"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [1,2,3,4,5,5,6,7]
    
    plotName = "%s/models_zoom.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    
    tini, tmax, dt = 0.0, 21.0, 0.1
    tt = np.arange(tini,tmax,dt)
    
    for filt, color, magidx in zip(filts,colors,magidxs):
    
        if opts.doEvent:
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)
   
    if opts.doCadence:
        for c in cadence:
            plt.plot([c,c],[-100,100],'k--')
    
    plt.xlim([0.0, 18.0])
    plt.ylim([-18.0,-10.0])
    
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Absolute Magnitude',fontsize=24)
    plt.legend(loc="best",prop={'size':16},numpoints=1)
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()

    filts = ["u","g","r","i","z","y","J","H","K"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [0,1,2,3,4,5,5,6,7]

    plotName = "%s/models_panels.pdf"%(plotDir)
    #plt.figure(figsize=(20,18))
    plt.figure(figsize=(20,28))

    tini, tmax, dt = 0.0, 21.0, 0.1
    tt = np.arange(tini,tmax,dt)

    cnt = 0
    for filt, color, magidx in zip(filts,colors,magidxs):
        cnt = cnt+1
        vals = "%d%d%d"%(len(filts),1,cnt)
        if cnt == 1:
            ax1 = plt.subplot(eval(vals))
        else:
            ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)

        if opts.doEvent:
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c=color,markersize=15)
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c=color,markersize=15)

        if filt == "u":
            app_mag = 27.0
            distance = 100.0
            lim_mag = app_mag - 5*(np.log10(distance*1e6) - 1)
            plt.plot([3,5],[lim_mag,lim_mag],'k')
            plt.text(3,lim_mag-0.5,'HST F336W')
        elif filt == "J":
            app_mag = 26.0
            distance = 100.0
            lim_mag = app_mag - 5*(np.log10(distance*1e6) - 1)
            plt.plot(5,lim_mag,'ko')
            plt.text(5-1,lim_mag-0.5,'HST F110W')
            plt.plot(10,lim_mag,'ko')
            plt.text(10-1,lim_mag-0.5,'HST F110W')
            app_mag = 27.0
            distance = 100.0
            lim_mag = app_mag - 5*(np.log10(distance*1e6) - 1)
            plt.plot(20,lim_mag,'ko')
            plt.text(20-1,lim_mag-0.5,'HST F110W')
        elif filt == "H":
            app_mag = 25.5
            distance = 100.0
            lim_mag = app_mag - 5*(np.log10(distance*1e6) - 1)
            plt.plot(5,lim_mag,'ko')
            plt.text(5-1,lim_mag-0.5,'HST F160W')
            plt.plot(10,lim_mag,'ko')
            plt.text(10-1,lim_mag-0.5,'HST F160W')
            app_mag = 26.5
            distance = 100.0
            lim_mag = app_mag - 5*(np.log10(distance*1e6) - 1)
            plt.plot(20,lim_mag,'ko')
            plt.text(20-1,lim_mag-0.5,'HST F160W')

        if opts.doCadence:
            for c in cadence:
                plt.plot([c,c],[-100,100],'k--')

        plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
        plt.xlim([0.0, 21.0])
        plt.ylim([-18.0,-7.0])
        plt.gca().invert_yaxis()
        plt.grid()
 
        if cnt == 1:
            ax1.set_yticks([-18,-16,-14,-12,-10,-8])
            plt.setp(ax1.get_xticklabels(), visible=False)
            #l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(filts):
            plt.setp(ax2.get_xticklabels(), visible=False)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

    ax1.set_zorder(1)
    plt.xlabel('Time [days]',fontsize=48)
    plt.savefig(plotName)
    plt.close()
   
elif opts.doSpec:

    if opts.doEvent:
        filename = "../spectra/%s_spectra_index.dat"%opts.event
        lines = [line.rstrip('\n') for line in open(filename)]
        filenames = []
        T0s = []
        for line in lines:
            lineSplit = line.split(" ")
            #if not lineSplit[0] == opts.event: continue
            filename = "%s/%s"%(spectraDir,lineSplit[1])
            filenames.append(filename)
            mjd = Time(lineSplit[2], format='isot').mjd
            T0s.append(mjd-opts.T0)
    
        distconv = (opts.distance*1e6/10)**2
        pctocm = 3.086e18 # 1 pc in cm
        distconv = 4*np.pi*(opts.distance*1e6*pctocm)**2
    
        data_out = {}
        cnt = 0
        for filename,T0 in zip(filenames,T0s):
            cnt = cnt + 1
            #if cnt > 5: continue
    
            data_out_temp = lightcurve_utils.loadEventSpec(filename)
            data_out[str(T0)] = data_out_temp
    
            data_out[str(T0)]["data"] = data_out[str(T0)]["data"]*distconv
            data_out[str(T0)]["error"] = data_out[str(T0)]["error"]*distconv
    
            data_out[str(T0)]["data"] = scipy.signal.medfilt(data_out[str(T0)]["data"],kernel_size=15)
            data_out[str(T0)]["error"] = scipy.signal.medfilt(data_out[str(T0)]["error"],kernel_size=15)

    keys = sorted(data_out.keys())
    colors=cm.rainbow(np.linspace(0,1,len(keys)))

    plotName = "%s/spec_panels.pdf"%(plotDir)
    plotNamePNG = "%s/spec_panels.png"%(plotDir)
    fig = plt.figure(figsize=(20,28))
    
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
    
        if opts.doEvent:
            plt.plot(data_out[key]["lambda"],np.log10(data_out[key]["data"]),'k--',linewidth=2)
    
        plt.fill_between([13500.0,14500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
        plt.fill_between([18000.0,19500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
    
        plt.ylabel('%.1f'%float(key),fontsize=48,rotation=0,labelpad=40)
        plt.xlim([5000, 25000])
        plt.ylim([35.5,37.9])
        plt.grid()
        plt.yticks(fontsize=36)
    
        if (not cnt == len(keys)) and (not cnt == 1):
            plt.setp(ax2.get_xticklabels(), visible=False)
        elif cnt == 1:
            plt.setp(ax1.get_xticklabels(), visible=False)
            #l = plt.legend(loc="upper right",prop={'size':40},numpoints=1,shadow=True, fancybox=True)
            l = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2, prop={'size':48})
    
        else:
            plt.xticks(fontsize=36)
   
    ax3 = fig.add_subplot(111)
    ax3.plot([0,1],[0.5,0.5],'--',c='coral',linewidth=5,clip_on=False)
    ax3.text(-0.15,0.485,"WFC3-IR",fontsize=36)
    ax3.plot([0,1],[-0.03,-0.03],'--',c='coral',linewidth=5,clip_on=False)
    ax3.text(-0.15,-0.04,"WFC3-IR",fontsize=36) 
    ax3.set_xlim([0,1])
    ax3.set_ylim([0,1])
    ax3.axis("off")
 
    ax1.set_zorder(1)
    ax2.set_xlabel(r'$\lambda [\AA]$',fontsize=48,labelpad=30)
    plt.savefig(plotNamePNG,bbox_inches='tight')
    plt.close()
    convert_command = "convert %s %s"%(plotNamePNG,plotName)
    os.system(convert_command)

elif opts.doLuminosity:

    names = opts.name.split(",")
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

    Lbols, names = lightcurve_utils.read_files_lbol(filenames)

    if opts.doModels:

        model_data = {}
        modelfiles = opts.modelfile.split(",")
        for modelfile in modelfiles:
            modelfile = os.path.join(opts.plotDir,modelfile)
            modelfileSplit = modelfile.split("/")
            model_out = np.loadtxt(modelfile)

            errorbudget = float(modelfileSplit[-2])
            modelType = modelfileSplit[-4]
            model = modelfileSplit[-6].split("_")[0]

            if model == "BNS" and modelType == "ejecta":
                t0_best, mej_best,vej_best,th_best,ph_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]

                tmag, lbol, mag = bns_model_ejecta(mej_best,vej_best,th_best,ph_best)
                tmag = tmag + t0_best
            elif model == "Blue" and modelType == "ejecta":
                t0_best, mej_best,vej_best,beta_best,kappa_r_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]

                tmag, lbol, mag, Tobs = blue_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best)
                tmag = tmag + t0_best
            elif model == "Arnett" and modelType == "ejecta":
                t0_best, mej_best,vej_best,slope_r_best,kappa_r_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]
                tmag, lbol, mag, Tobs = arnett_model_ejecta(mej_best,vej_best,slope_r_best,kappa_r_best)
                tmag = tmag + t0_best

            else:
                print "Not implemented..."
                exit(0)

            model_data[model] = {}
            model_data[model]["tmag"] = tmag
            model_data[model]["lbol"] = lbol
            model_data[model]["mag"] = mag
            model_data[model]["Tobs"] = Tobs

    if opts.doEvent:
        filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
        data_out = lightcurve_utils.loadEventLbol(filename)
        data_out["tt"] = data_out["tt"] - opts.T0

    #colors = ["g","r","c","y","m"]
    if opts.doModels:
        colors_names=cm.rainbow(np.linspace(0,1,len(names)+len(model_data.keys())))
    else:
        colors_names=cm.rainbow(np.linspace(0,1,len(names)))

    plotName = "%s/models_Lbol.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,name in enumerate(names):
        Lbol_d = Lbols[name]
        indexes = np.where(~np.isnan(Lbol_d["Lbol"]))[0]
        index1 = indexes[0]
        index2 = int(len(indexes)/2)
        offset = 0.0
        t = Lbol_d["tt"]
        plt.loglog(t,Lbol_d["Lbol"]+offset,'-',label=legend_names[ii],linewidth=2,color=colors_names[ii])
        plt.fill_between(t,Lbol_d["Lbol"]/2.5,Lbol_d["Lbol"]*2.5,facecolor=colors_names[ii],alpha=0.2)

    if opts.doModels:

        for ii, model in enumerate(model_data.keys()):
            tmag, lbol, mag = model_data[model]["tmag"], model_data[model]["lbol"], model_data[model]["mag"]

            tini, tmax, dt = 0.0, 14.0, 0.1
            tt = np.arange(tini,tmax,dt)
    
            idx = np.where(~np.isnan(lbol))[0]
            f = interp.interp1d(tmag[idx], lbol[idx], fill_value='extrapolate')
            lbolinterp = f(tt)
    
            if model == "BNS" and modelType == "ejecta":
                legend_name = "Dietrich and Ujevic (2017)"
            elif model == "Blue" and modelType == "ejecta":
                legend_name = "Metzger (2017)"
            elif model == "Arnett" and modelType == "ejecta":
                legend_name = "Anders (2017)"   
 
            zp_factor = 10**(zp_best/-2.5)
            plt.loglog(tt,zp_factor*lbolinterp,'--',c=colors_names[int(len(names)+ii)],linewidth=2,label=legend_name)
            plt.fill_between(tt,zp_factor*lbolinterp/2.5,zp_factor*lbolinterp*2.5,facecolor=colors_names[int(len(names)+ii)],alpha=0.2)

    if opts.doEvent:
        plt.errorbar(data_out["tt"],data_out["Lbol"],data_out["Lbol_err"],fmt='o',c="k")

    plt.xlim([10**-1,20])
    plt.ylim([10.0**39,10.0**46])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Bolometric Luminosity [erg/s]',fontsize=24)
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(plotName)
    plt.close()

    plotName = "%s/models_T.pdf"%(plotDir)
    plt.figure(figsize=(10,8))

    if opts.doModels:

        tini, tmax, dt = 0.0, 14.0, 0.1
        tt = np.arange(tini,tmax,dt)

        idx = np.where(~np.isnan(Tobs))[0]
        f = interp.interp1d(tmag[idx], Tobs[idx], fill_value='extrapolate')
        Tobsinterp = f(tt)

        if model == "BNS" and modelType == "ejecta":
            legend_name = "Dietrich and Ujevic (2017)"
        elif model == "Blue" and modelType == "ejecta":
            legend_name = "Metzger (2017)"

        plt.loglog(tt,Tobsinterp,'--',c=colors_names[len(names)],linewidth=2,label=legend_name)

    if opts.doEvent:
        plt.errorbar(data_out["tt"],data_out["T"],data_out["T_err"],fmt='o',c="k")

    plt.xlim([10**-2,50])
    plt.ylim([0.0,10000])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Temperature [K]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(plotName)
    plt.close()
