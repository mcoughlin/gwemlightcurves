
import os, sys
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
from astropy.time import Time

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from scipy.optimize import curve_fit

from gwemlightcurves import BHNSKilonovaLightcurve, BNSKilonovaLightcurve, SALT2
from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
 
    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")

    #parser.add_option("-n","--name",default="rpft_m005_v2,BHNS_H4M005V20,BNS_H4M005V20,neutron_precursor3,SED_ns12ns12_kappa10")
    #parser.add_option("-f","--outputName",default="fiducial")

    #parser.add_option("-n","--name",default="rpft_m005_v2,BHNS_H4M005V20,BNS_H4M005V20,neutron_precursor3,SED_ns12ns12_kappa10")
    #parser.add_option("-n","--name",default="rprocess")
    #parser.add_option("-f","--outputName",default="G298048_rprocess")
    #parser.add_option("-f","--outputName",default="G298048_lanthanides")
    #parser.add_option("-n","--name",default="rpft_m005_v2,SED_ns12ns12_kappa10,a80_leak_HR")
    #parser.add_option("-f","--outputName",default="fiducial_spec")
    #parser.add_option("-n","--name",default="rpft_m005_v2,SED_ns12ns12_kappa10,a80_leak_HR")
    #parser.add_option("-f","--outputName",default="fiducial_spec")
    #parser.add_option("-n","--name",default="a80_leak_HR,t000A3,t100A3p15_SD1e-2,t300A3p15,tInfA3p15")
    parser.add_option("-n","--name",default="a80_leak_HR")
    parser.add_option("-f","--outputName",default="kilonova_wind")    

    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("-e","--event",default="G298048_GROND")
    #parser.add_option("-e","--event",default="G298048_20170822")
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)

    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("-m","--modelfile",default="gws/BNS/i_z_y_J_H_K/ejecta/G298048_GROND/1.00/best.dat")

    #parser.add_option("-n","--name",default="H4Q3a0,H4Q3a25,H4Q3a50,H4Q3a75")
    #parser.add_option("-f","--outputName",default="spin")

    parser.add_option("--doAB",  action="store_true", default=False)
    parser.add_option("--doSpec",  action="store_true", default=False)

    parser.add_option("--errorbudget",default=1.0,type=float)
    parser.add_option("--filters",default="g,r,i,z")

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

# Parse command line
opts = parse_commandline()

filters = opts.filters.split(",")

lightcurvesDir = opts.lightcurvesDir
spectraDir = opts.spectraDir

outputDir = opts.outputDir
baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,opts.outputName)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","BHNS","BNS","SN","tanaka_compactmergers","macronovae-rosswog","Afterglow","metzger_rprocess"]
models_ref = ["Barnes et al. (2016)","Barnes and Kasen (2013)","Kasen et al. (2014)","Metzger et al. (2015)","Kawaguchi et al. (2016)","Dietrich and Ujevic (2017)","Guy et al. (2007)","Tanaka and Hotokezaka (2013)","Rosswog et al. (2017)","Van Eerten et al. (2012)","Metzger et al. (2010)"]

if opts.doAB:

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
    
    mags, names = lightcurve_utils.read_files(filenames)
    
    if opts.doModels:
    
        modelfile = os.path.join(opts.plotDir,opts.modelfile)
        modelfileSplit = modelfile.split("/")
        model_out = np.loadtxt(modelfile)
    
        errorbudget = float(modelfileSplit[-2])
        modelType = modelfileSplit[-4]
        model = modelfileSplit[-6] 
    
        if model == "BNS" and modelType == "ejecta":
            t0_best, mej_best,vej_best,th_best,ph_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]
     
            tmag, lbol, mag = bns_model_ejecta(mej_best,vej_best,th_best,ph_best)
            tmag = tmag + t0_best
    
        else:
            print "Not implemented..."
            exit(0)
    
    if opts.doEvent:
        filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
        data_out = lightcurve_utils.loadEvent(filename)
        for ii,key in enumerate(data_out.iterkeys()):
            if key == "t":
                continue
            else:
                data_out[key][:,0] = data_out[key][:,0] - opts.T0
                data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)
    
    colors = ["g","r","c","y","m"]
    plotName = "%s/models.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,name in enumerate(names):
        mag_d = mags[name]
        indexes = np.where(~np.isnan(mag_d["g"]))[0]
        index1 = indexes[0]
        index2 = int(len(indexes)/2)
        offset = -mag_d["g"][index2] + ii*3
        offset = 0.0
        t = mag_d["t"]
        linestyle = "%s-"%colors[ii]
        plt.semilogx(t,mag_d["i"]+offset,linestyle,label=legend_names[ii],linewidth=2)
        linestyle = "%s--"%colors[ii]
        plt.semilogx(t,mag_d["g"]+offset,linestyle,linewidth=2)
    
    if opts.doEvent:
    
        filt = "g"
        samples = data_out[filt]
        t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
        plt.errorbar(t,y,sigma_y,fmt='o',c="k",label='%s-band'%filt)    
    
        filt = "i"
        samples = data_out[filt]
        t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
        plt.errorbar(t,y,sigma_y,fmt='^',c="k",label='%s-band'%filt)  
    
    if opts.doModels:
    
        tini, tmax, dt = np.min(t), 10.0, 0.1
        tt = np.arange(tini,tmax,dt)
    
        magidx = 2
        ii = np.where(~np.isnan(mag[magidx]))[0]
        f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
        maginterp = f(tt)
        plt.plot(tt,maginterp+zp_best,'k',linewidth=2)
    
        magidx = 0
        ii = np.where(~np.isnan(mag[magidx]))[0]
        f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
        maginterp = f(tt)
        plt.plot(tt,maginterp+zp_best,'k--',linewidth=2)
    
    plt.xlim([10**-2,50])
    #plt.ylim([-15,5])
    plt.ylim([-20,5])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Absolute Magnitude',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()
    
    filts = ["g","r","i","z","y","J","H","K"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [1,2,3,4,5,5,6,7]
    
    plotName = "%s/models_zoom.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    
    tini, tmax, dt = 0.0, 10.0, 0.1
    tt = np.arange(tini,tmax,dt)
    
    for filt, color, magidx in zip(filts,colors,magidxs):
    
        if opts.doEvent:
            if not filt in data_out: continue
            samples = data_out[filt]
            t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            plt.errorbar(t,y,sigma_y,fmt='o',c=color,label='%s-band'%filt)
    
        if opts.doModels:
            if not filt in ["g","r"]:
                ii = np.where(~np.isnan(mag[magidx]))[0]
                f = interp.interp1d(tmag[ii], mag[magidx][ii], fill_value='extrapolate')
                maginterp = f(tt)
                plt.plot(tt,maginterp+zp_best,'--',c=color,linewidth=2)
                plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=color,alpha=0.2)
    
        for ii,name in enumerate(names):
            mag_d = mags[name]
            if not filt in mag_d: continue
            if not filt in filters: continue
            #if not filt in ["g","r"]: continue
            offset = 0.0
            t = mag_d["t"]
            linestyle = "%s-"%colors[ii]
               
            ii = np.where(~np.isnan(mag_d[filt]))[0]
            f = interp.interp1d(t[ii], mag_d[filt][ii], fill_value='extrapolate')
            maginterp = f(tt)
            zp_best_tmp = -7.0
            zp_best_tmp = -1.0
            #zp_best_tmp = 0.0
            plt.plot(tt,maginterp+zp_best_tmp,'--',c=color,linewidth=2)
            plt.fill_between(tt,maginterp+zp_best_tmp-opts.errorbudget,maginterp+zp_best_tmp+opts.errorbudget,facecolor=color,alpha=0.2)
    
    plt.xlim([0.0, 3.5])
    plt.ylim([-20.0,-10.0])
    
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Absolute Magnitude',fontsize=24)
    plt.legend(loc="best",prop={'size':16},numpoints=1)
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()
    
    colors = ["g","r","c","y","m"]
    plotName = "%s/models_iminusg.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,name in enumerate(names):
        mag_d = mags[name]
        indexes = np.where(~np.isnan(mag_d["g"]))[0]
        index1 = indexes[0]
        index2 = int(len(indexes)/2)
        offset = -mag_d["g"][index2] + ii*3
        offset = 0.0
        t = mag_d["t"]
        linestyle = "%s-"%colors[ii]
        plt.semilogx(t,mag_d["i"]-mag_d["g"],linestyle,label=legend_names[ii],linewidth=2)
    plt.xlim([10**-2,50])
    #plt.ylim([-15,5])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Absolute Magnitude [i-g]',fontsize=24)
    plt.legend(loc="best")
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
    Lbols, names = lightcurve_utils.read_files_lbol(filenames)
    
    plotName = "%s/models_Lbol.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,name in enumerate(names):
        Lbol_d = Lbols[name]
        indexes = np.where(~np.isnan(Lbol_d["Lbol"]))[0]
        index1 = indexes[0]
        index2 = int(len(indexes)/2)
        offset = 0.0
        t = Lbol_d["t"]
        linestyle = "%s-"%colors[ii]
        plt.loglog(t,Lbol_d["Lbol"]+offset,linestyle,label=legend_names[ii],linewidth=2)
    plt.xlim([10**-2,50])
    plt.ylim([10.0**39,10.0**43])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Bolometric Luminosity [erg/s]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(plotName)
    plt.close()

elif opts.doSpec:

    names = opts.name.split(",")
    filenames = []
    legend_names = []
    for name in names:
        for ii,model in enumerate(models):
            filename = '%s/%s/%s_spec.dat'%(outputDir,model,name)
            if not os.path.isfile(filename):
                continue
            filenames.append(filename)
            legend_names.append(models_ref[ii])
            break
    specs, names = lightcurve_utils.read_files_spec(filenames)

    if opts.doEvent:
        filename = "%s/%s.dat"%(spectraDir,opts.event)
        data_out = lightcurve_utils.loadEventSpec(filename)

    maxhist = -1e10
    colors = ["g","r","c","y","m"]
    plotName = "%s/models_spec.pdf"%(plotDir)
    plt.figure(figsize=(12,10))
    for ii,name in enumerate(names):
        spec_d = specs[name]
        spec_d_mean = np.mean(spec_d["data"],axis=0)
        linestyle = "%s-"%colors[ii]
        plt.loglog(spec_d["lambda"],np.abs(spec_d_mean),linestyle,label=legend_names[ii],linewidth=2)
        maxhist = np.max([maxhist,np.max(np.abs(spec_d_mean))])
 
    if opts.doEvent:
        plt.errorbar(data_out["lambda"],np.abs(data_out["data"])*maxhist/np.max(np.abs(data_out["data"])),fmt='--',c='k',label='event')

    plt.xlim([3000,30000])
    #plt.ylim([10.0**39,10.0**43])
    plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
    plt.ylabel('Fluence [erg/s/cm2/A]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(plotName)
    plt.close()

