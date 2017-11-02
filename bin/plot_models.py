
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
 
    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-s","--spectraDir",default="../spectra")

    #parser.add_option("-n","--name",default="rpft_m005_v2,BHNS_H4M005V20,BNS_H4M005V20,neutron_precursor3,SED_ns12ns12_kappa10")
    #parser.add_option("-f","--outputName",default="fiducial")

    #parser.add_option("-n","--name",default="rpft_m005_v2,BHNS_H4M005V20,BNS_H4M005V20,neutron_precursor3,SED_ns12ns12_kappa10")
    #parser.add_option("-n","--name",default="rprocess")
    #parser.add_option("-n","--name",default="rpft_m005_v2,SED_ns12ns12_kappa10,a80_leak_HR,APR4-1215_k1,SAd_magnitudes_m0.005_v0.2")
    parser.add_option("-n","--name",default="rpft_m05_v2,t300A3p15,APR4-1215_k1,SAd_magnitudes_m0.005_v0.2,Blue_H4M050V20")
    parser.add_option("-f","--outputName",default="G298048_all")
    #parser.add_option("-f","--outputName",default="G298048_rprocess")
    #parser.add_option("-f","--outputName",default="G298048_lanthanides")
    #parser.add_option("-n","--name",default="rpft_m005_v2,SED_ns12ns12_kappa10,a80_leak_HR")
    #parser.add_option("-f","--outputName",default="fiducial_spec")
    #parser.add_option("-n","--name",default="rpft_m005_v2,SED_ns12ns12_kappa10,a80_leak_HR")
    #parser.add_option("-f","--outputName",default="fiducial_spec")
    #parser.add_option("-n","--name",default="a80_leak_HR,t000A3,t100A3p15_SD1e-2,t300A3p15,tInfA3p15")
    #parser.add_option("-n","--name",default="a80_leak_HR")
    #parser.add_option("-f","--outputName",default="kilonova_wind")    

    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("-e","--event",default="G298048_PS1_GROND_SOFI")
    #parser.add_option("-e","--event",default="G298048_XSH_PESSTO")
    #parser.add_option("-e","--event",default="G298048_20170822")
    #parser.add_option("-e","--event",default="G298048_PESSTO_20170818,G298048_PESSTO_20170819,G298048_PESSTO_20170820,G298048_PESSTO_20170821,G298048_XSH_20170819,G298048_XSH_20170821")
    parser.add_option("--distance",default=40.0,type=float)
    parser.add_option("--T0",default=57982.5285236896,type=float)

    parser.add_option("--doModels",  action="store_true", default=False)
    #parser.add_option("-m","--modelfile",default="gws_luminosity/Arnett_FixZPT0/0_14/ejecta/G298048_XSH_PESSTO/1.00/best.dat")
   # parser.add_option("-m","--modelfile",default="gws_luminosity/Blue_FixZPT0/0_14/ejecta/G298048_XSH_PESSTO/1.00/best.dat,gws_luminosity/Arnett_FixZPT0/0_14/ejecta/G298048_XSH_PESSTO/1.00/best.dat")
    parser.add_option("-m","--modelfile",default="gws/Me2017x2_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/best.dat")
    #parser.add_option("-m","--modelfile",default="gws/Me2017_FixZPT0/u_g_r_i_z_y_J_H_K/0_14/ejecta/GW170817/1.00/best.dat")

    #parser.add_option("-n","--name",default="H4Q3a0,H4Q3a25,H4Q3a50,H4Q3a75")
    #parser.add_option("-f","--outputName",default="spin")

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

        model_data = {}
        modelfiles = opts.modelfile.split(",")
        for modelfile in modelfiles: 
            modelfile = os.path.join(opts.plotDir,modelfile)
            modelfileSplit = modelfile.split("/")
            model_out = np.loadtxt(modelfile)
   
            errorbudget = float(modelfileSplit[-2])
            modelType = modelfileSplit[-4]
            model = modelfileSplit[-7].split("_")[0]

            if model == "DiUj2017" and modelType == "ejecta":
                t0_best, mej_best,vej_best,th_best,ph_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]
     
                tmag, lbol, mag = bns_model_ejecta(mej_best,vej_best,th_best,ph_best)
                tmag = tmag + t0_best
            elif model == "Me2017" and modelType == "ejecta":   
                t0_best, mej_best,vej_best,beta_best,kappa_r_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]

                print mej_best,vej_best,beta_best,kappa_r_best

                tmag, lbol, mag = Me2017_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best)

                #tmag, lbol, mag, Tobs = blue_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best)
                tmag = tmag + t0_best
            elif model == "Me2017x2" and modelType == "ejecta":
                t0_best, mej_1_best,vej_1_best,beta_1_best,kappa_r_1_best,mej_2_best,vej_2_best,beta_2_best,kappa_r_2_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5], model_out[6], model_out[7], model_out[8], model_out[9]

                mej_1_best = 0.08329
                vej_1_best = 0.01252
                beta_1_best = 1.05949
                kappa_r_1_best = 0.4391
                
                mej_2_best = 1e-5
                vej_2_best = 0.2
                beta_2_best = 1.0
                kappa_r_2_best = 10.0

                tmag, lbol, mag = Me2017x2_model_ejecta(mej_1_best,vej_1_best,beta_1_best,kappa_r_1_best,mej_2_best,vej_2_best,beta_2_best,kappa_r_2_best)
                tmag = tmag + t0_best 
            elif model == "SmCh2017" and modelType == "ejecta":   
                t0_best, mej_best,vej_best,slope_r_best,kappa_r_best,zp_best  = model_out[0], model_out[1], model_out[2], model_out[3], model_out[4], model_out[5]
                kappa_r_best = 10**kappa_r_best

                tmag, lbol, mag, Tobs = arnett_model_ejecta(mej_best,vej_best,beta_best,kappa_r_best)
                tmag = tmag + t0_best
            else:
                print "Not implemented..."
                exit(0)

            model_data[model] = {}    
            model_data[model]["tmag"] = tmag
            model_data[model]["lbol"] = lbol
            model_data[model]["mag"] = mag
            #model_data[model]["Tobs"] = Tobs 

    if opts.doEvent:
        filename = "%s/%s.dat"%(lightcurvesDir,opts.event)
        data_out = lightcurve_utils.loadEvent(filename)
        for ii,key in enumerate(data_out.iterkeys()):
            if key == "t":
                continue
            else:
                data_out[key][:,0] = data_out[key][:,0] - opts.T0
                data_out[key][:,1] = data_out[key][:,1] - 5*(np.log10(opts.distance*1e6) - 1)
    
    #colors = ["g","r","c","y","m"]
    colors=cm.rainbow(np.linspace(0,1,len(names)))
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
        plt.semilogx(t,mag_d["i"]+offset,'-',label=legend_names[ii],linewidth=2,c=colors[ii])
        plt.semilogx(t,mag_d["g"]+offset,'--',linewidth=2,c=colors[ii])
        #plt.semilogx(t,mag_d["K"]+offset,'.-',linewidth=2,c=colors[ii])   
 
        #plt.plot(t,mag_d["i"]+offset,'-',label=legend_names[ii],linewidth=2,c=colors[ii])
        #plt.plot(t,mag_d["g"]+offset,'--',linewidth=2,c=colors[ii])
        #plt.plot(t,mag_d["K"]+offset,'.-',linewidth=2,c=colors[ii])

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
   
        filt = "K"
        samples = data_out[filt]
        t, y, sigma_y = samples[:,0], samples[:,1], samples[:,2]
        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
        #plt.errorbar(t,y,sigma_y,fmt='*',c="k",label='%s-band'%filt)
 
    if opts.doModels:
   
        for model in model_data:
            tmag, lbol, mag = model_data[model]["tmag"], model_data[model]["lbol"], model_data[model]["mag"] 
            tini, tmax, dt = np.min(t), 14.0, 0.1
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
    
    plt.xlim([10**-1,14])
    #plt.xlim([10**-2,50])
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
    
        if opts.doModels:
            for model in model_data:
                tmag, lbol, mag = model_data[model]["tmag"], model_data[model]["lbol"], model_data[model]["mag"] 

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
               
            ii = np.where(~np.isnan(mag_d[filt]))[0]
            f = interp.interp1d(t[ii], mag_d[filt][ii], fill_value='extrapolate')
            maginterp = f(tt)
            zp_best_tmp = -7.0
            zp_best_tmp = -1.0
            #zp_best_tmp = 0.0
            plt.plot(tt,maginterp+zp_best_tmp,'--',c=color,linewidth=2)
            plt.fill_between(tt,maginterp+zp_best_tmp-opts.errorbudget,maginterp+zp_best_tmp+opts.errorbudget,facecolor=color,alpha=0.2)
   
    plt.xlim([0.0, 18.0])
    plt.ylim([-18.0,-10.0])
    
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Absolute Magnitude',fontsize=24)
    plt.legend(loc="best",prop={'size':16},numpoints=1)
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()

    filts = ["g","r","i","z","y","J","H","K"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [1,2,3,4,5,5,6,7]
    if opts.doModels:
        colors_names=cm.rainbow(np.linspace(0,1,len(names)+1))
    else:
        colors_names=cm.rainbow(np.linspace(0,1,len(names)))

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
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='o',c='k',markersize=15)
            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(t[idx],y[idx],sigma_y[idx],fmt='v',c='k',markersize=15)

        if opts.doModels:
            for model in model_data:
                tmag, lbol, mag = model_data[model]["tmag"], model_data[model]["lbol"], model_data[model]["mag"] 

                idx = np.where(~np.isnan(mag[magidx]))[0]
                f = interp.interp1d(tmag[idx], mag[magidx][idx], fill_value='extrapolate')
                maginterp = f(tt)
       
                if model == "DiUj2017" and modelType == "ejecta":
                    legend_name = "Dietrich and Ujevic (2017)"
                elif model == "Me2017" and modelType == "ejecta":
                    legend_name = "Metzger (2017)"
                elif model == "Me2017x2" and modelType == "ejecta":
                    legend_name = "Metzger (2017)"

                plt.plot(tt,maginterp+zp_best,'--',c=colors_names[len(names)],linewidth=4,label=legend_name)
                plt.plot(tt,maginterp+zp_best-opts.errorbudget,'-',c=colors_names[len(names)],linewidth=4)
                plt.plot(tt,maginterp-zp_best+opts.errorbudget,'-',c=colors_names[len(names)],linewidth=4)

                plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=colors_names[len(names)],alpha=0.2,linewidth=4)

        for ii,name in enumerate(names):
            mag_d = mags[name]
            if not filt in mag_d: continue
            if not filt in filters: continue
            #if not filt in ["g","r"]: continue
            offset = 0.0
            t = mag_d["t"]

            idx = np.where(~np.isnan(mag_d[filt]))[0]
            f = interp.interp1d(t[idx], mag_d[filt][idx], fill_value='extrapolate')
            maginterp = f(tt)
            zp_best_tmp = -7.0
            zp_best_tmp = -1.0
            zp_best_tmp = 0.0


            plt.plot(tt,maginterp+zp_best_tmp,'--',c=colors_names[ii],label=legend_names[ii],linewidth=4)
            plt.plot(tt,maginterp+zp_best_tmp-opts.errorbudget,'-',c=colors_names[ii],linewidth=4)
            plt.plot(tt,maginterp+zp_best_tmp+opts.errorbudget,'-',c=colors_names[ii],linewidth=4)

            plt.fill_between(tt,maginterp+zp_best_tmp-opts.errorbudget,maginterp+zp_best_tmp+opts.errorbudget,facecolor=colors_names[ii],edgecolor=colors_names[ii],alpha=0.2,linewidth=4)
    
        plt.ylabel('%s'%filt,fontsize=48,rotation=0,labelpad=40)
        plt.xlim([0.0, 18.0])
        plt.ylim([-18.0,-10.0])
        plt.gca().invert_yaxis()
        plt.grid()
 
        if cnt == 1:
            ax1.set_yticks([-18,-16,-14,-12,-10])
            plt.setp(ax1.get_xticklabels(), visible=False)
            l = plt.legend(loc="upper right",prop={'size':36},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(filts):
            plt.setp(ax2.get_xticklabels(), visible=False)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

    ax1.set_zorder(1)
    plt.xlabel('Time [days]',fontsize=48)
    plt.savefig(plotName)
    plt.close()
   
    plotName = "%s/models_panels_optical.pdf"%(plotDir)
    plt.figure(figsize=(10,8))

    tini, tmax, dt = 0.0, 14.0, 0.1
    tt = np.arange(tini,tmax,dt)

    cnt = 0
    for filt, color, magidx in zip(filts,colors,magidxs):
        if not filt in ["u","g","r","i","z"]: continue
        cnt = cnt+1
        vals = "%d%d%d"%(4,1,cnt)
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
            plt.errorbar(t,y,sigma_y,fmt='o',c='k')
 
        if opts.doModels:
            for model in model_data:
                tmag, lbol, mag = model_data[model]["tmag"], model_data[model]["lbol"], model_data[model]["mag"] 

                idx = np.where(~np.isnan(mag[magidx]))[0]
                f = interp.interp1d(tmag[idx], mag[magidx][idx], fill_value='extrapolate')
                maginterp = f(tt)

                if model == "BNS" and modelType == "ejecta":
                    legend_name = "Dietrich and Ujevic (2017)"
                elif model == "Blue" and modelType == "ejecta":
                    legend_name = "Metzger (2017)"

                plt.plot(tt,maginterp+zp_best,'--',c=colors_names[len(names)],linewidth=2,label=legend_name)
                plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=colors_names[len(names)],alpha=0.2)

        for ii,name in enumerate(names):
            mag_d = mags[name]
            if not filt in mag_d: continue
            if not filt in filters: continue
            #if not filt in ["g","r"]: continue
            offset = 0.0
            t = mag_d["t"]

            idx = np.where(~np.isnan(mag_d[filt]))[0]
            f = interp.interp1d(t[idx], mag_d[filt][idx], fill_value='extrapolate')
            maginterp = f(tt)
            zp_best_tmp = -7.0
            zp_best_tmp = -1.0
            zp_best_tmp = 0.0
            plt.plot(tt,maginterp+zp_best_tmp,'--',c=colors_names[ii],linewidth=2,label=legend_names[ii])
            plt.fill_between(tt,maginterp+zp_best_tmp-opts.errorbudget,maginterp+zp_best_tmp+opts.errorbudget,facecolor=colors_names[ii],alpha=0.2)

        plt.ylabel('%s'%filt,fontsize=24,rotation=0,labelpad=20)
        plt.xlim([0.0, 14.0])
        plt.ylim([-18.0,-10.0])
        plt.gca().invert_yaxis()
        plt.grid()

        if cnt == 1:
            ax1.set_yticks([-18,-14,-10])
            plt.setp(ax1.get_xticklabels(), visible=False)
            #l = plt.legend(loc="upper right",prop={'size':16},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(filts):
            plt.setp(ax2.get_xticklabels(), visible=False)

    ax1.set_zorder(1)
    plt.xlabel('Time [days]',fontsize=24)
    plt.savefig(plotName)
    plt.close()

    plotName = "%s/models_panels_nir.pdf"%(plotDir)
    plt.figure(figsize=(10,8))

    tini, tmax, dt = 0.0, 14.0, 0.1
    tt = np.arange(tini,tmax,dt)

    cnt = 0
    for filt, color, magidx in zip(filts,colors,magidxs):
        if not filt in ["y","J","H","K"]: continue
        cnt = cnt+1
        vals = "%d%d%d"%(4,1,cnt)
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
            plt.errorbar(t,y,sigma_y,fmt='o',c='k')

        if opts.doModels:
            for model in model_data:
                tmag, lbol, mag = model_data[model]["tmag"], model_data[model]["lbol"], model_data[model]["mag"] 

                idx = np.where(~np.isnan(mag[magidx]))[0]
                f = interp.interp1d(tmag[idx], mag[magidx][idx], fill_value='extrapolate')
                maginterp = f(tt)

                if model == "BNS" and modelType == "ejecta":
                    legend_name = "Dietrich and Ujevic (2017)"
                elif model == "Blue" and modelType == "ejecta":
                    legend_name = "Metzger (2017)"

                plt.plot(tt,maginterp+zp_best,'--',c=colors_names[len(names)],linewidth=2,label=legend_name)
                plt.fill_between(tt,maginterp+zp_best-errorbudget,maginterp+zp_best+errorbudget,facecolor=colors_names[len(names)],alpha=0.2)

        for ii,name in enumerate(names):
            mag_d = mags[name]
            if not filt in mag_d: continue
            if not filt in filters: continue
            #if not filt in ["g","r"]: continue
            offset = 0.0
            t = mag_d["t"]

            idx = np.where(~np.isnan(mag_d[filt]))[0]
            f = interp.interp1d(t[idx], mag_d[filt][idx], fill_value='extrapolate')
            maginterp = f(tt)
            zp_best_tmp = -7.0
            zp_best_tmp = -1.0
            zp_best_tmp = 0.0
            plt.plot(tt,maginterp+zp_best_tmp,'--',c=colors_names[ii],linewidth=2,label=legend_names[ii])
            plt.fill_between(tt,maginterp+zp_best_tmp-opts.errorbudget,maginterp+zp_best_tmp+opts.errorbudget,facecolor=colors_names[ii],alpha=0.2)

        plt.ylabel('%s'%filt,fontsize=24,rotation=0,labelpad=20)
        plt.xlim([0.0, 14.0])
        plt.ylim([-18.0,-10.0])
        plt.gca().invert_yaxis()
        plt.grid()

        if cnt == 1:
            ax1.set_yticks([-18,-14,-10])
            plt.setp(ax1.get_xticklabels(), visible=False)
            #l = plt.legend(loc="upper right",prop={'size':16},numpoints=1,shadow=True, fancybox=True)
        elif not cnt == len(filts):
            plt.setp(ax2.get_xticklabels(), visible=False)

    ax1.set_zorder(1)
    plt.xlabel('Time [days]',fontsize=24)
    plt.savefig(plotName)
    plt.close()

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
        plt.semilogx(t,mag_d["i"]-mag_d["g"],'-',label=legend_names[ii],linewidth=2,color=colors[ii])
    plt.xlim([10**-2,50])
    #plt.ylim([-15,5])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Absolute Magnitude [i-g]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()
  
    colors = ["g","r","c","y","m"]

    plotName = "%s/models_gminusi.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    for ii,name in enumerate(names):
        mag_d = mags[name]
        indexes = np.where(~np.isnan(mag_d["g"]))[0]
        index1 = indexes[0]
        index2 = int(len(indexes)/2)
        offset = -mag_d["g"][index2] + ii*3
        offset = 0.0
        t = mag_d["t"]
        plt.semilogx(t,mag_d["g"]-mag_d["i"],'-',label=legend_names[ii],linewidth=2,color=colors[ii])
    plt.xlim([10**-2,50])
    #plt.ylim([-15,5])
    plt.xlabel('Time [days]',fontsize=24)
    plt.ylabel('Color [g-i]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    #plt.gca().invert_yaxis()
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
        events = opts.event.split(",")
        eventdata = {}
        for event in events:
            filename = "%s/%s.dat"%(spectraDir,event)
            data_out = lightcurve_utils.loadEventSpec(filename)
            eventdata[event] = data_out

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
    if maxhist < 0: maxhist = 1
 
    if opts.doEvent:
        events = eventdata.keys()
        colors=cm.rainbow(np.linspace(0,1,len(events)))
        for ii,event in enumerate(events):
            x = eventdata[event]["lambda"]
            y = np.abs(eventdata[event]["data"])/np.max(np.abs(eventdata[event]["data"]))
            #plt.loglog(x,y,'-',c=colors[ii],label=event)
            idx = np.where( (x >= 4000))[0]
            #s1 = InterpolatedUnivariateSpline(x[idx], y[idx], k=1.0) 
            #plt.loglog(x,s1(x),'--',c=colors[ii])           
            filtered = lowess(y[idx],x[idx], is_sorted=True, frac=0.10, it=0)
            plt.loglog(filtered[:,0],filtered[:,1],'--',c=colors[ii])

    from astropy.modeling.models import BlackBody1D
    from astropy.modeling.blackbody import FLAM
    from astropy import units as u
    from astropy.visualization import quantity_support

    bb = BlackBody1D(temperature=5000*u.K)
    wav = np.arange(1000, 110000) * u.AA
    flux = bb(wav).to(FLAM, u.spectral_density(wav))

    plt.semilogx(wav, flux/np.max(flux)) 

    plt.xlim([3000,30000])
    #plt.ylim([10.0**39,10.0**43])
    plt.xlabel(r'$\lambda [\AA]$',fontsize=24)
    plt.ylabel('Normalized Fluence [erg/s/cm2/A]',fontsize=24)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(plotName)
    plt.close()

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
