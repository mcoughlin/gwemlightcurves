
import os, sys
import numpy as np
import optparse

from astropy.table import Table, Column
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from gwemlightcurves.sampler import *
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
    parser.add_option("-b","--boxfitDir",default="../boxfit")
    parser.add_option("-m","--model",default="Ka2017")
    parser.add_option("-e","--eos",default="H4")
    parser.add_option("-q","--massratio",default=3.0,type=float)
    parser.add_option("-a","--chi_eff",default=0.1,type=float) 
    parser.add_option("--mej",default=0.05,type=float)
    parser.add_option("--mej_dyn",default=0.01,type=float)
    parser.add_option("--mej_wind",default=0.03,type=float)
    parser.add_option("--vej",default=0.2,type=float)
    parser.add_option("--m1",default=1.35,type=float)
    parser.add_option("--m2",default=1.35,type=float)
    parser.add_option("-z","--redshift",default=0.001,type=float)
    parser.add_option("--x0",default=1e-5,type=float)
    parser.add_option("--x1",default=0.1,type=float)
    parser.add_option("-c","--c",default=0.01,type=float)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("--theta_0",default=0.1,type=float)
    parser.add_option("--E",default=1e53,type=float)
    parser.add_option("--n",default=1.0,type=float)
    parser.add_option("--theta_obs",default=0.0,type=float)
    parser.add_option("--theta_r",default=0.0,type=float)
    parser.add_option("--beta",default=3.0,type=float)
    parser.add_option("--kappa_r",default=0.1,type=float)
    parser.add_option("--slope_r",default=-1.2,type=float)
    parser.add_option("--Xlan",default=1e-3,type=float)
    parser.add_option("--Ye",default=0.25,type=float)

    parser.add_option("--mej1",default=0.05,type=float)
    parser.add_option("--vej1",default=0.2,type=float)
    parser.add_option("--Xlan1",default=1e-3,type=float)
    parser.add_option("--mej2",default=0.05,type=float)
    parser.add_option("--vej2",default=0.2,type=float)
    parser.add_option("--Xlan2",default=1e-3,type=float)

    parser.add_option("--sd",default=0.034,type=float)
    parser.add_option("--rwind",default=0.2,type=float)
    parser.add_option("--a",default=2.0,type=float)

    parser.add_option("--T",default=6000.0,type=float)
    parser.add_option("--phi",default=45.0,type=float)
    parser.add_option("--iota",default=0.0,type=float)
    parser.add_option("--colormodel",default="a2.0")

    parser.add_option("--kappaLF",default=10,type=float)
    parser.add_option("--gammaLF",default=-1.0,type=float)
    parser.add_option("--kappaLR",default=10,type=float)
    parser.add_option("--gammaLR",default=-1.0,type=float)

    parser.add_option("--doAB",  action="store_true", default=False)
    parser.add_option("--doSpec",  action="store_true", default=False)
    parser.add_option("--doSaveModel",  action="store_true", default=False)

    parser.add_option("--doComparison",  action="store_true", default=False)
    parser.add_option("--comparisonFile",default="../output/kasen_kilonova_grid/knova_d1_n10_m0.050_vk0.20_fd1.0_Xlan1e-3.0.dat") 

    parser.add_option("--gptype",default="sklearn")

    opts, args = parser.parse_args()
 
    return opts

# Parse command line
opts = parse_commandline()

boxfitDir = opts.boxfitDir

m1 = opts.m1
m2 = opts.m2
q = opts.massratio
chi_eff = opts.chi_eff
mej = opts.mej
mej_dyn = opts.mej_dyn
mej_wind = opts.mej_wind
vej = opts.vej
theta_0 = opts.theta_0
theta_r = opts.theta_r
E = opts.E
n = opts.n
theta_obs = opts.theta_obs
beta = opts.beta
kappa_r = opts.kappa_r
slope_r = opts.slope_r
Xlan = opts.Xlan
Ye = opts.Ye
iota = opts.iota
phi = opts.phi
T = opts.T
a = opts.a
sd = opts.sd
rwind = opts.rwind


mej1 = opts.mej1
vej1 = opts.vej1
Xlan1 = opts.Xlan1
mej2 = opts.mej2
vej2 = opts.vej2
Xlan2 = opts.Xlan2

colormodel = opts.colormodel.split(",")

if opts.model!="SN" and opts.model!="Afterglow":
    if not opts.doAB and not opts.doSpec:
        print("ERROR! Must use --model SN, --model Afterglow, --doAB, or --doSpec, otherwise tmax will not be defined")

if opts.eos == "APR4":
    c = 0.180
    mb = 1.50
elif opts.eos == "ALF2":
    c = 0.161
    mb = 1.49
elif opts.eos == "H4":
    c = 0.147
    mb = 1.47
elif opts.eos == "MS1":
    c = 0.138
    mb = 1.46

mns = 1.35

tini = 0.1
dt = 0.5

if opts.model == "SN":
    tmax = 50.0
else:
    if opts.doAB:
        tmax = 21.0
    elif opts.doSpec:
        tmax = 10.0


lambdaini = 5000
#lambdaini = 3500
#lambdaini = 4500
lambdamax = 25000
#dlambda = 50.0 
dlambda = 500.0
#dlambda = 100.0

vave = 0.267
vmin = 0.02
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 1

p = 2.5    
epsilon_B = 1e-2 
epsilon_E = 1e-1
ksi_N = 1

#add default values from above to table
samples = {}
samples['tini'] = tini
samples['tmax'] = tmax
samples['dt'] = dt
samples['lambdaini'] = lambdaini
samples['lambdamax'] = lambdamax
samples['dlambda'] = dlambda
samples['vmin'] = vmin
samples['th'] = th
samples['ph'] = ph
samples['kappa'] = kappa
samples['eps'] = eps
samples['alp'] = alp
samples['eth'] = eth
samples['flgbct'] = flgbct
samples['beta'] = beta
samples['kappa_r'] = kappa_r
samples['slope_r'] = slope_r
samples['Xlan'] = Xlan
samples['Ye'] = Ye
samples['a'] = a

samples['theta_0'] = theta_0
samples['theta_r'] = theta_r
samples['E'] = E
samples['n'] = n
samples['theta_obs'] = theta_obs
samples['theta'] = theta_0

samples['mej_dyn'] = mej_dyn
samples['mej_wind'] = mej_wind

samples['mej_1'] = mej1
samples['vej_1'] = vej1
samples['Xlan_1'] = Xlan1
samples['mej_2'] = mej2
samples['vej_2'] = vej2
samples['Xlan_2'] = Xlan2
samples['iota'] = iota
samples['phi'] = phi
samples['T'] = T

samples['kappaLF'] = opts.kappaLF
samples['gammaLF'] = opts.gammaLF
samples['kappaLR'] = opts.kappaLR
samples['gammaLR'] = opts.gammaLR

#Wo2020 keys
samples['sd'] = sd 
samples['rwind'] = rwind

if len(colormodel) == 1:
    samples['colormodel'] = colormodel[0]
else:
    samples['colormodel'] = colormodel

samples["gptype"] = opts.gptype

if opts.doEjecta:
    samples['mej'] = opts.mej
    samples['vej'] = opts.vej
elif opts.doMasses:
    samples['m1'] = opts.m1
    samples['m2'] = opts.m2
    samples['q'] = opts.massratio
    samples['chi_eff'] = opts.chi_eff
    samples['c1'] = c
    samples['c2'] = c
    samples['mb1'] = mb
    samples['mb2'] = mb
else:
    print("Enable --doEjecta or --doMasses")
    exit(0)

ModelPath = '%s/svdmodels'%(opts.outputDir)
#ModelPath = '%s/svdmodels_remove'%(opts.outputDir)
if not os.path.isdir(ModelPath):
    os.makedirs(ModelPath)

if opts.doSaveModel:
    kwargs = {'SaveModel':True,'LoadModel':False,'ModelPath':ModelPath}
else:
    kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
kwargs["doAB"] = opts.doAB
kwargs["doSpec"] = opts.doSpec
kwargs["phi"] = phi

t = Table()
for key in samples.keys():
    val = samples[key]
    t.add_column(Column(data=[val],name=key))
samples = t

if not opts.model in ["SN", "Afterglow"]:
    model_table = KNTable.model(opts.model, samples, **kwargs)
    if opts.doAB:
        t, lbol, mag = model_table["t"][0], model_table["lbol"][0], model_table["mag"][0] 
    elif opts.doSpec:
        t, lambdas, spec = model_table["t"][0], model_table["lambda"][0], model_table["spec"][0]

if opts.model == "KaKy2016":
    if opts.doEjecta:
        name = "KaKy2016_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*100)
    elif opts.doMasses:
        name = "%sQ%.0fa%.0f"%(opts.eos,opts.massratio,opts.chi_eff*100)
elif opts.model == "DiUj2017":
    if opts.doEjecta:
        name = "DiUj2017_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*100)
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "Me2017":
    if opts.doEjecta:
        name = "Me2017_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*100)
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "SmCh2017":
    if opts.doEjecta:
        name = "SmCh2017_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*100)
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "WoKo2017":
    if opts.doEjecta:
        name = "WoKo2017_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*100)
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "BaKa2016":
    if opts.doEjecta:
        name = "BaKa2016_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*100)
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "Ka2017":
    if opts.doEjecta:
        name = "Ka2017_%sM%03dV%02dX%d"%(opts.eos,opts.mej*1000,opts.vej*100,np.log10(opts.Xlan))
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "Ka2017inc":
    if opts.doEjecta:
        name = "Ka2017inc_M%03dV%02dX%d_i%.0f"%(opts.mej*1000,opts.vej*100,np.log10(opts.Xlan),opts.iota)
    elif opts.doMasses:
        name = "%sM%.0fm%.0fi%.0f"%(opts.eos,opts.m1*100,opts.m2*100,opts.iota)
elif opts.model == "Ka2017x2":
    if opts.doEjecta:
        name = "Ka2017x2_M%03dV%02dX%d_M%03dV%02dX%d"%(opts.mej1*1000,opts.vej1*100,np.log10(opts.Xlan1),opts.mej2*1000,opts.vej2*100,np.log10(opts.Xlan2))
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "Ka2017x2inc":
    if opts.doEjecta:
        name = "Ka2017x2inc_M%03dV%02dX%d_M%03dV%02dX%d_i%d"%(opts.mej1*1000,opts.vej1*100,np.log10(opts.Xlan1),opts.mej2*1000,opts.vej2*100,np.log10(opts.Xlan2),opts.iota)
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "RoFe2017":
    if opts.doEjecta:
        name = "FoFe2017_%sM%03dV%02dX%d"%(opts.eos,opts.mej*1000,opts.vej*100,np.log10(opts.Ye))
    elif opts.doMasses:
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
elif opts.model == "Bu2019":
    if opts.doEjecta:
        name = "Bu2019_%sM%03dT%d"%(opts.eos,opts.mej*1000,np.log10(opts.T))
elif opts.model == "Bu2019inc":
    if opts.doEjecta:
        name = "Bu2019_%sM%03dP%d"%(opts.eos,opts.mej*1000,opts.phi)
elif opts.model in "Bu2019lf":
    if opts.doEjecta:
        name = "Bu2019lf_Mdyn%03dMwind%03dP%d"%(opts.mej_dyn*1000,opts.mej_wind*1000,opts.phi)
elif opts.model in "Bu2019lr":
    if opts.doEjecta:
        name = "Bu2019lr_Mdyn%03dMwind%03dP%d"%(opts.mej_dyn*1000,opts.mej_wind*1000,opts.phi)
elif opts.model in "Bu2019lm":
    if opts.doEjecta:
        name = "Bu2019lm_Mdyn%03dMwind%03dP%d"%(opts.mej_dyn*1000,opts.mej_wind*1000,opts.phi)
elif opts.model in "Bu2019lw":
    if opts.doEjecta:
        name = "Bu2019lw_Mwind%03dP%d"%(opts.mej_wind*1000,opts.phi)
elif opts.model in "Bu2019re":
    if opts.doEjecta:
        name = "Bu2019re_M%03dP%d"%(opts.mej*1000,opts.phi)
elif opts.model in "Bu2019op":
    if opts.doEjecta:
        name = "Bu2019bc_kappaLF%03dgammaLF%.1fkappaLR%03dgammaLR%.1f"%(opts.kappaLF,opts.gammaLF,opts.kappaLR,opts.gammaLR)
elif opts.model in "Bu2019ops":
    if opts.doEjecta:
        name = "Bu2019bc_kappaLF%03dkappaLR%03d"%(opts.kappaLF,opts.kappaLR)
elif opts.model in ["Bu2019rp", "Bu2019rps","Bu2019rpd"]:
    if opts.doEjecta:
        name = "Bu2019lm_M1%03dM2%03dP%d"%(opts.mej1*1000,opts.mej2*1000,opts.phi)
elif opts.model in "Wo2020dyn":
    if opts.doEjecta:
        name = "Wo2020dyn_M%03dV%02dP%d"%(opts.mej*1000,opts.vej*100,opts.phi)
elif opts.model in "Wo2020dw":
    if opts.doEjecta:
        name = "Wo2020dw_M%03dV%02dP%d"%(opts.mej*1000,opts.vej*100,opts.phi)
elif opts.model in "Bu2019nsbh":
    if opts.doEjecta:
        name = "Bu2019nsbh_M%03dV%02dP%d"%(opts.mej*1000,opts.vej*100,opts.phi)
elif opts.model == "SN":
    t0 = (tini+tmax)/2.0
    #t0 = 0.0
    t, lbol, mag = SALT2.lightcurve(tini,tmax,dt,opts.redshift,t0,opts.x0,opts.x1,opts.c)
    t = t - t[0]
    name = "z%.0fx0%.0fx1%.0fc%.0f"%(opts.redshift*100.0,opts.x0*10000.0,opts.x1*10000.0,opts.c*10000.0)

elif opts.model == "Afterglow":
    tini = 0.0
    tmax = 10.0
    dt = 0.1
    p = 2.5
    epsilon_B = 1e-2
    epsilon_E = 1e-1
    ksi_N = 1

    t, lbol, mag = BOXFit.lightcurve(boxfitDir,tini,tmax,dt,theta_0,E,n,theta_obs,p,epsilon_B,epsilon_E,ksi_N)

    name = "theta0%.0fE0%.0en%.0fthetaobs%.0f"%(theta_0*100,E,n*10,theta_obs*100)

else:
   print("Model must be either: DiUj2017,KaKy2016,Me2017,SmCh2017,WoKo2017,BaKa2016, Ka2017, Ka2017x2, SN, Afterglow,")
   exit(0)

if opts.doAB:
    if np.sum(lbol) == 0.0:
        print("No luminosity...")
        exit(0)
elif opts.doSpec:
    if np.sum(spec) == 0.0:
        print("No spectra...")
        exit(0)

baseoutputDir = opts.outputDir
if not os.path.isdir(baseoutputDir):
    os.makedirs(baseoutputDir)
outputDir = os.path.join(baseoutputDir,opts.model)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.makedirs(baseplotDir)
plotDir = os.path.join(baseplotDir,opts.model)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

if opts.doAB:
    filename = "%s/%s.dat"%(outputDir,name)
    fid = open(filename,'w')
    #fid.write('# t[days] g-band r-band  i-band z-band\n')
    fid.write('# t[days] u g r i z y J H K\n')
    for ii in range(len(t)):
        fid.write("%.2f "%t[ii])
        for jj in np.arange(0,9):
            fid.write("%.3f "%mag[jj][ii])
        fid.write("\n")
    fid.close()
    
    filts = ["u","g","r","i","z","y","J","H","K"]
    colors=cm.rainbow(np.linspace(0,1,len(filts)))
    magidxs = [0,1,2,3,4,5,6,7,8]

    plotName = "%s/%s.pdf"%(plotDir,name)
    plt.figure(figsize=(10,12))
    for filt, color, magidx in zip(filts,colors,magidxs):
        plt.plot(t,mag[magidx,:],alpha=1.0,c=color,label=filt)
    plt.xlabel('Time [days]')
    plt.ylabel('Absolute AB Magnitude')
    if opts.model in ["Ka2017inc","Ka2017x2inc"]:
        plt.xlim([0,7])
    if not opts.model in ["SN"]:
        plt.ylim([-20,10])
    if opts.model in ["Ka2017inc","Ka2017x2inc"]:
        plt.title('Inclination: %.1f' % opts.iota) 
    plt.legend(loc="lower center",ncol=5)
    plt.gca().invert_yaxis()
    plt.savefig(plotName, bbox_inches='tight')
    plotNamePNG = "%s/%s.png"%(plotDir,name)
    plt.savefig(plotNamePNG)
    plt.close()   

    color1 = 'coral'
    color2 = 'cornflowerblue'
   
    if opts.doComparison: 
        mag_d_comparison = lightcurve_utils.read_files([opts.comparisonFile])
        key = list(mag_d_comparison[0].keys())[0]
        mag_d_comparison = mag_d_comparison[0][key]

    tini, tmax, dt = 0.0, 21.0, 0.1
    tt = np.arange(tini,tmax,dt)
    zp_best1, zp_best2 = 0, 0
    errorbudget = 1.0

    plotName = "%s/%s_panels.pdf"%(plotDir,name)
    plotNamePNG = "%s/%s_panels.png"%(plotDir,name)
    plt.figure(figsize=(20,28))
   
    cnt = 0
    for filt, color, magidx in zip(filts,colors,magidxs):
        cnt = cnt+1
        vals = "%d%d%d"%(len(filts),1,cnt)
        if cnt == 1:
            ax1 = plt.subplot(eval(vals))
        else:
            ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
    
        magave1 = mag[magidx,:]
 
        if opts.doComparison:
            magave2 = mag_d_comparison[filt]

        ii = np.where(~np.isnan(magave1))[0]
        if len(ii) > 1:
            f = interp.interp1d(t[ii], magave1[ii], fill_value='extrapolate')
            maginterp1 = f(tt)
            plt.plot(tt,maginterp1+zp_best1,'--',c=color1,linewidth=2,label='1 Component')
            plt.plot(tt,maginterp1+zp_best1-errorbudget,'-',c=color1,linewidth=2)
            plt.plot(tt,maginterp1+zp_best1+errorbudget,'-',c=color1,linewidth=2)
            plt.fill_between(tt,maginterp1+zp_best1-errorbudget,maginterp1+zp_best1+errorbudget,facecolor=color1,alpha=0.2)
   
        if opts.doComparison: 
            ii = np.where(~np.isnan(magave2))[0]
            f = interp.interp1d(mag_d_comparison["t"][ii], magave2[ii], fill_value='extrapolate')
            maginterp2 = f(tt)
            plt.plot(tt,maginterp2+zp_best2,'--',c=color2,linewidth=2,label='2 Component')
            plt.plot(tt,maginterp2+zp_best2-errorbudget,'-',c=color2,linewidth=2)
            plt.plot(tt,maginterp2+zp_best2+errorbudget,'-',c=color2,linewidth=2)
            plt.fill_between(tt,maginterp2+zp_best2-errorbudget,maginterp2+zp_best2+errorbudget,facecolor=color2,alpha=0.2)
    
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

    filename = "%s/%s_Lbol.dat"%(outputDir,name)
    fid = open(filename,'w')
    fid.write('# t[days] Lbol[erg/s]\n')
    for ii in range(len(t)):
        fid.write("%.5f %.5e\n"%(t[ii],lbol[ii]))
    fid.close()
    
    Lbol_ds = np.loadtxt(filename)
    t = Lbol_ds[:,0]
    Lbol = Lbol_ds[:,1]
    
    plotName = "%s/%s_Lbol.pdf"%(plotDir,name)
    plt.figure(figsize=(12,8))
    plt.semilogy(t,Lbol,'k--')
    plt.xlabel('Time [days]')
    plt.ylabel('Bolometric Luminosity [erg/s]')
    plt.savefig(plotName)
    plt.close()

elif opts.doSpec:

    filename = "%s/%s_spec.dat"%(outputDir,name)
    fid = open(filename,'w')
    fid.write("nan")
    for jj in range(len(lambdas)):
        fid.write(" %.3f"%lambdas[jj])
    fid.write("\n")
    for ii in range(len(t)):
        fid.write("%.5f "%t[ii])
        for jj in range(len(lambdas)):
            fid.write("%.5e "%spec[jj][ii])
        fid.write("\n")
    fid.close()

    data_out = np.loadtxt(filename)
    t_d, lambda_d, spec_d = data_out[1:,0], data_out[0,1:], data_out[1:,1:]
    vmin, vmax = np.nanmin(np.log10(spec_d)), np.nanmax(np.log10(spec_d))
    vmin = vmax - 4.0
    spec_d_log10 = np.log10(spec_d)
    spec_d_log10[~np.isfinite(spec_d_log10)] = -100.0

    TGRID,LAMBDAGRID = np.meshgrid(t_d,lambda_d)
    plotName = "%s/%s_spec.pdf"%(plotDir,name)
    plt.figure(figsize=(12,10))
    plt.pcolormesh(TGRID,LAMBDAGRID,spec_d_log10.T,vmin=vmin,vmax=vmax)
    plt.xlabel('Time [days]')
    plt.ylabel(r'$\lambda [\AA]$')
    plt.colorbar()
    plt.savefig(plotName)
    plt.close()

    color1 = 'coral'
    color2 = 'cornflowerblue'

    tt = np.arange(0.5,10.5,1.0)
    colors=cm.rainbow(np.linspace(0,1,len(tt)))
    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=np.max(tt)))
    sm._A = []

    data_out = np.loadtxt(opts.comparisonFile)
    t_d_comparison, lambda_d_comparison, spec_d_comparison = \
        data_out[1:,0], data_out[0,1:], data_out[1:,1:]
    spec_d_comparison_log10 = np.log10(spec_d_comparison)
    spec_d_comparison_log10[~np.isfinite(spec_d_comparison_log10)] = -100.0

    plotName = "%s/%s_spec_panels.pdf"%(plotDir,name)
    plotNamePNG = "%s/%s_spec_panels.png"%(plotDir,name)
    fig = plt.figure(figsize=(22,28))

    cnt = 0
    for ii in range(len(tt)):
        cnt = cnt+1
        vals = "%d%d%d"%(len(tt),1,cnt)
        if cnt == 1:
            #ax1 = plt.subplot(eval(vals))
            ax1 = plt.subplot(len(tt),1,cnt)
        else:
            #ax2 = plt.subplot(eval(vals),sharex=ax1,sharey=ax1)
            ax2 = plt.subplot(len(tt),1,cnt,sharex=ax1,sharey=ax1)

        idx = np.argmin(np.abs(t_d-tt[ii]))
        plt.plot(lambda_d,spec_d_log10[idx,:],'-',c=color1)
        idx = np.argmin(np.abs(t_d_comparison-tt[ii]))
        plt.plot(lambda_d_comparison,spec_d_comparison_log10[idx,:],'-',c=color2)

        plt.fill_between([13500.0,14500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)
        plt.fill_between([18000.0,19500.0],[-100.0,-100.0],[100.0,100.0],facecolor='0.5',edgecolor='0.5',alpha=0.2,linewidth=3)

        plt.ylabel('%.1f'%float(tt[ii]),fontsize=48,rotation=0,labelpad=40)
        plt.xlim([3500, 25000])
        plt.ylim([33.0,38.0])
        plt.grid()

        if (not cnt == len(tt)) and (not cnt == 1):
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

