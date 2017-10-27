
import os, sys
import optparse
import numpy as np
import h5py
import bisect
from scipy.interpolate import interpolate as interp
import statsmodels.api as sm

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-m","--model",default="barnes_kilonova_spectra") 
    parser.add_option("-n","--name",default="rpft_m001_v1")
    parser.add_option("--doAB",  action="store_true", default=False)
    parser.add_option("--doSpec",  action="store_true", default=False)    

    opts, args = parser.parse_args()

    return opts

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

def getMagAbsAB(filename_AB,filename_bol,filtname,model):

    u = np.loadtxt(filename_AB)
    t = u[:,0]

    if filename_bol:
        u_bol = np.loadtxt(filename_bol)
        L = u_bol[:,1]
    else:
        L = u[:,1]

    if model == "tanaka_compactmergers": 
        if filtname == "y":
            wavelengths = [3543, 4775.6, 6129.5, 7484.6, 8657.8, 12350, 16620, 21590]
            wavelength_interp = 9603.1

            mag = np.zeros(t.shape)
            for ii in xrange(len(t)):
                mag[ii] = np.interp(wavelength_interp,wavelengths,u[ii,1:])
        else:
            cols = ["t","u","g","r","i","z","J","H","K"]
            idx = cols.index(filtname)
            mag = u[:,idx]
    elif model == "korobkin_kilonova":
        if filtname == "u":
            wavelengths = [4775.6, 6129.5, 7484.6, 8657.8, 9603.1, 12350, 16620, 21590]
            wavelength_interp = 3543

            mag = np.zeros(t.shape)
            for ii in xrange(len(t)):
                mag[ii] = np.interp(wavelength_interp,wavelengths,u[ii,2:])
        else:
            cols = ["t","g","r","i","z","y","J","H","K"]
            idx = cols.index(filtname)+1
            mag = u[:,idx]

    return t, mag, L

def getMagLbol(filename,band,model):

    if model == "ns_precursor_Lbol":
        w = 1e10/(np.array([1.2e15, 7.5e14, 4.46e14,3.7e14])/3e8)

    u = lines = [line.rstrip('\n') for line in open(filename)]
    # In AB mags at 200 Mpc
    D2 = 200*3.0857e16*1e6
    # Want at 10pc
    D1 = 10*3.0857e16
    #u[:,1:] = u[:,1:]-(5*np.log10(200*1e6)-5)
    # AB mags at 10pc
    #u[:,0] /= (24*3600) # time in days

    t_d = []
    mag_d = []
    L_d = []

    idx = np.arange(0,len(u),2)

    for ii in idx:
        i = u[ii].split(" ")
        i = filter(None,i)
        i = [float(x) for x in i]
        i = np.array(i)
        L = i[1:]-(5*np.log10(200*1e6)-5)
        spec = np.array(zip(w,L))
        effwave = np.sum(band[:,0]*band[:,1])/np.sum(band[:,1])
        mag = np.interp(effwave,w,L)
        #Lbol = np.trapz(spec[:,1],x=spec[:,0])
        Lbol = float(u[ii+1])

        if not np.isfinite(mag):
            mag = np.nan

        t=i[0]/(24*3600) # time in days
        t_d.append(t)
        mag_d.append(mag)
        L_d.append(Lbol)
    t_d = np.array(t_d)
    mag_d = np.array(mag_d)
    L_d = np.array(L_d)

    return t_d, mag_d, L_d

def getMagAB(filename,band,model):

    if model == "ns_precursor_AB":
        w = 1e10/(np.array([1.2e15, 7.5e14, 4.46e14,3.7e14])/3e8)

    u = np.loadtxt(filename)
    # In AB mags at 200 Mpc
    D2 = 200*3.0857e16*1e6
    # Want at 10pc
    D1 = 10*3.0857e16
    u[:,1:] = u[:,1:]-(5*np.log10(200*1e6)-5)
    # AB mags at 10pc
    u[:,0] /= (24*3600) # time in days

    t = u[:,0]

    t_d = []
    mag_d = []
    L_d = []

    for i in u:
        L = i[1:]
        spec = np.array(zip(w,L))
        effwave = np.sum(band[:,0]*band[:,1])/np.sum(band[:,1])
        mag = np.interp(effwave,w,L)
        Lbol = np.trapz(spec[:,1],x=spec[:,0])

        if not np.isfinite(mag):
            mag = np.nan
       
        t=i[0]
        t_d.append(t)
        mag_d.append(mag)
        L_d.append(Lbol)
    t_d = np.array(t_d)
    mag_d = np.array(mag_d)
    L_d = np.array(L_d)

    return t_d, mag_d, L_d

def getMagSpecH5(filename,band,model):
    fin    = h5py.File(filename,'r')
    # frequency in Hz
    nu    = np.array(fin['nu'],dtype='d')
    # array of time in seconds
    times = np.array(fin['time'])
    # covert time to days
    times = times/3600.0/24.0

    # specific luminosity (ergs/s/Hz) 
    # this is a 2D array, Lnu[times][nu]
    Lnu_all   = np.array(fin['Lnu'],dtype='d')

    S = 0.1089/band[:,0]**2

    S1 = S*band[:,1]

    ZP = np.trapz(S1,x=band[:,0])

    t_d = []
    mag_d = []
    L_d = []
    
    for t in times[:-1]:
        if (t < 0) or (t > 7): continue

        # index corresponding to t
        it = bisect.bisect(times,t)
        # spectrum at this epoch
        Lnu = np.flipud(Lnu_all[it,:])

        # if you want thing in Flambda (ergs/s/Angstrom)
        c    = 2.99e10
        lam  = np.flipud(c/nu*1e8)
        Llam = Lnu*np.flipud(nu)**2.0/c/1e8

        D_cm = 10*3.0857e16*100 # 10 pc in cm

        spec = np.array(zip(lam,Llam/(4*np.pi*D_cm**2)))
        spec1 = easyint(spec[:,0],spec[:,1],band[:,0])

        conv = spec1*band[:,1]
        flux = np.trapz(conv,x=band[:,0])
        mag = -2.5*np.log10(flux/ZP)
        Lbol = np.trapz(Llam,x=lam)

        if not np.isfinite(mag):
            mag = np.nan
        w=[]
        L=[]

        t_d.append(t)
        mag_d.append(mag)
        L_d.append(Lbol)

    t_d = np.array(t_d)
    mag_d = np.array(mag_d)
    L_d = np.array(L_d)

    ii = np.where(np.isfinite(np.log10(L_d)))[0]
    f = interp.interp1d(t_d[ii], np.log10(L_d[ii]), fill_value='extrapolate')
    L_d = 10**f(t_d)

    ii = np.where(~np.isnan(mag_d))[0]
    f = interp.interp1d(t_d[ii], mag_d[ii], fill_value='extrapolate')
    mag_d = f(t_d)

    mag_d_lowess = sm.nonparametric.lowess(mag_d, t_d, frac=0.1)
    L_d_lowess = sm.nonparametric.lowess(np.log10(L_d), t_d, frac=0.1)

    mag_d = mag_d_lowess[:,1]
    L_d = 10**L_d_lowess[:,1]

    return t_d, mag_d, L_d

def getMagSpec(filename,band,model):
    #u = np.genfromtxt(opts.name)
    u = np.loadtxt(filename,skiprows=1)
    if model == "kilonova_wind_spectra":
        u = u[u[:,2]==0.05] 

    D_cm = 10*3.0857e16*100 # 10 pc in cm

    if model == "kilonova_wind_spectra":
        u[:,3] /= (4*np.pi*D_cm**2) # F_lam (erg/s/cm2/A at 10pc)
        u[:,0] /= (24*3600) # time in days
    elif model == "macronovae-rosswog":
        u[:,2] /= 1.0
    else:
        u[:,2] /= (4*np.pi*D_cm**2) # F_lam (erg/s/cm2/A at 10pc)
        u[:,0] /= (24*3600) # time in days

    t = u[0,0]

    w=[]
    L=[]

    S = 0.1089/band[:,0]**2

    S1 = S*band[:,1]

    ZP = np.trapz(S1,x=band[:,0])

    t_d = []
    mag_d = []
    L_d = []

    for i in u:
        if i[0]==t:
            if model == "kilonova_wind_spectra":
                w.append(i[1])
                L.append(i[3])
            else:
                w.append(i[1])
                L.append(i[2])
        else:
            w = np.array(w)
            L = np.array(L)
            spec = np.array(zip(w,L))
            spec1 = easyint(spec[:,0],spec[:,1],band[:,0])
            conv = spec1*band[:,1]
            flux = np.trapz(conv,x=band[:,0])
            mag = -2.5*np.log10(flux/ZP)
            Lbol = np.trapz(spec[:,1]*(4*np.pi*D_cm**2),x=spec[:,0])

            if not np.isfinite(mag):
                mag = np.nan
            w=[]
            L=[]
            t=i[0]

            t_d.append(t)
            mag_d.append(mag)
            L_d.append(Lbol)
    t_d = np.array(t_d)
    mag_d = np.array(mag_d)
    L_d = np.array(L_d)

    return t_d, mag_d, L_d

def getSpecH5(filename,model):

    fin    = h5py.File(filename,'r')
    # frequency in Hz
    nu    = np.array(fin['nu'],dtype='d')
    # array of time in seconds
    times = np.array(fin['time'])
    # covert time to days
    times = times/3600.0/24.0

    # specific luminosity (ergs/s/Hz) 
    # this is a 2D array, Lnu[times][nu]
    Lnu_all   = np.array(fin['Lnu'],dtype='d')

    t_d = []
    lambda_d = []
    spec_d = []

    for t in times[:-1]:
        if (t < 0) or (t > 7): continue

        # index corresponding to t
        it = bisect.bisect(times,t)
        # spectrum at this epoch
        Lnu = Lnu_all[it,:]

        # if you want thing in Flambda (ergs/s/Angstrom)
        c    = 2.99e10
        lam  = c/nu*1e8
        Llam = Lnu*nu**2.0/c/1e8

        D_cm = 10*3.0857e16*100 # 10 pc in cm

        Llam = Llam / (4*np.pi*D_cm**2) # F_lam (erg/s/cm2/A at 10pc)

        t_d.append(t)
        lambda_d = lam
        spec_d.append(Llam)

    t_d = np.array(t_d)
    lambda_d = np.array(lambda_d)
    spec_d = np.array(spec_d)

    return t_d, lambda_d, spec_d

def getSpec(filename,model):
    #u = np.genfromtxt(opts.name)
    u = np.loadtxt(filename,skiprows=1)
    if model == "kilonova_wind_spectra":
        u = u[u[:,2]==0.05]

    D_cm = 10*3.0857e16*100 # 10 pc in cm

    if model == "kilonova_wind_spectra":
        u[:,3] /= (4*np.pi*D_cm**2) # F_lam (erg/s/cm2/A at 10pc)
        u[:,0] /= (24*3600) # time in days
    elif model == "macronovae-rosswog":
        u[:,2] /= 1.0
    else:
        u[:,2] /= (4*np.pi*D_cm**2) # F_lam (erg/s/cm2/A at 10pc)
        u[:,0] /= (24*3600) # time in days

    t = u[0,0]

    w=[]
    L=[]

    t_d = []
    lambda_d = []
    spec_d = []

    for i in u:
        if i[0]==t:
            if model == "kilonova_wind_spectra":
                w.append(i[1])
                L.append(i[3])
            else:
                w.append(i[1])
                L.append(i[2])
        else:
            w = np.array(w)
            L = np.array(L)

            t=i[0]
            t_d.append(t)
            lambda_d = w
            spec_d.append(L)

            w=[]
            L=[]

    t_d = np.array(t_d)
    lambda_d = np.array(lambda_d)
    spec_d = np.array(spec_d)

    return t_d, lambda_d, spec_d

# Parse command line
opts = parse_commandline()

baseoutputDir = opts.outputDir
outputDir = os.path.join(baseoutputDir,opts.model)
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

baseplotDir = opts.plotDir
plotDir = os.path.join(baseplotDir,opts.model)
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
dataDir = opts.dataDir

specmodels = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog"]
spech5models = ["kasen_kilonova_survey"]
ABmodels = ["ns_precursor_AB"]
Lbolmodels = ["ns_precursor_Lbol"]
absABmodels = ["tanaka_compactmergers","korobkin_kilonova"]

if opts.model == "kilonova_wind_spectra":
    filename = "%s/%s/%s.mod"%(dataDir,opts.model,opts.name)
elif opts.model == "macronovae-rosswog":
    filename = "%s/%s/%s.dat"%(dataDir,opts.model,opts.name)
elif opts.model == "korobkin_kilonova":
    filename_AB = "%s/%s/%s.dat"%(dataDir,opts.model,opts.name)
    filename_bol = []
elif opts.model == "kasen_kilonova_survey":
    filename = "%s/%s/%s.h5"%(dataDir,opts.model,opts.name)
elif opts.model in specmodels:
    filename = "%s/%s/%s.spec"%(dataDir,opts.model,opts.name)
elif opts.model in absABmodels:
    filename_AB = "%s/%s/%s_AB.txt"%(dataDir,opts.model,opts.name)
    filename_bol = "%s/%s/%s.txt"%(dataDir,opts.model,opts.name)
else:
    filename = "%s/%s/%s.dat"%(dataDir,opts.model,opts.name)

if opts.doAB:
    mag_ds = {}
    for ii in xrange(9):
        mag_ds[ii] = np.array([])
    
    #filts = np.genfromtxt('../input/PS1_filters.txt')
    #filtnames = ["g","r","i","z","y"]
    #g = filts[:,1], r=2, i=3, z=4, y=5
    filts = np.genfromtxt('../input/filters.dat')
    filtnames = ["u","g","r","i","z","y","J","H","K"]
    for ii in xrange(9):
        band = np.array(zip(filts[:,0]*10,filts[:,ii+1]))
        if opts.model in specmodels:
            t_d, mag_d, L_d = getMagSpec(filename,band,opts.model)
        elif opts.model in absABmodels:
            t_d, mag_d, L_d = getMagAbsAB(filename_AB,filename_bol,filtnames[ii],opts.model)
        elif opts.model == "kasen_kilonova_survey":
            t_d, mag_d, L_d = getMagSpecH5(filename,band,opts.model)
        elif opts.model in Lbolmodels:
            t_d, mag_d, L_d = getMagLbol(filename,band,opts.model)
        else:
            t_d, mag_d, L_d = getMagAB(filename,band,opts.model)
        mag_ds[ii] = mag_d

    if np.any(np.array(L_d) == 0.0):
        print "0's in bolometric luminosity.... quitting."
        exit(0)

    filename = "%s/%s.dat"%(outputDir,opts.name)
    fid = open(filename,'w')
    fid.write('# t[days] u g r i z y J H K\n')
    for ii in xrange(len(t_d)):
        fid.write("%.5f "%t_d[ii])
        for jj in xrange(9):
            fid.write("%.3f "%mag_ds[jj][ii])
        fid.write("\n")
    fid.close()
    
    mag_ds = np.loadtxt(filename)
    mag1 = mag_ds[:,1]
    indexes = np.where(~np.isnan(mag1))[0]
    index1 = indexes[0]
    index2 = indexes[-1]
    mag_ds = mag_ds[index1:index2,:]
    t = mag_ds[:,0]
    
    plotName = "%s/%s.pdf"%(plotDir,opts.name)
    plt.figure()
    plt.plot(t,mag_ds[:,2],'y',label='g-band')
    plt.plot(t,mag_ds[:,3],'g',label='r-band')
    plt.plot(t,mag_ds[:,4],'b',label='i-band')
    plt.plot(t,mag_ds[:,5],'c',label='z-band')
    plt.plot(t,mag_ds[:,6],'k',label='y-band')
    plt.xlabel('Time [days]')
    plt.ylabel('Absolute AB Magnitude')
    plt.legend(loc="best")
    plt.gca().invert_yaxis()
    plt.savefig(plotName)
    plt.close()
    
    filename = "%s/%s_Lbol.dat"%(outputDir,opts.name)
    fid = open(filename,'w')
    fid.write('# t[days] Lbol[erg/s]\n')
    for ii in xrange(len(t_d)):
        fid.write("%.5f %.5e\n"%(t_d[ii],L_d[ii]))
    fid.close()
    
    Lbol_ds = np.loadtxt(filename)
    t = Lbol_ds[:,0]
    Lbol = Lbol_ds[:,1]
    
    plotName = "%s/%s_Lbol.pdf"%(plotDir,opts.name)
    plt.figure()
    plt.semilogy(t,Lbol,'k--')
    plt.xlabel('Time [days]')
    plt.ylabel('Bolometric Luminosity [erg/s]')
    plt.savefig(plotName)
    plt.close()
elif opts.doSpec:

    if opts.model == "kasen_kilonova_survey":
        t_d, lambda_d, spec_d = getSpecH5(filename,opts.model)
    else:
        t_d, lambda_d, spec_d = getSpec(filename,opts.model)

    #if np.any(np.array(L_d) == 0.0):
    #    print "0's in bolometric luminosity.... quitting."
    #    exit(0)

    filename = "%s/%s_spec.dat"%(outputDir,opts.name)
    fid = open(filename,'w')
    fid.write("nan")
    for jj in xrange(len(lambda_d)):
        fid.write(" %.3f"%lambda_d[jj])
    fid.write("\n")
    for ii in xrange(len(t_d)):
        fid.write("%.5f "%t_d[ii])
        for jj in xrange(len(lambda_d)):
            fid.write("%.5e "%spec_d[ii][jj])
        fid.write("\n")
    fid.close()

    data_out = np.loadtxt(filename) 
    t_d, lambda_d, spec_d = data_out[1:,0], data_out[0,1:], data_out[1:,1:]

    TGRID,LAMBDAGRID = np.meshgrid(t_d,lambda_d)
    plotName = "%s/%s_spec.pdf"%(plotDir,opts.name)
    plt.figure(figsize=(12,10))
    plt.pcolormesh(TGRID,LAMBDAGRID,spec_d.T,vmin=np.min(spec_d),vmax=np.max(spec_d))
    plt.xlabel('Time [days]')
    plt.ylabel(r'$\lambda [\AA]$')
    plt.savefig(plotName)
    plt.close()

