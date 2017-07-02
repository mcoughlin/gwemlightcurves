
import os, sys
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp

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
    parser.add_option("-m","--model",default="barnes_kilonova_spec") 
    parser.add_option("-n","--name",default="rpft_m001_v1")

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
    cols = ["t","u","g","r","i","z","y","H","K"]
    idx = cols.index(filtname)

    t = u[:,0]
    mag = u[:,idx]
        
    u = np.loadtxt(filename_bol)
    L = u[:,1]

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

# Parse command line
opts = parse_commandline()

baseoutputDir = opts.outputDir
if not os.path.isdir(baseoutputDir):
    os.mkdir(baseoutputDir)
outputDir = os.path.join(baseoutputDir,opts.model)
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

baseplotDir = opts.plotDir
if not os.path.isdir(baseplotDir):
    os.mkdir(baseplotDir)
plotDir = os.path.join(baseplotDir,opts.model)
if not os.path.isdir(plotDir):
    os.mkdir(plotDir)
dataDir = opts.dataDir

specmodels = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog"]
ABmodels = ["ns_precursor_AB"]
Lbolmodels = ["ns_precursor_Lbol"]
absABmodels = ["tanaka_compactmergers"]

if opts.model == "kilonova_wind_spectra":
    filename = "%s/%s/%s.mod"%(dataDir,opts.model,opts.name)
elif opts.model == "macronovae-rosswog":
    filename = "%s/%s/%s.dat"%(dataDir,opts.model,opts.name)
elif opts.model in specmodels:
    filename = "%s/%s/%s.spec"%(dataDir,opts.model,opts.name)
elif opts.model in absABmodels:
    filename_AB = "%s/%s/%s_AB.txt"%(dataDir,opts.model,opts.name)
    filename_bol = "%s/%s/%s.txt"%(dataDir,opts.model,opts.name)
else:
    filename = "%s/%s/%s.dat"%(dataDir,opts.model,opts.name)

mag_ds = {}
for ii in xrange(5):
    mag_ds[ii] = np.array([])

filts = np.genfromtxt('../input/PS1_filters.txt')
filtnames = ["g","r","i","z","y"]
#g = filts[:,1], r=2, i=3, z=4, y=5
for ii in xrange(5):
    band = np.array(zip(filts[:,0]*10,filts[:,ii+1]))
    if opts.model in specmodels:
        t_d, mag_d, L_d = getMagSpec(filename,band,opts.model)
    elif opts.model in absABmodels:
        t_d, mag_d, L_d = getMagAbsAB(filename_AB,filename_bol,filtnames[ii],opts.model)
    elif opts.model in Lbolmodels:
        t_d, mag_d, L_d = getMagLbol(filename,band,opts.model)
    else:
        t_d, mag_d, L_d = getMagAB(filename,band,opts.model)
    mag_ds[ii] = mag_d

filename = "%s/%s.dat"%(outputDir,opts.name)
fid = open(filename,'w')
fid.write('# t[days] g-band  r-band i-band  z-band  w-band\n')
for ii in xrange(len(t_d)):
    fid.write("%.5f "%t_d[ii])
    for jj in xrange(5):
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
plt.plot(t,mag_ds[:,1],'y',label='g-band')
plt.plot(t,mag_ds[:,2],'g',label='r-band')
plt.plot(t,mag_ds[:,3],'b',label='i-band')
plt.plot(t,mag_ds[:,4],'c',label='z-band')
plt.plot(t,mag_ds[:,5],'k',label='y-band')
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
