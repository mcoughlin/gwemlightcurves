
import os, sys
import numpy as np
import optparse
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

from gwemlightcurves import BNSKilonovaLightcurve, BHNSKilonovaLightcurve, SALT2
from gwemlightcurves import BHNSKilonovaLightcurveOpt

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots") 
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("-e","--eos",default="H4")
    parser.add_option("-q","--massratio",default=5.0,type=float)
    parser.add_option("-a","--chi",default=0.9,type=float) 
    parser.add_option("--mej",default=0.005,type=float)
    parser.add_option("--vej",default=0.2,type=float)
    parser.add_option("--m1",default=1.35,type=float)
    parser.add_option("--m2",default=1.35,type=float)
    parser.add_option("-z","--redshift",default=0.5,type=float)
    parser.add_option("--x0",default=1.0,type=float)
    parser.add_option("--x1",default=1.0,type=float)
    parser.add_option("-c","--c",default=1.0,type=float)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)

    opts, args = parser.parse_args()
 
    return opts

# Parse command line
opts = parse_commandline()

m1 = opts.m1
m2 = opts.m2
q = opts.massratio
chi = opts.chi
mej = opts.mej
vej = opts.vej
i = 60.0

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
tmax = 14.0
dt = 0.1

vave = 0.267
vmin = 0.02
th = 1.0
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.0
eth = 0.5
flgbct = 0

if opts.model == "BHNS":
    if opts.doEjecta:
        t, lbol, mag = BHNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth)
        t1, lbol1, mag1 = BHNSKilonovaLightcurveOpt.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth)
        print np.nansum(mag1[0]-mag[0])
        name = "BHNS_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*10)
    elif opts.doMasses:
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi,i,c,mb,mns)
        name = "%sQ%.0fa%.0f"%(opts.eos,opts.massratio,opts.chi*100)
    else:
        print "Enable --doEjecta or --doMasses"
        exit(0)
elif opts.model == "BNS":
    if opts.doEjecta:
        t, lbol, mag = BNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth,flgbct)
        name = "BNS_%sM%03dV%02d"%(opts.eos,opts.mej*1000,opts.vej*10)
    elif opts.doMasses:
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb,c,m2,mb,c,flgbct)
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
    else:
        print "Enable --doEjecta or --doMasses"
        exit(0)
elif opts.model == "SN":
    t0 = (tini+tmax)/2.0
    t0 = 0.0
    t, lbol, mag = SALT2.lightcurve(tini,tmax,dt,opts.redshift,t0,opts.x0,opts.x1,opts.c)
    name = "z%.0fx0%.0fx1%.0fc%.0f"%(opts.redshift*100,opts.x0*10000,opts.x1*10000,opts.c*10000)
else:
   print "Model must be either: BHNS, BNS, SN"
   exit(0)

if np.sum(lbol) == 0.0:
    print "No luminosity..."
    exit(0)

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

#filename = 'bhns/%s.txt'%name
#fid = open(filename,'w')
#fid.write('# t[days]   Lbol[erg/s]  u-band  g-band  r-band i-band  z-band  J-band  H-band  K-band\n')
#for ii in xrange(len(t)):
#    fid.write("%.2f "%t[ii])
#    fid.write("%.3e "%lbol[ii])
#    for jj in xrange(8):
#        fid.write("%.3f "%mag[jj][ii])
#    fid.write("\n")
#fid.close()

filename = "%s/%s.dat"%(outputDir,name)
fid = open(filename,'w')
fid.write('# t[days] g-band r-band  i-band z-band\n')
for ii in xrange(len(t)):
    fid.write("%.2f "%t[ii])
    for jj in np.arange(1,5):
        fid.write("%.3f "%mag[jj][ii])
    fid.write("\n")
fid.close()

plotName = "%s/%s.pdf"%(plotDir,name)
plt.figure()
plt.plot(t,mag[0],'r',label='u-band')
plt.plot(t,mag[1],'y',label='g-band')
plt.plot(t,mag[2],'g',label='r-band')
plt.plot(t,mag[3],'b',label='i-band')
plt.plot(t,mag[4],'c',label='z-band')
plt.xlabel('Time [days]')
plt.ylabel('Absolute AB Magnitude')
plt.legend(loc="best")
plt.gca().invert_yaxis()
plt.savefig(plotName)
plt.close()

filename = "%s/%s_Lbol.dat"%(outputDir,name)
fid = open(filename,'w')
fid.write('# t[days] Lbol[erg/s]\n')
for ii in xrange(len(t)):
    fid.write("%.5f %.5e\n"%(t[ii],lbol[ii]))
fid.close()

Lbol_ds = np.loadtxt(filename)
t = Lbol_ds[:,0]
Lbol = Lbol_ds[:,1]

plotName = "%s/%s_Lbol.pdf"%(plotDir,name)
plt.figure()
plt.semilogy(t,Lbol,'k--')
plt.xlabel('Time [days]')
plt.ylabel('Bolometric Luminosity [erg/s]')
plt.savefig(plotName)
plt.close()
