
import os, sys
import numpy as np
import optparse
 
import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt

import BNSKilonovaLightcurve 
import BHNSKilonovaLightcurve

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="output")
    parser.add_option("-p","--plotDir",default="plots") 
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("-e","--eos",default="H4")
    parser.add_option("-q","--massratio",default=5.0,type=float)
    parser.add_option("-a","--chi",default=0.9,type=float) 
    parser.add_option("-b","--mej",default=0.005,type=float)
    parser.add_option("-c","--vej",default=0.2,type=float)
    parser.add_option("-d","--m1",default=1.35,type=float)
    parser.add_option("-f","--m2",default=1.35,type=float)
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
th = 0.2
ph = 3.14
kappa = 10.0
eps = 1.58*(10**10)
alp = 1.2
eth = 0.5
flgbct = 0

if opts.model == "BHNS":
    if opts.doEjecta:
        t, lbol, mag = BHNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth)
        name = "BHNS_%sM%04dV%02d"%(opts.eos,opts.mej*1000,opts.vej*10)
    elif opts.doMasses:
        t, lbol, mag = BHNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,q,chi,i,c,mb,mns)
        name = "%sQ%.0fa%.0f"%(opts.eos,opts.massratio,opts.chi*100)
    else:
        print "Enable --doEjecta or --doMasses"
        exit(0)
elif opts.model == "BNS":
    if opts.doEjecta:
        t, lbol, mag = BNSKilonovaLightcurve.calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth,flgbct)
        name = "BNS_%sM%04dV%02d"%(opts.eos,opts.mej*1000,opts.vej*10)
    elif opts.doMasses:
        t, lbol, mag = BNSKilonovaLightcurve.lightcurve(tini,tmax,dt,vmin,th,ph,kappa,eps,alp,eth,m1,mb,c,m2,mb,c,flgbct)
        name = "%sM%.0fm%.0f"%(opts.eos,opts.m1*100,opts.m2*100)
    else:
        print "Enable --doEjecta or --doMasses"
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
