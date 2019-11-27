
import numpy as np
import os, sys, glob
import optparse

condorDir = '../condor'
logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

errorbudgets = [0.01,0.10,1.00]
errorbudgets = [0.10, 0.25, 1.00]

filename = "../lightcurves/GRB.dat"
lines = [line.rstrip('\n') for line in open(filename)]
#lines.append("GW170817 u,g,r,i,z,y,J,H,K 57982.5285236896 40")

grbs = ["GRB150101B","GRB050709","GRB130603B","GRB051221A","GW170817"]
grbs = ["GW170817", "GRB150101B","GRB050709","GRB060614"]
job_number = 0

fid = open(os.path.join(condorDir,'condor.dag'),'w')
for line in lines:
    lineSplit = line.split(" ")
    grb = lineSplit[0]
    filts = lineSplit[1]
    mjd = float(lineSplit[2])
    dist = float(lineSplit[3])

    if not grb in grbs: continue
    if np.isnan(dist): continue

    #if not grb == "GRB130603B": continue

    if grb in ["GW170817"]:
        tmin, tmax = 0.0, 14.0
    else:
        tmin, tmax = 0.0, 10.0

    for errorbudget in errorbudgets:

        #model = "TrPi2018"
        #checkfile = "../plots/gws/%s_FixZPT0/%s/%d_%d/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        #if not os.path.isfile(checkfile):
        #    fid.write('JOB %d condor.sub\n'%(job_number))
        #    fid.write('RETRY %d 3\n'%(job_number))
        #    fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
        #    fid.write('\n\n')
        #    job_number = job_number + 1

        model = "Ka2017"
        checkfile = "../plots/gws/%s/%s/%d_%d/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        if not os.path.isfile(checkfile):
            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
            fid.write('\n\n')
            job_number = job_number + 1

        model = "Ka2017_TrPi2018"
        checkfile = "../plots/gws/%s/%s/%d_%d/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        if not os.path.isfile(checkfile):
            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
            fid.write('\n\n')
            job_number = job_number + 1       

        #if not grb in grbs: continue
        #model = "Ka2017"
        #checkfile = "../plots/gws/%s_FixZPT0/%s/%d_%d/ejecta/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        #if not os.path.isfile(checkfile):
        #    fid.write('JOB %d condor.sub\n'%(job_number))
        #    fid.write('RETRY %d 3\n'%(job_number))
        #    fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
        #    fid.write('\n\n')
        #    job_number = job_number + 1
 
        model = "Bu2019inc"
        checkfile = "../plots/gws/%s/%s/%d_%d/ejecta/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        if not os.path.isfile(checkfile):
            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
            fid.write('\n\n')
            job_number = job_number + 1

        #model = "Ka2017_TrPi2018"
        #checkfile = "../plots/gws/%s_FixZPT0/%s/%d_%d/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        #if not os.path.isfile(checkfile):
        #    fid.write('JOB %d condor.sub\n'%(job_number))
        #    fid.write('RETRY %d 3\n'%(job_number))
        #    fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
        #    fid.write('\n\n')
        #    job_number = job_number + 1

        model = "Bu2019inc_TrPi2018"
        checkfile = "../plots/gws/%s/%s/%d_%d/%s/%.2f/data.pkl"%(model,filts.replace(",","_"),tmin,tmax,grb,errorbudget)
        if not os.path.isfile(checkfile):
            fid.write('JOB %d condor.sub\n'%(job_number))
            fid.write('RETRY %d 3\n'%(job_number))
            fid.write('VARS %d jobNumber="%d" model="%s" distance="%.5f" T0="%.5f" filters="%s" errorbudget="%.2f" grb="%s" tmin="%.1f" tmax="%.1f"\n'%(job_number,job_number,model,dist,mjd,filts,errorbudget,grb,tmin,tmax))
            fid.write('\n\n')
            job_number = job_number + 1

fid = open(os.path.join(condorDir,'condor.sub'),'w')
fid.write('executable = /home/mcoughlin/gwemlightcurves/bin/run_lightcurves_models.py\n')
fid.write('output = logs/out.$(jobNumber)\n');
fid.write('error = logs/err.$(jobNumber)\n');
#fid.write('arguments = --doEvent --model $(model) --name $(grb) --tmin $(tmin) --tmax $(tmax) --distance $(distance) --T0 $(T0) --filters $(filters) --errorbudget $(errorbudget) --doFixZPT0 --doEjecta\n')
fid.write('arguments = --doEvent --model $(model) --name $(grb) --tmin $(tmin) --tmax $(tmax) --distance $(distance) --T0 $(T0) --filters $(filters) --errorbudget $(errorbudget) --doEjecta\n')
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 4000\n');
fid.write('request_cpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /usr1/mcoughlin/gwemlightcurves_errorbar_redo.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()


