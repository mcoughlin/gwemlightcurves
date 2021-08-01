
import os, sys
import glob
import numpy as np

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_Lbol","tanaka_compactmergers","macronovae-rosswog","kasen_kilonova_survey","kasen_kilonova_grid"]
models = ["kasen_kilonova_grid"]
models = ["bulla_1D"]
models = ["bulla_2D"]
#models = ["bulla_1D","bulla_2D"]
models = ["bulla_2D"]
#models = ["bulla_2D","bulla_2Component_lfree","bulla_2Component_lrich"]
models = ["bulla_2Component_lfree","bulla_2Component_lrich"]
models = ["bulla_2Component_lmid"]
models = ["bulla_blue_cone","bulla_red_ellipse"]
models = ["bulla_opacity"]
models = ["bulla_2D"]
models = ["bulla_reprocess"]
models = ["bulla_2Component_lmid"]
models = ["bulla_2Comp_kappas"]

ntheta = 11
costhetas = np.linspace(0,1,ntheta)
thetas = np.rad2deg(np.arccos(costhetas))
#thetas = [0.0]
#thetas = thetas[2:4]
#thetas = np.append(thetas,90)

condorDir = '../condor'
logDir = os.path.join(condorDir,'logs')
if not os.path.isdir(logDir):
    os.makedirs(logDir)

data = {}
job_number = 0

fid = open(os.path.join(condorDir,'condor.dag'),'w')

for model in models:
    if model == "bulla_2Comp_kappas":
        files = glob.glob("../data/%s/*spec*"%model)
    else:
        files = glob.glob("../data/%s/*"%model)
    for file in files:
        name = file.split("/")[-1].replace(".mod","").replace(".spec","").replace("_AB","").replace(".h5","").replace(".dat","").replace(".txt","")

        #if not "phi30" in name:
        #    continue
        #if (not "mej0.090" in name) and (not "mej0.080" in name):
        #    continue

        #filename = "../output/%s/%s.dat"%(model,name)
        #if os.path.isfile(filename): continue

        #system_call = "python run_models.py --doAB --model %s --name %s"%(model,name)
        #print(system_call)
        #os.system(system_call)

        for theta in thetas:
            filename = "../output/%s/%s_%.1f.dat"%(model,name,theta)
            if not os.path.isfile(filename):

            #if True:
                fid.write('JOB %d condor.sub\n'%(job_number))
                fid.write('RETRY %d 3\n'%(job_number))
                fid.write('VARS %d jobNumber="%d" model="%s" name="%s" theta="%.1f"\n'%(job_number,job_number,model,name,theta))
                fid.write('\n\n')
                job_number = job_number + 1
            else:
                print(filename)

fid = open(os.path.join(condorDir,'condor.sub'),'w')
fid.write('executable = /home/mcoughlin/gwemlightcurves/bin/run_models.py\n')
fid.write('output = logs/out.$(jobNumber)\n');
fid.write('error = logs/err.$(jobNumber)\n');
#fid.write('arguments = --doEvent --model $(model) --name $(grb) --tmin $(tmin) --tmax $(tmax) --distance $(distance) --T0 $(T0) --filters $(filters) --errorbudget $(errorbudget) --doFixZPT0 --doEjecta\n')
fid.write('arguments = --doAB --model $(model) --name $(name) --theta $(theta)\n') 
fid.write('requirements = OpSys == "LINUX"\n');
fid.write('request_memory = 4000\n');
fid.write('request_cpus = 1\n');
fid.write('accounting_group = ligo.dev.o2.burst.allsky.stamp\n');
fid.write('notification = never\n');
fid.write('getenv = true\n');
fid.write('log = /usr1/mcoughlin/gwemlightcurves_fit.log\n')
fid.write('+MaxHours = 24\n');
fid.write('universe = vanilla\n');
fid.write('queue 1\n');
fid.close()

print(stop)

#print stop

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog","kasen_kilonova_survey","kasen_kilonova_grid"]
models = ["kasen_kilonova_grid"]

for model in models:
    files = glob.glob("../data/%s/*"%model)
    for file in files:
        name = file.split("/")[-1].replace(".mod","").replace(".spec","").replace("_AB","").replace(".h5","").replace(".dat","").replace(".txt","")

        filename = "../output/%s/%s_spec.dat"%(model,name)
        if os.path.isfile(filename): continue
        system_call = "python run_models.py --doSpec --model %s --name %s"%(model,name)
        os.system(system_call)

