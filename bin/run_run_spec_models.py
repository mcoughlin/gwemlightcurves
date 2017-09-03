
import os, sys
import glob
import numpy as np

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog"]
names = ["rpft_m005_v2","bp_CaFN_hv_l","a80_leak_HR","SED_ns12ns12_kappa10"]

for model in models:
    for name in names:
        filename = "../plots/models_spec/%s/%s/1.00/samples.dat"%(model,name)
        if os.path.isfile(filename): continue
        system_call = "python run_spec_models.py --doModels --model %s --name %s"%(model,name)
        #os.system(system_call)

names = ["G298048_20170819","G298048_20170822"]

for model in models:
    for name in names:
        filename = "../plots/gws_spec/%s/%s/1.00/samples.dat"%(model,name)
        #if os.path.isfile(filename): continue
        system_call = "python run_spec_models.py --doEvent --model %s --name %s"%(model,name)
        os.system(system_call)


