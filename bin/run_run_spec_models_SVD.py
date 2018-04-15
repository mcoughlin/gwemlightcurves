
import os, sys
import glob
import numpy as np

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","macronovae-rosswog","kasen_kilonova_survey","kasen_kilonova_grid"]

files = glob.glob("../spectra/*.dat")
for model in models:
    for file in files:
        name = file.split("/")[-1].split(".")[0]

        system_call = "python run_spec_models_SVD.py --doEvent --name %s --model %s"%(name,model)
        os.system(system_call)


