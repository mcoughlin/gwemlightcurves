
import os, sys
import glob
import numpy as np

models = ["barnes_kilonova_spectra","ns_merger_spectra","kilonova_wind_spectra","ns_precursor_AB","tanaka_compactmergers","macronovae-rosswog"]

for model in models:
    files = glob.glob("../data/%s/*"%model)
    for file in files:
        name = file.replace("_AB","").split("/")[-1].split(".")[0]

        filename = "../output/%s/%s.dat"%(model,name)
        if os.path.isfile(filename): continue

        system_call = "python run_models.py --model %s --name %s"%(model,name)
        os.system(system_call)

