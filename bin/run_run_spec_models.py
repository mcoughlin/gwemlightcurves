
import os, sys
import glob
import numpy as np

files = glob.glob("../spectra/*.dat")
for file in files:
    name = file.split("/")[-1].split(".")[0]

    system_call = "python run_spec_models.py --doEvent --name %s"%(name)
    os.system(system_call)


