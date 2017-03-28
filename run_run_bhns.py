
import os, sys
import glob
import numpy as np

eoss = ["APR4","ALF2","H4","MS1"]
qs = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
chis = [0.0, 0.3, 0.6, 0.9]

for eos in eoss:
    for q in qs:
        for chi in chis:
            system_call = "python run_bhns.py --eos %s -q %.0f -a %.1f"%(eos,q,chi)
            os.system(system_call)


