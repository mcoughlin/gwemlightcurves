
import os, sys
import glob
import numpy as np

eoss = ["APR4","ALF2","H4","MS1"]
qs = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
chis = [0.0, 0.3, 0.6, 0.9]

for eos in eoss:
    for q in qs:
        for chi in chis:
            system_call = "python run_parameterized_models.py --doMasses --model BHNS --eos %s -q %.0f -a %.1f"%(eos,q,chi)
            os.system(system_call)

ms = [1.3,1.35,1.4]
for eos in eoss:
    for m1 in ms:
        for m2 in ms:
            system_call = "python run_parameterized_models.py --doMasses --model BNS --eos %s --m1 %.2f --m2 %.2f"%(eos,m1,m2)
            os.system(system_call)


