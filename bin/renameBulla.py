import os
import sys
#this is a renaming scheme NOT FOR ACTUAL FINAL ANALYSIS, BECAUSE IM SAYING MEJ = MEJWIND, which is awful. It's JUST to see what the output of run_run_models_bulla_color.py does.
fileList = os.listdir("../output/bulla_2D/")
print(fileList)
for item in fileList:
	params=item.split("_")
	mej0=float(params[2].replace("mejwind",""))
	phi0=float(params[3].replace("phi",""))
	theta=float(params[5].replace(".dat",""))
	if "Lbol" in item:
		name="%s_mej%.5f_phi%.5f_%.5f_Lbol.dat"%(params[0],mej0,phi0,theta)
	else:
		name="%s_mej%.5f_phi%.5f_%.5f.dat"%(params[0],mej0,phi0,theta)
	os.rename(r"../output/bulla_2D/"+item,r"../output/bulla_2D/"+name)
print("done")
