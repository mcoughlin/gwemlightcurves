
import numpy as np
import json
from astropy.time import Time
import pandas as pd

filename = "GW170817.json"
outfile = "GW170817.dat"
T0 = 57982.5285236896 

bands = []
mags = []
dmags = []
ts = []
tbands = []

with open(filename) as json_file:  
    data = json.load(json_file)
    data = data["GW170817"]
    data = data["photometry"]
     
    for dat in data:
        if not "band" in dat: continue
        band = dat["band"]
        mag = float(dat["magnitude"])
        mjd = float(dat["time"])
        t = Time(mjd, format='mjd').isot
        if mjd < T0: continue

        if 'upperlimit' in dat:
            dmag = np.inf
        else:
            if 'e_magnitude' in dat:
                dmag = float(dat['e_magnitude'])
            else:
                continue   

        if band not in ["u","g","r","i","z","Y","J","H","Ks"]: continue  
        if band == "Ks": 
            band = "K"
        if band == "Y":
            band = "y"  
 
        ts.append(t)
        bands.append(band)
        tbands.append(t+band)
        mags.append(mag)
        dmags.append(dmag)

table = [('t',ts),('band',bands),('tband',tbands),('mag',mags),('dmag',dmags)]
df = pd.DataFrame.from_items(table)
df = df.drop_duplicates(subset='tband')

fid = open(outfile,'w')
for index, row in df.iterrows():
    fid.write('%s %s %.5f %.5f\n'%(row["t"],row["band"],row["mag"],row["dmag"]))
fid.close()
