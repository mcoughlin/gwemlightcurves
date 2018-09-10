
import numpy as np
import json
from astropy.time import Time
import pandas as pd

filename = "ATLAS18qqn.dan"
outfile = "ATLAS18qqn.dat"
T0 = 58285.441

bands = []
mags = []
dmags = []
ts = []
tbands = []

lines = [line.rstrip('\n') for line in open(filename)]
lines = lines[1:]
for line in lines:
    lineSplit = list(filter(None,line.split(" ")))
    mjd = float(lineSplit[0])
    instrument = lineSplit[1] 
    band = lineSplit[2]   

    if ">" in lineSplit[4]:
        mag = float(lineSplit[4][1:])
        dmag = np.inf
    else:
        mag = float(lineSplit[4])
        dmag = float(lineSplit[5])

    t = Time(mjd, format='mjd').isot
    if mjd < T0: continue

    if band == "Ks":
        band = "K"
    if band == "Y":
        band = "y"
    if band == "U":
        band = "u"
  
    if band not in ["u","g","r","i","z","y","J","H","K"]: continue  

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
