
import numpy as np
import json
from astropy.time import Time
import pandas as pd

transient = "ATLAS18qqn"

filename = "%s.tmp"%transient
outfile = "%s.dat"%transient

bands = []
mags = []
dmags = []
ts = []
tbands = []
jds = []

bands_hold = []
mags_hold = []
dmags_hold = []
ts_hold = []
tbands_hold = []
jds_hold = []

lines = [line.rstrip('\n') for line in open(filename)]
for line in lines:
    lineSplit = line.split(",")
    mag = lineSplit[4]
    if mag == "None": continue
    if ">" in mag:
        mag = float(mag[1:])
        dmag = np.inf
    else:
        mag = float(mag)
        dmag = float(lineSplit[5])
    band = lineSplit[2].replace('"',"")
    jd = float(lineSplit[1])
    t = Time(jd, format='jd').isot

    ts.append(t)
    bands.append(band)
    tbands.append(t+band)
    mags.append(mag)
    dmags.append(dmag)
    jds.append(jd)        

table = [('t',ts),('band',bands),('tband',tbands),('mag',mags),('dmag',dmags)]
df = pd.DataFrame.from_items(table)
df = df.drop_duplicates(subset='tband')

fid = open(outfile,'w')
for index, row in df.iterrows():
    fid.write('%s %s %.5f %.5f\n'%(row["t"],row["band"],row["mag"],row["dmag"]))
fid.close()
