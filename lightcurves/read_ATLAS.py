
import numpy as np
import json
from astropy.time import Time
import pandas as pd

transient = "ATLAS17ddd"
transient = "ATLAS17fue"
transient = "ATLAS17kll"

filename = "%s.tmp"%transient
outfile = "%s.dat"%transient

bands = []
mags = []
dmags = []
ts = []
tbands = []
mjds = []

lines = [line.rstrip('\n') for line in open(filename)]
lines = lines[1:]
for line in lines:
    lineSplit = line.split(",")
    mag = lineSplit[3]
    if mag == "None": continue
    if ">" in mag:
        mag = float(mag[1:])
        dmag = np.inf
    else:
        mag = float(mag)
        dmag = float(lineSplit[4])
    band = lineSplit[6]
    mjd = float(lineSplit[8])
    t = Time(mjd, format='mjd').isot

    ts.append(t)
    bands.append(band)
    tbands.append(t+band)
    mags.append(mag)
    dmags.append(dmag)
    mjds.append(mjd)

table = [('t',ts),('band',bands),('tband',tbands),('mag',mags),('dmag',dmags)]
df = pd.DataFrame.from_items(table)
df = df.drop_duplicates(subset='tband')

fid = open(outfile,'w')
for index, row in df.iterrows():
    fid.write('%s %s %.5f %.5f\n'%(row["t"],row["band"],row["mag"],row["dmag"]))
fid.close()
