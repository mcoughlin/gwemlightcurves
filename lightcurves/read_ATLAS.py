
import numpy as np
import json
from astropy.time import Time
import pandas as pd

#transient = "ATLAS17ddd"
#transient = "ATLAS17fue"
transient = "ATLAS17kll"

filename = "%s.tmp"%transient
outfile = "%s.dat"%transient

bands = []
mags = []
dmags = []
ts = []
tbands = []
mjds = []

bands_hold = []
mags_hold = []
dmags_hold = []
ts_hold = []
tbands_hold = []
mjds_hold = []

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

    if (not ts_hold) or (np.abs(mjd-np.max(mjds_hold)) <0.5):
        ts_hold.append(t)
        bands_hold.append(band)
        tbands_hold.append(t+band)
        mags_hold.append(mag)
        dmags_hold.append(dmag)
        mjds_hold.append(mjd)    
        continue
    else:
        idx = np.argmin(dmags_hold)

        ts.append(ts_hold[idx])
        bands.append(bands_hold[idx])
        tbands.append(tbands_hold[idx])
        mags.append(mags_hold[idx])
        dmags.append(dmags_hold[idx])
        mjds.append(mjds_hold[idx])

        ts_hold = [t]
        bands_hold = [band]
        tbands_hold = [t+band]
        mags_hold = [mag]
        dmags_hold = [dmag]
        mjds_hold = [mjd]

    #ts.append(t)
    #bands.append(band)
    #tbands.append(t+band)
    #mags.append(mag)
    #dmags.append(dmag)
    #mjds.append(mjd)        

table = [('t',ts),('band',bands),('tband',tbands),('mag',mags),('dmag',dmags)]
df = pd.DataFrame.from_items(table)
df = df.drop_duplicates(subset='tband')

fid = open(outfile,'w')
for index, row in df.iterrows():
    fid.write('%s %s %.5f %.5f\n'%(row["t"],row["band"],row["mag"],row["dmag"]))
fid.close()
