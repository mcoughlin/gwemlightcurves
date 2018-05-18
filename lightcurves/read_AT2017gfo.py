
import numpy as np
import json
from astropy.time import Time
import pandas as pd

#transient = "ATLAS17ddd"
#transient = "ATLAS17fue"
transient = "AT2017gfo"

filename = "%s.tmp"%transient
outfile = "%s.dat"%transient

filters = ["u","g","r","i","z","y","J","H"]
filter_indexes = [2,4,6,8,10,12,14,16]
filtererr_indexes = [3,5,7,9,11,13,15,17]

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
    lineSplit = filter(None,line.split(" "))
    mjd = float(lineSplit[0])
    t = Time(mjd, format='mjd').isot

    for band, filt_index, filt_err_index in zip(filters,filter_indexes,filtererr_indexes):
        mag = float(lineSplit[filt_index])
        dmag = float(lineSplit[filt_err_index])

        if np.isnan(mag): continue        
        if dmag > 100:
            dmag = np.inf

        ts.append(t)
        bands.append(band)
        mags.append(mag)
        dmags.append(dmag)
        mjds.append(mjd)

    band = "K"
    if len(lineSplit) == 22:
        mag = float(lineSplit[20])
        dmag = float(lineSplit[21])
    elif len(lineSplit) == 20:
        mag = float(lineSplit[18])
        dmag = float(lineSplit[19])

    if np.isnan(mag): continue
    if dmag > 100:
        dmag = np.inf

    ts.append(t)
    bands.append(band)
    mags.append(mag)
    dmags.append(dmag)
    mjds.append(mjd)

table = [('t',ts),('band',bands),('mag',mags),('dmag',dmags)]
df = pd.DataFrame.from_items(table)

fid = open(outfile,'w')
for index, row in df.iterrows():
    fid.write('%s %s %.5f %.5f\n'%(row["t"],row["band"],row["mag"],row["dmag"]))
fid.close()
