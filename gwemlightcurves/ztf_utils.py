
import os, sys, json
from matplotlib import pyplot as plt
import requests
import numpy as np
from astropy.time import Time

try:
    from penquins import Kowalski
except:
    print('Please install penquins')
    pass

def get_ztf(filename, name, username, password,
            filetype = "default", z=0.0, zerr=0.0001, SN_Type="Ia"):

    k = Kowalski(username=username, password=password, verbose=True)

    q = {"query_type": "general_search",
     "query": "db['ZTF_alerts'].find({'objectId': {'$eq': '"+name+"'}})"
     }
    r = k.query(query=q,timeout=10)
    if len(r['result_data']['query_result']) >0:
        candidate = r['result_data']['query_result'][0]
        prevcandidates= r['result_data']['query_result'][0]['prv_candidates']

        print(candidate,prevcandidates)

        jd = [candidate['candidate']['jd']]
        mag = [candidate['candidate']['magpsf']]
        magerr = [candidate['candidate']['sigmapsf']]
        filt = [candidate['candidate']['fid']]

        for candidate in prevcandidates:
            jd.append(candidate['jd'])
            if not candidate['magpsf'] == None:
                mag.append(candidate['magpsf'])
            else:
                mag.append(candidate['diffmaglim'])
            if not candidate['sigmapsf'] == None:
                magerr.append(candidate['sigmapsf'])
            else:
                magerr.append(np.inf)

            filt.append(candidate['fid'])
        filtname = []
        for f in filt:
            if f == 1:
                filtname.append('g')
            elif f == 2:
                filtname.append('r')
            elif f == 3:
                filtname.append('i')
    idx = np.argsort(jd)

    if filetype == "lc":
        mjds, fluxs, fluxerrs, passband = [], [], [], []
        for ii in idx:
            t = Time(jd[ii], format='jd').mjd
            flux = 10**((mag[ii]+48.60)/(-2.5))
            fluxerr = magerr[ii]*flux
            mjds.append(t)
            fluxs.append(flux)
            fluxerrs.append(fluxerr)
            passband.append(filtname[ii])
        return mjds, fluxs, fluxerrs, passband

    fid = open(filename,'w')
    if filetype == "default":
        for ii in idx:
            t = Time(jd[ii], format='jd').isot
            fid.write('%s %s %.5f %.5f\n'%(t,filtname[ii],mag[ii],magerr[ii]))
    elif filetype == "snmachine":
        fid.write('HOST_GALAXY_PHOTO-Z:   %.4f  +- %.4f\n'%(z,zerr))
        fid.write('SIM_COMMENT:  SN Type = %s\n'%SN_Type)
        for ii in idx:
            t = Time(jd[ii], format='jd').mjd
            flux = 10**((mag[ii]+48.60)/(-2.5))
            fluxerr = magerr[ii]*flux
            fid.write('OBS: %.3f %s NULL %.3e %.3e %.2f %.5f %.5f\n'%(t,filtname[ii],flux,fluxerr,flux/fluxerr,mag[ii],magerr[ii]))
    fid.close()

def get_ztf_lc(filename, name, username, password,
               filetype = "default", z=0.0, zerr=0.0001, SN_Type="Ia"):

    r = requests.post('http://skipper.caltech.edu:8080/cgi-bin/growth/print_lc.cgi', auth=(username, password), data={'name' : name})
    lines = r.text.replace(" ","").replace("\n","").replace('"','').split("isdiffpos")[-1].split("<br>")
    jd, filtname, mag, magerr = [], [], [], []
    
    for line in lines:
        lineSplit = line.split(",")
        lineSplit = list(filter(None,lineSplit))
        if not lineSplit: continue
        if len(lineSplit) > 13:
            lineSplit = lineSplit[:13]

        if len(lineSplit) == 12:
            date,jdobs,filt,magpsf,sigmamagpsf,limmag,instrument,programid,reducedby,refsys,issub,isdiffpos = lineSplit
        else:
            date,jdobs,filt,absmag,magpsf,sigmamagpsf,limmag,instrument,programid,reducedby,refsys,issub,isdiffpos = lineSplit

        if not instrument in ["P48+ZTF","P60+SEDM"]: continue

        if not isdiffpos == "True":
            continue

        if float(magpsf) < -100:
            magpsf = "99.0"
        if float(limmag) < -100:
            limmag = "99.0"

        if np.isclose(float(magpsf),99.0):
            continue

        jd.append(float(jdobs))
        filtname.append(filt)

        if np.isclose(float(magpsf),99.0):
            mag.append(float(limmag))
            magerr.append(np.inf)
            continue
        else:
            mag.append(float(magpsf))
            magerr.append(float(sigmamagpsf))
    idx = np.argsort(jd)

    if filetype == "lc":
        mjds, fluxs, fluxerrs, passband = [], [], [], []
        for ii in idx:
            t = Time(jd[ii], format='jd').mjd
            flux = 10**((mag[ii]+48.60)/(-2.5))
            fluxerr = magerr[ii]*flux
            mjds.append(t)
            fluxs.append(flux)
            fluxerrs.append(fluxerr)
            passband.append(filtname[ii])

        maxfluxes = {}
        for filt in list(set(passband)):
            maxfluxes[filt] = -1
            for flux, thisfilt in zip(fluxs,passband):
                if thisfilt == filt:
                    if maxfluxes[filt] < flux:
                        maxfluxes[filt] = flux*1.0
        fluxs = [flux/maxfluxes[filt] for flux, filt in zip(fluxs,passband)]
        fluxerrs = [fluxerr/maxfluxes[filt] for fluxerr, filt in zip(fluxerrs,passband)]

        return mjds, mag, magerr, fluxs, fluxerrs, passband 

    fid = open(filename,'w')
    if filetype == "default":
        for ii in idx:
            t = Time(jd[ii], format='jd').isot
            fid.write('%s %s %.5f %.5f\n'%(t,filtname[ii],mag[ii],magerr[ii]))
    elif filetype == "snmachine":
        fid.write('HOST_GALAXY_PHOTO-Z:   %.4f  +- %.4f\n'%(z,zerr))
        fid.write('SIM_COMMENT:  SN Type = %s\n'%SN_Type)
        for ii in idx:
            t = Time(jd[ii], format='jd').mjd
            flux = 10**((mag[ii]+48.60)/(-2.5))
            fluxerr = magerr[ii]*flux
            if np.isclose(fluxerr,0.0,atol=1e-12):
                snr = np.inf
            else:
                snr = flux/fluxerr
            if not np.isfinite(magerr[ii]): continue
            fid.write('OBS: %.3f %s NULL %.3e %.3e %.2f %.5f %.5f\n'%(t,filtname[ii],flux,fluxerr,snr,mag[ii],magerr[ii]))
    fid.close()
