
import numpy as np
import sncosmo

def lightcurve(tini,tmax,dt,z,t0,x0,x1,c):
    model = sncosmo.Model(source='salt2')
    model.set(z=z, t0=t0, x0=x0,x1=x1,c=c)
    t = np.arange(tini,tmax+dt,dt)
    mag = np.nan*np.ones((9,len(t)))
    lbol = np.nan*np.ones((len(t),))

    filters = ['sdssu','sdssg','sdssr','sdssi','sdssz','f1000w','f1280w','f1500w','f2100w']
    for i, filt in enumerate(filters):
        try:
            mag[i,:] = model.bandmag(filt, 'ab', t)
            lbol = model.bandflux(filt,t)
        except:
            continue 

    return t, lbol, mag

