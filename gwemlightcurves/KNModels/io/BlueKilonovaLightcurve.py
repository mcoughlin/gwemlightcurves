# Kilonova Light Curve Calculator
# includes heating from r-process nuclei, free neutrons, and remnant magnetar
# based on physics in Metzger et al. 2010, Fernandez & Metzger 2014, Metzger 2017
# Brian Metzger, 2016

import os, sys
import numpy as np

from .model import register_model
from .. import KNTable

def get_BlueKilonovaLightcurve_model(table, **kwargs):
    table['mej'] = calc_meje(table['m1'], table['mb1'], table['c1'], table['m2'], table['mb2'], table['c2'])
    # Throw out smaples where the mass ejecta is less than zero.
    mask = (table['mej'] > 0)
    table = table[mask]
    # Log mass ejecta
    table['mej'] = np.log10(table['mej'])
    # calc the velocity of ejecta for those non-zero ejecta mass samples
    table['vej'] = calc_vej(table['m1'],table['c1'],table['m2'],table['c2'])
    # Calc lightcurve
    table['t'], table['lbol'], table['mag'], table['Tobs'] = calc_lc(table['tini'],table['tmax'],
                                                                     table['dt'], table['mej'],
                                                                     table['vej'],table['beta'], table['kappa_r'])
    return table

def lightcurve(tini,tmax,dt,beta,kappa_r,m1,mb1,c1,m2,mb2,c2):

    mej = calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = calc_vej(m1,c1,m2,c2)
    t, lbol, mag, Tobs = calc_lc(tini,tmax,dt,mej,vej,beta,kappa_r)

    return t, lbol, mag, Tobs

def calc_meje(m1,mb1,c1,m2,mb2,c2):

    a= -1.35695
    b=  6.11252
    c=-49.43355
    d=  16.1144
    n=  -2.5484

    tmp1=((mb1*((m2/m1)**(1.0/3.0))*(1.0-2.0*c1)/c1)+(mb2*((m1/m2)**(1.0/3.0))*(1.0-2.0*c2)/c2))*a
    tmp2=(mb1*((m2/m1)**n)+mb2*((m1/m2)**n))*b
    tmp3=(mb1*(1.0-m1/mb1)+mb2*(1.0-m2/mb2))*c

    meje_fit=np.maximum(tmp1+tmp2+tmp3+d,0)/1000.0

    return meje_fit

def calc_vrho(m1,c1,m2,c2):
    a=-0.219479
    b=0.444836
    c=-2.67385

    return ((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))*a+b

def calc_vz(m1,c1,m2,c2):
    a=-0.315585
    b=0.63808
    c=-1.00757

    return ((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))*a+b

def calc_vej(m1,c1,m2,c2):
    return np.sqrt(calc_vrho(m1,c1,m2,c2)**2.0+calc_vz(m1,c1,m2,c2)**2.0)

def calc_qej(m1,c1,m2,c2):
    vrho=calc_vrho(m1,c1,m2,c2)
    vz=calc_vz(m1,c1,m2,c2)
    vrho2=vrho*vrho
    vz2=vz*vz

    tmp1=3.*vz+np.sqrt(9*vz2+4*vrho2)
    qej=((2.0**(4.0/3.0))*vrho2+(2.*vrho2*tmp1)**(2.0/3.0))/((vrho**5.0)*tmp1)**(1.0/3.0)

    return qej

def calc_phej(m1,c1,m2,c2):
  return 4.0*calc_qej(m1,c1,m2,c2)*np.pi/2.0

def calc_lc(tini,tmax,dt,mej,vej,beta,kappa_r):

    # ** define constants **
    c = 3.0e10
    mp = 1.67e-24
    Msun = 2.0e33
    kb = 1.38e-16
    sigSB = 5.67e-5
    h = 6.63e-27
    arad = 7.56e-15
    Mpc = 3.08e24
    
    # ** define parameters **
    
    # fiducial redshift/distance
    #z = 0.01
    #D = 39.5*Mpc
    z = 0.00
    D = 1e-5*Mpc
    
    # define desired observer band wavelengths (nm) 
    # u (0), b (1), v (2), r (3), i (4), z (5), y(6), j (7), k (8), l (9)
    #lambdaobs = np.array([365., 445., 551., 658., 806., 900., 1020., 1220., 2190., 3450.])
    
    # u (0) g (1) r (2) i (3) z (4) y (5) J (6) H (7) K (8)
    lambdaobs = np.array([354.3, 477.56, 612.95, 748.46, 865.78, 960.31, 1235.0, 1662.0, 2159.0])
    
    nuobs = c/(1.0e-7*lambdaobs)
    nuobs = nuobs/(1.0 + z)
    
    # total ejecta mass 
    M0 = mej*Msun
    # minimum initial velocity
    v0 = vej*c
    # velocity index (M ~ v**-beta)
    #beta = 3.
    # initial thermal energy of bulk
    E0 = (M0)*(v0**2.0)/2.0
    # normalization of opacity of r-process matter (~10 for lanthanides, ~1 for non-lanthanides)
    #kappa_r = 10.
    # IGNORE PARAMETERS BELOW THIS LINE
    # mass cut of free neutrons
    Mn = 1.0e-8*Msun
    # electron fraction & initial neutron mass fraction in outermost layers
    Ye = 0.1
    Xn0max = 1.0-2.0*Ye
    # engine (0 = off, 1 = on)
    engine_switch = 0
    # BH (0 = magnetar, 1 = BH)
    BH_switch = 0
    ej = 0.1
    # magnetar period (in seconds) and magnetic field (G)
    P = 0.7e-3
    B = 1.0e15
    # magnetar collapse time (in units of initial spin-down times)
    tcollapse = 10000000.
    
    # ** define time array in seconds **
    #tprec = 10000
    #tmin = np.log(0.1)
    #tmax = np.log(1.0e6)
    #t = np.arange(tprec)*(tmax-tmin)/(tprec-1.0) + tmin
    #t = np.exp(t)
    #tdays = t/(3600.*24.)
    
    tdays = np.arange(tini,tmax+dt,dt)
    t = tdays*(3600.*24.)
    tprec = len(t)
    
    # ** define mass/velocity array of outer ejecta, comprised of half of mass **
    mmin = np.log(1.0e-8)
    mmax = np.log(M0/Msun)
    mprec = 300
    m = np.arange(mprec)*(mmax-mmin)/(mprec-1.0) + mmin
    m = np.exp(m)
    
    #vm(where(m gt 0.5*M0/Msun)) = v0
    #vm(where(m le 0.5*M0/Msun)) = v0*(m(where(m le 0.5*M0/Msun))/(0.5*M0/Msun))^(-1./beta)
    vm = v0*(m/(M0/Msun))**(-1./beta)
    vm[vm > c] = c
    
    # define thermalization efficiency from Barnes+16
    # 1e-2 Msun, 0.2 c
    ca3 = 1.3
    cb3 = 0.2
    cd3 = 1.1
    # 1e-3, 0.3 c
    ca2 = 8.2
    cb2 = 1.2
    cd2 = 1.52
    # 1e-2, 0.1 c
    ca = 0.56
    cb = 0.17
    cd = 0.74
    eth = 0.36*(np.exp(-ca*tdays) + np.log(1.0+2*cb*(tdays**(cd)))/(2*cb*tdays**(cd)))
    eth2 = 0.36*(np.exp(-ca2*tdays) + np.log(1.0+2*cb2*(tdays**(cd2)))/(2*cb2*tdays**(cd2)))
    eth3 = 0.36*(np.exp(-ca3*tdays) + np.log(1.0+2*cb3*(tdays**(cd3)))/(2*cb3*tdays**(cd3)))
    
    # ** calculate magnetar power **
    Rns = 12.e5
    # moment of inertia
    Ins = 1.3e20
    Ins = Ins*1.0e25
    # magnetic moment
    mu = B*(Rns**(3.0))
    # angular rotation rate
    omega = 2.0*np.pi/P
    # rotational energy
    Erot = 0.5*Ins*omega**(2.0)
    # maximum spin-down luminosity
    Lsd0 = mu**(2.0)*(omega**(4.0))/c**(3.0)
    tsd0 = Erot/Lsd0
    Lsd = Lsd0/(1.0 + t/tsd0)**(2.0)
    Lsd[t > tcollapse*tsd0] = 0.0
    Lsd = Lsd/1.0e20
    Lsd = Lsd/1.0e20
    Lsd2 = Lsd
    
    if BH_switch:
        #*** calculate BH fall-back power
        Lsd = 2.0e11*(ej/0.1)*(t/0.1)**(-5./3.)
    if not engine_switch:
        Lsd[:] = 0.0
    
    # ** define diffusive mass depth (assumed beta = 3) **
    Mdiff = (4.0*np.pi*(M0)**(1./3.)*(v0*c*t**2.)/(3.0*kappa_r))**(3./4.)
    Mdiff[Mdiff > M0] = M0
    Mdiff = Mdiff/Msun
    
    # ** define radioactive heating rates **
    # neutron and r-process mass fractions
    Xn0 = Xn0max*2*np.arctan((Mn/(m*Msun))**(1.0))/np.pi
    Xr = 1.0-Xn0
    
    # define arrays in mass layer and time
    Xn = np.zeros((mprec,tprec))
    edotn = np.zeros((mprec,tprec))
    edotr = np.zeros((mprec,tprec))
    edot = np.zeros((mprec,tprec))
    kappa = np.zeros((mprec,tprec))
    kappan = np.zeros((mprec,tprec))
    kappar = np.zeros((mprec,tprec))

    # define specific heating rates and opacity of each mass layer
    t0 = 1.3
    sig = 0.11

    tarray = np.tile(t,(mprec,1))
    Xn0array = np.tile(Xn0,(tprec,1)).T
    Xrarray = np.tile(Xr,(tprec,1)).T
    etharray = np.tile(eth,(mprec,1))
    Xn = Xn0array*np.exp(-tarray/900.)
    edotn = 3.2e14*Xn
    edotr = 4.0e18*Xrarray*(0.5 - (1./np.pi)*np.arctan((tarray-t0)/sig))**(1.3)*etharray
    edotr = 2.1e10*etharray*((tarray/(3600.*24.))**(-1.3))
    edot = edotn + edotr
    kappan = 0.4*(1.0-Xn-Xrarray)
    kappar = kappa_r*Xrarray
    kappa = kappan + kappar
    
    # define total r-process heating of inner layer
    Lr = M0*4.0e18*(0.5 - (1./np.pi)*np.arctan((t-t0)/sig))**(1.3)*eth
    Lr = Lr/1.0e20
    Lr = Lr/1.0e20
    
    # *** define arrays by mass layer/time arrays ***
    ene = np.zeros((mprec,tprec))
    lum = np.zeros((mprec,tprec))
    lumpdv = np.zeros((mprec,tprec))
    lumedot = np.zeros((mprec,tprec))
    tdiff  = np.zeros((mprec,tprec))
    tau = np.zeros((mprec,tprec))
    # properties of photosphere
    Rphoto = np.zeros((tprec,))
    vphoto = np.zeros((tprec,))
    mphoto = np.zeros((tprec,))
    kappaphoto = np.zeros((tprec,))
    
    # *** define arrays for total ejecta (1 zone = deepest layer) ***
    # thermal energy
    E = np.zeros((tprec,))
    # kinetic energy
    Ek = np.zeros((tprec,))
    # velocity
    v = np.zeros((tprec,))
    R = np.zeros((tprec,))
    taues = np.zeros((tprec,))
    Lrad = np.zeros((tprec,))
    temp = np.zeros((tprec,))
    # setting initial conditions
    E[0] = E0/1.0e20
    E[0] = E[0]/1.0e20
    Ek[0] = E0/1.0e20
    Ek[0] = Ek[0]/1.0e20
    v[0] = v0
    R[0] = t[0]*v[0]

    dt = t[1:]-t[:-1]   
    dm = m[1:]-m[:-1]
    marray = np.tile(m,(tprec,1)).T
    dmarray = np.tile(dm,(tprec,1)).T

    for j in xrange(tprec-1):
        # one zone calculation
        temp[j] = 1.0e10*(3.0*E[j]/(arad*4.0*np.pi*R[j]**(3.0)))**(0.25)
        if (temp[j] > 4000.):
            kappaoz = kappa_r
        if (temp[j] < 4000.):
            kappaoz = kappa_r*(temp[j]/4000.)**(5.5)
        kappaoz = kappa_r
        LPdV = E[j]*v[j]/R[j]
        tdiff0 = 3.0*kappaoz*M0/(4.0*np.pi*c*v[j]*t[j])
        tlc0 = R[j]/c
        tdiff0 = tdiff0+tlc0
        Lrad[j] = E[j]/tdiff0
        Ek[j+1] = Ek[j] + LPdV*(dt[j])
        v[j+1] = 1.0e20*(2.0*Ek[j]/(M0))**(0.5)
        E[j+1] = (Lr[j] + Lsd[j]-LPdV-Lrad[j])*(dt[j]) + E[j]
        R[j+1] = v[j+1]*(dt[j]) + R[j]
        taues[j+1] = (M0)*0.4/(4.0*R[j+1]**(2.0))
   
        templayer = (3.0*ene[:-1,j]*dm*Msun/(arad*4.0*np.pi*(t[j]*vm[:-1])**(3.0)))**(0.25) 
        kappa_correction = np.ones(templayer.shape)
        kappa_correction[templayer > 4000.] = 1.0
        kappa_correction[templayer < 4000.] = 1.0*(templayer[templayer < 4000.]/4000.)**(5.5)
        kappa_correction[:] = 1.0

        tdiff[:-1,j] = 0.08*kappa[:-1,j]*m[:-1]*Msun*3*kappa_correction/(vm[:-1]*c*t[j]*beta)
        tau[:-1,j] = m[:-1]*Msun*kappa[:-1,j]/(4.0*np.pi*(t[j]*vm[:-1])**(2.0))
        lum[:-1,j] = ene[:-1,j]/(tdiff[:-1,j] + t[j]*(vm[:-1]/c))
        ene[:-1,j+1] = (edot[:-1,j] - (ene[:-1,j]/t[j]) - lum[:-1,j])*(dt[j]) + ene[:-1,j]
        lum[:-1,j] = lum[:-1,j]*(dm)*Msun

        tau[mprec-1,j] = tau[mprec-2,j]
        # photosphere 
        pig1 = np.argmin(np.abs(tdiff[:,j]-t[j]))
        pig = np.argmin(np.abs(tau[:,j]-1.0))
        vphoto[j] = vm[pig]
        Rphoto[j] = vphoto[j]*t[j]
        mphoto[j] = m[pig]
        kappaphoto[j] = kappa[pig,j]
      
    Ltotm = np.sum(lum,axis=0)
    Ltotm = Ltotm/1.0e20
    Ltotm = Ltotm/1.0e20
    
    if engine_switch:
        Ltot = Lrad
        Tobs = 1.0e10*(Ltot/(4.0*np.pi*(R)**(2.0)*sigSB))**(0.25)
        if not BH_switch:
            tlife = (Lsd/1.0e5)**(0.5)*(v/(0.3*c))**(0.5)*(t/(3600.*24.))**(-0.5)
            Ltot = Ltot/(1.0+tlife)
    if not engine_switch:
        Ltot = Ltotm  
        Tobs = 1.0e10*(Ltot/(4.0*np.pi*(Rphoto)**(2.0)*sigSB))**(0.25)
 
    nuobsarray = np.tile(nuobs,(tprec,1)).T    
    expo = np.exp(h*nuobsarray/(kb*Tobs))-1.0 
    F = (2.0*np.pi*(h*nuobsarray)*((nuobsarray/c)**(2.0))/expo)*(Rphoto/D)*(Rphoto/D)

    mAB = -2.5*np.log10(F) - 48.6
    
    # distance modulus
    muD = 5.0*np.log10(D/(3.08e18))-5.

    return tdays, Ltotm*1e40, mAB, Tobs
   

register_model('BlueKilonovaLightcurve', KNTable, get_BlueKilonovaLightcurve_model,
                 usage="table")
