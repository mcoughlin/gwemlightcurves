# Kilonova Light Curve Calculator
# includes heating from r-process nuclei, free neutrons, and remnant magnetar
# based on physics in Metzger et al. 2010, Fernandez & Metzger 2014, Metzger 2017
# Brian Metzger, 2016

import os, sys
import numpy as np
import scipy.interpolate

def lightcurve(tini,tmax,dt,theta_r,kappa_r,m1,mb1,c1,m2,mb2,c2):

    mej = calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = calc_vej(m1,c1,m2,c2)
    t, lbol, mag, Tobs = calc_lc(tini,tmax,dt,mej,vej,theta_r,kappa_r)

    return t, lbol, mag, Tobs

def calc_meje(m1,mb1,c1,m2,mb2,c2):

    a= -1.35695
    b=  6.11252
    c=-49.43355
    d=  16.1144
    n=  -2.5484

    tmp1=a*((mb1*((m2/m1)**(1.0/3.0))*(1.0-2.0*c1)/c1)+(mb2*((m1/m2)**(1.0/3.0))*(1.0-2.0*c2)/c2))
    tmp2=b*(mb1*((m2/m1)**n)+mb2*((m1/m2)**n))
    tmp3=c*(mb1*(1.0-m1/mb1)+mb2*(1.0-m2/mb2))

    meje_fit=np.max([tmp1+tmp2+tmp3+d,0])/1000.0

    return meje_fit

def calc_vrho(m1,c1,m2,c2):
    a=-0.219479
    b=0.444836
    c=-2.67385

    return a*((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))+b

def calc_vz(m1,c1,m2,c2):
    a=-0.315585
    b=0.63808
    c=-1.00757

    return a*((m1/m2)*(1.0+c*c1)+(m2/m1)*(1.0+c*c2))+b

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

def calc_lc(tini,tmax,dt,mej,vej,theta_r,kappa_r,model="DZ2"):

    mejconst = [-1.13,-1.01,-0.94,-0.94,-0.93,-0.93,-0.95,-0.99]
    vejconst = [-1.28,-1.60,-1.52,-1.56,-1.61,-1.61,-1.55,-1.33]
    kappaconst = [2.65,2.27,2.02,1.87,1.76,1.56,1.33,1.13]

    if model == "DZ2":
        mej0 = 0.013+0.005
        vej0 = 0.132+0.08
        kappa0 = 1.0
        modelfile = "../data/macronova_models_wollaeger2017/DZ2_mags_2017-03-20.dat"
    elif model == "gamA2":
        mej0 = 0.013+0.005
        vej0 = 0.132+0.08
        kappa0 = 1.0
        modelfile = "../data/macronova_models_wollaeger2017/gamA2_mags_2017-03-20.dat"
    elif model == "gamB2":
        mej0 = 0.013+0.005
        vej0 = 0.132+0.08
        kappa0 = 1.0
        modelfile = "../data/macronova_models_wollaeger2017/gamB2_mags_2017-03-20.dat"

    data_out = np.loadtxt(modelfile)
    ndata, nslices = data_out.shape    
    ints = np.arange(0,ndata,ndata/9)

    tvec_days = np.arange(tini,tmax+dt,dt)
    mAB = np.zeros(len(tvec_days),8)

    for ii in xrange(len(ints)):
        idx = np.arange(ii,ii+ndata/9)
        data_out_slice = data_out[idx,:]

        t = data_out_slice[:,1]
        data = data_out_slice[:,2:]
        nt, nbins = data.shape
       
        a_i = (360/(2*np.pi))*np.arccos(1 - np.arange(nbins)*2/float(nbins))
        b_i = (360/(2*np.pi))*np.arccos(1 - (np.arange(nbins)+1)*2/float(nbins)) 
        bins = (a_i + b_i)/2.0

        if ii == 0:
            f     = scipy.interpolate.interp2d(bins,t,np.log10(data), kind='linear')
        else:
            f     = scipy.interpolate.interp2d(bins,t,data, kind='linear')

        fam    = f(theta_r,tvec_days)

        if ii == 0:
            lbol = 10**fam
        else: 
            mAB[:,int(ii-1)] = fam + mejconst[int(ii-1)]*np.log10(mej/mej0) + vejconst[int(ii-1)]*np.log10(vej/vej0) + kappaconst[int(ii-1)]*np.log10(kappa_r/kappa0)

    print mAB
    print stop
 
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
   
    # engine (0 = off, 1 = on)
    engine_switch = 0
 
    # define desired observer band wavelengths (nm) 
    # u (0), b (1), v (2), r (3), i (4), z (5), y(6), j (7), k (8), l (9)
    #lambdaobs = np.array([365., 445., 551., 658., 806., 900., 1020., 1220., 2190., 3450.])
    
    # u (0) g (1) r (2) i (3) z (4) y (5) J (6) H (7) K (8)
    lambdaobs = np.array([354.3, 477.56, 612.95, 748.46, 865.78, 960.31, 1235.0, 1662.0, 2159.0])
    
    nuobs = c/(1.0e-7*lambdaobs)
    nuobs = nuobs/(1.0 + z)

    dotrap = 0
    M_ej = mej  # Ejecta mass (Msun)
    V_ej = vej*c
    E_51 = 1/((10./(V_ej**2))*1e51/(3*M_ej*2e33))
    kappa = kappa_r  # Initial spin period (ms)
    slope = slope_r  # Shift in days of light curve (t_sinceexplosion = tvec_days + shift_days)
    t0 = 1

    # Constants
    c      = 2.998e10   # Speed of light (cm/s) CHECKED
    e_ni0   = 3.90e10    # Energy release in Ni56 decay (erg/s/g)  CHECKED
    e_co0   = 6.78e9     # Energy release in Co56 decay (erg/s/g)  CHECKED
    m_sol  = 2e33    # Solar mass (g)  CHECKED
    tau_ni = 6.077*24*60*60/np.log(2)    # Decay time (half life / ln2) for Ni56 (s) 6.077  CHECKED
    tau_co = 77.27*24*60*60/np.log(2)    # Decay time (half life / ln2) for Co56 (s) 77.27d  CHECKED
    kappa_gamma = 0.03  # CHECKED
   
    tau_m = 1.05*((kappa/(13.7*c))**0.5) * (((((M_ej*m_sol)**3))/(E_51*1e51))**0.25)    # Diffusion time (Arnett 1982) Eq 18, 19, 22, 23 CHECKED
    y = tau_m/(2*tau_ni)   # Arnett 1982 Eq 33 CHECKED
    yp = tau_m/(2*tau_co)    # Arnet 1982 Eq 33, modified to 56Co decay  CHECKED

    Nintegrate = 5000  # Number of time steps to run integrals over

    tvec_days = np.arange(tini,tmax+dt,dt)
    Ntimes = len(tvec_days)
    Ltotm = np.zeros(tvec_days.shape)
    Rphoto = np.zeros(tvec_days.shape)

    for i in xrange(Ntimes):  

        t = tvec_days[i]*24*3600         # Time in seconds
        x = t/tau_m     # Arnett 1982 Eq 32 CHECKED
        z = np.linspace(0.000001,x,Nintegrate)        # Define limits of intergration for A(z)  CHECKED
    
        # Compute gamma ray deposition
        R = V_ej*z*tau_m     # Radius vector...z*tau_m = time  CHECKED
        rho = M_ej*2e33/(4*np.pi/3*R**3) # CHECKED
        tau_56co_gamma = kappa_gamma*rho*R  # CHECKED
        G = tau_56co_gamma/(tau_56co_gamma + 1.6)    # Arnett 1982 Eq 51 CHECKED
        D_gamma = G*(1 + 2*G*(1-G)*(1-0.75*G))    # Arnett 1982 eq 50  CHECKED

        power = np.zeros((z.shape))
        ind = np.where((z*tau_m > 0.0001*24*3600) & (z*tau_m <= t_break*24*3600))[0]
        slopeuse = slope
        ts = 1.3
        sigma = 0.11
        eth = 0.36*(np.exp(-0.56*z*tau_m/(24*3600)) + (np.log(1 + 2*0.17*(z*tau_m/(24*3600))**0.74))/(2*0.17*(z*tau_m/(24*3600))**0.74))
         
        power[ind] = eth[ind]*1.6e10*(M_ej*m_sol)*(z[ind]*tau_m/(t0*24*3600))**(slopeuse);
        ind = np.where(z*tau_m > t_break*24*3600)[0]
        slopeuse = slope_break
        power[ind] = 10**(slope-slopeuse)*eth[ind]*1.6e10*(M_ej*m_sol)*(z[ind]*tau_m/(t0*24*3600))**(slopeuse)
    
        # Kilnova part
        taudiff = 1.05/(13.7*3e10)**0.5*kappa**0.5*(M_ej*2e33)**0.75*(E_51*1e51)**(-0.25)/(24*3600)
        if (tvec_days[i] <= 2.5*taudiff):
            integrand_rprocess = power*np.exp(z**2-x**2)*2*z
            Lambda_kilonova = np.sum(integrand_rprocess*(x/Nintegrate))
        else:
            Lambda_kilonova = power[Nintegrate-1]
        Ltotm[i] = Lambda_kilonova   # Calculate luminosity
        Rphoto[i] = V_ej*tvec_days[i]*86400
 
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
    
    nuobsarray = np.tile(nuobs,(Ntimes,1)).T    
    expo = np.exp(h*nuobsarray/(kb*Tobs))-1.0 
    F = (2.0*np.pi*(h*nuobsarray)*((nuobsarray/c)**(2.0))/expo)*(Rphoto/D)*(Rphoto/D)

    mAB = -2.5*np.log10(F) - 48.6
    
    # distance modulus
    muD = 5.0*np.log10(D/(3.08e18))-5.

    return tvec_days, Ltotm*1e40, mAB, Tobs
   
