
import numpy as np

"""
PYTHON IMPLEMENTATION FOR THE REMNANT BLACK HOLE OF BH-NS MERGERS
from https://arxiv.org/pdf/1903.11622.pdf
(c) Francesco Zappa, Sebastiano Bernuzzi

This python implementation is distributed under GNU General Public
License as published by the Free Software Foundation

The code below also includes verbatim the IUB routines for the remnant BH
of BH-BH mergers released from 
 * https://arxiv.org/abs/1611.00332
 * https://arxiv.org/abs/1612.09566
"""


""" 
BBH final-sTate fitting formulas from https://arxiv.org/abs/1611.00332
(c) 2016-2017 Xisco Jimenez-Forteza, David Keitel, Sascha Husa, Mark Hannam, Sebastian Khan, Michael Puerrer
also included in LALInference under GPL at
https://versions.ligo.org/cgit/lalsuite/tree/lalinference/python/lalinference/imrtgr/nrutils.py
"""


def bbh_UIBfits_setup(m1, m2, chi1, chi2):
    """
    Common setup function for UIB final-state and luminosity fit functions
    """
    # Vectorize the function if arrays are provided as input
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    chi2 = np.vectorize(float)(np.array(chi2))
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(abs(chi1)>1):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2)>1):
      raise ValueError("chi2 has to be in [-1, 1]")
    # binary masses
    m    = m1+m2
    if np.any(m<=0):
      raise ValueError("m1+m2 must be positive")
    msq  = m*m
    m1sq = m1*m1
    m2sq = m2*m2
    # symmetric mass ratio
    eta  = m1*m2/msq
    if np.any(eta>0.25):
      print("Truncating eta from above to 0.25. This should only be necessary in some rounding corner cases, but better check your m1 and m2 inputs...")
      eta = np.minimum(eta,0.25)
    if np.any(eta<0.0):
      print("Truncating negative eta to 0.0. This should only be necessary in some rounding corner cases, but better check your m1 and m2 inputs...")
      eta = np.maximum(eta,0.0)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta2*eta2
    # spin variables (in m = 1 units)
    S1    = chi1*m1sq/msq # spin angular momentum 1
    S2    = chi2*m2sq/msq # spin angular momentum 2
    Stot  = S1+S2         # total spin
    Shat  = (chi1*m1sq+chi2*m2sq)/(m1sq+m2sq) # effective spin, = msq*Stot/(m1sq+m2sq)
    Shat2 = Shat*Shat
    Shat3 = Shat2*Shat
    Shat4 = Shat2*Shat2
    # spin difference, assuming m1>m2
    chidiff  = chi1 - chi2
    if np.any(m2>m1): # fit assumes m1>m2
      chidiff = np.sign(m1-m2)*chidiff
    chidiff2 = chidiff*chidiff
    # typical squareroots and functions of eta
    sqrt2 = 2.**0.5
    sqrt3 = 3.**0.5
    sqrt1m4eta = (1. - 4.*eta)**0.5
    return m, eta, eta2, eta3, eta4, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta

def bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2, version="v2"):
    """
    Calculate the final mass with the aligned-spin NR fit
    by Xisco Jimenez Forteza, David Keitel, Sascha Husa et al.
    [LIGO-P1600270] [https://arxiv.org/abs/1611.00332]
    versions v1 and v2 use the same ansatz,
    with v2 calibrated to additional SXS and RIT data

    m1, m2: component masses
    chi1, chi2: dimensionless spins of two BHs
    Note: Here it is assumed that m1>m2.
    """

    m, eta, eta2, eta3, eta4, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta = bbh_UIBfits_setup(m1, m2, chi1, chi2)

    if version == "v1":
        # rational-function Pade coefficients (exact) from Eq. (22) of 1611.00332v1
        b10 = 0.487
        b20 = 0.295
        b30 = 0.17
        b50 = -0.0717
        # fit coefficients from Tables VII-X of 1611.00332v1
        # values at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/EradUIB2016_pyform_coeffs.txt
        # git commit 7b47e0f35a8f960b99b24caf3ffea2ddefdc4e29
        a2 = 0.5635376058169299
        a3 = -0.8661680065959881
        a4 = 3.181941595301782
        b1 = -0.15800074104558132
        b2 = -0.15815904609933157
        b3 = -0.14299315232521553
        b5 = 8.908772171776285
        f20 = 3.8071100104582234
        f30 = 25.99956516423936
        f50 = 1.552929335555098
        f10 = 1.7004558922558886
        f21 = 0.
        d10 = -0.12282040108157262
        d11 = -3.499874245551208
        d20 = 0.014200035799803777
        d30 = -0.01873720734635449
        d31 = -5.1830734185518725
        f11 = 14.39323998088354
        f31 = -232.25752840151296
        f51 = -0.8427987782523847

    elif version == "v2":
        # rational-function Pade coefficients (exact) from Eq. (22) of 1611.00332v2
        b10 = 0.346
        b20 = 0.211
        b30 = 0.128
        b50 = -0.212
        # fit coefficients from Tables VII-X of 1611.00332v2
        # values at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/EradUIB2016v2_pyform_coeffs.txt
        # git commit f490774d3593adff5bb09ae26b7efc6deab76a42
        a2 = 0.5609904135313374
        a3 = -0.84667563764404
        a4 = 3.145145224278187
        b1 = -0.2091189048177395
        b2 = -0.19709136361080587
        b3 = -0.1588185739358418
        b5 = 2.9852925538232014
        f20 = 4.271313308472851
        f30 = 31.08987570280556
        f50 = 1.5673498395263061
        f10 = 1.8083565298668276
        f21 = 0.
        d10 = -0.09803730445895877
        d11 = -3.2283713377939134
        d20 = 0.01118530335431078
        d30 = -0.01978238971523653
        d31 = -4.91667749015812
        f11 = 15.738082204419655
        f31 = -243.6299258830685
        f51 = -0.5808669012986468

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # Calculate the radiated-energy fit from Eq. (27) of 1611.00332
    Erad = (((1. + -2.0/3.0*sqrt2)*eta + a2*eta2 + a3*eta3 + a4*eta4)*(1. + b10*b1*Shat*(f10 + f11*eta + (16. - 16.*f10 - 4.*f11)*eta2) + b20*b2*Shat2*(f20 + f21*eta + (16. - 16.*f20 - 4.*f21)*eta2) + b30*b3*Shat3*(f30 + f31*eta + (16. - 16.*f30 - 4.*f31)*eta2)))/(1. + b50*b5*Shat*(f50 + f51*eta + (16. - 16.*f50 - 4.*f51)*eta2)) + d10*sqrt1m4eta*eta2*(1. + d11*eta)*chidiff + d30*Shat*sqrt1m4eta*eta*(1. + d31*eta)*chidiff + d20*eta3*chidiff2
    # Convert to actual final mass
    Mf = m*(1.-Erad)
    ##return Mf
    return Erad, Mf, m

def bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, chi2, version="v2"):
    """
    Calculate the final spin with the aligned-spin NR fit
    by Xisco Jimenez Forteza, David Keitel, Sascha Husa et al.
    [LIGO-P1600270] [https://arxiv.org/abs/1611.00332]
    versions v1 and v2 use the same ansatz,
    with v2 calibrated to additional SXS and RIT data

    m1, m2: component masses
    chi1, chi2: dimensionless spins of two BHs
    Note: Here it is assumed that m1>m2.
    """

    m, eta, eta2, eta3, eta4, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta = bbh_UIBfits_setup(m1, m2, chi1, chi2)

    if version == "v1":
        # rational-function Pade coefficients (exact) from Eqs. (7) and (8) of 1611.00332v1
        a20 = 5.28
        a30 = 1.27
        a50 = 2.89
        b10 = -0.194
        b20 = 0.075
        b30 = 0.00782
        b50 = -0.527
        # fit coefficients from Tables I-IV of 1611.00332v1
        # evalues at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/FinalSpinUIB2016_pyform_coeffs.txt
        # git commit 7b47e0f35a8f960b99b24caf3ffea2ddefdc4e29
        a2 = 3.772362507208651
        a3 = -9.627812453422376
        a5 = 2.487406038123681
        b1 = 1.0005294518146604
        b2 = 0.8823439288807416
        b3 = 0.7612809461506448
        b5 = 0.9139185906568779
        f21 = 8.887933111404559
        f31 = 23.927104476660883
        f50 = 1.8981657997557002
        f11 = 4.411041530972546
        f52 = 0.
        d10 = 0.2762804043166152
        d11 = 11.56198469592321
        d20 = -0.05975750218477118
        d30 = 2.7296903488918436
        d31 = -3.388285154747212
        f12 = 0.3642180211450878
        f22 = -40.35359764942015
        f32 = -178.7813942566548
        f51 = -5.556957394513334

    elif version == "v2":
        # rational-function Pade coefficients (exact) from Eqs. (7) and (8) of 1611.00332v2
        a20 = 5.24
        a30 = 1.3
        a50 = 2.88
        b10 = -0.194
        b20 = 0.0851
        b30 = 0.00954
        b50 = -0.579
        # fit coefficients from Tables I-IV of 1611.00332v2
        # values at increased numerical precision copied from
        # https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/FinalSpinUIB2016v2_pyform_coeffs.txt
        # git commit f490774d3593adff5bb09ae26b7efc6deab76a42
        a2 = 3.8326341618708577
        a3 = -9.487364155598392
        a5 = 2.5134875145648374
        b1 = 1.0009563702914628
        b2 = 0.7877509372255369
        b3 = 0.6540138407185817
        b5 = 0.8396665722805308
        f21 = 8.77367320110712
        f31 = 22.830033250479833
        f50 = 1.8804718791591157
        f11 = 4.409160174224525
        f52 = 0.
        d10 = 0.3223660562764661
        d11 = 9.332575956437443
        d20 = -0.059808322561702126
        d30 = 2.3170397514509933
        d31 = -3.2624649875884852
        f12 = 0.5118334706832706
        f22 = -32.060648277652994
        f32 = -153.83722669033995
        f51 = -4.770246856212403

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # Calculate the fit for the Lorb' quantity from Eq. (16) of 1611.00332
    Lorb = (2.*sqrt3*eta + a20*a2*eta2 + a30*a3*eta3)/(1. + a50*a5*eta) + (b10*b1*Shat*(f11*eta + f12*eta2 + (64. - 16.*f11 - 4.*f12)*eta3) + b20*b2*Shat2*(f21*eta + f22*eta2 + (64. - 16.*f21 - 4.*f22)*eta3) + b30*b3*Shat3*(f31*eta + f32*eta2 + (64. - 16.*f31 - 4.*f32)*eta3))/(1. + b50*b5*Shat*(f50 + f51*eta + f52*eta2 + (64. - 64.*f50 - 16.*f51 - 4.*f52)*eta3)) + d10*sqrt1m4eta*eta2*(1. + d11*eta)*chidiff + d30*Shat*sqrt1m4eta*eta3*(1. + d31*eta)*chidiff + d20*eta3*chidiff2
    # Convert to actual final spin
    chif = Lorb + Stot
    return chif


"""
Peak luminosity fitting formula calibrated to numerical relativity and
perturbative results from https://arxiv.org/abs/1612.09566
(c) David Keitel, Xisco Jimenez Forteza, Sascha Husa, Lionel London, Alessandro Nagar, Sebastiano Bernuzzi, Enno Harms, Mark Hannam, Sebastian Khan, Michael Puerrer, Vivek Chaurasia, Geraint Pratten

This python implementation is equivalent to code contributed by the authors to the nrutils.py package of LALInference,
under GNU General Public License as published by the Free Software Foundation,
original package authors Archisman Ghosh, Nathan K. Johnson-McDaniel, P. Ajith, 2015-04-09
Additional thanks for code comments to Nathan Johnson-McDaniel, Ofek Birnholtz and Aaron Zimmerman.
"""


def bbh_UIBfits_setup_2(m1, m2, chi1, chi2):
    """
    Common setup function for UIB final-state and luminosity fit functions
    """
    # Vectorize the function if arrays are provided as input
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    chi2 = np.vectorize(float)(np.array(chi2))
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(abs(chi1)>1):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2)>1):
      raise ValueError("chi2 has to be in [-1, 1]")
    # binary masses
    m    = m1+m2
    if np.any(m<=0):
      raise ValueError("m1+m2 must be positive")
    msq  = m*m
    m1sq = m1*m1
    m2sq = m2*m2
    # symmetric mass ratio
    eta  = m1*m2/msq
    if np.any(eta>0.25):
      print("Truncating eta from above to 0.25. This should only be necessary in some rounding corner cases, but better check your m1 and m2 inputs...")
      eta = np.minimum(eta,0.25)
    if np.any(eta<0.0):
      print("Truncating negative eta to 0.0. This should only be necessary in some rounding corner cases, but better check your m1 and m2 inputs...")
      eta = np.maximum(eta,0.0)
    eta2 = eta*eta
    eta3 = eta2*eta
    eta4 = eta2*eta2
    eta5 = eta3*eta2
    # spin variables (in m = 1 units)
    S1    = chi1*m1sq/msq # spin angular momentum 1
    S2    = chi2*m2sq/msq # spin angular momentum 2
    Stot  = S1+S2         # total spin
    Shat  = msq*Stot/(m1sq+m2sq) # effective spin
    Shat2 = Shat*Shat
    Shat3 = Shat2*Shat
    Shat4 = Shat2*Shat2
    # spin difference, assuming m1>m2
    chidiff  = chi1 - chi2
    if np.any(m2>m1): # fit assumes m1>m2
      chidiff = np.sign(m1-m2)*chidiff
    chidiff2 = chidiff*chidiff
    # typical squareroots and functions of eta
    sqrt2 = 2.**0.5
    sqrt3 = 3.**0.5
    sqrt1m4eta = (1. - 4.*eta)**0.5
    return m, eta, eta2, eta3, eta4, eta5, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta

def LpeakUIB2016(m1, m2, chi1, chi2):
    """
    Peak luminosity fit function by Keitel, Jimenez-Forteza, Husa, London et al (2016)
    m1, m2: component masses
    chi1, chi2: dimensionless spins of two BHs
    Note: Here it is assumed that m1>m2.
    """

    m, eta, eta2, eta3, eta4, eta5, Stot, Shat, Shat2, Shat3, Shat4, chidiff, chidiff2, sqrt2, sqrt3, sqrt1m4eta = bbh_UIBfits_setup_2(m1, m2, chi1, chi2)

    # fit coefficients from Tables
    a0 = 0.8742169580717333
    a1 = -2.111792574893241
    a2 = 35.214103272783646
    a3 = -244.94930678226913
    a4 = 877.1061892200927
    a5 = -1172.549896493467
    b1 = 0.9800204548606681
    b2 = -0.1779843936224084
    b4 = 1.7859209418791981
    f71 = 0.
    d10 = 3.789116271213293
    d20 = 0.40214125006660567
    d30 = 4.273116678713487
    f10 = 1.6281049269810424
    f11 = -3.6322940180721037
    f20 = 31.710537408279116
    f21 = -273.84758785648336
    f30 = -0.23470852321351202
    f31 = 6.961626779884965
    f40 = 0.21139341988062182
    f41 = 1.5255885529750841
    f60 = 3.0901740789623453
    f61 = -16.66465705511997
    f70 = 0.8362061463375388
    # calculate
    Lpeak = a0 + a1*eta + a2*eta2 + a3*eta3 + a4*eta4 + a5*eta5 + (0.456*Shat*(f10 + f11*eta + (16. - 16.*f10 - 4.*f11)*eta2) - 0.019*Shat2*(f20 + f21*eta + (16. - 16.*f20 - 4.*f21)*eta2) + 1.*Shat3*(f30 + f31*eta + (-16.*f30 - 4.*f31)*eta2) + 1.*Shat4*(f40 + f41*eta + (-16.*f40 - 4.*f41)*eta2))/(1. - 0.586*Shat*(f60 + f61*eta + (16. - 16.*f60 - 4.*f61)*eta2) + 1.*Shat2*(f70 + f71*eta + (-16.*f70 - 4.*f71)*eta2)) + d10*sqrt1m4eta*eta3*chidiff + d30*Shat*sqrt1m4eta*eta3*chidiff + d20*eta3*chidiff2
    L0 = 0.01637919720310354
    # Convert to actual luminosity
    Lpeak = Lpeak*eta2*L0
    # Convert to 10^56 ergs/s units
    # We first define the "Planck luminosity" of c^5/G in 10^56 ergs/s units. Note: 10^56 ergs/s = 10^49 J/s
    LAL_LUMPL_SI = 3.628504984913064522721519179529402840e52
    LumPl_ergs_per_sec = LAL_LUMPL_SI*1e-49
    ##return LumPl_ergs_per_sec*Lpeak
    return Lpeak, LumPl_ergs_per_sec*Lpeak


"""
BH-NS remnant routines
"""

def model1a(x, p):
    """
    1D model a
    """
    return (1 + x * p[0] + x**2 * p[1]) / (1 + x * p[2]**2)**2

def model1b(x, p):
    """
    1D model b
    """
    return (1 + x * p[0] + x**2 * p[1])**2 / (1 + x * p[2]**2)**4

def pijk_to_pk(nu, ai, par):
    """
    Calculate parameters for 1D model
    """
    p = [0, 0, 0]
    p110 = par[0]
    p111 = par[1]
    p120 = par[2]
    p121 = par[3]
    p210 = par[4]
    p211 = par[5]
    p220 = par[6]
    p221 = par[7]
    p310 = par[8]
    p311 = par[9]
    p320 = par[10]
    p321 = par[11]
    p11 = p110*ai + p111
    p12 = p120*ai + p121
    p21 = p210*ai + p211
    p22 = p220*ai + p221
    p31 = p310*ai + p311
    p32 = p320*ai + p321
    p[0] = (p11 + p12*nu)*nu
    p[1] = (p21 + p22*nu)*nu
    p[2] = (p31 + p32*nu)*nu
    return p

def model3a(nu, ai, lam, par):
    """
    3D model a for BHNS remnant
    """
    return model1a(lam, pijk_to_pk(nu, ai, par))

def model3b(nu, ai, lam, par):
    """
    3D model b for BHNS remnant
    """
    return model1b(lam, pijk_to_pk(nu, ai, par))

def m1m2_to_mnu(m1, m2):
    """
    From individual masses to total mass and sym mass ratio
    """
    m = m1+m2
    nu  = m1*m2/(m*m)
    if np.any(nu>0.25):
        nu = np.minimum(nu, 0.25)
    if np.any(nu<0.0):
        nu = np.maximum(nu, 0.0)
    return m,nu

def BHNS_mass_precessing(m1, m2, chi1, lam, beta):
    """
    Compute final black hole mass for precessing binaries
    m1, m2  : respectively black hole and neutron star masses
    chi1    : modulus of the dimensionless spin   
    lam     : neutron star tidal polarizability 
    beta    : angle between orb. ang. mom. and BH spin (degrees)
    """
    
    ## Vectorization
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    lam  = np.vectorize(float)(np.array(lam))
    beta = np.vectorize(float)(np.array(beta))
    
    ## Initial checks
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(chi1>1) or np.any(chi1<0):
      raise ValueError("chi1 has to be in [0, 1], for spin below the orbital plane specify  90 < beta < 180.")
    if np.any(lam<0):
      raise ValueError("Lambda must not be negative")
    
    ## Calculate effective chi1 for precessing binaries
    chi1 = np.cos(beta*np.pi/180.)*chi1
    
    ## Fit parameters
    massc = [-1.83417425e-03,  2.39226041e-03,  4.29407902e-03,  9.79775571e-03,
              2.33868869e-07, -8.28090025e-07, -1.64315549e-06, 8.08340931e-06,
             -2.00726981e-02,  1.31986011e-01,  6.50754064e-02, -1.42749961e-01]
    
    m, nu = m1m2_to_mnu(m1, m2)
    
    ## Enforce BBH values
    mask1 = np.logical_and(chi1<0, nu<0.188)
    mask2 = chi1<-0.5
    model = model3a(nu, chi1, lam, massc)
    model[model > 1] = 1
    model[mask1] = 1
    model[mask2] = 1
    
    return bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, 0.)[1]*model

def BHNS_mass_aligned(m1, m2, chi1, lam):
    """
    Compute final black hole mass for aligned black hole spin
    m1, m2  : respectively black hole and neutron star masses
    chi1    : dimensionless spin, [-1,1] where negative value means antialigned to the orbital ang. mom.   
    lam     : neutron star tidal polarizability 
    """


    ## Vectorization
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    lam  = np.vectorize(float)(np.array(lam))

    ## Initial checks
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(abs(chi1)>1):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(lam<0):
      raise ValueError("Lambda must not be negative")

    ## Fit parameters
    massc = [-1.83417425e-03,  2.39226041e-03,  4.29407902e-03,  9.79775571e-03,
              2.33868869e-07, -8.28090025e-07, -1.64315549e-06, 8.08340931e-06,
             -2.00726981e-02,  1.31986011e-01,  6.50754064e-02, -1.42749961e-01]

    m, nu = m1m2_to_mnu(m1, m2)

    ## Enforce BBH values
    mask1 = np.logical_and(chi1<0, nu<0.188)
    mask2 = chi1<-0.5
    model = model3a(nu, chi1, lam, massc)
    model[model > 1] = 1
    model[mask1] = 1
    model[mask2] = 1

    return bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, 0.)[1]*model

def final_angle(m1, chi1, beta, momega_0):
    '''Compute approximate final angle for precessing spin using Keplerian formulas as
       the initial angle between the total initial ang. mom. J and the initial orb. ang. mom. L   
       (degrees)
    '''
    degr_conv = 180./np.pi
    S_i = chi1*(m1)**2
    L   = m1**2*momega_0**(-1./3.)
    gamma = np.arctan(S_i*np.sin(beta/degr_conv)/(S_i*np.cos(beta/degr_conv)+L))*degr_conv
    return gamma
    
def BHNS_spin_precessing(m1, m2, chi1, lam, beta, momega_0):
    """
    Compute final black hole spin for precessing binaries
    m1, m2  : respectively black hole and neutron star masses
    chi1    : modulus of the dimensionless spin   
    lam     : neutron star tidal polarizability 
    beta    : angle between orb. ang. mom. and BH spin (degrees)
    momega_0: initial orbital frequency multiplied by the total mass of the binary
    """

    ## Vectorization
    degr_conv = 180./np.pi
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    lam  = np.vectorize(float)(np.array(lam))
    beta = np.vectorize(float)(np.array(beta))
    
    ## Initial checks
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(chi1>1) or np.any(chi1<0):
      raise ValueError("chi1 has to be in [0, 1], for spin below the orbital plane specify  90 < beta < 180.")
    if np.any(lam<0):
      raise ValueError("Lambda must not be negative")
    if np.any(beta<0) or np.any(beta>180):
      raise ValueError("beta has to be in [0, 180]")
    
    ## Calculate effective chi1 for precessing binaries
    chi1 = np.cos(beta/degr_conv)*chi1
    
    ## Fit parameters
    spinc=[-5.44187381e-03,  7.91165608e-03,  2.33362046e-02,  2.47764497e-02,                                                                           
           -8.56844797e-07, -2.81727682e-06,  6.61290966e-06,  4.28979016e-05,                                                                           
           -3.04174272e-02,  2.54889050e-01,  1.47549350e-01, -4.27905832e-01] 
    
    m, nu = m1m2_to_mnu(m1, m2)
    
    ## Enforce BBH values
    mask1 = np.logical_and(chi1<0, nu<0.188)
    mask2 = chi1<-0.5
    model = model3a(nu, chi1, lam, spinc)
    model[model > 1] = 1
    model[mask1] = 1
    model[mask2] = 1

    return bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, 0.)*model, final_angle(m1, chi1, beta, momega_0)

def BHNS_spin_aligned(m1, m2, chi1, lam):
    """
    Compute final black hole spin for aligned black hole spin
    m1, m2  : respectively black hole and neutron star masses
    chi1    : dimensionless spin, [-1,1] where negative value means antialigned to the orbital ang. mom.   
    lam     : neutron star tidal polarizability 
    """
    
    ## Vectorization
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    lam  = np.vectorize(float)(np.array(lam))
    
    ## Initial checks
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(abs(chi1)>1):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(lam<0):
      raise ValueError("Lambda must not be negative")
    
    ## Fit parameters
    spinc=[-5.44187381e-03,  7.91165608e-03,  2.33362046e-02,  2.47764497e-02,                                                                           
           -8.56844797e-07, -2.81727682e-06,  6.61290966e-06,  4.28979016e-05,                                                                           
           -3.04174272e-02,  2.54889050e-01,  1.47549350e-01, -4.27905832e-01] 
    
    m, nu = m1m2_to_mnu(m1, m2)
    
    ## Enforce BBH values
    mask1 = np.logical_and(chi1<0, nu<0.188)
    mask2 = chi1<-0.5
    model = model3a(nu, chi1, lam, spinc)
    model[mask1] = 1
    model[mask2] = 1
    
    return bbh_final_spin_non_precessing_UIB2016(m1, m2, chi1, 0.)*model


def BHNS_luminosity(m1, m2, chi1, lam):
    """
    Compute GW peak luminosity for aligned black hole spin
    m1, m2  : respectively black hole and neutron star masses
    chi1    : dimensionless spin, [-1,1] where negative value means antialigned to the orbital ang. mom.   
    lam     : neutron star tidal polarizability 
    """
    
    ## Vectorization
    m1   = np.vectorize(float)(np.array(m1))
    m2   = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    lam  = np.vectorize(float)(np.array(lam))
    
    ## Initial checks
    if np.any(m1<0):
      raise ValueError("m1 must not be negative")
    if np.any(m2<0):
      raise ValueError("m2 must not be negative")
    if np.any(abs(chi1)>1):
      raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(lam<0):
      raise ValueError("Lambda must not be negative")
    
    ## Fit parameters
    lumc=[3.07830084e-02, -4.17548813e-02, -5.17020750e-02,  3.19114445e-01,
         -1.23314189e-05,  8.83850420e-06,  1.04997531e-04, -3.87759407e-05,
          3.30442649e-01, -3.75556588e-02, -9.20177581e-01,  1.43524717e+00]
    
    m, nu = m1m2_to_mnu(m1, m2)
        
    ## Enforce BBH values
    mask1 = np.logical_and(chi1<0, nu<0.188)
    mask2 = chi1<-0.5
    model = model3b(nu, chi1, lam, lumc)
    model[mask1] = 1
    model[mask2] = 1
    
    return LpeakUIB2016(m1, m2, chi1, 0)[0]*model

if __name__=='__main__':
    m1 = np.array([2.8, 5., 6.])
    m2 = np.array([1.4, 1.6, 1.5])
    chi1 = np.array([0.9, 0.9, 0.9])
    lam = np.array([500., 1000., 1000.])
    beta = np.array([80., 80., 80.])	
    M_omega0 = np.array([0.025, 0.025, 0.025])
    '''Small examples'''
 	
    ## Final BH mass for aligned BH initial spins
    print(BHNS_mass_aligned(m1, m2, chi1, lam))
    
    ## Final BH mass for precessing binaries
    print(BHNS_mass_precessing(m1, m2, chi1, lam, beta))
   	
    ## Final BH spin for aligned BH initial spins
    print(BHNS_spin_aligned(m1, m2, chi1, lam))
    
    ## Final BH spin for precessing binaries
    print(BHNS_spin_precessing(m1, m2, chi1, lam, beta, M_omega0))

    ## GW luminosity 
    print(BHNS_luminosity(m1, m2, chi1, lam))

    Xdot = BHNS_mass_aligned(m1, m2, chi1, lam)
    Egw = BHNS_luminosity(m1, m2, chi1, lam)
  
    Mdisk = (1-(Xdot+Egw))*m1
    print(Mdisk)
