
# coding: utf-8

# # Calculation of local r-process matter due to BNS mergers

# The first detection of a BNS merger by LIGO presents the opportunity to compare the estimated BNS merger rates to the local r-process element abundances and attempt to answer some key questions:
# * What fraction of r-process elements are produced by BNS mergers?
# * How much mass is ejected from BNS mergers?
# * Can we put a tighter constraint on the BNS rate by combining the LIGO observed rate with r-process abundances?
# 
# While EM observations will give the most complete picture, estimates of the ejected mass can be made with just the NS masses.  Roughly speaking, the EM observations will just give a better estimate of the ejected mass (although they could help constrain the rate as well)
# 
# As a first step, we can estimate the r-process material density due to BNS mergers by simply counting up all the mass ejected from BNS's over the lifetime of the universe:
# 
# $$\rho_{\rm r-process} = \int_0^{t_{\rm H}} R(t)M_{\rm ej} {\rm d}t = \int_\infty^0 {\rm d}z \frac{{\rm d}t}{{\rm d}z}R(z)M_{\rm ej} =  t_{\rm H}\int_0^\infty {\rm d}z \frac{R(z)M_{\rm ej}}{(1+z)E(z)}$$
# 
# $R(z)$ is the rate density of BNS mergers, $M_{\rm ej}$ is the mass ejected from a BNS merger, and $t_{\rm H}$ is the Hubble time.
# 
# To find $R(z)$, we need a few quantities:
# * The distribution of delay times $t_{\rm d}$ after birth of a binary neutron star system
# * Fraction $F_{\rm BNS}$ of systems that result in BNS systems 
# * The distribution of formation times of BNS systems from ZAMS
# * Cosmological star formation rate 
# * The local rate $R(0)$, which we get from the LIGO detection
# 
# Unfortunately, getting $R(z)$ will require a good population synthesis model, which we don't exactly have.  What we can do instead for simplicity is calculate the local r-process material production, which is naively given by
# 
# $$ \Gamma [M_\odot {\rm Gpc}^{-3} {\rm yr}^{-1}] = R(0)M_{\rm ej}$$
# 
# and assume that it is constant over some time $\Delta t$, so that 
# 
# $$\rho_{\rm r-process} = R(0)M_{\rm ej}\Delta t$$
# 
# To find $M_{\rm ej}$, we need some model that connects the values from the PE posteriors to $M_{\rm ej}$.  
# We can use the models from [Dietrich et al.](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.95.024029) and use the relations from Figure 6.  For the BNS parameters, we'll just use the [initial PE results](https://ldas-jobs.ligo.caltech.edu/~carl-johan.haster/O2/G298048/inital_HLgatedV_flow24/lalinferencemcmc/IMRPhenomPv2pseudoFourPN/128s/1187008882.45-1/V1H1L1_PRELPREL_v2/posplots.html)
# 
# The relevant PE quantities are:
# * $m_1 = 1.84 M_\odot$, maP
# * $m_2 = 1.02M_\odot$, maP
# * $M = m_1 + m_2 = 2.86M_\odot$
# * $q = 0.552$, maP
# 
# Eyeballing Figure 6 of Dietrich et al., the ejected mass can be estimated as $\sim 0.04 M_\odot$.
# 
# The BNS merger rate densities calculated by Will Farr ([here](https://git.ligo.org/RatesAndPopulations/O2Populations/blob/master/bns/GW170817Rate.ipynb)) are between $600 {\rm Gpc}^{-3} {\rm yr}^{-1}$, and $8060 {\rm Gpc}^{-3} {\rm yr}^{-1}$.  Combining this with the ejecta masses from Dietrich et al. we get:

# In[14]:

R_low = 600 #Gpc^-3 yr^-1
R_high = 8060 #Gpc^-3 yr^-1
M_ej = 0.4 # M_o
Gamma_low = R_low*M_ej
Gamma_high = R_high*M_ej
print('The low r-process material production rate is {:.1f} M_solar/(Gpc^3 yr) or {:.2f} M_solar/(Mpc^3 Myr)'.format(Gamma_low,Gamma_low/1e3))
print('The high r-process material production rate is {:.1f} M_solar/(Gpc^3 yr) or {:.2f} M_solar/(Mpc^3 Myr)'.format(Gamma_high,Gamma_high/1e3))


# We'll assume that the BNS merger rate has been constant for the last 13 billion years (this is a pretty bad assumption, but just for this exercise).  Thus we get that the total r-process material mass in the galaxy is around....

# In[23]:

rho_low = (Gamma_low/1e3)*13e3 # 13e3 Myr
rho_high = (Gamma_high/1e3)*13e3 # 13e3 Myr
print('low r-process density is {:e} M_solar/Mpc^3'.format(rho_low))
print('high r-process density is {:e} M_solar/Mpc^3'.format(rho_high))


# For comparison, we can look at the total r-process mass in the galaxy of $\sim 10^4 M_\odot$, taken from a [paper by Qian](https://arxiv.org/pdf/astro-ph/0003242.pdf).  If we assume that there is one galaxy per cubic megaparsec, this estimate is in the right ballpark! Obviously there are a *ton* of effects that haven't been taken into account.  I name some of the obvious ones here:
# * The distribution of BNS merger masses
# * The cosmological BNS rate density
# * The full PE posteriors
# * The uncertainties in ejected mass

# In[ ]:



