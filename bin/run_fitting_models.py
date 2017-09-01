
import os, sys, glob, copy
import optparse
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt

import corner

import scipy.stats as ss
import plotutils.plotutils as pu

import pymultinest
from gwemlightcurves import BHNSKilonovaLightcurve, BNSKilonovaLightcurve, SALT2
from gwemlightcurves import lightcurve_utils

def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-o","--outputDir",default="../output")
    parser.add_option("-p","--plotDir",default="../plots")
    parser.add_option("-d","--dataDir",default="../data")
    parser.add_option("-l","--lightcurvesDir",default="../lightcurves")
    parser.add_option("-n","--name",default="1087")
    parser.add_option("-m","--model",default="BHNS")
    parser.add_option("--mej",default=0.005,type=float)
    parser.add_option("--vej",default=0.25,type=float)
    parser.add_option("-e","--errorbudget",default=1.0,type=float)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doSimulation",  action="store_true", default=False)
    parser.add_option("--doFixMChirp",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMassGap",  action="store_true", default=False)
    parser.add_option("--doEvent",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
    parser.add_option("--doLoveC",  action="store_true", default=False)
    parser.add_option("-f","--filters",default="g,r,i,z")

    opts, args = parser.parse_args()

    return opts

def q2eta(q):
    return q/(1+q)**2

def mc2ms(mc,eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    root = np.sqrt(0.25-eta)
    fraction = (0.5+root) / (0.5-root)
    invfraction = 1/fraction

    m2= mc * np.power((1+fraction),0.2) / np.power(fraction,0.6)

    m1= mc* np.power(1+invfraction,0.2) / np.power(invfraction,0.6)
    return (m1,m2)

def ms2mc(m1,m2):
    eta = m1*m2/( (m1+m2)*(m1+m2) )
    mchirp = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))
    q = m2/m1

    return (mchirp,eta,q)

# Give the compactness-Love and Love-compactness relations
# NKJ-M, 08.2017

import numpy as np

def CLove(lmbda):
    """
    Compactness-Love relation for neutron stars from Eq. (78) of Yagi and Yunes, Phys. Rep. 681, 1 (2017), using the YY coefficients and capping the compactness at the Buchdahl limit of 4/9 = 0.44... (since the fit diverges as lambda \to 0). We also cap the compactness at zero, since it becomes negative for large lambda, though these lambdas are so large that they are unlikely to be encountered in practice. In both cases, we raise an error if it runs up against either of the bounds.

    Input: Dimensionless quadrupolar tidal deformability lmbda
    Output: Compactness (mass over radius, in geometrized units, so the result is dimensionless)
    """

    # Give coefficients
    a0 = 0.360
    a1 = -0.0355
    a2 = 0.000705

    # Compute fit
    lmbda = np.atleast_1d(lmbda)
    ll = np.log(lmbda)
    cc = a0 + (a1 + a2*ll)*ll

    for kk in np.arange(len(cc)):
        if cc[kk] > 4./9.:
            print("Warning: Returned a compactness of 4/9 = 0.44... though the fit gives a compactness of %f for the input value of lambda = %f"%(cc[kk], lmbda[kk]))
            cc[kk] = 4./9.
        elif cc[kk] < 0.:
            print("Warning: Returned a compactness of 0. though the fit gives a compactness of %f for the input value of lambda = %f"%(cc[kk], lmbda[kk]))
            cc[kk] = 0.
    return cc

def LoveC(cc):
    """
    Invert the compactness-Love relation given above.
    """
    # Give coefficients
    a0 = 0.360
    a1 = -0.0355
    a2 = 0.000705
    ll = -(a1 + (a1*a1 - 4.*a2*(a0 - cc))**0.5)/(2.*a2)

    return np.exp(ll)

def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2, :]
    den_pts = pts[Npts/2:, :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir

def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2]
    den_pts = pts[Npts/2:]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

def kde_eval_single(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    td = kde(truth)

    return td

def hist_results(samples,Nbins=16,bounds=None):

    if not bounds==None:
        bins = np.linspace(bounds[0],bounds[1],Nbins)
    else:
        bins = np.linspace(np.min(samples),np.max(samples),Nbins)
    hist1, bin_edges = np.histogram(samples, bins=bins, density=True)
    hist1[hist1==0.0] = 1e-3
    #hist1 = hist1 / float(np.sum(hist1))
    bins = (bins[1:] + bins[:-1])/2.0

    return bins, hist1

def bhns_model(q,chi_eff,mns,mb,c):

    meje = BHNSKilonovaLightcurve.calc_meje(q,chi_eff,c,mb,mns)
    vave = BHNSKilonovaLightcurve.calc_vave(q)
  
    return meje, vave

def bns_model(m1,mb1,c1,m2,mb2,c2):

    mej = BNSKilonovaLightcurve.calc_meje(m1,mb1,c1,m2,mb2,c2)
    vej = BNSKilonovaLightcurve.calc_vej(m1,c1,m2,c2)

    return mej, vej

def get_post_file(basedir):
    filenames = glob.glob(os.path.join(basedir,'2-post*'))
    if len(filenames)>0:
        filename = filenames[0]
    else:
        filename = []
    return filename

def myprior_bhns(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        cube[1] = cube[1]*0.75
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*0.1 + 0.1

def myprior_bns(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.16 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*0.16 + 0.08

def myprior_bhns_EOSFit(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        cube[1] = cube[1]*0.75
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.1 + 0.1

def myprior_bns_EOSFit(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*0.16 + 0.08
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.16 + 0.08

def myprior_combined_ejecta(cube, ndim, nparams):
        cube[0] = cube[0]*5.0 - 5.0
        cube[1] = cube[1]*1.0

def myprior_combined_masses_bns(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*2.0 + 0.0

def myprior_combined_masses_bhns(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        cube[1] = cube[1]*5.0 + 0.0

def prior_bns(m1,mb1,c1,m2,mb2,c2):
        if m1 < m2:
            return 0.0
        else:
            return 1.0

def prior_bhns(q,chi_eff,mns,mb,c):
        return 1.0

def myloglike_bns(cube, ndim, nparams):
        m1 = cube[0]
        mb1 = cube[1]
        c1 = cube[2]
        m2 = cube[3]
        mb2 = cube[4]
        c2 = cube[5]

        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob(mej, vej)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_gw(cube, ndim, nparams):
        m1 = cube[0]
        mb1 = cube[1]
        c1 = cube[2]
        m2 = cube[3]
        mb2 = cube[4]
        c2 = cube[5]

        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob_gw(m1,m2)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_EOSFit(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob(mej, vej)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_EOSFit_FixMChirp(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob1 = calc_prob(mej, vej)
        prob2 = calc_prob_mchirp(m1, m2)
        prob = prob1+prob2
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_gw_EOSFit(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        prob = calc_prob_gw(m1, m2)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        mb = cube[3]
        c = cube[4]

        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob = calc_prob(mej, vej)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_EOSFit(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        c = cube[3]

        mb = EOSfit(mns,c)
        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob = calc_prob(mej, vej)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_EOSFit_FixMChirp(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        c = cube[3]

        mb = EOSfit(mns,c)
        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob1 = calc_prob(mej, vej)
        prob2 = calc_prob_mchirp(q*mns, mns)
        prob = prob1+prob2

        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf

        return prob

def myloglike_bhns_gw_EOSFit(cube, ndim, nparams):
        q = cube[0]
        chi_eff = cube[1]
        mns = cube[2]
        c = cube[3]

        mb = EOSfit(mns,c)
        mej, vej = bhns_model(q,chi_eff,mns,mb,c)

        prob = calc_prob_gw(q*mns, mns)
        prior = prior_bhns(q,chi_eff,mns,mb,c)
        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_combined(cube, ndim, nparams):
        var1 = cube[0]
        var2 = cube[1]
        vals = np.array([var1,var2]).T

        kdeeval_gw = kde_eval(kdedir_gw,vals)[0]
        prob_gw = np.log(kdeeval_gw)
        kdeeval_em = kde_eval(kdedir_em,vals)[0]
        prob_em = np.log(kdeeval_em)
        prob = prob_gw + prob_em

        if np.isnan(prob):
            prob = -np.inf

        return prob

def calc_prob(mej, vej): 

        if (mej==0.0) or (vej==0.0):
            prob = np.nan
        else:
            vals = np.array([mej,vej]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def calc_prob_gw(m1, m2):

        if (m1==0.0) or (m2==0.0):
            prob = np.nan
        else:
            vals = np.array([m1,m2]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob = np.log(kdeeval)

        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def Gaussian(x, mu, sigma):
    return (1.0/np.sqrt(2.0*np.pi*sigma*sigma))*np.exp(-(x-mu)*(x-mu)/2.0/sigma/sigma)

def calc_prob_mchirp(m1, m2):

        if (m1==0.0) or (m2==0.0):
            prob = np.nan
        else:
            mchirp,eta,q = ms2mc(m1,m2)
            prob = np.log(Gaussian(mchirp, mchirp_mu, mchirp_sigma))
 
        if np.isnan(prob):
            prob = -np.inf

        #if np.isfinite(prob):
        #    print mej, vej, prob
        return prob

def EOSfit(mns,c):
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb

# Parse command line
opts = parse_commandline()

if not opts.model in ["BHNS", "BNS"]:
   print "Model must be either: BHNS, BNS"
   exit(0)

filters = opts.filters.split(",")

baseplotDir = opts.plotDir
if opts.doModels:
    basename = 'fitting_models'
elif opts.doGoingTheDistance:
    basename = 'fitting_going-the-distance'
elif opts.doMassGap:
    basename = 'fitting_massgap'
elif opts.doEvent:
    basename = 'fitting_gws'
elif opts.doSimulation:
    basename = 'fitting'
else:
    print "Need to enable --doModels, --doEvent, --doSimulation, --doMassGap, or --doGoingTheDistance"
    exit(0)
plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
else:
    plotDir = os.path.join(plotDir,'%s'%opts.model)
if opts.doModels:
    plotDir = os.path.join(plotDir,"_".join(filters))
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    plotDir = os.path.join(plotDir,opts.name)
    plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)
    dataDir = plotDir.replace("fitting_models","models").replace("_EOSFit","")
elif opts.doSimulation:
    plotDir = os.path.join(plotDir,'M%03dV%02d'%(opts.mej*1000,opts.vej*100))
    plotDir = os.path.join(plotDir,"%.3f"%(opts.errorbudget*100.0))
elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    plotDir = os.path.join(plotDir,"_".join(filters))
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    plotDir = os.path.join(plotDir,opts.name)
    #dataDir = plotDir.replace("fitting_","").replace("_EOSFit","")
    dataDir = plotDir.replace("fitting_","")
    if opts.doEjecta:
        dataDir = dataDir.replace("_EOSFit","")
    dataDir = os.path.join(dataDir,"%.2f"%opts.errorbudget)
    plotDir = os.path.join(plotDir,"%.2f"%opts.errorbudget)

if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

errorbudget = opts.errorbudget
n_live_points = 1000
evidence_tolerance = 0.5

seed = 1
np.random.seed(seed=seed)

if opts.doSimulation:
    mejvar = (opts.mej*opts.errorbudget)**2
    vejvar = (opts.vej*opts.errorbudget)**2

    nsamples = 1000
    mean = [opts.mej,opts.vej]
    cov = [[mejvar,0],[0,vejvar]]
    pts = np.random.multivariate_normal(mean, cov, nsamples)
elif opts.doModels:
    multifile = get_post_file(dataDir)
    data = np.loadtxt(multifile) 

    if opts.doEjecta:
        mej = 10**data[:,1]
        vej = data[:,2]
    elif opts.doMasses:
        print "Masses not implemented..."
        exit(0)
    pts = np.vstack((mej,vej)).T

elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.doGoingTheDistance:
        data_out = lightcurve_utils.going_the_distance(opts.dataDir,opts.name)
        eta = q2eta(data_out["q"])
        m1, m2 = mc2ms(data_out["mc"], eta)
    elif opts.doMassGap:
        data_out, truths = lightcurve_utils.massgap(opts.dataDir,opts.name)
        m1, m2 = data_out["m1"], data_out["m2"]
    elif opts.doEvent:
        data_out = lightcurve_utils.event(opts.dataDir,"G298048")
        m1, m2 = data_out["m1"], data_out["m2"]

    pts = np.vstack((m1,m2)).T

    multifile = get_post_file(dataDir)
    data = np.loadtxt(multifile)

    filename = os.path.join(dataDir,"truth_mej_vej.dat")
    truths_mej_vej = np.loadtxt(filename)
    truths_mej_vej[0] = np.log10(truths_mej_vej[0])

    filename = os.path.join(dataDir,"truth.dat")
    truths = np.loadtxt(filename)

    if opts.doEjecta:
        mej_em = data[:,1]
        vej_em = data[:,2]

        mej_true = truths_mej_vej[0]
        vej_true = truths_mej_vej[1]

    elif opts.doMasses:
        if opts.model == "BNS":
            if opts.doEOSFit:
                mchirp_em,eta_em,q_em = ms2mc(data[:,1],data[:,3])
                mchirp_true,eta_true,q_true = ms2mc(truths[0],truths[2])
            else:
                mchirp_em,eta_em,q_em = ms2mc(data[:,1],data[:,4])
                mchirp_true,eta_true,q_true = ms2mc(truths[0],truths[2])
        elif opts.model == "BHNS":
            if opts.doEOSFit:
                mchirp_em,eta_em,q_em = ms2mc(data[:,1]*data[:,3],data[:,3])
                mchirp_true,eta_true,q_true = ms2mc(truths[0]*truths[4],truths[4])
            else:
                mchirp_em,eta_em,q_em = ms2mc(data[:,1]*data[:,3],data[:,3])
                mchirp_true,eta_true,q_true = ms2mc(truths[0]*truths[4],truths[4])
        q_em = 1/q_em  
        q_true = 1/q_true

kdedir = greedy_kde_areas_2d(pts)
kdedir_pts = copy.deepcopy(kdedir)

if opts.doModels or opts.doSimulation:
    if opts.model == "BHNS":
        if opts.doEOSFit:
            parameters = ["q","chi_eff","mns","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$C$"]
            n_params = len(parameters)        
            pymultinest.run(myloglike_bhns_EOSFit, myprior_bhns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["q","chi_eff","mns","mb","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$m_{\rm b}$",r"$C$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model == "BNS":
        if opts.doEOSFit:
            parameters = ["m1","c1","m2","c2"]
            labels = [r"$m_1$",r"$C_1$",r"$m_2$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["m1","mb1","c1","m2","mb2","c2"]
            labels = [r"$m_1$",r"$m_{\rm b1}$",r"$C_1$",r"$m_2$",r"$m_{\rm b2}$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.model == "BHNS":
        if opts.doEOSFit:
            parameters = ["q","chi_eff","mns","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$C$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns_gw_EOSFit, myprior_bhns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["q","chi_eff","mns","mb","c"]
            labels = [r"$q$",r"$\chi_{\rm eff}$",r"$M_{\rm NS}$",r"$m_{\rm b}$",r"$C$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bhns_gw, myprior_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
    elif opts.model == "BNS":
        if opts.doEOSFit:
            parameters = ["m1","c1","m2","c2"]
            labels = [r"$m_1$",r"$C_1$",r"$m_2$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_gw_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        else:
            parameters = ["m1","mb1","c1","m2","mb2","c2"]
            labels = [r"$m_1$",r"$m_{\rm b1}$",r"$C_1$",r"$m_2$",r"$m_{\rm b2}$",r"$C_2$"]
            n_params = len(parameters)
            pymultinest.run(myloglike_bns_gw, myprior_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False)

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='%s/2-'%plotDir)
s = a.get_stats()

import json
# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
            json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
            json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

multifile = get_post_file(plotDir)
data = np.loadtxt(multifile)

if (opts.doModels or opts.doSimulation) and opts.model == "BHNS" and opts.doEOSFit:
    m1 = data[:,0]*data[:,2]
    m2 = data[:,2]
    mchirp,eta,q = ms2mc(data[:,0],data[:,2])

    if opts.doFixMChirp:
        mchirp_mu, mchirp_sigma = np.mean(mchirp), 0.001*np.mean(mchirp)

        mchirpDir = os.path.join(plotDir,"mchirp")
        if not os.path.isdir(mchirpDir):
            os.makedirs(mchirpDir)

        pymultinest.run(myloglike_bhns_EOSFit_FixMChirp, myprior_bhns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%mchirpDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        multifile = get_post_file(mchirpDir)
        data_mchirp = np.loadtxt(multifile)

        plotName = "%s/corner_mchirp.pdf"%(plotDir)
        figure = corner.corner(data_mchirp[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".2f")
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

elif (opts.doModels or opts.doSimulation) and opts.model == "BNS" and opts.doEOSFit:
    data_new = np.zeros(data.shape)
    labels = [r"$q$",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$"] 
    mchirp,eta,q = ms2mc(data[:,0],data[:,2])
    data_new[:,0] = 1/q
    data_new[:,1] = mchirp
    data_new[:,2] = data[:,1]
    data_new[:,3] = data[:,3]
    data = data_new

    if opts.doLoveC:
        labels = [r"$q$",r"$M_{\rm c}$",r"$\log_{\rm 10} \lambda_{\rm 1}$",r"$\log_{\rm 10} \lambda_{\rm 2}$"]
        data[:,2] = np.log10(LoveC(data[:,2]))
        data[:,3] = np.log10(LoveC(data[:,3]))

    if opts.doFixMChirp:
        mchirp_mu, mchirp_sigma = np.mean(mchirp), 0.001*np.mean(mchirp)

        mchirpDir = os.path.join(plotDir,"mchirp")
        if not os.path.isdir(mchirpDir):
            os.makedirs(mchirpDir)

        pymultinest.run(myloglike_bns_EOSFit_FixMChirp, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%mchirpDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        multifile = get_post_file(mchirpDir)
        data_mchirp = np.loadtxt(multifile)
        data_new = np.zeros(data_mchirp.shape)
        labels_mchirp = [r"q",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$"]
        mchirp,eta,q = ms2mc(data_mchirp[:,0],data_mchirp[:,2])
        data_new[:,0] = 1/q
        data_new[:,1] = mchirp
        data_new[:,2] = data_mchirp[:,1]
        data_new[:,3] = data_mchirp[:,3]
        data_mchirp = data_new

        plotName = "%s/corner_mchirp.pdf"%(plotDir)
        figure = corner.corner(data_mchirp[:,:-1], labels=labels_mchirp,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".2f")
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

elif opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    if opts.model == "BNS":
        if opts.doEOSFit:
            mchirp_gw,eta_gw,q_gw = ms2mc(data[:,0],data[:,2])
            mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,c1,m2,c2 in data[:,:-1]:
                mb1 = EOSfit(m1,c1)
                mb2 = EOSfit(m2,c2)
                mej_gw[ii], vej_gw[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        else:
            mchirp_gw,eta_gw,q_gw = ms2mc(data[:,0],data[:,3])
            mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,mb1,c1,m2,mb2,c2 in data[:,:-1]:
                mej_gw[ii], vej_gw[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        q_gw = 1/q_gw 
        mej_gw = np.log10(mej_gw)
    elif opts.model == "BHNS":
        if opts.doEOSFit:
            mchirp_gw,eta_gw,q_gw = ms2mc(data[:,0]*data[:,2],data[:,2])
            mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for q,chi,mns,c in data[:,:-1]:
                mb = EOSfit(mns,c)
                mej_gw[ii], vej_gw[ii] = bhns_model(q,chi,mns,mb,c)
                ii = ii + 1
        else:
            mchirp_gw,eta_gw,q_gw = ms2mc(data[:,0]*data[:,2],data[:,3])
            mej_gw, vej_gw = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for q,chi,mns,mb,c in data[:,:-1]:
                mej_gw[ii], vej_gw[ii] = bhns_model(q,chi,mns,mb,c)
                ii = ii + 1
        q_gw = 1/q_gw
        mej_gw = np.log10(mej_gw)

    combinedDir = os.path.join(plotDir,"combined")
    if not os.path.isdir(combinedDir):
        os.makedirs(combinedDir)       

    if opts.doEjecta:
        pts_em = np.vstack((mej_em,vej_em)).T
        pts_gw = np.vstack((mej_gw,vej_gw)).T
        kdedir_em = greedy_kde_areas_2d(pts_em)
        kdedir_gw = greedy_kde_areas_2d(pts_gw)

        parameters = ["mej","vej"]
        n_params = len(parameters)
        pymultinest.run(myloglike_combined, myprior_combined_ejecta, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        labels_combined = [r"log10 ${\rm M}_{\rm ej}$",r"${\rm v}_{\rm ej}$"]
        multifile = get_post_file(combinedDir)
        data_combined = np.loadtxt(multifile)
        mej_combined = data_combined[:,0]
        vej_combined = data_combined[:,1]
        data_combined = np.vstack((mej_combined,vej_combined)).T

        plotName = "%s/corner_combined.pdf"%(plotDir)
        figure = corner.corner(data_combined, labels=labels_combined,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   truths=[mej_true,vej_true])
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()
    elif opts.doMasses:
        pts_em = np.vstack((q_em,mchirp_em)).T
        pts_gw = np.vstack((q_gw,mchirp_gw)).T

        kdedir_em = greedy_kde_areas_2d(pts_em)
        kdedir_gw = greedy_kde_areas_2d(pts_gw)

        parameters = ["q","mchirp"]
        n_params = len(parameters)
        if opts.model == "BNS":
            pymultinest.run(myloglike_combined, myprior_combined_masses_bns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)
        elif opts.model == "BHNS":
            pymultinest.run(myloglike_combined, myprior_combined_masses_bhns, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        labels_combined = [r"$q$",r"${\rm M}_{\rm c}$"]
        multifile = get_post_file(combinedDir)
        data_combined = np.loadtxt(multifile)
        q_combined = data_combined[:,0]
        mchirp_combined = data_combined[:,1]
        data_combined = np.vstack((q_combined,mchirp_combined)).T

        plotName = "%s/corner_combined.pdf"%(plotDir)
        figure = corner.corner(data_combined, labels=labels_combined,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_kwargs={"fontsize": 24},
                   label_kwargs={"fontsize": 28}, title_fmt=".2f",
                   truths=[q_true,mchirp_true])
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

#loglikelihood = -(1/2.0)*data[:,1]
#idx = np.argmax(loglikelihood)

plotName = "%s/corner.pdf"%(plotDir)
if opts.doGoingTheDistance:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".2f",
                       truths=truths)
else:
    figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".2f")
figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

if opts.doGoingTheDistance or opts.doMassGap or opts.doEvent:
    colors = ['b','g','r','m','c']
    linestyles = ['-', '-.', ':','--']

    if opts.doEjecta:
        if opts.doEvent:
            if opts.model == "BHNS":
                bounds = [-3.0,0.0]
                xlims = [-3.0,0.0]
                ylims = [1e-1,10]
            elif opts.model == "BNS":
                bounds = [-3.0,-1.0]
                xlims = [-3.0,-1.0]
                ylims = [1e-1,10]
        else:
            if opts.model == "BHNS":
                bounds = [-3.0,0.0]
                xlims = [-3.0,0.0]
                ylims = [1e-1,10]
            elif opts.model == "BNS":
                bounds = [-3.0,-1.0]
                xlims = [-3.0,-1.3]
                ylims = [1e-1,10]

        plotName = "%s/mej.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        bins, hist1 = hist_results(mej_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,label="GW")
        bins, hist1 = hist_results(mej_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,label="EM")
        bins, hist1 = hist_results(mej_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,label="GW-EM")
        plt.semilogy([mej_true,mej_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()
   
        if opts.model == "BHNS":
            bounds = [0.0,1.0]
            xlims = [0.0,1.0]
            ylims = [1e-1,20]
        elif opts.model == "BNS":
            bounds = [0.0,1.0]
            xlims = [0.0,1.0]
            ylims = [1e-1,10]
 
        plotName = "%s/vej.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        bins, hist1 = hist_results(vej_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,label="GW")
        bins, hist1 = hist_results(vej_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,label="EM")
        bins, hist1 = hist_results(vej_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,label="GW-EM")
        plt.semilogy([vej_true,vej_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()
   
        plotName = "%s/combined.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        h = pu.plot_kde_posterior_2d(data_combined,cmap='viridis')
        #pu.plot_greedy_kde_interval_2d(pts_em,np.array([0.5]),colors='b')
        #pu.plot_greedy_kde_interval_2d(pts_gw,np.array([0.5]),colors='g')
        pu.plot_greedy_kde_interval_2d(data_combined,np.array([0.5]),colors='r')
        plt.plot(mej_true,vej_true,'kx',markersize=20)
        plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$")
        plt.ylabel(r"${v}_{\rm ej}$")
        plt.savefig(plotName)
        plt.close('all')

    elif opts.doMasses:
        if opts.model == "BHNS":
            bounds = [0.8,4.0]
            xlims = [0.8,4.0]
            ylims = [1e-1,10]
        elif opts.model == "BNS":
            bounds = [0.8,2.0]
            xlims = [0.8,2.0]
            ylims = [1e-1,10]

        plotName = "%s/mchirp.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        bins, hist1 = hist_results(mchirp_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,label="GW")
        bins, hist1 = hist_results(mchirp_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,label="EM")
        bins, hist1 = hist_results(mchirp_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,label="GW-EM")
        plt.semilogy([mchirp_true,mchirp_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"${\rm M}_{\rm c}$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()

        if opts.model == "BHNS":
            bounds = [2.9,9.1]
            xlims = [2.9,9.1]
            ylims = [1e-1,10]
        elif opts.model == "BNS":
            bounds = [0.0,2.0]
            xlims = [0.9,2.0]
            ylims = [1e-1,10]

        plotName = "%s/q.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        bins, hist1 = hist_results(q_gw,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'b-',linewidth=3,label="GW")
        bins, hist1 = hist_results(q_em,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'g-.',linewidth=3,label="EM")
        bins, hist1 = hist_results(q_combined,Nbins=25,bounds=bounds)
        plt.semilogy(bins,hist1,'r:',linewidth=3,label="GW-EM")
        plt.semilogy([q_true,q_true],[1e-3,10],'k--',linewidth=3,label="True")
        plt.xlabel(r"$q$",fontsize=24)
        plt.ylabel('Probability Density Function',fontsize=24)
        plt.legend(loc="best",prop={'size':24})
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig(plotName)
        plt.close()

        plotName = "%s/combined.pdf"%(plotDir)
        plt.figure(figsize=(10,8))
        h = pu.plot_kde_posterior_2d(data_combined,cmap='viridis')
        #pu.plot_greedy_kde_interval_2d(pts_em,np.array([0.5]),colors='b')
        #pu.plot_greedy_kde_interval_2d(pts_gw,np.array([0.5]),colors='g')
        pu.plot_greedy_kde_interval_2d(data_combined,np.array([0.5]),colors='r')
        plt.plot(q_true,mchirp_true,'kx',markersize=20)
        plt.ylabel(r"${\rm M}_{\rm c}$")
        plt.xlabel(r"$q$")
        plt.savefig(plotName)
        plt.close('all')

if opts.model == "BHNS":
    q_min = 3.0
    q_max = 9.0
    chi_min = -1.0
    chi_max = 1.0

    qlin = np.linspace(q_min,q_max,50)
    chilin = np.linspace(chi_min,chi_max,51)

    qlin = (qlin[:-1] + qlin[1:])/2.0
    chilin = (chilin[:-1] + chilin[1:])/2.0

    QGRID,CHIGRID = np.meshgrid(qlin,chilin)
    MGRID = np.zeros(QGRID.shape).T
    VGRID = np.zeros(QGRID.shape).T

    c = 0.147
    mb = 1.47
    mns = 1.35
    for ii in xrange(len(qlin)):
        for jj in xrange(len(chilin)):
            MGRID[ii,jj] = BHNSKilonovaLightcurve.calc_meje(qlin[ii],chilin[jj],c,mb,mns)
            VGRID[ii,jj] = BHNSKilonovaLightcurve.calc_vave(qlin[ii])

    plt.figure(figsize=(12,10))
    plt.pcolormesh(QGRID,CHIGRID,MGRID.T,vmin=np.min(MGRID),vmax=np.max(MGRID))
    plt.xlabel("Mass Ratio")
    plt.ylabel(r"$\chi$")
    plt.xlim([q_min,q_max])
    plt.ylim([chi_min,chi_max])
    cbar = plt.colorbar()
    cbar.set_label(r'log_{10} ($M_{ej})$')
    plotName = os.path.join(plotDir,'mej_pcolor.pdf')
    plt.savefig(plotName)
    plt.close('all')

    plt.figure(figsize=(12,10))
    plt.pcolormesh(QGRID,CHIGRID,VGRID.T,vmin=np.min(VGRID),vmax=np.max(VGRID))
    plt.xlabel("Mass Ratio")
    plt.ylabel(r"$\chi$")
    plt.xlim([q_min,q_max])
    plt.ylim([chi_min,chi_max])
    cbar = plt.colorbar()
    cbar.set_label(r'$v_{ej}$')
    plotName = os.path.join(plotDir,'vej_pcolor.pdf')
    plt.savefig(plotName)
    plt.close('all')
