
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
    parser.add_option("-e","--errorbudget",default=0.2,type=float)
    parser.add_option("--doReduced",  action="store_true", default=False)
    parser.add_option("--doFixZPT0",  action="store_true", default=False)
    parser.add_option("--doEOSFit",  action="store_true", default=False)
    parser.add_option("--doSimulation",  action="store_true", default=False)
    parser.add_option("--doFixMChirp",  action="store_true", default=False)
    parser.add_option("--doModels",  action="store_true", default=False)
    parser.add_option("--doGoingTheDistance",  action="store_true", default=False)
    parser.add_option("--doMasses",  action="store_true", default=False)
    parser.add_option("--doEjecta",  action="store_true", default=False)
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
        #cube[1] = cube[1]*2.0 - 1.0
        cube[1] = cube[1]*0.75
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*2.0 + 1.0
        #cube[4] = cube[4]*0.17 + 0.08
        cube[4] = cube[4]*0.1 + 0.1

def myprior_bns(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*2.0 + 1.0
        cube[2] = cube[2]*0.17 + 0.08
        cube[3] = cube[3]*2.0 + 1.0
        cube[4] = cube[4]*2.0 + 1.0
        cube[5] = cube[5]*0.17 + 0.08

def myprior_bhns_EOSFit(cube, ndim, nparams):
        cube[0] = cube[0]*6.0 + 3.0
        #cube[1] = cube[1]*2.0 - 1.0
        #cube[1] = cube[1]*1.5 - 0.75
        cube[1] = cube[1]*0.75
        cube[2] = cube[2]*2.0 + 1.0
        #cube[3] = cube[3]*0.17 + 0.08
        cube[3] = cube[3]*0.1 + 0.1

def myprior_bns_EOSFit(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 + 1.0
        cube[1] = cube[1]*0.17 + 0.08
        cube[2] = cube[2]*2.0 + 1.0
        cube[3] = cube[3]*0.17 + 0.08

def myprior_combined(cube, ndim, nparams):
        cube[0] = cube[0]*5.0 - 5.0
        cube[1] = cube[1]*1.0

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

        print m1, m2, prob
        if prior == 0.0:
            prob = -np.inf
        if mej == 0.0:
            prob = -np.inf

        return prob

def myloglike_bns_gw_EOSFit_FixMChirp(cube, ndim, nparams):
        m1 = cube[0]
        c1 = cube[1]
        m2 = cube[2]
        c2 = cube[3]

        mb1 = EOSfit(m1,c1)
        mb2 = EOSfit(m2,c2)
        mej, vej = bns_model(m1,mb1,c1,m2,mb2,c2)

        #prob = calc_prob_mchirp(m1, m2)
        prob = calc_prob_gw(m1, m2)
        prior = prior_bns(m1,mb1,c1,m2,mb2,c2)

        print m1, m2, prob
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

def myloglike_combined(cube, ndim, nparams):
        mej = cube[0]
        vej = cube[1]
        vals = np.array([mej,vej]).T

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

def calc_prob_mchirp(m1, m2):

        if (m1==0.0) or (m2==0.0):
            prob = np.nan
        else:
            vals = np.array([m1,m2]).T
            kdeeval = kde_eval(kdedir_pts,vals)[0]
            prob1 = np.log(kdeeval)
            kdeeval = kde_eval(kdedir_mchirp,vals)[0]
            prob2 = np.log(kdeeval)
            prob = prob1+prob2
 
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
elif opts.doSimulation:
    basename = 'fitting'
else:
    print "Need to enable --doModels, --doSimulation, or --doGoingTheDistance"
    exit(0)
plotDir = os.path.join(baseplotDir,basename)
if opts.doEOSFit:
    plotDir = os.path.join(plotDir,'%s_EOSFit'%opts.model)
else:
    plotDir = os.path.join(plotDir,'%s'%opts.model)
if opts.doModels:
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
elif opts.doGoingTheDistance:
    plotDir = os.path.join(plotDir,"_".join(filters))
    if opts.doMasses:
        plotDir = os.path.join(plotDir,'masses')
    elif opts.doEjecta:
        plotDir = os.path.join(plotDir,'ejecta')
    plotDir = os.path.join(plotDir,opts.name)
    dataDir = plotDir.replace("fitting_","").replace("_EOSFit","")
    dataDir = os.path.join(dataDir,"%.2f"%opts.errorbudget)

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

elif opts.doGoingTheDistance:
    data_out = lightcurve_utils.going_the_distance(opts.dataDir,opts.name)
    eta = q2eta(data_out["q"])
    m1, m2 = mc2ms(data_out["mc"], eta)
    pts = np.vstack((m1,m2)).T

    multifile = get_post_file(dataDir)
    data = np.loadtxt(multifile)

    if opts.doEjecta:
        mej_measured = data[:,1]
        vej_measured = data[:,2]
    elif opts.doMasses:
        print "Masses not implemented..."
        exit(0)

    filename = os.path.join(dataDir,"truth_mej_vej.dat")
    truths_mej_vej = np.loadtxt(filename)
    truths_mej_vej[0] = np.log10(truths_mej_vej[0])

    filename = os.path.join(dataDir,"truth.dat")
    truths = np.loadtxt(filename)

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
elif opts.doGoingTheDistance:
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

#multifile= os.path.join(plotDir,'2-.txt')
multifile = get_post_file(plotDir)
data = np.loadtxt(multifile)

if (opts.doModels or opts.doSimulation) and opts.model == "BNS" and opts.doEOSFit:
    data_new = np.zeros(data.shape)
    labels = [r"q",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$"] 
    mchirp,eta,q = ms2mc(data[:,0],data[:,2])
    data_new[:,0] = 1/q
    data_new[:,1] = mchirp
    data_new[:,2] = data[:,1]
    data_new[:,3] = data[:,3]
    data = data_new

    if opts.doFixMChirp:
        nsamples = 1000
        mchirp = np.mean(mchirp) + 0.01*np.mean(mchirp)*np.random.randn(nsamples,)
        q = 1.0 + 2.0*np.random.rand(nsamples,)
        q = 1/q
        eta = q/(1+q)**2
        m1_mchirp, m2_mchirp = mc2ms(mchirp,eta)
        pts_mchirp = np.vstack((m1_mchirp,m2_mchirp)).T
        kdedir_mchirp = greedy_kde_areas_2d(pts_mchirp)

        mchirpDir = os.path.join(plotDir,"mchirp")
        if not os.path.isdir(mchirpDir):
            os.makedirs(mchirpDir)

        pymultinest.run(myloglike_bns_gw_EOSFit, myprior_bns_EOSFit, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%mchirpDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        print stop

elif opts.doGoingTheDistance:
    if opts.model == "BNS":
        labels_mej_vej = [r"log10 ${\rm M}_{\rm ej}$",r"${\rm v}_{\rm ej}$"]
        if opts.doEOSFit:
            mej, vej = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,c1,m2,c2 in data[:,:-1]:
                mb1 = EOSfit(m1,c1)
                mb2 = EOSfit(m2,c2)
                mej[ii], vej[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        else:
            mej, vej = np.zeros(data[:,0].shape), np.zeros(data[:,0].shape)
            ii = 0
            for m1,mb1,c1,m2,mb2,c2 in data[:,:-1]:
                mej[ii], vej[ii] = bns_model(m1,mb1,c1,m2,mb2,c2)
                ii = ii + 1
        mej = np.log10(mej)

        pts_em = np.vstack((mej_measured,vej_measured)).T
        pts_gw = np.vstack((mej,vej)).T
        kdedir_em = greedy_kde_areas_2d(pts_em)
        kdedir_gw = greedy_kde_areas_2d(pts_gw)

        combinedDir = os.path.join(plotDir,"combined")
        if not os.path.isdir(combinedDir):
            os.makedirs(combinedDir)

        parameters = ["mej","vej"]
        n_params = len(parameters)
        pymultinest.run(myloglike_combined, myprior_combined, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%combinedDir, evidence_tolerance = evidence_tolerance, multimodal = False)

        multifile = get_post_file(combinedDir)
        data_combined = np.loadtxt(multifile)
        mej_combined = data_combined[:,0]
        vej_combined = data_combined[:,1]

        data_mej_vej = np.vstack((mej_combined,vej_combined)).T
        plotName = "%s/corner_mej_vej.pdf"%(plotDir)
        figure = corner.corner(data_mej_vej, labels=labels_mej_vej,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 24},
                       label_kwargs={"fontsize": 28}, title_fmt=".2f",
                       truths=truths_mej_vej)
        figure.set_size_inches(14.0,14.0)
        plt.savefig(plotName)
        plt.close()

        data_new = np.zeros(data.shape)
        labels = [r"q",r"$M_{\rm c}$",r"$C_{\rm 1}$",r"$C_{\rm 2}$"]
        mchirp,eta,q = ms2mc(data[:,0],data[:,2])
        data_new[:,0] = 1/q
        data_new[:,1] = mchirp
        data_new[:,2] = data[:,1]
        data_new[:,3] = data[:,3]
        data = data_new
        truths_new = np.zeros(truths.shape)
        mchirp,eta,q = ms2mc(truths[0],truths[2])
        truths_new[0] = 1/q
        truths_new[1] = mchirp
        truths_new[2] = truths[1]
        truths_new[3] = truths[3]
        truths = truths_new

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

if opts.doGoingTheDistance:
    colors = ['b','g','r','m','c']
    linestyles = ['-', '-.', ':','--']

    plotName = "%s/mej.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    bins, hist1 = hist_results(mej,Nbins=25,bounds=[-3.0,-1.0])
    plt.semilogy(bins,hist1,'b-',linewidth=3,label="GW")
    bins, hist1 = hist_results(mej_measured,Nbins=25,bounds=[-3.0,-1.0])
    plt.semilogy(bins,hist1,'g-.',linewidth=3,label="EM")
    bins, hist1 = hist_results(mej_combined,Nbins=25,bounds=[-3.0,-1.0])
    plt.semilogy(bins,hist1,'r:',linewidth=3,label="GW-EM")
    plt.semilogy([truths_mej_vej[0],truths_mej_vej[0]],[1e-3,10],'k--',linewidth=3,label="True")
    plt.xlabel(r"${\rm log}_{10} (M_{\rm ej})$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim([-3.0,-1.3])
    plt.ylim([1e-1,10])
    plt.savefig(plotName)
    plt.close()

    plotName = "%s/vej.pdf"%(plotDir)
    plt.figure(figsize=(10,8))
    bins, hist1 = hist_results(vej,Nbins=25,bounds=[0.0,1.0])
    plt.semilogy(bins,hist1,'b-',linewidth=3,label="GW")
    bins, hist1 = hist_results(vej_measured,Nbins=25,bounds=[0.0,1.0])
    plt.semilogy(bins,hist1,'g-.',linewidth=3,label="EM")
    bins, hist1 = hist_results(vej_combined,Nbins=25,bounds=[0.0,1.0])
    plt.semilogy(bins,hist1,'r:',linewidth=3,label="GW-EM")
    plt.semilogy([truths_mej_vej[1],truths_mej_vej[1]],[1e-3,10],'k--',linewidth=3,label="True")
    plt.xlabel(r"${v}_{\rm ej}$",fontsize=24)
    plt.ylabel('Probability Density Function',fontsize=24)
    plt.legend(loc="best",prop={'size':24})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim([1e-1,10])
    plt.savefig(plotName)
    plt.close()


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
    plotName = os.path.join(plotDir,'mej.pdf')
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
    plotName = os.path.join(plotDir,'vej.pdf')
    plt.savefig(plotName)
    plt.close('all')
