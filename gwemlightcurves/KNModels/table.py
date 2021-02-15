# Copyright (C) Scott Coughlin (2017)
#
# This file is part of gwemlightcurves.
#
# gwemlightcurves is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gwemlightcurves is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwemlightcurves.  If not, see <http://www.gnu.org/licenses/>.

"""Extend :mod:`astropy.table` with the `KNTable`
"""

import os
import numpy as np
import math
import scipy
import h5py
import pandas as pd

import astropy.coordinates
from astropy.table import (Table, Column, vstack)
from astropy import units as u
from distutils.spawn import find_executable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, RationalQuadratic

from gwemlightcurves import lightcurve_utils
from scipy import interpolate

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['KNTable', 'tidal_lambda_from_tilde', 'CLove', 'EOSfit', 'get_eos_list', 'get_lalsim_eos', 'construct_eos_from_polytrope']


def marginalize_eos_spec(row_sample, low_latency_flag=False):
        if (low_latency_flag):
                m1s, m2s, dists_mbta, chi_effs, weights_mbta, lambda1s, lambda2s, r1s, r2s, mb1s, mb2s, mbnss = [], [], [], [], [], [], [], [], [], [], [], []
                m1, m2, dist_mbta, chi_eff, weight_mbta = row_sample["m1"], row_sample["m2"], row_sample["dist_mbta"], row_sample["chi_eff"], row_sample["weight_mbta"]
                nsamples = 2396
                for jj in range(nsamples):
                        lambda1, lambda2, radius1, radius2, mbaryon1, mbaryon2, mbns = -1, -1, -1, -1, -1, -1, -1
                        eospath = "/home/philippe.landry/nseos/eos/spec/macro_nsstruc/macro-spec_%dcr.csv" % jj
                        data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                        marray, larray, rarray, mbararray = data_out["M"], data_out["Lambda"], data_out["R"], data_out["Mb"]
                        mmax   = np.argmax(marray)
                        marray, larray, rarray, mbararray = marray[0:mmax], larray[0:mmax], rarray[0:mmax], mbararray[0:mmax]
                        f_lambda = interpolate.interp1d(marray, larray, fill_value=0, bounds_error=False)
                        f_radius = interpolate.interp1d(marray, rarray, fill_value=0, bounds_error=False)
                        f_mbaryon = interpolate.interp1d(marray, mbararray, fill_value=0, bounds_error=False)
                        if float(f_lambda(m1)) > lambda1: lambda1 = f_lambda(m1)
                        if float(f_lambda(m2)) > lambda2: lambda2 = f_lambda(m2)
                        if float(f_radius(m1)) > radius1: radius1 = f_radius(m1)
                        if float(f_radius(m2)) > radius2: radius2 = f_radius(m2)
                        if float(f_mbaryon(m1)) > mbaryon1: mbaryon1 = f_mbaryon(m1)
                        if float(f_mbaryon(m2)) > mbaryon2: mbaryon2 = f_mbaryon(m2)
                        radius1, radius2 = radius1 * 1000, radius2 * 1000 #radius in meter
                        if np.max(marray) > mbns: mbns = np.max(marray)
                        
                        if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.) or (radius1 < 0.) or (radius2 < 0.) or (mbaryon1 < 0.) or (mbaryon2 < 0.):
                                continue
                        m1s.append(m1)
                        m2s.append(m2)
                        dists_mbta.append(dist_mbta)
                        chi_effs.append(chi_eff)
                        weights_mbta.append(weight_mbta)
                        lambda1s.append(lambda1)
                        lambda2s.append(lambda2)
                        r1s.append(radius1)
                        r2s.append(radius2)
                        mb1s.append(mbaryon1)
                        mb2s.append(mbaryon2)
                        mbnss.append(mbns)     
                data = np.vstack((m1s, m2s, dists_mbta, chi_effs, weights_mbta, lambda1s, lambda2s, r1s, r2s, mb1s, mb2s, mbnss)).T 
                results_samples = KNTable(data, names=('m1', 'm2', 'dist_mbta', 'chi_eff', 'weight_mbta', 'lambda1', 'lambda2', 'r1', 'r2', 'mb1', 'mb2', 'mbns'))
        else:
                m1s, m2s, chi_effs, lambda1s, lambda2s, r1s, r2s, mb1s, mb2s, mbnss = [], [], [], [], [], [], [], [], [], [] 
                m1, m2, chi_eff = row_sample["m1"], row_sample["m2"], row_sample["chi_eff"]           
                nsamples = 2396
                for jj in range(nsamples):
                        lambda1, lambda2, radius1, radius2, mbaryon1, mbaryon2, mbns = -1, -1, -1, -1, -1, -1, -1
                        eospath = "/home/philippe.landry/nseos/eos/spec/macro_nsstruc/macro-spec_%dcr.csv" % jj
                        data_out = np.genfromtxt(eospath, names=True, delimiter=",")
                        marray, larray, rarray, mbararray = data_out["M"], data_out["Lambda"], data_out["R"], data_out["Mb"]
                        mmax   = np.argmax(marray)
                        marray, larray, rarray, mbararray = marray[0:mmax], larray[0:mmax], rarray[0:mmax], mbararray[0:mmax]
                        f_lambda = interpolate.interp1d(marray, larray, fill_value=0, bounds_error=False)
                        f_radius = interpolate.interp1d(marray, rarray, fill_value=0, bounds_error=False)
                        f_mbaryon = interpolate.interp1d(marray, mbararray, fill_value=0, bounds_error=False)
                        if float(f_lambda(m1)) > lambda1: lambda1 = f_lambda(m1)
                        if float(f_lambda(m2)) > lambda2: lambda2 = f_lambda(m2)
                        if float(f_radius(m1)) > radius1: radius1 = f_radius(m1)
                        if float(f_radius(m2)) > radius2: radius2 = f_radius(m2)
                        if float(f_mbaryon(m1)) > mbaryon1: mbaryon1 = f_mbaryon(m1)
                        if float(f_mbaryon(m2)) > mbaryon2: mbaryon2 = f_mbaryon(m2)
                        radius1, radius2 = radius1 * 1000, radius2 * 1000 #radius in meter
                        if np.max(marray) > mbns: mbns = np.max(marray)

                        if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.) or (radius1 < 0.) or (radius2 < 0.) or (mbaryon1 < 0.) or (mbaryon2 < 0.):
                                continue
                        m1s.append(m1)
                        m2s.append(m2)
                        chi_effs.append(chi_eff)
                        lambda1s.append(lambda1)
                        lambda2s.append(lambda2)
                        r1s.append(radius1)
                        r2s.append(radius2)
                        mb1s.append(mbaryon1)
                        mb2s.append(mbaryon2)
                        mbnss.append(mbns)
                data = np.vstack((m1s, m2s, chi_effs, lambda1s, lambda2s, r1s, r2s, mb1s, mb2s, mbnss)).T                     
                results_samples = KNTable(data, names=('m1', 'm2', 'chi_eff', 'lambda1', 'lambda2', 'r1', 'r2', 'mb1', 'mb2', 'mbns'))                 
        return results_samples          



def tidal_lambda_from_tilde(mass1, mass2, lam_til, dlam_til):
    """
    Determine physical lambda parameters from effective parameters.
    See Eqs. 5 and 6 from
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.89.103012
    """
    mt = mass1 + mass2
    eta = mass1 * mass2 / mt**2
    q = np.sqrt(1 - 4*eta)

    a = (8./13) * (1 + 7*eta - 31*eta**2)
    b = (8./13) * q * (1 + 9*eta - 11*eta**2)
    c = 0.5 * q * (1 - 13272*eta/1319 + 8944*eta**2/1319)
    d = 0.5 * (1 - 15910*eta/1319 + 32850*eta**2/1319 + 3380*eta**3/1319)

    lambda1 = 0.5 * ((c - d) * lam_til - (a - b) * dlam_til)/(b*c - a*d)
    lambda2 = 0.5 * ((c + d) * lam_til - (a + b) * dlam_til)/(a*d - b*c)

    return lambda1, lambda2

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

    if (cc > 4./9.).any():
        print("Warning: Returned compactnesses > 4/9 = 0.44 ... setting = 4/9")
        print("setting compact value of {0} for lambda {1} to 4/9".format(cc[cc > 4./9.], lmbda[cc > 4./9.]))
        cc[cc > 4./9.] = 4./9.
    if (cc < 0.).any():
        print("Warning: Returned compactnesses < 0 ... setting = 0.")
        cc[cc < 0.0] = 0.0

    return cc

def EOSfit(mns,c):
    """
    # Equation to relate EOS and neutron star mass to Baryonic mass
    # Eq 8: https://arxiv.org/pdf/1708.07714.pdf
    """
    mb = mns*(1 + 0.8857853174243745*c**1.2082383572002926)
    return mb


def get_eos_list(TOV):
    """
    Populates lists of available EOSs for each set of TOV solvers
    """
    import os
    if TOV not in ['Monica', 'Wolfgang', 'lalsim']:
        raise ValueError('You have provided a TOV '
                         'for which we have no data '
                         'and therefore cannot '
                         'calculate the radius.')
    try:
        path = find_executable('ap4_mr.dat')
        path = path[:-10]
    except:
       raise ValueError('Check to make sure EOS mass-radius '
                        'tables have been installed correctly '
                        '(try `which ap4_mr.dat`)')
    if TOV == 'Monica':
        EOS_List=[file_name[:-7] for file_name in os.listdir(path) if file_name.endswith("_mr.dat") and 'lalsim' not in file_name]
    if TOV == 'Wolfgang':
        EOS_List=[file_name[:-10] for file_name in os.listdir(path) if file_name.endswith("seq")]
    if TOV == 'lalsim':
        EOS_List=[file_name[:-14] for file_name in os.listdir(path) if file_name.endswith("lalsim_mr.dat")]
    return EOS_List

def construct_eos_from_polytrope(eos_name):
    """
    Uses lalsimulation to read polytrope parameters from table
    """
    import lalsimulation as lalsim
    from astropy.io import ascii
    polytrope_table=np.genfromtxt(find_executable('polytrope_table.dat'), dtype=("|S10", '<f8','<f8','<f8','<f8'), names=True)
  
    #convert all eos names to lower case
    for i in range(0,len(polytrope_table['eos'])):
        polytrope_table['eos'][i]=polytrope_table['eos'][i].lower()

    #convert logp from cgs to si
    for i in range(0, len(polytrope_table['logP1'])):
        polytrope_table['logP1'][i]=np.log10(10**(polytrope_table['logP1'][i])*0.1)

    eos_indx=np.where(polytrope_table['eos']==eos_name.encode('utf-8'))[0][0]

    eos=lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(polytrope_table['logP1'][eos_indx], polytrope_table['gamma1'][eos_indx], polytrope_table['gamma2'][eos_indx], polytrope_table['gamma3'][eos_indx])
    fam=lalsim.CreateSimNeutronStarFamily(eos)

    return eos, fam


def get_lalsim_eos(eos_name):
    """
    EOS tables described by Ozel `here <https://arxiv.org/pdf/1603.02698.pdf>`_ and downloadable `here <http://xtreme.as.arizona.edu/NeutronStars/data/eos_tables.tar>`_. LALSim utilizes this tables, but needs some interfacing (i.e. conversion to SI units, and conversion from non monotonic to monotonic pressure density tables)
    """
    import os
    import lalsimulation
    import lal
    obs_max_mass = 2.01 - 0.04
    print("Checking %s" % eos_name)
    eos_fname = ""
    if os.path.exists(eos_name):
        # NOTE: Adapted from code by Monica Rizzo
        print("Loading from %s" % eos_name)
        bdens, press, edens = np.loadtxt(eos_name, unpack=True)
        press *= 7.42591549e-25
        edens *= 7.42591549e-25
        eos_name = os.path.basename(eos_name)
        eos_name = os.path.splitext(eos_name)[0].upper()

        if not np.all(np.diff(press) > 0):
            keep_idx = np.where(np.diff(press) > 0)[0] + 1
            keep_idx = np.concatenate(([0], keep_idx))
            press = press[keep_idx]
            edens = edens[keep_idx]
        assert np.all(np.diff(press) > 0)
        if not np.all(np.diff(edens) > 0):
            keep_idx = np.where(np.diff(edens) > 0)[0] + 1
            keep_idx = np.concatenate(([0], keep_idx))
            press = press[keep_idx]
            edens = edens[keep_idx]
        assert np.all(np.diff(edens) > 0)

        print("Dumping to %s" % eos_fname)
        eos_fname = "./." + eos_name + ".dat"
        np.savetxt(eos_fname, np.transpose((press, edens)), delimiter='\t')
        eos = lalsimulation.SimNeutronStarEOSFromFile(eos_fname)
        fam = lalsimulation.CreateSimNeutronStarFamily(eos)

    else:
        eos = lalsimulation.SimNeutronStarEOSByName(eos_name)
        fam = lalsimulation.CreateSimNeutronStarFamily(eos)

    mmass = lalsimulation.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
    print("Family %s, maximum mass: %1.2f" % (eos_name, mmass))
    if np.isnan(mmass) or mmass > 3. or mmass < obs_max_mass:
        return

    return eos, fam



class KNTable(Table):
    """A container for a table of events


    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `KNTable`
    """
    # -- i/o ------------------------------------
    @classmethod
    def read_samples(cls, filename_samples, Nsamples=100):
        """
        Read LALinference posterior_samples
        """
        import os
        if not os.path.isfile(filename_samples):
            raise ValueError("Sample file supplied does not exist")

        if "hdf" in filename_samples:
            samples_out = h5py.File(filename_samples, 'r')
            samples_out = samples_out['lalinference']

            data_out = Table(samples_out)
            data_out['q'] = data_out['m1'] / data_out['m2']
            data_out['mchirp'] = (data_out['m1'] * data_out['m2'])**(3./5.) / (data_out['m1'] + data_out['m2'])**(1./5.)
            
            data_out['theta'] = data_out['iota']
            idx = np.where(data_out['theta'] > 90.)[0]
            data_out['theta'][idx] = 180 - data_out['theta'][idx]


            data_out["eta"] = lightcurve_utils.q2eta(data_out["q"])
            data_out["m1"], data_out["m2"] = lightcurve_utils.mc2ms(data_out["mchirp"],data_out["eta"])
            data_out['q'] = 1.0/data_out['q']

 
        else:
            data_out = Table.read(filename_samples, format='ascii')
    
            if 'mass_1_source' in list(data_out.columns):
                data_out['m1'] = data_out['mass_1_source']
                print('setting m1 to m1_source')
            if 'mass_2_source' in list(data_out.columns):
                data_out['m2'] = data_out['mass_2_source']
                print('setting m2 to m2_source')

            if 'm1_detector_frame_Msun' in list(data_out.columns):
                data_out['m1'] = data_out['m1_detector_frame_Msun']
                print('setting m1 to m1_source')
            if 'm2_detector_frame_Msun' in list(data_out.columns):
                data_out['m2'] = data_out['m2_detector_frame_Msun']
                print('setting m2 to m2_source')

            if 'dlam_tilde' in list(data_out.columns):
                data_out['dlambdat'] = data_out['dlam_tilde']
                print('setting dlambdat to dlam_tilde')
            if 'lam_tilde' in list(data_out.columns):
                data_out['lambdat'] = data_out['lam_tilde']
                print('setting lambdat to lam_tilde')   

            if 'delta_lambda_tilde' in list(data_out.columns):
                data_out['dlambdat'] = data_out['delta_lambda_tilde']
                print('setting dlambdat to delta_lambda_tilde')
            if 'lambda_tilde' in list(data_out.columns):
                data_out['lambdat'] = data_out['lambda_tilde']
                print('setting lambdat to lambda_tilde')

            if 'm1' not in list(data_out.columns):
                eta = lightcurve_utils.q2eta(data_out['mass_ratio'])
                m1, m2 = lightcurve_utils.mc2ms(data_out["chirp_mass"], eta)
                data_out['m1'] = m1
                data_out['m2'] = m2

            data_out['mchirp'], data_out['eta'], data_out['q'] = lightcurve_utils.ms2mc(data_out['m1'], data_out['m2'])
            data_out['q'] = 1.0/data_out['q']
            if ('spin1' in data_out) and ('spin2' in data_out):
                data_out['chi_eff'] = ((data_out['m1'] * data_out['spin1'] +
                                           data_out['m2'] * data_out['spin2']) /
                                          (data_out['m1'] + data_out['m2']))
            elif ('chi1' in data_out) and ('chi2' in data_out):
                data_out['chi_eff'] = ((data_out['m1'] * data_out['chi1'] +
                                           data_out['m2'] * data_out['chi2']) /
                                          (data_out['m1'] + data_out['m2']))
            else:
                data_out['chi_eff'] = 0.0

            if "luminosity_distance_Mpc" in data_out:
                data_out["dist"] = data_out["luminosity_distance_Mpc"] 
            elif "luminosity_distance" in data_out:
                data_out["dist"] = data_out["luminosity_distance"]

        data_out = KNTable(data_out)
        data_out = data_out.downsample(Nsamples)
        return data_out

    @classmethod
    def read_mchirp_samples(cls, filename_samples, Nsamples=100, twixie_flag=False):
    #def read_mchirp_samples(cls, filename_samples, Nsamples=100, twixie_flag=False):
                """
                Read low latency posterior_samples
                """
                import os
                if not os.path.isfile(filename_samples):
                        raise ValueError("Sample file supplied does not exist")

                try:
                    names = ['SNRdiff', 'erf', 'weight', 'm1', 'm2',
                             'spin1', 'spin2', 'dist_mbta']
                    data_out = Table.read(filename_samples,
                                          names=names,
                                          format='ascii')
                except:
                    names = ['SNRdiff', 'erf', 'weight',
                             'm1', 'm2', 'dist']
                    data_out = Table.read(filename_samples,
                                          names=names,
                                          format='ascii')
                    data_out['spin1'] = 0.0
                    data_out['spin2'] = 0.0

                data_out['mchirp'], data_out['eta'], data_out['q'] = lightcurve_utils.ms2mc(data_out['m1'], data_out['m2'])

                data_out['chi_eff'] = ((data_out['m1'] * data_out['spin1'] +
                                       data_out['m2'] * data_out['spin2']) /
                                      (data_out['m1'] + data_out['m2']))

                #modify 'weight' using twixie informations
                if (twixie_flag):
                          from twixie import backends, distributions, utils
 
                          twixie_file = "/home/reed.essick/mass-dip/production/O1O2-ALL_BandpassPowerLaw-MassDistBeta/twixie-sample-emcee_O1O2-ALL_MassDistBandpassPowerLaw1D-MassDistBeta2D_CLEAN.hdf5"
                          (data_twixie, logprob_twixie, params_twixie), (massDist1D_twixie, massDist2D_twixie), (ranges_twixie, fixed_twixie), (posteriors_twixie, injections_twixie) = backends.load_emcee_samples(twixie_file, backends.DEFAULT_EMCEE_NAME)
                          nstp_twixie, nwlk_twixie, ndim_twixie = data_twixie.shape
                          num_1D_params_twixie = len(distributions.KNOWN_MassDist1D[massDist1D_twixie]._params)
                          mass_model_twixie = distributions.KNOWN_MassDist1D[massDist1D_twixie](*data_twixie[0,0,:num_1D_params_twixie]) ### assumes 1D model params always come first, which should be OK
                          mass_model_twixie = distributions.KNOWN_MassDist2D[massDist2D_twixie](mass_model_twixie, *data_twixie[0,0,num_1D_params_twixie:])
                          min_mass_twixie, max_mass_twixie = 1.0, 100.0
                          m_grid_twixie = np.linspace(min_mass_twixie, max_mass_twixie, 100)
                          ans_twixie = utils.qdist(data_twixie, mass_model_twixie, m_grid_twixie, np.median(data_out['q']), num_points=100)
                          ans_twixie = np.array([list(item) for item in ans_twixie])
                          twixie_func = interpolate.interp1d(ans_twixie[:,0], ans_twixie[:,1])
                          data_out['weight'] = data_out['weight'] * twixie_func(data_out['q'])
               

                data_out['weight'] = data_out['weight'] / np.max(data_out['weight'])
                data_out = data_out[data_out['weight'] > 0]
                #kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
                #gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
                #params = np.vstack((data_out['mchirp'],data_out['q'],data_out['chi_eff'],data_out['dist_mbta'])).T
                #data = np.array(data_out['weight'])
                #gp.fit(params, data)

                #mchirp_min, mchirp_max = np.min(data_out['mchirp']), np.max(data_out['mchirp'])
                #q_min, q_max = np.min(data_out['q']), np.max(data_out['q'])
                #chi_min, chi_max = np.min(data_out['chi_eff']), np.max(data_out['chi_eff'])
                #dist_mbta_min, dist_mbta_max = np.min(data_out['dist_mbta']), np.max(data_out['dist_mbta'])

                #cnt = 0
                #samples = []
                #while cnt < Nsamples:
                #    mchirp = np.random.uniform(mchirp_min, mchirp_max)
                #    q = np.random.uniform(q_min, q_max)
                #    chi_eff = np.random.uniform(chi_min, chi_max)
                #    dist_mbta = np.random.uniform(dist_mbta_min, dist_mbta_max)
                #    samp = np.atleast_2d(np.array([mchirp,q,chi_eff,dist_mbta]))
                #    weight = gp.predict(samp)[0]
                #    thresh = np.random.uniform(0,1)
                #    if weight > thresh:
                #        samples.append([mchirp,q,chi_eff,dist_mbta])
                #        cnt = cnt + 1


                #samples = [] 
                #for i in range(len(data_out)):
                #        samples = samples + [[data_out[i]['mchirp'], data_out[i]['q'], data_out[i]['chi_eff'], data_out[i]['dist_mbta']]] * math.ceil(Nsamples * data_out[i]['weight'] / np.sum(data_out['weight'])) 
                samples = np.zeros((len(data_out), 5))
                samples[:,0], samples[:,1], samples[:,2], samples[:,3], samples[:,4] = data_out['mchirp'], data_out['q'], data_out['chi_eff'], data_out['dist_mbta'], data_out['weight']



                #samples = np.array(samples)
                #data_out = Table(data=samples, names=['mchirp','q','chi_eff','dist_mbta'])
                data_out = Table(data=samples, names=['mchirp','q','chi_eff','dist_mbta', 'weight_mbta'])
                data_out["eta"] = lightcurve_utils.q2eta(data_out["q"])
                data_out["m1"], data_out["m2"] = lightcurve_utils.mc2ms(data_out["mchirp"],data_out["eta"])
                data_out["q"] = 1.0 / data_out["q"]

                #if 'm1_source' in list(data_out.columns):
                #        data_out['m1'] = data_out['m1_source']
                #        print('setting m1 to m1_source')
                #if 'm2_source' in list(data_out.columns):
                #        data_out['m2'] = data_out['m2_source']
                #        print('setting m2 to m2_source')

                #if 'dlam_tilde' in list(data_out.columns):
                #        data_out['dlambdat'] = data_out['dlam_tilde']
                #        print('setting dlambdat to dlam_tilde')
                #if 'lam_tilde' in list(data_out.columns):
                #        data_out['lambdat'] = data_out['lam_tilde']
                #        print('setting lambdat to lam_tilde')

                return KNTable(data_out)

    @classmethod
    def initialize_object(cls, input_samples, Nsamples=1000, twixie_flag=False):
                """
                Read low latency posterior_samples
                """
                names = ['weight', 'm1', 'm2', 'spin1', 'spin2', 'dist_mbta']
                data_out = Table(input_samples, names=names)
              
                data_out['mchirp'], data_out['eta'], data_out['q'] = lightcurve_utils.ms2mc(data_out['m1'], data_out['m2'])

                data_out['chi_eff'] = ((data_out['m1'] * data_out['spin1'] +
                                       data_out['m2'] * data_out['spin2']) /
                                      (data_out['m1'] + data_out['m2']))

                #modify 'weight' using twixie informations
                if (twixie_flag):
                          twixie_file = "/home/reed.essick/mass-dip/production/O1O2-ALL_BandpassPowerLaw-MassDistBeta/twixie-sample-emcee_O1O2-ALL_MassDistBandpassPowerLaw1D-MassDistBeta2D_CLEAN.hdf5"
                          (data_twixie, logprob_twixie, params_twixie), (massDist1D_twixie, massDist2D_twixie), (ranges_twixie, fixed_twixie), (posteriors_twixie, injections_twixie) = backends.load_emcee_samples(twixie_file, backends.DEFAULT_EMCEE_NAME)
                          nstp_twixie, nwlk_twixie, ndim_twixie = data_twixie.shape
                          num_1D_params_twixie = len(distributions.KNOWN_MassDist1D[massDist1D_twixie]._params)
                          mass_model_twixie = distributions.KNOWN_MassDist1D[massDist1D_twixie](*data_twixie[0,0,:num_1D_params_twixie]) ### assumes 1D model params always come first, which should be OK
                          mass_model_twixie = distributions.KNOWN_MassDist2D[massDist2D_twixie](mass_model_twixie, *data_twixie[0,0,num_1D_params_twixie:])
                          min_mass_twixie, max_mass_twixie = 1.0, 100.0
                          m_grid_twixie = np.linspace(min_mass_twixie, max_mass_twixie, 100)
                          ans_twixie = utils.qdist(data_twixie, mass_model_twixie, m_grid_twixie, np.median(data_out['q']), num_points=100)
                          ans_twixie = np.array([list(item) for item in ans_twixie])
                          twixie_func = interpolate.interp1d(ans_twixie[:,0], ans_twixie[:,1])
                          data_out['weight'] = data_out['weight'] * twixie_func(data_out['q'])
               

                data_out['weight'] = data_out['weight'] / np.max(data_out['weight'])
                kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
                gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
                params = np.vstack((data_out['mchirp'],data_out['q'],data_out['chi_eff'],data_out['dist_mbta'])).T
                data = np.array(data_out['weight'])
                gp.fit(params, data)

                mchirp_min, mchirp_max = np.min(data_out['mchirp']), np.max(data_out['mchirp'])
                q_min, q_max = np.min(data_out['q']), np.max(data_out['q'])
                chi_min, chi_max = np.min(data_out['chi_eff']), np.max(data_out['chi_eff'])
                dist_mbta_min, dist_mbta_max = np.min(data_out['dist_mbta']), np.max(data_out['dist_mbta'])

                cnt = 0
                samples = []
                while cnt < Nsamples:
                    mchirp = np.random.uniform(mchirp_min, mchirp_max)
                    q = np.random.uniform(q_min, q_max)
                    chi_eff = np.random.uniform(chi_min, chi_max)
                    dist_mbta = np.random.uniform(dist_mbta_min, dist_mbta_max)
                    samp = np.atleast_2d(np.array([mchirp,q,chi_eff,dist_mbta]))
                    weight = gp.predict(samp)[0]
                    thresh = np.random.uniform(0,1)
                    if weight > thresh:
                        samples.append([mchirp,q,chi_eff,dist_mbta])
                        cnt = cnt + 1
                samples = np.array(samples)
                data_out = Table(data=samples, names=['mchirp','q','chi_eff','dist_mbta'])
                data_out["eta"] = lightcurve_utils.q2eta(data_out["q"])
                data_out["m1"], data_out["m2"] = lightcurve_utils.mc2ms(data_out["mchirp"],data_out["eta"])
                data_out["q"] = 1.0 / data_out["q"]

                return KNTable(data_out)
    

    @classmethod
    def read_cbc_list(cls, filename_samples):
        """
        Read CBC list
        """
        if not os.path.isfile(filename_samples):
            raise ValueError("Sample file supplied does not exist")

        data_out = Table.read(filename_samples, format='ascii',
                      names = ("idx","type","t0","tc","m1",
                               "m2","Xi1",
                               "Xi2","z","dist","ra",
                               "decl","polarization",
                               "inclination","phase at t0",
                               "snrET","snrCE","snr3G"))

        if 'm1_source' in list(data_out.columns):
                data_out['m1'] = data_out['m1_source']
                print('setting m1 to m1_source')
        if 'm2_source' in list(data_out.columns):
                data_out['m2'] = data_out['m2_source']
                print('setting m2 to m2_source')

        if 'dlam_tilde' in list(data_out.columns):
                data_out['dlambdat'] = data_out['dlam_tilde']
                print('setting dlambdat to dlam_tilde')
        if 'lam_tilde' in list(data_out.columns):
                data_out['lambdat'] = data_out['lam_tilde']
                print('setting lambdat to lam_tilde')

        data_out['chi_eff'] = (data_out['m1']*data_out['Xi1']+data_out['m1']*data_out['Xi1'])/(data_out['m1']+data_out['m2'])
        data_out['q'] = data_out['m2'] / data_out['m1']

        return KNTable(data_out)

    @classmethod
    def read_multinest_samples(cls, filename_samples, model):
        """
        Read LALinference posterior_samples
        """
        import os
        if not os.path.isfile(filename_samples):
            raise ValueError("Sample file supplied does not exist")

        if model == "Ka2017":
            names=['t0', 'mej', 'vej', 'Xlan', 'zp', 'loglikelihood']
        elif model == "Ka2017x2":
            names=['t0', 'mej_1', 'vej_1', 'Xlan_1', 'mej_2', 'vej_2', 'Xlan_2', 'zp', 'loglikelihood']
        elif model == "Ka2017x2inc":
                        names=['t0', 'mej_1', 'vej_1', 'Xlan_1', 'mej_2', 'vej_2', 'Xlan_2', 'inclination', 'zp', 'loglikelihood']
        elif model == "Ka2017_TrPi2018":
                names = ["t0","mej","vej","Xlan","theta_v","E0","theta_c","theta_w","n","p","epsilon_E","epsilon_B","zp", 'loglikelihood']
        elif model == "Ka2017_A":
                names=['t0', 'mej', 'vej', 'Xlan', 'A', 'zp', 'loglikelihood']
        elif model == "Bu2019inc":
                        names=['t0', 'mej', 'phi', 'theta', 'zp', 'loglikelihood']
        elif model in ["Bu2019lf","Bu2019lr","Bu2019lm"]:
                        names=['t0', 'mej_dyn', 'mej_wind', 'phi', 'theta', 'zp', 'loglikelihood']
        elif model in ["Bu2019lw"]:
                        names=['t0', 'mej_wind', 'phi', 'theta', 'zp', 'loglikelihood']
        elif model == "Bu2019inc_TrPi2018":
                        names=['t0', 'mej', 'phi', 'theta', "E0","theta_c","theta_w","n","p","epsilon_E","epsilon_B", 'zp', 'loglikelihood']
        else:
            print("Model not implemented...")
            exit(0)
        data_out = Table.read(filename_samples, format='ascii', names = names)
        if model == "Ka2017":
            data_out['mej'] = 10**data_out['mej']
            data_out['Xlan'] = 10**data_out['Xlan']
        elif model == "Ka2017_A":
                data_out['mej'] = 10**data_out['mej']
                data_out['Xlan'] = 10**data_out['Xlan']
                data_out['A'] = 10**data_out['A']
        elif model in ["Ka2017x2","Ka2017x2inc"]:
            data_out['mej_1'] = 10**data_out['mej_1']
            data_out['Xlan_1'] = 10**data_out['Xlan_1']
            data_out['mej_2'] = 10**data_out['mej_2']
            data_out['Xlan_2'] = 10**data_out['Xlan_2']
        elif model == "Ka2017_TrPi2018":
                data_out['mej'] = 10**data_out['mej']
                data_out['Xlan'] = 10**data_out['Xlan']
                data_out['E0'] = 10**data_out['E0']
                data_out['n'] = 10**data_out['n']
                data_out['epsilon_E'] = 10**data_out['epsilon_E']
                data_out['epsilon_B'] = 10**data_out['epsilon_B']
        elif model in ["Bu2019","Bu2019inc"]:
                        data_out['mej'] = 10**data_out['mej']
        elif model in ["Bu2019lf","Bu2019lr","Bu2019lm"]:
                        data_out['mej_dyn'] = 10**data_out['mej_dyn']
                        data_out['mej_wind'] = 10**data_out['mej_wind']
        elif model in ["Bu2019lw"]:
                        data_out['mej_wind'] = 10**data_out['mej_wind']
        elif model == "Bu2019inc_TrPi2018":
                        data_out['mej'] = 10**data_out['mej']
                        data_out['E0'] = 10**data_out['E0']
                        data_out['n'] = 10**data_out['n']
                        data_out['epsilon_E'] = 10**data_out['epsilon_E']
                        data_out['epsilon_B'] = 10**data_out['epsilon_B']

        #zp_mu, zp_std = 0.0, 5.0
        #data_out['zp'] = scipy.stats.norm(zp_mu, zp_std).ppf(data_out['zp'])

        return KNTable(data_out)

    def calc_tidal_lambda(self, remove_negative_lambda=False):
        """
        Takes posterior samples and calculates lambda1 and lambda2 from
        lambdat and dlambdat.
        """

        if (not 'lambda1' in list(self.columns)) and (not 'lambda2' in list(self.columns)):
            self['lambda1'], self['lambda2'] = tidal_lambda_from_tilde(
                                          self["m1"], self["m2"],
                                          self["lambdat"], self["dlambdat"])
        if remove_negative_lambda:
            print('You have requested to remove negative lambda values')
            mask = (self["lambda1"] < 0) | (self["lambda2"] < 0)
            self = self[~mask]
            print("Removing %d/%d due to negative lambdas"%(np.sum(mask),len(mask)))

        return self


    def calc_compactness(self, fit=False):
        """
        calculate compactness of objects from lambda1 and lambda2
        """
        try:
            import lal
            G = lal.G_SI; c = lal.C_SI; msun = lal.MSUN_SI
        except:
            import astropy.units as u
            import astropy.constants as C
            G = lal.G_SI; c = C.c.value; msun = u.M_sun.to(u.kg)

        if fit:
            print('You have chose to calculate compactness from fit.')
            print('you are therefore choosing to be EOS agnostic')
            self["c1"] = CLove(self["lambda1"])
            self["c2"] = CLove(self["lambda2"])
        else:
            print('You have chose to calculate compactness from radius.')
            print('you are therefore must have selected a EOS')
            self['c1'] = self['m1'] / self['r1'] * G / c**2 * msun
            self['c2'] = self['m2'] / self['r2'] * G / c**2 * msun
            
            if (self['c1'] > 4./9.).any():
                print("Warning: Returned compactnesses > 4/9 = 0.44 ... setting = 4/9")
                self['c1'][self['c1'] > 4./9.] = 4./9.
            if (self['c1'] < 0.).any():
                print("Warning: Returned compactnesses < 0 ... setting = 0.")
                self['c1'][self['c1'] < 0.0] = 0.0
                

            
            if (self['c2'] > 4./9.).any():
                print("Warning: Returned compactnesses > 4/9 = 0.44 ... setting = 4/9")
                self['c2'][self['c2'] > 4./9.] = 4./9.
            if (self['c2'] < 0.).any():
                print("Warning: Returned compactnesses < 0 ... setting = 0.")
                self['c2'][self['c2'] < 0.0] = 0.0
            return self



    def calc_baryonic_mass(self, EOS, TOV, fit=False):
        """
        if fit=True then the fit from
        Equation to relate EOS and neutron star mass to Baryonic mass
        Eq 8: https://arxiv.org/pdf/1708.07714.pdf
        """
        if fit:
            self["mb1"] = EOSfit(self["m1"], self["c1"])
            self["mb2"] = EOSfit(self["m2"], self["c2"])
            return self

        if TOV not in ['Monica', 'Wolfgang']:
            raise ValueError('You have provided a TOV '
                             'for which we have no data '
                             'and therefore cannot '
                             'calculate the Baryonic mass.')

        if EOS not in get_eos_list(TOV):
            raise ValueError('You have provided a EOS '
                            'for which we have no data '
                            'and therefore cannot '
                            'calculate the Baryonic mass.')

        if TOV == 'Monica':
            import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
            import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_mr.dat'), format='ascii')
            baryonic_mass_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['mb'])
            # after obtaining the baryonic_mass_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
            self['mb1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['mb'], baryonic_mass_of_mass_const)
            self['mb2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['mb'], baryonic_mass_of_mass_const)

        if TOV == 'Wolfgang':

            import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
            import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '.tidal.seq'), format='ascii')
            baryonic_mass_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['baryonic_mass'])
            # after obtaining the baryonic_mass_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
            self['mb1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['baryonic_mass'], baryonic_mass_of_mass_const)
            self['mb2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['baryonic_mass'], baryonic_mass_of_mass_const)

        return self


    def calc_radius(self, EOS, TOV, polytrope=False):
        """
        """
        if TOV not in ['Monica', 'Wolfgang', 'lalsim']:
            raise ValueError('You have provided a TOV '
                             'for which we have no data '
                             'and therefore cannot '
                             'calculate the radius.')

        if EOS not in get_eos_list(TOV):
            raise ValueError('You have provided a EOS '
                            'for which we have no data '
                            'and therefore cannot '
                            'calculate the radius.')

        if TOV == 'Monica':

            import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
            import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_mr.dat'), format='ascii')
            radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'])
            # after obtaining the radius_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
            # also radius is in km in table. need to convert to SI (i.e. meters)
            self['r1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3
            self['r2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3

        elif TOV == 'Wolfgang':

            import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
            import gwemlightcurves.EOS.TOV.Monica.eos_tools as et

            try:
                import lal
                G = lal.G_SI; c = lal.C_SI; msun = lal.MSUN_SI
            except:
                import astropy.units as u
                import astropy.constants as C
                G = C.G.value; c = C.c.value; msun = u.M_sun.to(u.kg)

            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '.tidal.seq'), format='ascii')
            radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'])
            unit_conversion = (msun * G / c**2)
            self['r1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'], radius_of_mass_const) * unit_conversion
            self['r2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'], radius_of_mass_const) * unit_conversion

        elif TOV == 'lalsim':
            import lalsimulation as lalsim
            if polytrope==True:
                try:
                    import lal
                    G = lal.G_SI; c = lal.C_SI; msun = lal.MSUN_SI
                except:
                    import astropy.units as u
                    import astropy.constants as C
                    G = C.G.value; c = C.c.value; msun = u.M_sun.to(u.kg)

                ns_eos, eos_fam=construct_eos_from_polytrope(EOS)
                self['r1'] = np.vectorize(lalsim.SimNeutronStarRadius)(self["m1"] * msun, eos_fam)
                self['r2'] = np.vectorize(lalsim.SimNeutronStarRadius)(self["m2"] * msun, eos_fam)

            else:
                import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
                import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
                MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_lalsim_mr.dat'), format='ascii')
                radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'])
                # after obtaining the radius_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
                # also radius is in km in table. need to convert to SI (i.e. meters)
                self['r1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3
                self['r2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3

        return self


    def calc_radius_and_epsilon_c(self, EOS, TOV):
        if TOV not in ['Monica', 'Wolfgang', 'lalsim']:
            raise ValueError('You have provided a TOV '
                             'for which we have no data '
                             'and therefore cannot '
                             'calculate the radius.')

        if EOS not in get_eos_list(TOV):
            raise ValueError('You have provided a EOS '
                            'for which we have no data '
                            'and therefore cannot '
                            'calculate the radius.')

        if TOV == 'Monica':

            import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
            import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
            import numpy as np
            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_mr.dat'), format='ascii')

            radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'])
            energy_density_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'],    np.log10(MassRadiusBaryMassTable['rho_c']))

            # after obtaining the radius_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
            # also radius is in km in table. need to convert to SI (i.e. meters)
            self['r1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3
            self['r2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3
            self['eps01'] = 10**(et.values_from_table(self['m1'], MassRadiusBaryMassTable['mass'], np.log10(MassRadiusBaryMassTable['rho_c']), energy_density_of_mass_const))
            self['eps02'] = 10**(et.values_from_table(self['m2'], MassRadiusBaryMassTable['mass'], np.log10(MassRadiusBaryMassTable['rho_c']), energy_density_of_mass_const))
   
        return self

    def downsample(self, Nsamples=100):
        """
        randomly down samples the number os posterior samples used for calculating lightcurves
        plotting etc
        """
        print('You are requesting to downsample the number of posterior samples to {0}'.format(Nsamples))
        idx = np.random.permutation(len(self))
        idx = idx[:Nsamples]
        return self[idx]

    @classmethod
    def plot_mag_panels(cls, table_dict, distance, filts=["g","r","i","z","y","J","H","K"],  magidxs=[0,1,2,3,4,5,6,7,8], figsize=(20, 28)):
        """
        This allows us to take the lightcurves from the KNModels samples table and plot it
        using a supplied set of filters. Default: filts=["g","r","i","z","y","J","H","K"]
        """
        # get legend determines the names to add to legend based on KN model
        def get_legend(model):

            if model == "DiUj2017":
                legend_name = "Dietrich and Ujevic (2017)"
            if model == "KaKy2016":
                legend_name = "Kawaguchi et al. (2016)"
            elif model == "Me2017":
                legend_name = "Metzger (2017)"
            elif model == "SmCh2017":
                legend_name = "Smartt et al. (2017)"
            elif model == "WoKo2017":
                legend_name = "Wollaeger et al. (2017)"
            elif model == "BaKa2016":
                legend_name = "Barnes et al. (2016)"
            elif model == "Ka2017":
                legend_name = "Kasen (2017)"
            elif model == "RoFe2017":
                legend_name = "Rosswog et al. (2017)"

            return legend_name

        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams.update({'font.size': 16})
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm


        # Initialize variables and arrays
        models = table_dict.keys()
        colors_names = cm.rainbow(np.linspace(0, 1, len(models)))
        tt = np.arange(table_dict[models[0]]['tini'][0], table_dict[models[0]]['tmax'][0] + table_dict[models[0]]['dt'][0], table_dict[models[0]]['dt'][0])

        # Initialize plot
        plt.figure(figsize = figsize)

        cnt = 0
        for filt, magidx in zip(filts, magidxs):
            cnt = cnt + 1
            vals = "%d%d%d"%(len(filts), 1, cnt)
            if cnt == 1:
                ax1 = plt.subplot(eval(vals))
            else:
                ax2 = plt.subplot(eval(vals), sharex=ax1, sharey=ax1)

            for ii, model in enumerate(models):
                legend_name = get_legend(model)

                magmed = np.median(table_dict[model]["mag_%s"%filt], axis=0)
                magmax = np.max(table_dict[model]["mag_%s"%filt], axis=0)
                magmin = np.min(table_dict[model]["mag_%s"%filt], axis=0)

                plt.plot(tt, magmed, '--', c=colors_names[ii], linewidth=2, label=legend_name)
                plt.fill_between(tt, magmin, magmax, facecolor=colors_names[ii], alpha=0.2)

            plt.ylabel('%s'%filt, fontsize=48, rotation=0, labelpad=40)
            plt.xlim([0.0, 14.0])
            plt.ylim([-18.0, -10.0])
            plt.gca().invert_yaxis()
            plt.grid()
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)

            if cnt == 1:
                ax1.set_yticks([-18,-16,-14,-12,-10])
                plt.setp(ax1.get_xticklabels(), visible=False)
                l = plt.legend(loc="upper right", prop={'size':24}, numpoints=1, shadow=True, fancybox=True)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)

                ax3 = ax1.twinx()    # mirror them
                ax3.set_yticks([16,12,8,4,0])
                app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(distance*1e6) - 1))
                ax3.set_yticklabels(app.astype(int))

                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
            else:
                ax4 = ax2.twinx()    # mirror them
                ax4.set_yticks([16,12,8,4,0])
                app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(distance*1e6) - 1))
                ax4.set_yticklabels(app.astype(int))

                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)

            if (not cnt == len(filts)) and (not cnt == 1):
                plt.setp(ax2.get_xticklabels(), visible=False)

        ax1.set_zorder(1)
        ax2.set_xlabel('Time [days]',fontsize=48)
        return plt

    def mass_cut(self, mass1=None,mass2=None,mtotmin=None,mtotmax=None):
        """
        Perform mass cut on table.
        """
        #print('You are requesting to remove samples with m1 above %.2f solar masses and m2 above %.2f solar masses'%(mass1,mass2))

        if not mass1 == None:
            idx = np.where(self["m1"] <= mass1)
            self = self[idx]
        if not mass2 == None:
            idx = np.where(self["m2"] <= mass2)
            self = self[idx]
        if not mtotmin == None:
            idx = np.where(self["m1"] + self["m2"] >= mtotmin)
            self = self[idx]
        if not mtotmax == None:
            idx = np.where(self["m1"] + self["m2"] <= mtotmax)
            self = self[idx]

        return self

    @classmethod
    def model(cls, format_, *args, **kwargs):
        """Fetch a table of events from a database

        Parameters
        ----------

        *args
            all other positional arguments are specific to the
            data format, see below for basic usage

        **kwargs
            all other positional arguments are specific to the
            data format, see the online documentation for more details


        Returns
        -------
        table : `KNTable`
            a table of events recovered from the remote database

        Examples
        --------

        Notes
        -----"""
        # standard registered fetch
        from .io.model import get_model
        model = get_model(format_, cls)
        return model(*args, **kwargs)
