# -*- coding: utf-8 -*-
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gwemlightcurves.  If not, see <http://www.gnu.org/licenses/>.

"""Extend :mod:`astropy.table` with the `KNTable`
"""

import numpy as np

from astropy.table import (Table, Column, vstack)
from distutils.spawn import find_executable

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'
__all__ = ['KNTable', 'tidal_lambda_from_tilde', 'CLove', 'EOSfit', 'get_eos_list']


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


class KNTable(Table):
    """A container for a table of events

    This differs from the basic `~astropy.table.Table` in two ways

    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `KNTable`
    """
    # -- i/o ------------------------------------
    @classmethod
    def read_samples(cls, filename_samples):
        """
        Read LALinference posterior_samples
        """
        import os
        if not os.path.isfile(filename_samples):
            raise ValueError("Sample file supplied does not exist")

        data_out = Table.read(filename_samples, format='ascii')

        if 'm1_source' in list(data_out.columns):
            data_out['m1'] = data_out['m1_source']
            print 'setting m1 to m1_source'
        if 'm2_source' in list(data_out.columns):
            data_out['m2'] = data_out['m2_source']
            print 'setting m2 to m2_source'

        if 'dlam_tilde' in list(data_out.columns):
            data_out['dlambdat'] = data_out['dlam_tilde']
            print 'setting dlambdat to dlam_tilde'
        if 'lam_tilde' in list(data_out.columns):
            data_out['lambdat'] = data_out['lam_tilde']
            print 'setting lambdat to lam_tilde'

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
            print 'You have requested to remove negative lambda values'
            mask = (self["lambda1"] < 0) | (self["lambda2"] < 0)
            self = self[~mask]
            print "Removing %d/%d due to negative lambdas"%(np.sum(mask),len(mask))

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
            print 'You have chose to calculate compactness from fit.'
            print 'you are therefore choosing to be EOS agnostic'
            self["c1"] = CLove(self["lambda1"])
            self["c2"] = CLove(self["lambda2"])
        else:
            print 'You have chose to calculate compactness from radius.'
            print 'you are therefore must have selected a EOS'
            self['c1'] = self['m1'] / self['r1'] * G / c**2 * msun
            self['c2'] = self['m2'] / self['r2'] * G / c**2 * msun
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


    def calc_radius(self, EOS, TOV):
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
                G = lal.G_SI; c = C.c.value; msun = u.M_sun.to(u.kg)

            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '.tidal.seq'), format='ascii')
            radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'])
            unit_conversion = (msun * G / c**2)
            self['r1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'], radius_of_mass_const) * unit_conversion
            self['r2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'], radius_of_mass_const) * unit_conversion

        elif TOV == 'lalsim':

            import lalsimulation as lalsim
            MassRadiusiBaryMassTable = Table.read(find_executable(EOS + '_lalsim_mr.dat'), format='ascii')
            radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'])
            # after obtaining the radius_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
            # also radius is in km in table. need to convert to SI (i.e. meters)
            self['r1'] = et.values_from_table(self['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3
            self['r2'] = et.values_from_table(self['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const)*10**3

        return self


    def downsample(self, Nsamples=100):
        """
        randomly down samples the number os posterior samples used for calculating lightcurves
        plotting etc
        """
        print('You are requesting to downsample the number of posterior samples to {0}'.format(Nsamples))
        idx = np.random.permutation(len(self["m1"]))
        idx = idx[:Nsamples]
        return self[idx]


    def plot_mag_panels(self, model, color_name='blue', filts=["g","r","i","z","y","J","H","K"], name='mag_panels', figsize=(20, 28)):
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
        colors = cm.rainbow(np.linspace(0, 1, len(filts)))
        magidxs, filts  = enumerate(filters)
        tt = np.arange(self['tini'], self['tmax'] + self['dt'], self['dt'])
        # Initialize empty arrays for all the mags
        for filt, color, magidx in zip(filts,colors,magidxs):
            mag_all[model][filt] = np.empty((0,len(tt)))

        # Initialize plot
        plotName = "{1}.pdf".format(name)
        plt.figure(figsize = figsize)

        cnt = 0
        for filt, color, magidx in zip(filts, colors, magidxs):
            cnt = cnt + 1
            vals = "%d%d%d"%(len(filts), 1, cnt)
            if cnt == 1:
                ax1 = plt.subplot(eval(vals))
            else:
                ax2 = plt.subplot(eval(vals), sharex=ax1, sharey=ax1)

            for ii, model in enumerate(models):
                legend_name = get_legend(model)

                magmed = np.median(mag_all[model][filt],axis=0)
                magmax = np.max(mag_all[model][filt],axis=0)
                magmin = np.min(mag_all[model][filt],axis=0)

                plt.plot(tt, magmed, '--', c=colors_names[ii], linewidth=2, label=legend_name)
                plt.fill_between(tt, magmin, magmax, facecolor=colors_names[ii], alpha=0.2)
            plt.ylabel('%s'%filt, fontsize=48, rotation=0, labelpad=40)
            plt.xlim([0.0, 14.0])
            plt.ylim([-18.0, -10.0])
            plt.gca().invert_yaxis()
            plt.grid()
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)


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

