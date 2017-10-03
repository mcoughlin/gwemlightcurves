.. _examples:

######################
Simulating lightcurves
######################

============
Introduction
============
The first thing you will need in order to generate a light curve is a system that is expected to have some mass ejecta. Once you have one of these systems you can calculate the masss ejects if you have information on the mass of the objects and if it is a binary nuetron star the compactness and baryonic masses of both systems. Here we display some ways to get information on the compactness and baryonic mass of neutron stars through using certain Equation of State (EOS)

Reading and using KNTable
-------------------------

Say you have run parameter estimation on a BNS signal

The `KNTable` object comes with a :meth:`KNTable.read_samples` method, allowing
trivial reading of samples::

    >>> from gwemlightcurves.KNModels import KNTable
    >>> t = KNTable.read_samples('posterior_samples.dat')

The results should look like this::

    >>> print(t)
       dlambdat     costheta_jn   ...      dec       matched_filter_snr
    -------------- -------------- ... -------------- ------------------
    -424.431604646 0.848696760069 ... 0.577824127959      33.5477773932
    -147.120517819 0.991997169733 ... 0.576384671836      33.5566982929
    -240.912505017 0.976047933498 ... 0.577896147804      33.6263488512
    -186.290687306 0.727047513491 ... 0.582878134524      33.4518918366
    -210.457602464 0.967621722464 ... 0.580780890411      33.5696948608
    0.377168713485 0.653066039319 ...  0.57000694353      33.5307165083
    -350.294734561 0.891785128102 ... 0.585910104721      33.5685077643
               ...            ... ...            ...                ...
    -37.5863972893 0.975612607455 ... 0.575280574082      33.6140080488
    -128.526058992 0.698513990317 ... 0.581271675244      33.5713366289
    -10.3419873355 0.971274339543 ... 0.576536076406      33.6291236021
     202.992550229 0.967479803846 ... 0.571265399682      33.5992585584
    -21.5943515138 0.673295017122 ... 0.578523325358      33.3765394973
     82.6149783838 0.941362067895 ... 0.582807999184      33.5504875571
     -490.19334394  0.83602530718 ... 0.564811075217      33.6087938033
    Length = 1996 rows


After loading the table, one can caluclate lambda1 and lambda2 from dtilde if it is not already in the samples.
:meth:`~KNTable.calc_tidal_lambda`::

    >>> t = t.calc_tidal_lambda(remove_negative_lambda=True)

After accomplishing the reading of the sample, let's say we want to calculate 
compactness from radius. This would require calculating the radius from a mass radius curve. We can use :meth:`~KNTable.calc_radius`. In this module we have a number of ways to accomplish this::


    >>> t_sly_mon = t.calc_radius(EOS='sly', TOV='Monica')
    >>> t_sly_wolf = t.calc_radius(EOS='sly', TOV='Wolfgang')
    >>> t_sly_lalsim = t.calc_radius(EOS='sly', TOV='lalsim')


After this we can now calculate the compactness :meth:`~KNTable.calc_compactness`.::

    >>> t_sly_mon = t_sly_mon.calc_compactness()
    >>> t_sly_wolf = t_sly_wolf.calc_compactness()
    >>> t_sly_lalsim = t_sly_lalsim.calc_compactness()

After this we can calulcate the baryonic mass. Now we can either use the calculated compactness and have it be EOS dependent of calculate the baryonic mass using a fit using :meth:`~KNTable.calc_baryonic_mass`::

    >>> t_sly_mon = t_sly_mon.calc_baryonic_mass(EOS='ap4', TOV='Monica')
    >>> t_sly_wolf = t_sly_wolf.calc_baryonic_mass(EOS='ap4', TOV='Wolfgang')
    >>> t_sly_lalsim = t_sly_lalsim.calc_baryonic_mass(EOS='ap4', TOV='lalsim')
    >>> t_sly_mon_bary_from_fit = t_sly_mon.calc_baryonic_mass(EOS=None, TOV=None, fit=True)


Calculating Compactness
-----------------------

Let's Demonstrate some of the differences between calculating compactness from fit (i.e. being EOS agnostic) versus calculating it from a EOS.

.. plot::
   :include-source:

    >>> from gwemlightcurves.KNModels import KNTable
    >>> from gwpy.table import EventTable
    >>> t = KNTable.read_samples('posterior_samples.dat')
    >>> t = t.calc_tidal_lambda(remove_negative_lambda=True)
    >>> t_sly_mon = t.calc_radius(EOS='sly', TOV='Monica'); t_H4_mon = t.calc_radius(EOS='H4', TOV='Monica'); t_mpa1_mon = t.calc_radius(EOS='mpa1', TOV='Monica'); t_ms1_mon = t.calc_radius(EOS='ms1', TOV='Monica'); t_ms1b_mon = t.calc_radius(EOS='ms1b', TOV='Monica');
    >>> t_sly_mon = t_sly_mon.calc_compactness(); t_H4_mon = t_H4_mon.calc_compactness(); t_mpa1_mon = t_mpa1_mon.calc_compactness(); t_ms1_mon = t_ms1_mon.calc_compactness(); t_ms1b_mon = t_ms1b_mon.calc_compactness()
    >>> t_comp_fit = t.calc_compactness(fit=True)
    >>> t_sly_mon = EventTable(t_sly_mon); t_H4_mon = EventTable(t_H4_mon); t_mpa1_mon = EventTable(t_mpa1_mon); t_ms1_mon = EventTable(t_ms1_mon); t_ms1b_mon = EventTable(t_ms1b_mon); t_comp_fit = EventTable(t_comp_fit);
    >>> plot = t_sly_mon.hist('c1', bins=20, histtype='stepfilled', label='Compactness Monica Sly')
    >>> ax = plot.gca()
    >>> ax.hist(t_H4_mon['c1'], logbins=True, bins=20, histtype='stepfilled', label='Compactness Monica H4'); ax.hist(t_mpa1_mon['c1'], logbins=True, bins=20, histtype='stepfilled', label='Compactness Monica mpa1'); ax.hist(t_ms1_mon['c1'], logbins=True, bins=20, histtype='stepfilled', label='Compactness Monica ms1'); ax.hist(t_ms1b_mon['c1'], logbins=True, bins=20, histtype='stepfilled', label='Compactness Monica ms1b'); ax.hist(t_comp_fit['c1'], logbins=True, bins=20, histtype='stepfilled', label='Compactness From Fit')
    >>> ax.set_xlabel('Compactness')
    >>> ax.set_ylabel('Rate')
    >>> ax.set_title('Compactness Values')
    >>> plot.add_legend()
    >>> ax.autoscale(axis='x', tight=True)

Calculating Baryonic Mass
-------------------------

Let's demonstrate some of the differences between calculating the baryonic_mass from fit versus calculating it from an EOS table.

.. plot::
   :include-source:

    >>> from gwemlightcurves.KNModels import KNTable
    >>> from gwpy.table import EventTable
    >>> t = KNTable.read_samples('posterior_samples.dat')
    >>> t = t.calc_tidal_lambda(remove_negative_lambda=True)
    >>> t_sly_mon = t.calc_radius(EOS='sly', TOV='Monica'); t_H4_mon = t.calc_radius(EOS='H4', TOV='Monica'); t_mpa1_mon = t.calc_radius(EOS='mpa1', TOV='Monica'); t_ms1_mon = t.calc_radius(EOS='ms1', TOV='Monica'); t_ms1b_mon = t.calc_radius(EOS='ms1b', TOV='Monica');
    >>> t_sly_mon = t_sly_mon.calc_compactness(); t_H4_mon = t_H4_mon.calc_compactness(); t_mpa1_mon = t_mpa1_mon.calc_compactness(); t_ms1_mon = t_ms1_mon.calc_compactness(); t_ms1b_mon = t_ms1b_mon.calc_compactness()
    >>> t_sly_mon = t_sly_mon.calc_baryonic_mass(EOS='sly', TOV='Monica'); t_H4_mon = t_H4_mon.calc_baryonic_mass(EOS='H4', TOV='Monica'); t_mpa1_mon = t_mpa1_mon.calc_baryonic_mass(EOS='mpa1', TOV='Monica'); t_ms1_mon = t_ms1_mon.calc_baryonic_mass(EOS='ms1', TOV='Monica'); t_ms1b_mon = t_ms1b_mon.calc_baryonic_mass(EOS='ms1b', TOV='Monica')
    >>> t_sly_mon_bary_fit = t_sly_mon.calc_baryonic_mass(EOS=None, TOV=None, fit=True); t_H4_mon_bary_fit = t_H4_mon.calc_baryonic_mass(EOS=None, TOV=None, fit=True); t_mpa1_mon_bary_fit = t_mpa1_mon.calc_baryonic_mass(EOS=None, TOV=None, fit=True); t_ms1_mon_bary_fit = t_ms1_mon.calc_baryonic_mass(EOS=None, TOV=None, fit=True); t_ms1b_mon_bary_fit = t_ms1b_mon.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
    >>> t_sly_mon = EventTable(t_sly_mon); t_H4_mon = EventTable(t_H4_mon); t_mpa1_mon = EventTable(t_mpa1_mon); t_ms1_mon = EventTable(t_ms1_mon); t_ms1b_mon = EventTable(t_ms1b_mon); t_sly_mon_bary_fit = EventTable(t_sly_mon_bary_fit); t_H4_mon_bary_fit = EventTable(t_H4_mon_bary_fit); t_mpa1_mon_bary_fit = EventTable(t_mpa1_mon_bary_fit); t_ms1_mon_bary_fit = EventTable(t_ms1_mon_bary_fit); t_ms1b_mon_bary_fit = EventTable(t_ms1b_mon_bary_fit)
    >>> plot = t_sly_mon.plot('m1','mb1', label='M1 MB1 Monica Sly Bary From Table')
    >>> ax = plot.gca()
    >>> ax.scatter(t_sly_mon_bary_fit['m1'], t_sly_mon_bary_fit['mb1'], label='M1 MB1 Monica Sly Bary From Fit'); ax.scatter(t_H4_mon['m1'], t_H4_mon['mb1'], label='M1 MB1 Monica H4 Bary From Table'); ax.scatter(t_H4_mon_bary_fit['m1'], t_H4_mon_bary_fit['mb1'], label='M1 MB1 Monica H4 Bary From Fit'); ax.scatter(t_ms1_mon['m1'], t_ms1_mon['mb1'], label='M1 MB1 Monica ms1 Bary From Table'); ax.scatter(t_ms1_mon_bary_fit['m1'], t_ms1_mon_bary_fit['mb1'], label='M1 MB1 Monica ms1 Bary From Fit')
    >>> ax.set_xlabel('M1')
    >>> ax.set_ylabel('MB1')
    >>> ax.set_title('M1 by MB1')
    >>> plot.add_legend()
    >>> ax.autoscale(axis='x', tight=True)


Generating Light Curves
-----------------------

Finally, let's calculate a lightcurve being EOS agnostic. That is, we calculate both the compactness and baryonic masses from fits. Also let us look at a Metzer 2017 and DiUj2017 models.

.. plot::
   :include-source:

    >>> from gwemlightcurves.KNModels import KNTable
    >>> from gwemlightcurves import lightcurve_utils
    >>> t = KNTable.read_samples('posterior_samples.dat')
    >>> t = t.calc_tidal_lambda(remove_negative_lambda=True)
    >>> t = t.calc_compactness(fit=True)
    >>> t = t.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
    >>> t = t.downsample(Nsamples=100)
    >>> tini = 0.1; tmax = 50.0; dt = 0.1; vmin = 0.02; th = 0.2; ph = 3.14; kappa = 1.0; eps = 1.58*(10**10); alp = 1.2; eth = 0.5; flgbct = 1; beta = 3.0; kappa_r = 1.0; slope_r = -1.2; theta_r = 0.0; Ye = 0.3
    >>> t['tini'] = tini; t['tmax'] = tmax; t['dt'] = dt; t['vmin'] = vmin; t['th'] = th; t['ph'] = ph; t['kappa'] = kappa; t['eps'] = eps; t['alp'] = alp; t['eth'] = eth; t['flgbct'] = flgbct; t['beta'] = beta; t['kappa_r'] = kappa_r; t['slope_r'] = slope_r; t['theta_r'] = theta_r; t['Ye'] = Ye

    >>> # Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
    >>> models = ["DiUj2017","Me2017"]
    >>> model_tables = {}
    >>> for model in models:
    >>>     model_tables[model] = KNTable.model(model, t)
    >>> # Now we need to do some interpolation
    >>> for model in models:
    >>>    model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model])
    >>>    model_tables[model] = lightcurve_utils.interpolate_mags_lbol(model_tables[model])

    >>> distance = 100 #Mpc
    >>> plot = KNTable.plot_mag_panels(model_tables, distance=distance)
    >>> plot.show()
