.. _examples:

######################
Simulating lightcurves
######################

============
Introduction
============
The first thing you will need in order to generate a light curve is a system that is expected to have some mass ejecta. Once you have one of these systems you can calculate the masss ejects if you have information on the mass of the objects and if it is a binary nuetron star the compactness and baryonic masses of both systems. Here we display some ways to get information on the compactness and baryonic mass of neutron stars through using certain Equation of State (EOS)

Reading
-------

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

    >>> t_sly_mon = t_sly_mon.calc_compactness(EOS='ap4', TOV='Monica')
    >>> t_sly_wolf = t_sly_wolf.calc_compactness(EOS='ap4', TOV='Wolfgang')
    >>> t_sly_lalsim = t_sly_lalsim.calc_compactness(EOS='ap4', TOV='lalsim')

After this we can calulcate the baryonic mass. Now we can either use the calculated compactness and have it be EOS dependent of calculate the baryonic mass using a fit using :meth:`~KNTable.calc_baryonic_mass`::

    >>> t_sly_mon = t_sly_mon.calc_baryonic_mass(EOS='ap4', TOV='Monica')
    >>> t_sly_wolf = t_sly_wolf.calc_baryonic_mass(EOS='ap4', TOV='Wolfgang')
    >>> t_sly_lalsim = t_sly_lalsim.calc_baryonic_mass(EOS='ap4', TOV='lalsim')
    >>> t_sly_mon_bary_from_fit = t_sly_mon.calc_baryonic_mass(EOS='ap4', TOV='Monica', fit=True)
