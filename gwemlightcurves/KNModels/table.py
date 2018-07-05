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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
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
__all__ = ['KNTable', 'tidal_lambda_from_tilde', 'CLove', 'EOSfit', 'get_eos_list', 'get_lalsim_eos', 'construct_eos_from_polytrope']


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

	eos_indx=np.where(polytrope_table['eos']==eos)[0][0]

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
	print "Checking %s" % eos_name
	eos_fname = ""
	if os.path.exists(eos_name):
		# NOTE: Adapted from code by Monica Rizzo
		print "Loading from %s" % eos_name
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

		print "Dumping to %s" % eos_fname
		eos_fname = "./." + eos_name + ".dat"
		np.savetxt(eos_fname, np.transpose((press, edens)), delimiter='\t')
		eos = lalsimulation.SimNeutronStarEOSFromFile(eos_fname)
		fam = lalsimulation.CreateSimNeutronStarFamily(eos)

	else:
		eos = lalsimulation.SimNeutronStarEOSByName(eos_name)
		fam = lalsimulation.CreateSimNeutronStarFamily(eos)

	mmass = lalsimulation.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
	print "Family %s, maximum mass: %1.2f" % (eos_name, mmass)
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

        @classmethod
        def read_cbc_list(cls, filename_samples):
                """
                Read CBC list
                """
                import os
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
                elif model == "Ka2017_TrPi2018":
                        names = ["t0","mej","vej","Xlan","theta_v","E0","theta_c","theta_w","n","p","epsilon_E","epsilon_B","zp", 'loglikelihood']
                elif model == "Ka2017_A":
                        names=['t0', 'mej', 'vej', 'Xlan', 'A', 'zp', 'loglikelihood']
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
		elif model == "Ka2017x2":
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
				self['r1']=lalsim.SimNeutronStarRadius(self['m1']*msun, eos_fam)
				self['r2']=lalsim.SimNeutronStarRadius(self['m2']*msun, eos_fam)

			else:
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
			energy_density_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'],	np.log10(MassRadiusBaryMassTable['rho_c']))

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

				ax3 = ax1.twinx()	# mirror them
				ax3.set_yticks([16,12,8,4,0])
				app = np.array([-18,-16,-14,-12,-10])+np.floor(5*(np.log10(distance*1e6) - 1))
				ax3.set_yticklabels(app.astype(int))

				plt.xticks(fontsize=28)
				plt.yticks(fontsize=28)
			else:
				ax4 = ax2.twinx()	# mirror them
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
