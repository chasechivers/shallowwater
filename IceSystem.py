# Author: Chase Chivers
# Last updated: 7/10/19
# Modular build for 2d heat diffusion problem
#   applied to liquid water in the ice shell of Europa

import numpy as np
from scipy import optimize
from HeatSolver import HeatSolver


# Comment out for pace runs
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import seaborn as sns
# sns.set(palette='colorblind', color_codes=1, context='notebook', style='ticks')


class IceSystem(HeatSolver):
	"""
	Class with methods to set up initial conditions for two-dimensional, two-phase thermal diffusion model that
	includes temperature-dependent conductivity and salinity. Includes the HeatSolver class used to solve the heat
	equation utilizing an enthalpy method (Huber et al., 2008) to account for latent heat from phase change as well
	as a parameterization for a saline system.
	"""
	def __init__(self, Lx, Lz, dx, dz, kT=True, cpT=False, use_X_symmetry=False):
		"""
		Initialize the system.
		Parameters:
			Lx : float
				length of horizontal spatial domain, m
			Lz : float
				thickness of shell, length of vertical spatial domain, m
			dx : float
				horizontal spatial step size, m
			dz : float
				vertical spatial step size, m
			cpT : bool
			    choose whether to use temperature-depedent specific heat,
			    default = False.
			    True: temperature-dependent, cp_i ~ 185 + 7*T (Hesse et al., 2019)
			kT : bool
			    choose whether to use temperature-dependent thermal conductivity,
			    default = True, temperature-dependent, k=ac/T (Petrenko, Klinger, etc.)
			use_X_symmetry : bool
				assume the system is symmetric about the center of the intrusion
				* NOTE: Must use Reflecting boundary condition for sides if using this
			issalt : bool
				declare whether salinity will be used in this system, necessary for declaring fit functions and
				melting temperature calculations
		Usage:
			Ice Shell is 40 km thick and 40 km wide at a spatial discretization of 50 m.
				model = IceSystem(40e3, 40e3, 50, 50)

			See README
		"""

		self.Lx, self.Lz = Lx, Lz
		self.dx, self.dz = dx, dz
		self.nx, self.nz = int(Lx / dx + 1), int(Lz / dz + 1)
		self.Z = np.linspace(0, self.Lz, self.nz)  # z domain starts at zero, z is positive down
		if use_X_symmetry:
			self.symmetric = True
			self.Lx = self.Lx / 2
			self.nx = int(self.Lx / self.dx + 1)
			self.X = np.linspace(0, self.Lx, self.nx)
			self.X, self.Z = np.meshgrid(self.X, self.Z)  # create spatial grid
		elif use_X_symmetry is False:
			self.X = np.linspace(-self.Lx / 2, self.Lx / 2, self.nx)  # x domain centered on 0
			self.X, self.Z = np.meshgrid(self.X, self.Z)  # create spatial grid
		self.T = np.ones((self.nz, self.nx))  # initialize domain at one temperature
		self.S = np.zeros((self.nz, self.nx))  # initialize domain with no salt
		self.phi = np.zeros((self.nz, self.nx))  # initialize domain as ice
		self.kT, self.cpT = kT, cpT  # k(T), cp_i(T) I/O
		self.issalt = False  # salt I/O

	class constants:
		"""
		No-methods class used for defining constants in a simulation. May be changed inside here or as an
		instance during simulation runs.
		"""
		styr = 3.154e7  # s/yr, seconds in a year

		g = 1.32  # m/s2, Europa surface gravity

		# Thermal properties
		rho_i = 917.  # kg/m3, pure ice density
		rho_w = 1000.  # kg/m3 pure water density
		cp_i = 2.11e3  # J/kgK, pure ice specific heat
		cp_w = 4.19e3  # J/kgK, pure water specific heat
		ki = 2.3  # W/mK, pure ice thermal conductivity
		kw = 0.56  # W/mK, pure water thermal conductivity
		ac = 567  # W/m, ice thermal conductivity constant, ki = ac/T
		Tm = 273.15  # K, pure ice melting temperature at 1 atm
		Lf = 333.6e3  # J/kg, latent heat of fusion of ice
		expans = 1.6e-4  # 1/K, thermal expansivity

		rho_s = 0.  # kg/m3, salt density, assigned only when salinity is used

		# Radiation properties
		emiss = 0.97  # pure ice emissivity
		stfblt = 5.67e-8  # W/m2K4 stefan-boltzman constant

		# Constants for viscosity dependent tidalheating
		#   from Mitri & Showman (2005)
		act_nrg = 26.  # activation energy for diffusive regime
		Qs = 60e3  # J/mol, activation energy of ice (Goldsby & Kohlstadt, 2001)
		Rg = 8.3144598  # J/K*mol, gas constant
		eps0 = 1e-5  # maximum tidal flexing strain
		omega = 2.5e-5  # 1/s, tidal flexing frequency
		visc0i = 1e13  # Pa s, minimum reference ice viscosity at T=Tm
		visc0w = 1.3e-3  # Pa s, dynamic viscosity of water at 0 K

		# Mechanical properties of ice
		G = 3.52e9  # Pa, shear modulus/rigidity (Moore & Schubert, 2000)
		E = 1e6  # Pa, Young's Modulus

	def save_initials(self):
		""" Save initial values to compare with simulation results. """
		self.T_initial = self.T.copy()
		self.Tm_initial = self.Tm.copy()
		self.phi_initial = self.phi.copy()
		self.S_initial = self.S.copy()

		if self.kT:
			self.k_initial = self.phi_initial * self.constants.kw + (1 - self.phi_initial) * \
			                 self.constants.ac / self.T_initial
		else:
			self.k_initial = self.phi_initial * self.constants.kw + (1 - self.phi_initial) * self.constants.ki

	def init_volume_averages(self):
		"""
		Initialize volume averaged values over the domain. In practice, this is automatically called by any future
		function that are changing physical parameters such as liquid fraction, salinity or temperature.
		"""
		if self.kT:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * self.constants.ac / self.T + self.phi * self.constants.kw

		if self.cpT is True:
			# what comes out given this function and a time derivative of rhoc
			self.cp_i = 185. + 2 * 7.037 * self.T
		else:
			self.cp_i = self.constants.cp_i

		# this is very unimportant overall
		if self.issalt:
			self.rhoc = (1 - self.phi) * (self.constants.rho_i + self.Ci_rho * self.S) * self.cp_i \
			            + self.phi * (self.constants.rho_w + self.C_rho * self.S) * self.constants.cp_w

		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w

		self.save_initials()

	def init_T(self, Tsurf, Tbot, profile='non-linear', real_Lz=0):
		"""
		Initialize temperature profile
			Parameters:
				Tsurf : float
					surface temperature
				Tbot : float
					temperature at bottom of domain
				profile : string
					-> defaults to 'non-linear'
					prescribed temperature profile
					'non-linear' -- expected equilibrium thermal gradient with k(T)
					'linear'     -- equilibirium thermal gradient for constant k
					'stefan'     -- sets up the freezing stefan problem temperature profile
									in this instance, Tbot should be the melting temperature
				real_Lz : float
					used if you want to simulate some portion of a much larger shell, so this parameter is used to
					make the temperature profile that of the much larger shell than the one being simulated.
					For example, a 40 km conductive shell (real_Lz = 40e3) discretized at 10 m can be computationally
					expensive.However, if we assume that any temperature anomaly at shallow depths (~1-5km) won't
					reach to 40km within the model time, we can reduce the computational domain to ~5km to speed up
					the simulation. This will take the Tbot as the Tbot of a 40km and find the temperature at 5km to
					account for the reduced domain size.
					Usage case down below
			Returns:
				T : (nz,nx) grid
					grid of temperature values

			Usage :
				Default usage:
					model.init_T(Tsurf=75, Tbot=273.15)

				Linear profile:
					model.init_T(Tsurf = 50, Tbot = 273.15, profile='linear')

				Cheated domain:
					realLz = 50e3
					modelLz = 5e3
					model = IceSystem(Lz=modelLz, ...)
					model.init_T(Tsurf=110,Tbot=273.15,real_lz=realLz)
		"""
		# set melting temperature to default
		self.Tm = self.constants.Tm * self.T.copy()

		if isinstance(profile, str):
			if profile == 'non-linear':
				if real_Lz > 0:
					Tbot = Tsurf * (Tbot / Tsurf) ** (self.Lz / real_Lz)
				self.T = Tsurf * (Tbot / Tsurf) ** (abs(self.Z / self.Lz))

			elif profile == 'linear':
				if real_Lz > 0:
					Tbot = (Tbot - Tsurf) * (self.Lz / real_Lz) + Tsurf
				self.T = (Tbot - Tsurf) * abs(self.Z / self.Lz) + Tsurf

			elif profile == 'stefan':
				self.T[0, :] = Tsurf  # set the very top of grid to surface temperature
				self.T[1:, :] = Tbot  # everything below is at the melting temperature
				self.phi[:, :] = 1  # everything starts as liquid
				profile += ' plus domain all water'

			print('init_T(Tsurf = {}, Tbot = {})'.format(Tsurf, Tbot))
			print('\t Temperature profile initialized to {}'.format(profile))

		else:
			self.T = profile
			print('init_T: custom profile implemented')

		# save boundaries for dirichlet or other
		# left and right boundaries
		self.Tedge = self.T[:, 0] = self.T[:, -1]
		self.Tsurf = Tsurf
		self.Tbot = Tbot
		self.init_volume_averages()

	def set_intrusion_geom(self, depth, thickness, radius, geometry='ellipse'):
		"""
		Sets geometry of intrusion. In practice, is automatically called by init_intrusion() and generally unneeded
		to be called in simulation script. Creates tuple IceSystem.geom that holds the initial intrusion grid indices for
		manipulation inside simulation and outside for more customization.
		"""

		if isinstance(geometry, str):
			if geometry == 'ellipse':
				center = thickness / 2 + depth
				try:
					if self.symmetric:  # adjust geometry to make sure the center of the intrusion isn't on the boundary
						_R_ = self.X - self.dx
						thickness += self.dz
				except AttributeError:
					_R_ = self.X
				self.geom = np.where((_R_ / radius) ** 2 + (self.Z - center) ** 2 / ((thickness / 2) ** 2) <= 1)
				del center, _R_
			elif geometry == 'box':
				try:
					if self.symmetric:  # adjust geometry to make sure the center of the intrusion isn't on the boundary
						radius += self.dx
				except AttributeError:
					radius = radius
				r = np.where(abs(self.X[0, :]) <= radius)[0]
				z = np.intersect1d(np.where(self.Z[:, 0] <= thickness + depth), np.where(self.Z[:, 0] >= depth))
				tmp = np.zeros(np.shape(self.T))
				tmp[z.min():z.max(), r.min():r.max() + 1] = 1
				self.geom = np.where(tmp == 1)
				del tmp, r, z

		# option for a custom geometry
		else:
			self.geom = geometry

	def init_intrusion(self, T, depth, thickness, radius, phi=1, geometry='ellipse'):
		"""
		Initialize intrusion properties. Updates volume averages after initialization: means we can just initialize
		temperature and intrusion to get all thermal properties set.
		**So far this only accounts for a single intrusion at the center of the domain
			should be simple to add multiples in the future if necessary

		Parameters:
			T : float
				set intrusion to single Temperature value, assuming that it is well mixed
			depth : float
				set depth of upper edge of the intrusion, m
			thickness : float
				set thickness of intrusion, m
			radius : float
				set radius of intrusion, m
			phi : float [0,1]
				set liquid fraction of intrusion, generally interested in totally liquid bodies so default = 1
			geometry : string (see set_intrusion_geom()), array
				set geometry of intrusion, default is an ellipse

		Usage:
			Intrusion at pure water melting temperature (273.15 K), emplaced at 2 km depth in the shell, 2 km thick
			and a radius of 4 km:
				model.init_intrusion(T=273.15, depth=2e3, thickness=2e3, radius=4e3)
		"""

		if phi < 0 or phi > 1:
			raise Exception('liquid fraction must be between 0 and 1')

		# save intrusion properties
		self.T_int = T
		self.depth, self.thickness, self.R_int = depth, thickness, radius
		self.set_intrusion_geom(depth, thickness, radius, geometry)  # get chosen geometry
		self.T[self.geom] = T  # set intrusion temperature to chosen temperature
		self.phi[self.geom] = phi  # set intrusion to chosen liquid fraction
		self.init_volume_averages()  # update volume averages

	# define a bunch of useful functions for salty systems, unused otherwise
	# non-linear fit, for larger dT
	def shallow_fit(self, dT, a, b, c, d):
		return a + b * (dT + c) * (1 - np.exp(-d / dT)) / (1 + dT)

	# linear fit, for small dT
	def linear_fit(self, dT, a, b):
		return a + b * dT

	# FREEZCHEM quadratic fit for liquidus curve
	def Tm_func(self, S, a, b, c):
		return a * S ** 2 + b * S + c

	def init_salinity(self, S=None, composition='MgSO4', concentration=12.3, rejection_cutoff=0.25, shell=False,
	                  in_situ=False, T_match=True):
		"""
		Initialize salinity properties for simulations.
		Parameters:
			S : (nz,nx) grid
				Necessary for a custom background salinity or other configurations, e.g. a super saline layer
				-> though this could be done outside of this command so....
			composition : string
				Choose which composition the liquid should be.
				Options: 'MgSO4', 'NaCl'
			concentration : float
				Initial intrusion concentration and/or ocean concentration; if using the shell option (below),
				this assumes that the shell was frozen out of an ocean with this concentration and composition
			rejection_cutoff : float > 0
				Liquid fraction (phi) below which no more salt will be accepted into the remaining liquid or
				interstitial liquid. Note: should be greater than 0
			shell : bool
				Option to include background salinity in the shell given the chosen composition and concentration.
				This will automatically adjust the temperature profile to account for a salty ocean near the melting
				temperature. If assuming something else, such as a slightly cooler convecting layer between the
				brittle shell and the ocean, this can be adjusted afterward by calling init_T()
			in_situ : bool
				Note: shell option must be True.
				Assumes the intrusion is from an event that melted the shell in-situ, thus have the same
				concentration and composition as the shell at that depth.
			T_match : bool
				Option to adjust the temperature profile to make the bottom be at the melting temperature of an ocean
				with the same composition and concentration. This is mostly used if making the assumption that the
				brittle layer simulated here is directly above the ocean.

		Usage:
			Pure shell, saline intrusion: Intrusion with 34 ppt NaCl salinity
				model.init_intrusion(composition='NaCl',concentration=12.3)

			Saline shell, in-situ melting: Ocean began with 100 ppt MgSO4 and intrusion has been created by in-situ
			melting
				model.init_intrusion(composition='MgSO4', concentration=100., shell=True, in_situ=True)
		"""

		self.issalt = True  # turn on salinity for solvers
		self.saturated = 0  # whether liquid is saturated
		self.rejection_cutoff = rejection_cutoff  # minimum liquid fraction of cell to accept rejected salt

		# composition and concentration coefficients for fits from Buffo et al. (2019)
		# others have been calculated by additional runs using the model from Buffo et al. (2019)

		# dic structure {composition: [a,b,c]}
		# Liquidus curves derived from Liquius 1.0 (Buffo et al. 2019 and FREEZCHEM) for MgSO4 and NaCl
		self.Tm_consts = {'MgSO4': [-1.333489497e-5, -0.01612951864, 273.055175687],
		                  'NaCl': [-9.1969758e-5, -0.03942059, 272.63617665]
		                  }

		# dict structure {composition: {concentration: [a,b,c,d]}}
		self.shallow_consts = {'MgSO4': {0: [0., 0., 0., 0.],
		                                 12.3: [12.21, -8.3, 1.836, 20.2],
		                                 100: [22.19, -11.98, 1.942, 21.91],
		                                 282: [30.998, -11.5209, 2.0136, 21.1628]},
		                       'NaCl': {0: [0., 0., 0., 0.],
		                                10: [7.662, -4.936, 2.106, 24.8],
		                                34: [11.1, -4.242, 1.91, 22.55],
		                                100: [0., 0., 0., 0.],
		                                260: [0., 0., 0., 0.]}
		                       }

		# dict structure {composition: {concentration: [a,b]}}
		self.linear_consts = {'MgSO4': {0: [0., 0.],
		                                12.3: [1.0375, 0.40205],
		                                100: [5.4145, 0.69992],
		                                282: [14.737, 0.62319]},
		                      'NaCl': {0: [0., 0.],
		                               10: [0.6442, 0.2279],
		                               34: [1.9231, 0.33668],
		                               100: [0., 0.],
		                               260: [0., 0.]}
		                      }

		# dict structure {composition: {concentration: [a,b,c]}}
		self.depth_consts = {'MgSO4': {12.3: [1.0271, -74.0332, -4.2241],
		                               100: [5.38, -135.096, -8.2515],
		                               282: [14.681, -117.429, -5.4962]},
		                     'NaCl': {10: [0., 0., 0.],
		                              34: [1.8523, -72.4049, -10.6679],
		                              100: [0., 0., 0.],
		                              260: [0., 0., 0.]}
		                     }

		self.composition = composition
		self.concentration = concentration

		if self.composition == 'MgSO4':
			# Liquidus curve derived from Liquius 1.0 (Buffo et al. 2019 and FREEZCHEM) for MgSO4
			# changing from lambda notation to def notation for better pickling?

			# def self.Tm_func = lambda S: (-(1.333489497 * 1e-5) * S ** 2) - 0.01612951864 * S + 273.055175687
			# density changes for water w/ concentration of salt below
			self.C_rho = 1.145
			self.Ci_rho = 7.02441855e-01

			self.saturation_point = 282.  # ppt, saturation concentration of MgSO4 in water
			self.constants.rho_s = 2660.  # kg/m^3, density of MgSO4

		elif self.composition == 'NaCl':
			# Liquidus curve derived from Liquius 1.0 (Buffo et al. 2019 and FREEZCHEM) for NaCl
			# linear fit for density change due to salinity S
			self.C_rho = 0.8644
			self.Ci_rho = 6.94487270e-01

			self.saturation_point = 260.  # ppt, saturation concentration of NaCl in water
			self.constants.rho_s = 2160.  # kg/m^3, density of NaCl

		if S is not None:
			# method for custom salinity + brine inclusion
			self.S = S
			self.S += self.phi * concentration

		if shell:
			# automatically set the bottom temperature to Tm for salinity at the bottom
			T_match = True
			# method for a salinity/depth profile via Buffo et al. 2019
			s_depth = lambda z, a, b, c: a + b / (c - z)
			self.S = s_depth(self.Z, *self.depth_consts[composition][concentration])

			if in_situ is False:  # for water emplaced in a salty shell
				self.S += self.phi * concentration
			else:  # must redistribute the salt evenly to simulate real in-situ melting
				print('Redistributing salt in intrusion')
				try:
					S_int_tot = self.S[self.geom].sum()
					self.S[self.geom] = S_int_tot / np.shape(self.geom)[1]
					if self.S[self.geom].sum() / S_int_tot > 1.0 + 1e-15 or \
							self.S[self.geom].sum() / S_int_tot < 1.0 - 1e-15:
						print('S_int_new/Si =', self.S[self.geom].sum() / S_int_tot)
						raise Exception('problem with salt redistribution')
				except AttributeError:
					pass

			# update temperature profile to reflect bottom boundary condition
			if T_match:
				self.Tbot = self.Tm_func(s_depth(self.Lz, *self.depth_consts[composition][concentration]),
				                         *self.Tm_consts[composition])
				print('--Adjusting temperature profile: Tsurf = {}, Tbot = {}'.format(self.Tsurf, self.Tbot))
				self.init_T(Tsurf=self.Tsurf, Tbot=self.Tbot)
		else:
			# homogenous brine, pure ice shell
			self.S = self.phi * concentration
			if T_match:
				self.Tbot = self.Tm_func(concentration, *self.Tm_consts[composition])
				print('--Pure shell; adjusting temperature profile: Tsurf = {}, Tbot = {}'.format(self.Tsurf,
				                                                                                  self.Tbot))
				self.init_T(Tsurf=self.Tsurf, Tbot=self.Tbot)

		# update initial melting temperature
		self.Tm = self.Tm_func(self.S, *self.Tm_consts[composition])
		# update volume average with included salt
		self.init_volume_averages()
		# begin tracking mass
		self.total_salt = [self.S.sum()]
		# begin tracking amount of salt removed from system
		self.removed_salt = [0]
		# update temperature of liquid to reflect salinity
		try:
			self.T_int = self.Tm_func(self.S[self.geom], *self.Tm_consts[composition])[0]
			print('--Updating intrusion temperature to reflect initial salinity, Tint =', self.T_int)
			self.T[self.geom] = self.T_int
		except AttributeError:
			pass
		self.save_initials()

	def entrain_salt(self, dT, S, composition='MgSO4'):
		"""
		Calculate the amount of salt entrained in newly frozen ice that is dependent on the thermal gradient across
		the ice (Buffo et al., 2019).
		Parameters:
			dT : float, array
				temperature gradient across cell, or array of temperature gradients
			S : float, array
				salinity (ppt) of newly frozen cell, or array of salinities
			composition : string
				salt composition
				options: 'MgSO4', 'NaCl'
		Returns:
			amount of salt entrained in ice, ppt
			or array of salt entrained in ice, ppt

		Usage:
			See HeatSolver.update_salinity() function.
		"""
		if composition != 'MgSO4':
			raise Exception('Run some Earth tests you dummy')

		# get list of concentrations with known constants for constitutive equations
		concentrations = [key for key in self.shallow_consts[composition]]
		concentrations = np.sort(concentrations)

		if isinstance(dT, (int, float)):  # if dT (and therefore S) is a single value
			if S in concentrations:
				# determine whether to use linear or shallow fit
				switch_dT = optimize.root(lambda x: self.shallow_fit(x, *self.shallow_consts[composition][S]) \
				                                    - self.linear_fit(x, *self.linear_consts[composition][S]), 3)['x'][
					0]
				if dT > switch_dT:
					return self.shallow_fit(dT, *self.shallow_consts[composition][S])
				elif dT <= switch_dT:
					return self.linear_fit(dT, *self.linear_consts[composition][S])

			elif S not in concentrations:
				c_min = concentrations[S > concentrations].max()
				c_max = concentrations[S < concentrations].min()

				# linearly interpolate between the two concentrations at gradient dT
				m, b = np.polyfit([c_max, c_min], [self.entrain_salt(dT, c_max, composition),
				                                   self.entrain_salt(dT, c_min, composition)], 1)

				return m * S + b

		else:  # recursively call this function to fill an array of the same length as input array
			ans = np.zeros(len(dT))
			for i in range(len(dT)):
				ans[i] = self.entrain_salt(dT[i], S[i], composition)
			return ans
