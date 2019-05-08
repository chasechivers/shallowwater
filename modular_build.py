# Author: Chase Chivers
# Last updated: 5/8/19, 11:15pm
# Modular build for 2d heat diffusion problem
#   applied to liquid water in the ice shell of Europa

# TO DO:
#   add stress calculations due to volume change
#   add a way to cheater the domain so we don't have to model the full thing => less time ?

import numpy as np
import matplotlib.pyplot as plt
import time as _timer_
import seaborn as sns

sns.set(palette='colorblind', color_codes=1, context='notebook', style='ticks')


class Plotter:
	def __init__(self):
		self.plot = 0


class HeatSolver:
	'''
	Solves two-phase thermal diffusivity problem with a temperature-dependent thermal conductivity of ice in
	two-dimensions. Sources and sinks include latent heat of fusion and tidal heating
	Options:
		tidalheat -- binary; turns on/off viscosity-dependent tidal heating from Mitri & Showman (2005), default = 0
		Ttol -- convergence tolerance for temperature, default = 0.1 K
		phitol -- convergence tolerance for liquid fraction, default = 0.01
		latentheat -- binary; default use Huber et al. (2008), else use Michaut & Manga (2016) method, default = 1
		freezestop -- binary; stop when sill is frozen, default = 0
	Usage:
		Assuming, model = IceSystem(...)
		- Turn on tidal heating component
			model.tidalheat = True

		- Change tolerances
			model.Ttol = 0.001
			model.phitol = 0.0001
	'''
	# off and on options
	tidalheat = 0  # turns off or on tidalheating component
	Ttol = 0.1  # temperature tolerance
	phitol = 0.01  # liquid fraction tolerance
	latentheat = 1  # choose enthalpy method to use
	freezestop = 0  # stop simulation upon total solidification of sill

	def stefan_solution(self, t, T1, T0):
		'''
		Analytical solution to Stefan problem, compares analytical solution to numerical solution calculated here
		f
		'''
		from scipy import optimize
		from scipy.special import erf
		if T1 > T0:
			kappa = self.constants.kw / (self.constants.cp_w * self.constants.rho_w)
			Stf = self.constants.cp_w * (T1 - T0) / self.constants.Lf
		elif T1 < T0:
			T1, T0 = T0, T1
			kappa = self.constants.ki / (self.constants.cp_i * self.constants.rho_i)
			Stf = self.constants.cp_i * (T1 - T0) / self.constants.Lf
		lam = optimize.root(lambda x: x * np.exp(x ** 2) * erf(x) - Stf / np.sqrt(np.pi), 1)['x'][0]

		self.stefan_zm = 2 * lam * np.sqrt(kappa * t)
		self.stefan_zm_func = lambda time: 2 * lam * np.sqrt(kappa * time)
		self.stefan_zm_const = 2 * lam * np.sqrt(kappa)
		# self.stefan_time_frozen = (self.thickness / (2 * lam)) ** 2 / kappa
		self.stefan_z = np.linspace(0, self.stefan_zm)
		self.stefan_T = T1 - (T1 - T0) * erf(self.stefan_z / (2 * np.sqrt(kappa * t))) / erf(lam)

	def set_boundayconditions(self, top=True, bottom=True, sides=True):
		'''
			Set boundary conditions for heat solver
			top : top boundary conditions
				default: Dirichlet, Ttop = Tsurf chosen earlier
				'Radiative': radiative flux
				'NoFlux': no flux boundary condition
				---Other options?---
			bottom: bottom boundary condition
				default: Dirichlet, Tbottom = Tbot chosen earlier
				'NoFlux': no flux boundary condition
				----Other options?----
			sides: left and right boundary conditions, forced symmetric
				default: Dirichlet, Tleft = Tright =  Tedge (see init_T)
				'NoFlux': no flux boundary condition
		'''
		self.topBC = top
		self.botBC = bottom
		self.sidesBC = sides

	def update_salinity(self, phi_last):
		if self.issalt:
			new_ice = np.where((phi_last != 0) and (self.phi == 0))
			if self.S[new_ice] == self.saturation_point:
				self.saturated = 1
				return 0
			if len(new_ice[0]) > 0:
				S_old = self.S.copy()
				dTx = (self.T[1:-1, new_ice[0] + 1] - self.T[1:-1, new_ice[0] - 1]) / self.dx
				dTz = (self.T[new_ice[0] + 1, 1:-1] - self.T[new_ice[0] - 1, 1:-1]) / self.dz
				dT = max(dTx, dTz)
				# use maximum gradient in x or z directions to determine amount of entrained salt in ice
				self.S[new_ice] = self.entrained_salt(dT, S_old[new_ice])
				rejected_salt = sum(S_old[new_ice] - self.S[new_ice])
				water = np.where(self.phi >= self.rejection_cutoff)
				self.S[water] = self.S[water] + rejected_salt / len(self.S[water])
				self.total_salt.append(sum(self.S))

	def update_liquid_fraction(self, phi_last):
		if self.issalt == True:
			self.Tm = self.Tm_func(self.S)
		# calculate new enthalpy of solid ice
		Hs = self.cp_i * self.Tm
		H = self.cp_i * self.T + self.constants.Lf * phi_last
		self.phi[H >= Hs] = (H[H >= Hs] - Hs[H >= Hs]) / self.constants.Lf
		self.phi[H <= Hs + self.constants.Lf] = (H[H <= Hs + self.constants.Lf] - Hs[
			H <= Hs + self.constants.Lf]) / self.constants.Lf
		self.phi[H < Hs] = 0.
		self.phi[H > Hs + self.constants.Lf] = 1
		'''
		# enthalpy lower than solid ice => ice
		self.phi[H < Hs] = 0.
		# enthalpy higher than liquid enthalpy => water
		self.phi[H > Hs + self.constants.Lf] = 1.
		# find in-between
		idx = np.where((H >= Hs) and (H <= Hs + self.constants.Lf))
		if type(Hs) != type(self.phi):
			self.phi[idx] = (H[idx] - Hs) / self.constants.Lf
		else:
			self.phi[idx] = (H[idx] - Hs[idx]) / self.constants.Lf
		'''

	def update_volume_averages(self):
		if self.kT == True:
			self.k = (1 - self.phi) * (self.constants.ac / self.T) + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
		if self.cpT == True:
			self.cp_i = 185. + 7.037 * self.T

		if self.issalt == True:
			self.rhoc = (1 - self.phi) * (self.constants.rho_i) * self.cp_i \
			            + self.phi * (self.constants.rho_w + self.a_rho * self.S + self.b_rho * self.T) * \
			            self.constants.cp_w
		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w

	def update_sources_sinks(self, phi_last, T_last):
		# probably need to use T last rather than the newly
		self.latent_heat = self.constants.rho_i * self.constants.Lf * \
		                   (self.phi[1:-1, 1:-1] - phi_last[1:-1, 1:-1]) / self.dt

		if self.tidalheat == True:
			# ICE effective viscosity follows an Arrenhius law
			#   viscosity = reference viscosity * exp[C/Tm * (Tm/T - 1)]
			# if cell is water, just use reference viscosity for pure ice at 0 K
			self.visc = (1 - self.phi[1:-1, 1:-1]) * self.constants.visc0i \
			            * np.exp(self.constants.Qs * (self.Tm[1:-1, 1:-1] / T_last[1:-1, 1:-1] - 1) / \
			                     (self.constants.Rg * self.Tm[1:-1, 1:-1])) + self.phi[1:-1,
			                                                                  1:-1] * self.constants.visc0w
			self.tidal_heat = (self.constants.eps0 ** 2 * self.constants.omega ** 2 * self.visc) / (
					1 + self.constants.omega ** 2 * self.visc ** 2 / (self.constants.G ** 2))
		else:
			self.tidal_heat = 0

		self.Q = self.tidal_heat - self.latent_heat

	def apply_boundary_conditions(self, T):
		# apply chosen boundary conditions at bottom of domain
		if self.botBC == True:
			self.T[-1, :] = self.Tbot

		# apply chosen boundary conditions at top of domain
		if self.topBC == True:
			self.T[0, :] = self.Tsurf

		elif self.topBC == 'Radiative':
			c = self.dt / (2 * self.rhoc[0, 1:-1])
			Ttopx = c / self.dz ** 2 * ((self.k[0, 1:-1] + self.k[0, 2:]) * (T[0, 2:] - T[0, 1:-1]) \
			                            - (self.k[0, 1:-1] + self.k[0, :-2]) * (T[0, 1:-1] - T[0, :-2]))
			Ttopz = c / self.dz ** 2 * ((self.k[1, 1:-1] + self.k[0, 1:-1]) * (T[1, 1:-1] - T[0, 1:-1]) \
			                            - 2 * self.dz * self.constants.emiss * self.constants.stfblt * (T[0,
			                                                                                            1:-1] - self.Tsurf) ** 4)

			self.T[0, 1:-1] = T[0, 1:-1] + Ttopx + Ttopz + self.Q[0, :] * self.dt / self.rhoc[0, 1:-1]

		# apply chosen boundary conditions at sides of domain
		if self.sidesBC == True:
			self.T[:, 0] = self.Tedge
			self.T[:, self.nx - 1] = self.Tedge

		elif self.sidesBC == 'NoFlux':
			# not sure this works quite right
			# left side
			c = self.dt / (2 * self.rhoc[1:-1, 0])
			Tlx = c / self.dx ** 2 * (self.k[1:-1, 1] + self.k[1:-1, 0]) * (T[1:-1, 1] - T[1:-1, 0])
			Tlz = c / self.dz ** 2 * ((self.k[2:, 0] + self.k[1:-1, 0]) * (T[2:, 0] - T[1:-1, 0]) \
			                          - (self.k[1:-1, 0] + self.k[:-2, 0]) * (T[1:-1, 0] - T[:-2, 0]))
			self.T[1:-1, 0] = T[1:-1, 0] + Tlx + Tlz + self.dt * self.Q[:, 0] / self.rhoc[1:-1, 0]

			# right side
			c = self.dt / (2 * self.rhoc[1:-1, -1])
			Trx = c / self.dx ** 2 * -(self.k[1:-1, -1] + self.k[1:-1, - 1]) \
			      * (T[1:-1, -1] - T[1:-1, -2])
			Trz = c / self.dz ** 2 * ((self.k[2:, - 1] + self.k[1:-1, - 1]) \
			                          * (T[2:, - 1] - T[1:-1, - 1]) - (self.k[1:-1, - 1] + self.k[:-2, -1]) \
			                          * (T[1:-1, -1] - T[:-2, -1]))
			self.T[1:-1, -1] = T[1:-1, - 1] + Trx + Trz \
			                   + self.Q[:, -1] * self.dt / self.rhoc[1:-1, -1]

			if self.topBC == 'Radiative':
				# make sure cell in top left corner doesn't get fucked up
				c = self.dt / (2 * self.rhoc[0, 0])
				T_TLCz = c / self.dz ** 2 * (
						(self.k[1, 0] + self.k[0, 0]) * (T[1, 0] - T[0, 0]) - 2 * self.dz * self.constants.emiss * \
						self.constants.stfblt * (T[0, 0] - self.Tsurf) ** 4)
				T_TLCx = c / self.dx ** 2 * (self.k[0, 1] + self.k[0, 0]) * (T[0, 1] - T[0, 0])
				self.T[0, 0] = T[0, 0] + T_TLCx + T_TLCz + self.dt * self.Q[0, 0] / self.rhoc[0, 0]

				# make sure cell in top right corner doesn't get fucked up
				c = self.dt / (2 * self.rhoc[0, -1])
				T_TRCz = c / self.dz ** 2 * ((self.k[1, -1] + self.k[0, -1]) * (T[1, -1] - T[0, -1]) - 2 * self.dz *
				                             self.constants.emiss * self.constants.stfblt * (T[0, -1] - self.Tsurf)
				                             ** 4)
				T_TRCx = -c / self.dx ** 2 * (self.k[0, -2] + self.k[0, -1]) * (T[0, -1] - T[1, -1])
				self.T[0, -1] = T[0, -1] + T_TRCx + T_TRCz + self.dt * self.Q[0, -1] / self.rhoc[0, -1]

		elif self.sidesBC == 'Reflect':
			self.T[:, 0] = self.T[:, 1]
			self.T[:, -1] = self.T[:, -2]

	def outputs(self):
		'''
		choose which outputs to track
		'''
		return 0

	def print_at_start(self, nt):
		def stringIO(bin):
			if bin:
				return 'on'
			else:
				return 'off'

		def stringBC(BC):
			if isinstance(BC, str):
				return BC
			elif BC:
				return 'Dirichlet'

		print('Starting simulation with\n-------------------------')
		print('\t total model time:  {}s, {}yr'.format(nt * self.dt, (nt * self.dt) / self.constants.styr))
		print('\t surface temperature: {} K'.format(self.Tsurf))
		print('\t bottom temperature:  {} K'.format(self.Tbot))
		print('\t boundary conditions:')
		print('\t    top:     {}'.format(stringBC(self.topBC)))
		print('\t    bottom:  {}'.format(stringBC(self.botBC)))
		print('\t    sides:   {}'.format(stringBC(self.sidesBC)))
		print('\t sources/sinks:')
		print('\t    tidal heating:  {}'.format(stringIO(self.tidalheat)))
		print('\t    latent heat:    {}'.format(stringIO(self.latentheat)))
		print('\t tolerances:')
		print('\t    temperature:     {}'.format(self.Ttol))
		print('\t    liquid fraction: {}'.format(self.phitol))
		print('\t other:')
		print('\t    ki(T):    {}'.format(stringIO(self.kT)))
		print('\t    ci(T):    {}'.format(stringIO(self.cpT)))
		print('\t    salinity: {}'.format(stringIO(self.issalt)))
		print('-------------------------')

	def solve_heat(self, nt, dt):
		self.dt = dt
		self.model_time, self.run_time = 0, 0
		start_time = _timer_.clock()

		# self.print_at_start(nt)

		for n in range(nt):
			TErr, phiErr = np.inf, np.inf
			iter_k = 0
			while (TErr > self.Ttol) and (phiErr > self.phitol):
				T_last, phi_last = self.T.copy(), self.phi.copy()

				self.update_liquid_fraction(phi_last=phi_last)
				self.update_salinity(phi_last=phi_last)
				self.update_volume_averages()

				# constant in front of x-terms
				Cx = self.dt / (2 * self.rhoc[1:-1, 1:-1] * self.dx ** 2)
				# constant in front of z-terms
				Cz = self.dt / (2 * self.rhoc[1:-1, 1:-1] * self.dz ** 2)
				# temperature terms in z direction
				Tz = Cz * ((self.k[1:-1, 1:-1] + self.k[2:, 1:-1]) * (T_last[2:, 1:-1] - T_last[1:-1, 1:-1]) \
				           - (self.k[1:-1, 1:-1] + self.k[:-2, 1:-1]) * (T_last[1:-1, 1:-1] - T_last[:-2, 1:-1]))
				# temperature terms in x direction
				Tx = Cx * ((self.k[1:-1, 1:-1] + self.k[1:-1, 2:]) * (T_last[1:-1, 2:] - T_last[1:-1, 1:-1]) \
				           - (self.k[1:-1, 1:-1] + self.k[1:-1, :-2]) * (T_last[1:-1, 1:-1] - T_last[1:-1, :-2]))

				self.update_sources_sinks(phi_last=phi_last, T_last=T_last)

				self.T[1:-1, 1:-1] = T_last[1:-1, 1:-1] + Tx + Tz + self.Q * self.dt / self.rhoc[1:-1, 1:-1]
				self.apply_boundary_conditions(T=self.T)

				TErr = np.amax(abs(self.T[1:-1, 1:-1] - T_last[1:-1, 1:-1]))
				phiErr = np.amax(abs(self.phi[1:-1, 1:-1] - phi_last[1:-1, 1:-1]))

				iter_k += 1
				# kill statement when parameters won't allow solution to converge
				if iter_k > 1000:
					raise Exception('solution not converging')
			# outputs here

			self.num_iter = iter_k
			self.model_time += self.dt
			if self.freezestop:
				if len(self.phi[self.phi > 0]) == 0:
					print('sill frozen at {0:0.04f}s'.format(self.model_time))
					return self.model_time

			if self.issalt and self.saturated:
				print('sill is saturated')
				return self.model_time

			self.run_time += _timer_.clock() - start_time

	def stefan_compare(self, dt):
		'''
		Compare simulation freezing front propagation zm(t) with stefan solution
		Parameters:
				dt : float
					time step
		'''
		self.dt = dt
		self.set_boundayconditions(top=True, bottom=True, sides='Reflect')
		self._time_ = [0]
		self.freeze_front = [0]
		midx = int(np.floor(len(self.X[0, :]) / 2))
		n = 1

		while self.freeze_front[-1] <= 0.9 * self.Lz:
			self.solve_heat(nt=1, dt=self.dt)
			if (self.phi == 0).any():
				idx = np.max(np.where(self.phi == 0))
				if self.Z[idx, midx] != self.freeze_front[-1]:
					self.freeze_front.append(self.Z[idx, midx])
					print(idx, midx)
					self._time_.append(n * self.dt)
			else:
				self.freeze_front.append(0)
				self._time_.append(n * self.dt)
			n += 1

		self.stefan_solution(t=n * self.dt, T1=self.Tsurf, T0=self.Tbot)


class IceSystem(HeatSolver, Plotter):
	def __init__(self, Lx, Lz, dx, dz, kT=True, cpT=True, issalt=False):
		'''
		Parameters:
			Lx : float
				length of horizontal spatial domain, m
			Lz : float
				"depth of shell", length of vertical spatial domain, m
			dx : float
				horizontal spatial step size, m
			dz : float
				vertical spatial step size, m
			cpT : binary
			    choose whether to use temperature-depedent specific heat,
			    default = 1, temperature-dependent, cp_i = 185 + 7*T (citation)
			kT : binary
			    choose whether to use temperature-dependent thermal conductivity,
			    default = 1, temperature-dependent, k=ac/T (Petrenko, Klinger, etc.)
			issalt : binary
				choose whether salinity will be used in solution
		'''
		self.Lx, self.Lz = Lx, Lz
		self.dx, self.dz = dx, dz
		self.nx, self.nz = int(Lx / dx), int(Lz / dz)
		X = np.linspace(-Lx / 2, Lx / 2, self.nx)  # x domain centered on 0
		Z = np.linspace(0, Lz, self.nz)  # z domain starts at zero
		self.X, self.Z = np.meshgrid(X, Z)  # create spatial grid
		self.T = np.ones((self.nz, self.nx))  # initialize domain at one temperature
		self.S = np.zeros((self.nz, self.nx))  # initialize domain with no salt
		self.phi = np.zeros((self.nz, self.nx))  # initialize domain as ice
		self.issalt = issalt  # salt IO
		self.kT, self.cpT = kT, cpT  # k(T), cp_i(T) I/O

	class constants:
		styr = 3.14e7  # s/yr, seconds in a year

		g = 1.32  # m/s2, Europa surface gravity

		# Thermal properties
		rho_i = 910.  # kg/m3, pure ice density
		rho_w = 1000.  # kg/m3 pure water density
		cp_i = 2.11e3  # J/kgK, pure ice specific heat
		cp_w = 4.19e3  # J/kgK, pure water specific heat
		ki = 2.3  # W/mK, pure ice thermal conductivity
		kw = 0.56  # W/mK, pure water thermal conductivity
		ac = 567  # W/m, ice thermal conductivity constant, ki = ac/T
		Tm = 273.15  # K, pure ice melting temperature at 1 atm
		Lf = 333.6e3  # J/kg, latent heat of fusion of ice
		expans = 1.6e-4  # 1/K, thermal expansivity

		# Radiation properties
		emiss = 0.97  # pure ice emissivity
		stfblt = 5.67e-8  # W/m2K4 stefan-boltzman constant

		# Constants for viscosity dependent tidalheating
		#   from Howell & Pappalardo (2018)
		act_nrg = 26.  # activation energy for diffusive regime
		Qs = 60e3  # J/mol, activation energy of ice (Goldsby & Kohlstadt, 2001)
		Rg = 8.3144598  # J/K*mol, gas constant
		eps0 = 2e-5  # maximum tidal flexing strain
		omega = 1.5e-5  # 1/s, tidal flexing frequency
		visc0i = 1e13  # Pa s, minimum reference ice viscosity at T=Tm
		visc0w = 1.7e6  # Pa s, dynamic viscosity of water at 0 K

		# Mechanical properties of ice
		G = 3.52e9  # Pa, shear modulus/rigidity (Moore & Schubert, 2000)
		E = 1e6  # Pa, Young's Modulus

	def save_initials(self):
		self.T_initial = self.T.copy()
		self.Tm_initial = self.Tm.copy()
		self.phi_initial = self.phi.copy()
		self.S_initial = self.S.copy()
		if self.kT:
			self.k_initial = self.phi_initial * self.constants.kw + (1 - self.phi_initial) * \
			                 self.constants.ac / self.T_initial

	def init_T(self, Tsurf, Tbot, profile='non-linear'):
		'''
			Initialize temperature profile
			Parameters:
				Tsurf : float
					surface temperature
				Tbot : float
					temperature at bottom of domain
				profile: defaults 'non-linear', 'linear'
					prescribed temperature profile
					'non-linear' -- expected equilibrium thermal gradient with k(T)
					'linear'     -- equilibirium thermal gradient for constant k
			Returns:
				T : (nz,nx) array
					grid of temperature values
		'''
		# set melting temperature to default
		self.Tm = self.constants.Tm * self.T.copy()

		if isinstance(profile, str):
			if profile == 'non-linear':
				self.T = Tsurf * (Tbot / Tsurf) ** (abs(self.Z / self.Lz))
			elif profile == 'linear':
				self.T = (Tbot - Tsurf) * abs(self.Z / self.Lz) + Tsurf
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
		# self.save_initials()

	def init_volume_averages(self):
		'''
		Initialize volume averaged values over the domain. Must be used if not using an intrusion
		'''

		if self.kT:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * self.constants.ac / self.T + self.phi * self.constants.kw

		if self.cpT:
			self.cp_i = 185. + 7.037 * self.T
		else:
			self.cp_i = self.constants.cp_i

		# if using salinity, use a water density with salinity and temperature dependence
		if self.issalt:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * (self.constants.rho_w + self.a_rho * self.S + self.b_rho * self.T) * \
			            self.constants.cp_w
		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w
		self.save_initials()

	def set_intrusion_geom(self, depth, thickness, radius, geometry='ellipse'):
		'''
		Set geometry of intrusion
		'''
		if isinstance(geometry, str):
			if geometry == 'ellipse':
				center = thickness / 2 + depth
				self.geom = np.where((self.X / radius) ** 2 + (self.Z - center) ** 2 / ((thickness / 2) ** 2) <= 1)
			elif geometry == 'box':
				self.geom = np.where((self.Z >= depth) and (self.Z <= depth + thickness) \
				                     and (abs(self.X) <= radius / 2))
		else:
			self.geom = geometry

	def init_intrusion(self, T, depth, thickness, radius, phi=1, geometry='ellipse'):
		'''
		Initialize intrusion properties.
		**So far this only accounts for a single intrusion at the center of the domain
			should be simple to add
		Updates volume averages after initialization: means we can just initialize temperature and intrusion to get
		all thermal properties set
		Parameters:
			T : float
				set intrusion to single Temperature value, assuming that it is well mixed
			depth : float
				set depth of upper edge of the intrusion, m
			thickness : float
				set thickness of intrusion, m
			radius : float
				set radius of intrusion, m
			geometry : string (see set_sill_geom()), array
				set geometry of intrusion, default is an ellipse
			cpT : binary
			    choose whether to use temperature-depedent specific heat,
			    default = 1, temperature-dependent, cp_i = 185 + 7*T (citation)
			kT : binary
			    choose whether to use temperature-dependent thermal conductivity,
			    default = 1, temperature-dependent, k=ac/T (Petrenko, Klinger, etc.)

		'''
		if phi < 0 or phi > 1:
			raise Exception('liquid fraction must be between 0 and 1')

		self.Tsill = T
		self.depth, self.thickness, self.R_int = depth, thickness, radius
		self.set_intrusion_geom(depth, thickness, radius, geometry)
		self.T[self.geom] = T
		self.phi[self.geom] = phi
		self.init_volume_averages()

	def init_salinity(self, S=None, composition='Europa', concentration=12.3, a_rho=0.75, b_rho=-0.0375,
	                  rejection_cutoff=0.9):
		'''
		Initialize salinity in the model
		-- add a way to include background salinity profile ala Buffo et. al (2019)
		'''

		self.issalt = 1  # turn on salinity for solvers
		self.a_rho = a_rho  # water density salinity coefficient
		self.b_rho = b_rho  # water density temperature coefficient
		self.saturation_point = 282.  # ppt, saturation point of water
		self.saturated = 0  # whether liquid is saturated
		self.rejection_cutoff = rejection_cutoff  # how much water can accept rejected salt

		# composition and concentration coefficients for fits from Buffo et al. (2019)
		self.shallow_consts = {'Europa': {12.3: [12.21, -8.3, 1.836, 20.2],
		                                  100: [22.19, -11.98, 1.942, 21.91],
		                                  282: [30.998, -11.5209, 2.0136, 21.1628]
		                                  },
		                       'Earth': {34: [10.27, -5.97, 1.977, 22.33]}
		                       }
		self.linear_consts = {'Europa': {12.3: [1.0375, 0.40205],
		                                 100: [5.4145, 0.69992],
		                                 282: [14.737, 0.62319]
		                                 },
		                      'Earth': {34: [1.9231, 0.33668]}
		                      }
		if composition == 'Earth':
			concentration = 34

		self.composition = composition
		self.concentration = concentration

		if S is not None:
			# use input S as background salinity?
			self.S = S
		else:
			self.shallow_fit = lambda dT, a, b, c, d: a + b * (dT + c) * \
			                                          (1 - np.exp(-d / dT)) / (1 + dT)
			# linear fit, for small dT/dz
			self.linear_fit = lambda dT, a, b: a + b * dT

			# homogenous brine, pure ice shell
			self.S = self.phi * concentration

			# Melting temperature based on curve fit from FREEZCHEM for compositions
			if self.composition == 'Europa':
				self.Tm_func = lambda S: (-(1.333489497 * 1e-5) * S ** 2) - 0.01612951864 * S + self.constants.Tm
			elif self.composition == 'Earth':
				self.Tm_func = lambda S: (-(9.1969758 * 1e-5) * S ** 2) - 0.03942059 * S + self.constants.Tm

			# update initial melting temperature
			self.Tm = self.Tm_func(self.S)
			self.init_volume_averages()
			self.total_salt = [sum(self.S)]

	def entrain_salt(self, dT, S, composition='Europa'):
		from scipy import optimize
		if composition != 'Europa':
			raise Exception('Run some Earth tests you dummy')

		if isinstance(dT, float) or isistance(dT, int):
			if S in self.shallow_consts[composition].keys():
				switch_dT = optimize.root(lambda x: self.shallow_fit(x, *self.shallow_consts[composition][S]) \
				                                    - self.linear_fit(x, *self.linear_consts[composition][S]), 3)['x'][
					0]
				if dT > switch_dT:
					return self.shallow_fit(dT, *self.shallow_consts[composition][S])
				elif dT < switch_dT:
					return self.linear_fit(dT, *self.linear_consts[composition][S])
			elif 12.3 < S < 100:
				ans = S * (self.entrained_S(dT, 100, composition) - self.entrained_S(dT, 12.3, composition)) / (
						100 - 12.3)
				return ans + self.entrained_S(dT, 12.3, composition)
			elif 100 < S < 282:
				ans = S * (self.entrained_S(dT, 282, composition) - self.entrained_S(dT, 100, composition)) / (
						282 - 100)
				return ans + self.entrained_S(dT, 100, composition)

			else:
				ans = np.zeros(len(dT))
				for i in range(len(dT)):
					ans[i] = self.entrained_S(dT[i], S[i], composition)
				return ans

	def cheater_domain(self):
		# find a way to cheat on domain size to reduce run time
		return 0
