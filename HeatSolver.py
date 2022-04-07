from Outputs import Outputs
import dill as pickle
import time as _timer_
import numpy as np


def div0(a, b, fill=np.nan):
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
	if np.isscalar(c):
		return c if np.isfinite(c) \
			else fill
	else:
		c[~ np.isfinite(c)] = fill
		return c


class HeatSolver:
	"""Class object that contains all of the necessary functions and variables to solve the 2D heat conduction
	equation."""

	def __init__(self):
		"""Set default simulation parameters"""
		self.TIDAL_HEAT = True  # tidal heating source
		self.FREEZE_STOP = True  # stop simulation when water body is frozen, i.e. sum(phi) = 0

		# the Courant-Fredrichs-Lewy condition when choosing a time step: dt = CFL * min(dx,dz)^2 / Kappa
		#  this should be much smaller for colder surface temperatures (i.e. CFL ~ 1/50 for Tsurf ~ 50 K)
		self.CFL = 1 / 24.
		self.ADAPT_DT = True  # adapt the time step with

		self.TTOL = 0.1  # temperature tolerance
		self.PHITOL = 0.01  # liquid fraction tolerance
		self.STOL = 1  # salinity tolerance

		self.saturation_time = 0
		self.num_iter = []
		self._niter = self.num_iter.append
		self.ITER_KILL = 2000

	def set_outputs(self, output_frequency: int, tmp_dir: str = "./tmp/", tmp_file_name: str = "",
	                outlist: list = None):
		"""Initialize the outputs class for outputting transient results during simulation.

		Parameters
		----------
		output_frequency : int
			Number of steps at which to save the current model state (i.e. every 2000 time steps)

		tmp_dir : str
			Path to directoy for which to save the current model state. Defaults to a directory called "tmp" in the
			current working directory

		outlist : list(str)
			A list of variables to track. Defaults to output only time, temperature ("T"), liquid fraction ("phi"),
			and salinity if applicable ("S").
			Options:
				"time"  -- 1D array of model times, s
				"T"     -- 2D temperature grid, K
				"phi"   -- 2D liquid fraction grid
				"S"     -- 2D salinity grid, ppt
				"Q"     -- 2D grid of sources and sinks W/m^3

				These can be calculated afterward and so are generally left out of simulations:
				"viscosity" -- 2D grid of viscosity, Pa s
				"Tm"        -- 2D grid of salinity-dependent melting temperatures, K
				"h"     -- 1D array of water column thickness at z=dz, m
				"r"     -- 1D array of the radius of current extent of water, m
				"freeze fronts" -- 1D array of position of the upper and lower freeze fronts with time, m

		Returns
		----------
		None
		"""
		if outlist is None:
			outlist = ["phi", "T"]
		self.outputs = Outputs(output_frequency, tmp_dir, name=tmp_file_name)
		self.outputs.choose(outlist, self.issalt)

	def set_boundaryconditions(self, top=True, bottom=True, sides=True, **kwargs):
		"""Choose the boundary conditions for the simulation

		Parameters
		----------
		top: str, bool
			Boundary condition assumed for the top boundary.
			Options:  True       -- (DEFAULT) Constant temperature (Dirichlect). Set at Tsurf from init_temperature
				      "Radiative" -- Assumes that the energy balance at the surface (z=0) is given by:
									  radiative losses + sublimation + flux to z=0 = solar flux + flux from z=dz
									    radiative losses = ε*σ*T^4,
									    ice sublimation = Lf * dm/dt = Lf * pv(T) * (H20 molar mass/2/pi/T/Rg)^0.5
									    solar flux = (1 - A)*c*S/r^2,
									  where T is the temperature at the surface, σ being the Stefan-Boltzmann
									  constant, and ε is ice emissivity, pv is equilibrium vapor pressure at T, A
									  is the albedo, S is the solar (flux) constant, r is distance from the sun in
									  AU, and c is a paramter determined by the orbit.

									  If this option is used, can use the "use_orbit" argument to turn on orbital
									  parameters.

		bottom : str, bool
			Boundary condition assumed for the bottom boundary.
			Options: Constant temperature (Dirichlect). Set at Tbot from init_temperature
					  "ConstFlux" -- Constant flux given to the bottom boundary. Must input this value when
									  setting boundary conditions as qbot (in W/m^2)
									  e.g. model.set_boundaryconditions(bottom="ConstFlux", qbot=10)
					  "IceFlux"   -- Creates a "ghost row" of ice cells below the bottom boundary at a consatnt
					                  temperature that provides heat to the bottom boundary. Requires temperature of
					                  ice when using this setting (in K)
					                  e.g. model.set_boundaryconditions(bottom="IceFlux", Tbot=261.0)
					  "OceanFlux" -- Creates a "ghost row" of water cells below the bottom boundary at a constant
					                  temperature that provides heat to the bottom boundary. Automatically assumes
					                  that this temperature is the melting temperature of the ocean that the shell
					                  came from, i.e., if the shell/injected sill is assumed to come from a 12.3 ppt
					                  MgSO4 ocean, the bottom boundary is assumed as T ~ 271 K
		sides: str, bool
			Boundary conditions for the left and right walls
			Options:   True     -- (DEFAULT) Constant temperature (Dirichlect). Set at the "equilibrium" temperature
									profile from init_temperature
					  "NoFlux"  -- No flux (insulating) boundary conditions on both left and right boundaries
					  "LNoFlux" -- Left boundary is kept constant while right boundary is "NoFlux" (insulating)
					  "RFlux"   -- Allow heat loss at the right boundary to a column of ice at the equilibrium
					                temperature profile from init_profile at a distance dL away. Must input dL value
					                when setting boundary conditions (in m)
					                e.g. model.set_boundaryconditions(sides="RFlux", dL=100.0)
					                Note: currently does not work in cylindrical coordniates

		Returns
		----------
		None
		"""
		self.topBC = top
		self.botBC = bottom
		self.sidesBC = sides

		if top is True:
			self.TtopBC = self.T_initial[0, :].copy()

		if top == "Radiative":
			try:
				self.use_orbits = kwargs['use_orbits']
			except KeyError:
				self.use_orbits = False
			if self.use_orbits is True:
				self.nu = 0
				x = (self.constants.rAU) * (1 - self.constants.ecc * self.constants.ecc)
				r = x / (1 + self.constants.ecc * np.cos(self.nu))
				self.nudot = (self.constants.GM * x) ** 0.5 / (r * r)

		if top == "ConstFlux":
			try:
				if isinstance(kwargs["qtop"], (float, int)):
					self.qtop = kwargs["qtop"] * np.ones(self.nx, dtype=float)
				else:
					self.qtop = kwargs["qtop"]
			except KeyError:
				raise Exception("Value for flux at base of ice shell not given\n\t->e.g., model.set_boundaryconditions("
				                "top='ConstFlux' qtop=1e-3)")

		if bottom is True:
			self.TbotBC = self.T_initial[-1, :].copy()

		if bottom == "ConstFlux":
			try:
				if isinstance(kwargs["qbot"], (float, int)):
					self.qbot = kwargs["qbot"] * np.ones(self.nx, dtype=float)
				else:
					self.qbot = kwargs["qbot"]
			except KeyError:
				raise Exception("Value for flux at base of ice shell not given\n\t->e.g., model.set_boundaryconditions("
				                "bottom='ConstFlux' qbot=1e-3)")
		elif bottom == "IceFlux":
			try:
				self.T_bottom_boundary = kwargs["Tbot"]
			except KeyError:
				raise Exception("Ice temperature for ghost cells not given\n\t->e.g.,model.set_boundaryconditions("
				                "bottom='IceFlux', Tbot=261.0)")
		elif bottom == "OceanFlux":
			if self.issalt:
				self.T_bottom_boundary = self.Tm_func(self.concentration, *self.Tm_consts)
			else:
				self.T_bottom_boundary = self.constants.Tm

		elif bottom == "GhostIce":
			try:
				self.TBotGhost = kwargs["TBotGhost"]
				if isinstance(self.TBotGhost, (float, int)):
					self.TBotGhost *= np.ones(self.nx, dtype=float)
			except KeyError:
				raise Exception("Please input TBotGhost\n\t-> e.g., model.set_boundaryconditions(bottom='GhostIce', "
				                "TBotGhost=273)")

		if sides == "RFlux":
			if self.coords in ["cylindrical", "zr", "rz"]:
				raise Exception("Please choose a new boundary condition, this currently does not work in this "
				                "coordinate system")
			else:
				pass

			try:
				self.dL = kwargs["dL"]
			except KeyError:
				raise Exception("Length for right side flux not chosen\n\t->model.set_boundaryconditions("
				                "sides='RFlux', dL=500e3)")

	def _update_liquid_fraction(self, phi_last):
		"""Updates the phase at each time step. Generally internal use only.

		Parameters
		----------
		phi_last : np.ndarray, float
			Liquid fraction at iteration k-1 or time step n-1

		Returns
		----------
		None
		"""
		if self.issalt:
			self.Tm = self.Tm_func(self.S, *self.Tm_consts)

		# calculate enthalpies
		Hs = self.cpi(self.T, self.S) * self.Tm
		Hl = Hs + self.constants.Lf * ((1 - self.p) if "p" in self.__dict__ else 1)
		H = self.cpi(self.T, self.S) * self.T + self.constants.Lf * phi_last

		self.phi[H >= Hs] = (H[H >= Hs] - Hs[H >= Hs]) / self.constants.Lf
		self.phi[H <= Hl] = (H[H <= Hl] - Hs[H <= Hl]) / self.constants.Lf
		# all ice
		self.phi[H < Hs] = 0.
		# all water
		self.phi[H > Hl] = 1.

	def _xdir(self, z_ix, x_ix, T):
		if self.symmetric and x_ix in [0, self.nx - 1]:
			return 0
		else:
			return abs(T[z_ix, x_ix - 1] - T[z_ix, x_ix + 1]) / 2 / self.dx

	def _zdir(self, z_ix, x_ix, T):
		if z_ix == self.nz - 1:
			return (T[z_ix - 1, x_ix] - T[z_ix, x_ix]) / self.dz
		else:
			return (T[z_ix - 1, x_ix] - T[z_ix + 1, x_ix]) / 2 / self.dz

	@np.vectorize
	def _entrain_param(self, S, dTz, dTx):
		if (dTz > 0 if self.composition != "HCN" else dTz < 0):
			return S
		else:
			dT = (dTx + abs(dTz)) / 2.
			return self.entrain_salt(dT=dT, S=S)

	def _update_salinity(self, phi_last):
		"""Parameterization for brine drainage, entrainment, and chemical mixing in the liquid portion.

		Parameters
		----------
		phi_last : np.ndarray, float
			Liquid fraction at iteration k-1 or time step n-1

		Returns
		----------
		None
		"""
		z_newice, x_newice = np.where((phi_last > 0) & (self.phi == 0))
		water = np.where(self.phi >= self.rejection_cutoff)
		volume = water[1].shape[0]
		# self.removed_salt.append(0)
		self._rsapp(0)

		if len(z_newice) > 0 and volume != 0 and self.phi[self.phi > 0].any():
			S_old = self.S.copy()
			dTx = [self._xdir(i, j, self.T) for i, j in zip(z_newice, x_newice)]
			dTz = [self._zdir(i, j, self.T) for i, j in zip(z_newice, x_newice)]
			self.S[z_newice, x_newice] = self._entrain_param(self, S_old[z_newice, x_newice], dTz, dTx)

			if self.salinate:
				self.S[water] = self.S[water] + abs((S_old - self.S).sum()) / volume

				self.removed_salt[-1] += (self.S[self.S >= self.saturation_point] - self.saturation_point).sum()

			self.S[self.S > self.saturation_point] = self.saturation_point

		if self.salinate:
			if sum(self.removed_salt) > 0:
				z_newwater, x_newwater = np.where((phi_last == 0) & (self.phi > 0))  # find where ice may have melted
				if len(z_newwater) > 0:
					self.S[water] = self.S[water].sum() / volume

			total_S_new = self.S.sum() + sum(self.removed_salt)

			if abs(total_S_new - self.total_salt[0]) > self.STOL:
				# self.total_salt.append(total_S_new)
				self._tsapp(total_S_new)
			# raise Exception('Salt mass not being conserved')

			else:
				# self.total_salt.append(total_S_new)
				self._tsapp(total_S_new)

		if (self.S >= self.saturation_point).any() and water[0].shape[0] > 0 and self.saturation_time == 0:
			self.saturation_time = self.model_time

	def _update_sources_sinks(self, phi_last, T, Tm):
		"""Updates the sources and sinks for heat in the model from internal heat production or external sources.

		Parameters
		----------
		phi_last : np.ndarray, float
			Liquid fraction at iteration k-1 or time step n-1
		T : np.ndarray, float
			Temperature at iteration k-1 or time step n-1, K
		Tm : np.ndarray, float
			Melting temperature at iteration k-1 or timestep n-1, K

		Returns
		----------
		Q : np.ndarray, float
			Internal heat production or external heat sources, W/m^3
		"""
		latent_heat = - self.constants.rho_i * self.constants.Lf * (self.phi - phi_last) / self.dt
		tidal_heat = 0

		if self.TIDAL_HEAT:
			tidal_heat = self.tidal_heating(phi_last, T, Tm)

		return tidal_heat + latent_heat

	def _sublimation_heat_loss(self, T=None):
		"""Heat loss at the surface from ice sublimation.

		Parameters
		----------
		T : np.ndarray, float
			Surface temperature, K

		Returns
		----------
		q : np.ndarray, float
			Heat flux out of the surface by mass loss, W/m^2
		"""
		if T is None:
			T = self.T
		q = self.constants.Lv * self.pv(T) * (self.constants.m_H2O / 2. / np.pi / T / self.constants.Rg) ** 0.5
		return q

	def _radiative_heat_loss(self, T=None):
		"""Heat loss at the surface due to infrared radation, q = ε*σ*T^4

		Parameters
		----------
		T : np.ndarray, float
			Surface temperature, K

		Returns
		----------
		q : np.ndarray, float
			Heat flux out of the surface by infrared radiation, W/m^2
		"""
		if T is None:
			T = self.T
		q = self.constants.stfblt * self.constants.emiss * T ** 4
		return q

	def _solar_flux(self):
		"""Heat flux from solar insolation.

		Parameters
		----------
		None

		Returns
		----------
		q : float
			Heat flux to surface from the sun, W/m^2
		"""
		q = (1 - self.constants.albedo) * self.constants.solar_const / (self.constants.rAU * self.constants.rAU)
		return q

	def _apply_boundaryconditions(self, T, k, rhoc):
		"""Applies boundary conditions chosen in set_boundaryconditions() during a simulation. Internal usage.

		Parameters
		----------
		T : np.ndarray
			2D grid of temperatures, K
		k : np.ndarray
			2D grid of thermal conductivity, W/m/K
		rhoc : np.ndarray
			2D grid of specific heat (density * specific heat capacity), J/K/m^3

		Returns
		----------
		None
		"""
		#######################################
		# APPLY BOUNDARY CONDITIONS AT BOTTOM #
		if self.botBC is True:
			self.T[-1, 1:-1] = self.TbotBC[1:-1]

		elif self.botBC == "NoFlux":
			self.T[-1, 1:-1] = self.T[-2, 1:-1]

		elif self.botBC == "ConstFlux":
			if self.coords in ["cartesian", "xz", "zx"]:
				dqx = self._conductivity_average(k[-1, 2:], k[-1, 1:-1]) * (T[-1, 2:] - T[-1, 1:-1]) / (
							self.dx * self.dx) \
				      - self._conductivity_average(k[-1, 1:-1], k[-2, :-2]) * (T[-1, 1:-1] - T[-1, :-2]) / (
							      self.dx * self.dx)

			elif self.coords in ["cylindrical", "zr", "rz"]:
				rph = 0.5 * (self.X[-1, 1:-1] + self.X[-1, 2:])
				rmh = 0.5 * (self.X[-1, 1:-1] + self.X[-1, :-2])
				dqx = (rph * self._conductivity_average(k[-1, 2:], k[-1, 1:-1]) * (T[-1, 2:] - T[-1, 1:-1])
				       - rmh * self._conductivity_average(k[-1, 1:-1], k[-1, :-2]) * (T[-1, 1:-1] - T[-1, :-2])
				       ) / self.X[-1, 1:-1] / (self.dx * self.dx)
			dqx *= self.dt / rhoc[-1, 1:-1]

			dqz = self.qbot[1:-1] / self.dz \
			      - self._conductivity_average(k[-1, 1:-1], k[-2, 1:-1]) * (T[-1, 1:-1] - T[-2, 1:-1]) / (
						      self.dz * self.dz)
			dqz *= self.dt / rhoc[-1, 1:-1]
			# dqz = self.T[-2, 1:-1] + self.dz * self.qbot[1:-1] / k[-1, 1:-1]

			self.T[-1, 1:-1] = T[-1, 1:-1] + dqx + dqz + self.dt * self.Q[-1, 1:-1] / rhoc[-1, 1:-1]

		elif self.botBC == "IceFlux" or self.botBC == "OceanFlux":
			kbot = self.k(0.0 if self.botBC == "IceFlux" else 1.0,
			              self.T_bottom_boundary,
			              self.S[-1, :] if self.botBC == "IceFlux" else (self.concentration if self.issalt else 0))

			if self.coords in ["cartesian", "xz", "zx"]:
				dqx = self._conductivity_average(k[-1, 2:], k[-1, 1:-1]) * (T[-1, 2:] - T[-1, 1:-1]) / (
							self.dx * self.dx) \
				      - self._conductivity_average(k[-1, 1:-1], k[-2, :-2]) * (T[-1, 1:-1] - T[-1, :-2]) / (
							      self.dx * self.dx)

			elif self.coords in ["cylindrical", "zr", "rz"]:
				rph = 0.5 * (self.X[-1, 1:-1] + self.X[-1, 2:])
				rmh = 0.5 * (self.X[-1, 1:-1] + self.X[-1, :-2])
				dqx = (rph * self._conductivity_average(k[-1, 2:], k[-1, 1:-1]) * (T[-1, 2:] - T[-1, 1:-1])
				       - rmh * self._conductivity_average(k[-1, 1:-1], k[-1, :-2]) * (T[-1, 1:-1] - T[-1, :-2])
				       ) / self.X[-1, 1:-1] / (self.dx * self.dx)

			dqx *= self.dt / rhoc[-1, 1:-1]

			dqz = self._conductivity_average(kbot, k[-1, 1:-1]) * (self.T_bottom_boundary - T[-1, 1:-1]) \
			      - self._conductivity_average(k[-1, 1:-1], k[-2, 1:-1]) * (T[-1, 1:-1] - T[-2, 1:-1])
			dqz *= self.dt / rhoc[-1, 1:-1] / (self.dz * self.dz)

			self.T[-1, 1:-1] = T[-1, 1:-1] + dqx + dqz + self.dt * self.Q[-1, 1:-1] / rhoc[-1, 1:-1]

		elif self.botBC == "GhostIce":
			kghost = self.ki(self.TBotGhost, self.S)
			if self.coords in ["cartesian", "xz", "zx"]:
				dqx = self._conductivity_average(k[-1, 2:], k[-1, 1:-1]) * (T[-1, 2:] - T[-1, 1:-1]) / (
							self.dx * self.dx) \
				      - self._conductivity_average(k[-1, 1:-1], k[-2, :-2]) * (T[-1, 1:-1] - T[-1, :-2]) / (
							      self.dx * self.dx)

			elif self.coords in ["cylindrical", "zr", "rz"]:
				rph = 0.5 * (self.X[-1, 1:-1] + self.X[-1, 2:])
				rmh = 0.5 * (self.X[-1, 1:-1] + self.X[-1, :-2])
				dqx = (rph * self._conductivity_average(k[-1, 2:], k[-1, 1:-1]) * (T[-1, 2:] - T[-1, 1:-1])
				       - rmh * self._conductivity_average(k[-1, 1:-1], k[-1, :-2]) * (T[-1, 1:-1] - T[-1, :-2])
				       ) / self.X[-1, 1:-1] / (self.dx * self.dx)

			dqx *= self.dt / rhoc[-1, 1:-1]

			dqz = self._conductivity_average(kghost[1:-1], k[-1, 1:-1]) * (self.TBotGhost[1:-1] - T[-1, 1:-1]) \
			      - self._conductivity_average(k[-1, 1:-1], k[-2, 1:-1]) * (T[-1, 1:-1] - T[-2, 1:-1])
			dqz *= self.dt / rhoc[-1, 1:-1] / (self.dz * self.dz)

			self.T[-1, 1:-1] = T[-1, 1:-1] + dqx + dqz + self.dt * self.Q[-1, 1:-1] / rhoc[-1, 1:-1]

		# END BOTTOM BOUNDARY CONDITIONS   #
		####################################

		####################################
		# APPLY BOUNDARY CONDITIONS AT TOP #
		####################################
		if self.topBC is True:
			self.T[0, 1:-1] = self.TtopBC[1:-1]

		elif self.topBC == "NoFlux":
			self.T[0, 1:-1] = self.T[1, 1:-1]

		elif self.topBC == "Radiative":
			if self.coords in ["cartesian", "xz", "zx"]:
				dqx = self._conductivity_average(k[0, 2:], k[0, 1:-1]) * (T[0, 2:] - T[0, 1:-1]) / (self.dx * self.dx) \
				      - self._conductivity_average(k[0, 1:-1], k[0, :-2]) * (T[0, 1:-1] - T[0, :-2]) / (
							      self.dx * self.dx)

			elif self.coords in ["cylindrical", "zr", "rz"]:
				rph = 0.5 * (self.X[0, 1:-1] + self.X[0, 2:])
				rmh = 0.5 * (self.X[0, 1:-1] + self.X[0, :-2])
				dqx = (rph * self._conductivity_average(k[0, 2:], k[0, 1:-1]) * (T[0, 2:] - T[0, 1:-1])
				       - rmh * self._conductivity_average(k[0, 1:-1], k[0, :-2]) * (T[0, 1:-1] - T[0, :-2])
				       ) / self.X[0, 1:-1] / (self.dx * self.dx)

			dqx *= self.dt / rhoc[0, 1:-1]

			skindepth = self.skin_depth(self.Tsurf, self.S[0, :], self.dt)  # self.constants.solar_day)

			# heat lost to space due to infrared radiation
			radiation = self._radiative_heat_loss(T[0, :])

			# heat lost to space by sublimation of surface ice
			sublimation = self._sublimation_heat_loss(T[0, :])

			# source term for surface so that heat lost by sublimation/radiation are balanced by the heat flow from
			#  the cell below. This ideally helps to keep the equilibrium surface temperature chosen at the start of
			#  the simulation.
			kinit = self.k(self.phi_initial, self.T_initial, self.S_initial)
			q0 = self._conductivity_average(kinit[0, :], kinit[1, :]) \
			     * (self.T_initial[1, :] - self.T_initial[0, :]) / self.dz
			qout = (radiation - self._radiative_heat_loss(self.T_initial[0, :])
			        + sublimation - self._sublimation_heat_loss(self.T_initial[0, :])
			        + q0)

			if self.use_orbits:
				########################################################################################
				# More complex implementation of radiative boundary that takes time/orbit into account #
				# Currently not used as its not very applicable at typical time/space steps            #
				########################################################################################
				# convert all degs to rads
				obl = np.deg2rad(self.constants.obliq)
				Lp = np.deg2rad(self.constants.lonPer)
				lat = np.deg2rad(self.constants.latitude)
				# heat imparted on surface ice by solar radiation
				solar_flux = self._solar_flux()
				_h = (2 * np.pi * self.model_time / self.constants.solar_day) % (2 * np.pi)
				_x = self.constants.rAU * (1 - self.constants.ecc * self.constants.ecc)

				self.nu = self.nudot * self.dt
				_r = _x / (1 + self.constants.ecc * np.cos(self.nu))
				self.nudot = (self.constants.GM * _x) ** 0.5 / (_r * _r)

				_dec = np.arcsin(np.sin(obl) * np.sin(self.nu + Lp))
				_c = np.sin(lat) * np.sin(_dec) \
				     + np.cos(lat) * np.cos(_dec) * np.cos(_h)
				_c = 0.5 * (_c + np.abs(_c))
				solar_flux *= _c * (self.constants.rAU / _r) * (self.constants.rAU / _r)
				# account for thermal inertia?
				solar_flux *= skindepth / self.dz
				qout -= solar_flux

			# ratio = skindepth / self.dz
			# dqz = (qn - q0) * (1 - ratio) - (radiation + sublimation - solar_flux) * ratio

			dqz = self._conductivity_average(k[1, 1:-1], k[0, 1:-1]) * (T[1, 1:-1] - T[0, 1:-1]) / (self.dz *
			                                                                                        self.dz) - qout[
			                                                                                                   1:-1] / self.dz

			dqz *= self.dt / rhoc[0, 1:-1]

			self.T[0, 1:-1] = T[0, 1:-1] + dqz + dqx + self.dt * self.Q[0, 1:-1] / rhoc[0, 1:-1]

		# END TOP BOUNDARY CONDITIONS      #
		####################################

		#######################################
		# APPLY BOUNDARY CONDITIONS AT SIDES #
		if self.sidesBC == True:
			self.T[:, 0] = self.T_initial[:, 0]
			self.T[:, -1] = self.T_initial[:, -1]

		elif self.sidesBC == "NoFlux":
			self.T[:, 0] = self.T[:, 1]
			self.T[:, -1] = self.T[:, -2].copy()

		elif self.sidesBC == "LNoFlux":
			self.T[:, 0] = self.T[:, 1]
			self.T[:, -1] = self.T_initial[:, -1]

		elif self.sidesBC == "RFlux":
			# left boundary
			self.T[:, 0] = self.T[:, 1]

			# right boundary
			kinit_R = self.ki(self.T_initial[1:-1, -1], self.S_initial[1:-1, -1])

			if self.coords in ["cartesian", "xz", "zx"]:
				dqx = self._conductivity_average(k[1:-1, -1], kinit_R) * (self.T_initial[1:-1, -1] - T[1:-1, -1]) \
				      * self.dx / self.dL \
				      - self._conductivity_average(k[1:-1, -1], k[1:-1, -2]) * (T[1:-1, -1] - T[1:-1, -2])
				dqx /= self.dx * self.dx

			if self.coords in ["cylindrical", "zr", "rz"]:
				rph = 0.5 * (self.X[1:-1, -1] + self.X[1:-1, -1] + self.dL) * self.dx / self.dL
				rmh = 0.5 * (self.X[1:-1, -1] + self.X[1:-1, -2])
				dqx = (rph * self._conductivity_average(k[1:-1, -1], kinit_R) * (self.T_initial[1:-1, -1] - T[1:-1, -1])
				       - rmh * self._conductivity_average(k[1:-1, -1], k[1:-1, -2]) * (T[1:-1, -1] - T[1:-1, -2])
				       ) / self.X[1:-1, -1] / (self.dx * self.dx)

			dqx *= self.dt / rhoc[1:-1, -1]

			dqz = self._conductivity_average(k[2:, -1], k[1:-1, -1]) * (T[2:, -1] - T[1:-1, -1]) \
			      - self._conductivity_average(k[1:-1, -1], k[:-2, -1]) * (T[1:-1, -1] - T[:-2, -1])
			dqz *= self.dt * dqz / (self.dz * self.dz) / rhoc[1:-1, -1]

			self.T[1:-1, -1] = T[1:-1, -1] + dqz + dqx + self.dt * self.Q[1:-1, -1] / rhoc[1:-1, -1]

	# END SIDES BOUNDARY CONDITIONS    #
	####################################

	def _conductivity_average(self, k1, k2):
		"""Averages the thermal conductivity across adjacent cells for flux conservation. Generally for internal use
		during simulation.

		Note: other means are commented out in the function so you can use whichever. Geometric mean tends to be <
		arithmetic mean, meaning slower heat transfer. Thus, arithmetic mean was chosen so that rate of heat transfer
		was highest and thus gives a faster freezing rate

		Parameters
		----------
		k1 : np.ndarray, float
			Thermal conductivity, W/m/K
		k2 : np.ndarray, float
			Thermal conductivity, W/m/K

		Returns
		---------
			Arithmetic mean of thermal conductivity k1 and k2, W/m/K
		"""
		# Geometric mean
		# return (k1 * k2) ** 0.5
		# Harmonic mean
		# return 2 * k1 * k2 / (k1 + k2)
		# Arithmetic mean
		return 0.5 * (k1 + k2)

	def _get_gradients(self, T, k, rhoc, Tll=None):
		"""Calculates the flux-conservative thermal gradients (∇q = ∇(k∇T)) for a multi-phase system with different
		thermophysical properties and a temperature-dependent thermal conductivity in the internal nodes of the mesh.
		Internal use only.

		Parameters
		----------
		T : np.ndarray
			(k-1) iteration or current time step of temperature grid, K
		k : np.ndarray
			Phase and temperature-dependent thermal conductivity grid, W/m/K
		rhoc : np.ndarray
			Phase and temperature-dependent density * specific heat grid, J/K/m^3
		Tll : np.ndarray
			Temperature at time step n-1, used only when coordinates are cylindrical, K

		Returns
		----------
		out : np.ndarray
			Heat fluxes
		"""
		if self.coords in ["cartesian", "xz", "zx"]:
			dqx = self._conductivity_average(k[1:-1, 2:], k[1:-1, 1:-1]) * (T[1:-1, 2:] - T[1:-1, 1:-1]) / (
						self.dx * self.dx) \
			      - self._conductivity_average(k[1:-1, 1:-1], k[1:-1, :-2]) * (
					      T[1:-1, 1:-1] - T[1:-1, :-2]) / (self.dx * self.dx)
			self._get_gradients = self._get_gradients_cart

		elif self.coords in ["cylindrical", "zr", "rz"]:
			# Must account for curvature in a cylindrical system
			#  see pg. 252 in Langtangen & Linge (2017) "Finite Difference Computing with PDEs"
			dqx = (self._rph * self._conductivity_average(k[1:-1, 2:], k[1:-1, 1:-1]) * (T[1:-1, 2:] - T[1:-1, 1:-1])
			       - self._rmh * self._conductivity_average(k[1:-1, 1:-1], k[1:-1, :-2]) * (
					       T[1:-1, 1:-1] - T[1:-1, :-2])
			       ) / (self.dx * self.dx)
			self._get_gradients = self._get_gradients_cyl

		dqz = self._conductivity_average(k[2:, 1:-1], k[1:-1, 1:-1]) * (T[2:, 1:-1] - T[1:-1, 1:-1]) / (
					self.dz * self.dz) \
		      - self._conductivity_average(k[1:-1, 1:-1], k[:-2, 1:-1]) * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / (
					      self.dz * self.dz)

		return dqx + dqz

	def _get_gradients_cart(self, T, k, rhoc, Tll=None):
		dqx = self._conductivity_average(k[1:-1, 2:], k[1:-1, 1:-1]) * (T[1:-1, 2:] - T[1:-1, 1:-1]) / (
					self.dx * self.dx) \
		      - self._conductivity_average(k[1:-1, 1:-1], k[1:-1, :-2]) * (T[1:-1, 1:-1] - T[1:-1, :-2]) / (
					      self.dx * self.dx)
		dqz = self._conductivity_average(k[2:, 1:-1], k[1:-1, 1:-1]) * (T[2:, 1:-1] - T[1:-1, 1:-1]) / (
					self.dz * self.dz) \
		      - self._conductivity_average(k[1:-1, 1:-1], k[:-2, 1:-1]) * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / (
					      self.dz * self.dz)
		return dqx + dqz

	def _get_gradients_cyl(self, T, k, rhoc, Tll=None):
		dqx = (self._rph * self._conductivity_average(k[1:-1, 2:], k[1:-1, 1:-1]) * (T[1:-1, 2:] - T[1:-1, 1:-1])
		       - self._rmh * self._conductivity_average(k[1:-1, 1:-1], k[1:-1, :-2]) * (T[1:-1, 1:-1] - T[1:-1, :-2])
		       ) / (self.dx * self.dx)
		dqz = self._conductivity_average(k[2:, 1:-1], k[1:-1, 1:-1]) * (T[2:, 1:-1] - T[1:-1, 1:-1]) / (
					self.dz * self.dz) \
		      - self._conductivity_average(k[1:-1, 1:-1], k[:-2, 1:-1]) * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / (
					      self.dz * self.dz)
		return dqx + dqz

	def _print_all_opts(self, nt):
		"""Prints options chosen for simulation.

		Parameters
		----------
		nt : int
			Number of time steps to perform

		Returns
		----------
		None
		"""
		stringIO = lambda bin: 'on' if bin else 'off'
		stringBC = lambda BC: BC if isinstance(BC, str) else 'Dirichlect'
		print("=" * 72)
		print('Starting simulation with')
		print(f'\t Ice shell thickness: {self.D} m')
		print(f'\t Lateral domain size: {self.w} m')
		print(f'\t    dz = {self.dz} m\n\t    dx = {self.dx} m')
		print(f'\t Surface temperature: {self.Tsurf} K')
		print(f'\t Bottom temperature:  {self.Tbot} K')
		print("-" * 72)
		print('\t Intrusion/sill:')
		try:
			self.geom
			print(f'\t    radius:    {self.radius} m')
			print(f'\t    thickness: {self.thickness} m')
			print(f'\t    depth:     {self.depth} m')
		except AttributeError:
			pass
		print("-" * 72)
		print('\t    Salinity: {}'.format(stringIO(self.issalt)))
		if self.issalt:
			print(f'\t       composition:    {self.composition}')
			print(f'\t       concentration:  {self.concentration}ppt')
		print("-" * 72)
		print('\t Thermal properties:')
		print("\t   Ice")
		print(f"\t    ki(T, S)   = {str(self.constants.ac) + '/T' if self.kT is True else self.constants.ki} ["
		      f"W/m/K]")
		conc = self.concentration if self.issalt else 0
		_ = [self.rhoi(self.Tsurf + 1, conc) - self.rhoi(self.Tsurf, conc),
		     self.rhoi(self.Tsurf, conc + 1) - self.rhoi(self.Tsurf, conc)]
		print(f"\t    rhoi(T, S) = {self.constants.rho_i}{f'+ {_[0]:0.04f}T' if _[0] != 0 else ''}"
		      f"{f'+ {_[1]:0.04f}S' if _[1] != 0 else ''} [kg/m^3]")
		_ = [self.cpi(self.Tsurf + 1, conc) - self.cpi(self.Tsurf, conc),
		     self.cpi(self.Tsurf, conc + 1) - self.cpi(self.Tsurf, conc)]
		print(f"\t    cpi(T, S)  = {self.constants.cp_i}{f'+ {_[0]:0.04f}T' if _[0] != 0 else ''} "
		      f"{f'+ {_[1]:0.04f}S' if _[1] != 0 else ''} [J/kg/K]")
		print("\t   Water")
		print(f"\t    kw(T, S)   = {self.constants.kw} [W/m/K]")
		_ = [self.rhow(self.Tsurf + 1, conc) - self.rhow(self.Tsurf, conc),
		     self.rhow(self.Tsurf, conc + 1) - self.rhow(self.Tsurf, conc)]
		print(f"\t    rhow(T, S) = {self.constants.rho_w}{f'+ {_[0]:0.04f}T' if _[0] != 0 else ''}"
		      f"{f'+ {_[1]:0.04f}S' if _[1] != 0 else ''} [kg/m^3]")
		_ = [self.cpw(self.Tsurf + 1, conc) - self.cpw(self.Tsurf, conc),
		     self.cpw(self.Tsurf, conc + 1) - self.cpw(self.Tsurf, conc)]
		print(f"\t    cpw(T, S)  = {self.constants.cp_i}{f'+ {_[0]:0.04f}T' if _[0] != 0 else ''} "
		      f"{f'+ {_[1]:0.04f}S' if _[1] != 0 else ''} [J/kg/K]")
		print("-" * 72)
		print('\t Sources/sinks:')
		print(f'\t    tidal heating:  {stringIO(self.TIDAL_HEAT)}')
		# print('\t    latent heat:    {}'.format(stringIO(self.L)))
		print("-" * 72)
		print('\t Boundary conditions:')
		print(f'\t    top:     {stringBC(self.topBC)}')
		print(f'\t    bottom:  {stringBC(self.botBC)}')
		print(f'\t    sides:   {stringBC(self.sidesBC)}')
		print("-" * 72)
		print('\t Outputs:')
		try:
			print(f'\t     temporary directory:  {self.outputs.tmp_data_directory}')
			print(f'\t     file ID#:             {self.outputs.tmp_data_file_name.split("tmp_data_")[1]}')
			print(f'\t     requested outputs:    {list(self.outputs.outputs.keys())}')
			print(f'\t     output frequency:     {self.outputs.output_frequency} steps')
			print(f'\t     expected # outputs:   {int(np.ceil(nt / self.outputs.output_frequency))}')
		except AttributeError:
			print('No outputs requested')
		print("-" * 72)
		print('\t Other:')
		print(f'\t    stop on freeze:  {stringIO(self.FREEZE_STOP)}')
		print(f'\t    adapt time step: {stringIO(self.ADAPT_DT)}')
		print('\t  Convergence tolerances:')
		print(f'\t    temperature:     {self.TTOL}')
		print(f'\t    liquid fraction: {self.PHITOL}')
		if self.issalt:
			print(f'\t    salinity:        {self.STOL}')
		print(f'\t Total model run time: {nt * self.dt} s, {(nt * self.dt) / self.constants.styr} yr')
		print(f'\t             with dt = {self.dt} s')
		print("=" * 72)

	def _error_check(self):
		"""Checks whether simulation is within assumptions, physically consistent, and with no major errors.

		Parameters
		----------
		None

		Returns
		----------
		None
		"""
		# check all dependent variables
		if (np.isnan(self.T).any() or np.isnan(self.S).any() or
				np.isnan(self.phi).any()):
			raise Exception("Something went wrong... Check time step? may be too large")
		if (self.T[self.T <= 0].shape[0] > 0 or
				self.T[self.T >= 1e3].shape[0] > 0):
			if self.T[self.T <= 0].shape[0] > 0:
				ERROR_MSG = "T < 0 K"
			if (self.T[self.T >= 1e3].shape[0] > 0):
				ERROR_MSG = "T > 1000 K. Probably doesn't make sense for ice. Check boundary conditions, Q, etc."
			raise Exception(f"Unphysical temperatures: {ERROR_MSG}. Check time step, may be too large")
		if (self.phi[self.phi < 0].shape[0] > 0 or
				self.phi[self.phi > 1].shape[0] > 0):
			raise Exception("Unphysical phase: 0 <= phi <= 1")
		if self.issalt:
			if (self.S[self.S < 0].shape[0] > 0 or
					self.S[self.S > self.saturation_point].shape[0] > 0):
				if self.S[self.S < 0].shape[0] > 0:
					ERROR_MSG = 'S < 0.'
				elif self.S[self.S > self.saturation_point].shape[0] > 0:
					ERROR_MSG = f'S > {self.saturation_point} ppt. Model assumes ice can only entrain as much as the ' \
					            f'saturation_point variable.'
				raise Exception(f"Unphysical salt content: {ERROR_MSG}  Check fit constants in SalinityConstants.py")
		# check all independent/input variables
		if self.model_time == 0:
			for k, v in self.constants.__dict__.items():
				if isinstance(v, (float, int)):
					if v < 0:
						raise Exception(f"Consistency error: Constant {k} is < 0 ({v})")

	def time_step(self):
		"""Perform one time step during simulation. Calculates all physical properties, phase, etc. and iterates until
		temperature and phase fields converge on an error for ITER_KILL iterations for a more accurate solution.

		Parameters
		----------
		None

		Returns
		----------
		None
		"""
		# set errors to the largest possible value
		TErr, phiErr = np.inf, np.inf
		# save n-1 time step grids for iteration
		T_last, phi_last = self.T.copy(), self.phi.copy()
		# k_last, rhoc_last = self.k.copy(), self.rhoc.copy()
		rhoc_last = self.rhoc(phi_last, T_last, self.S)
		k_last = self.k(phi_last, T_last, self.S)
		iter_k = 0
		# save the n-2 time step grid for temperature if using cylindrical coordinate system
		if self.coords in ["cylindrical", "zr", "rz"]:
			Tll = T_last.copy()
		while (TErr > self.TTOL or phiErr > self.PHITOL):
			# calculate the fluxes in each grid cell
			dflux = self._get_gradients(T_last, k_last, rhoc_last, None)

			# update the liquid fraction
			self._update_liquid_fraction(phi_last=phi_last)
			# update salinity: entrain, reject, and mix salts
			if self.issalt:
				self._update_salinity(phi_last=phi_last)

			# update the sources and sinks
			self.Q = self._update_sources_sinks(phi_last=phi_last, T=T_last, Tm=self.Tm)

			# calculate new temperatures at time step n
			self.T[1:-1, 1:-1] = T_last[1:-1, 1:-1] + self.dt * dflux / rhoc_last[1:-1, 1:-1]
			self.T += self.Q * self.dt / rhoc_last

			# apply the boundary conditions
			self._apply_boundaryconditions(T_last, k_last, rhoc_last)

			# update the specific heat
			rhoc_last = self.rhoc(self.phi, self.T, self.S)

			# update the thermal conductivity
			k_last = self.k(self.phi, self.T, self.S)

			# calculate the error for convergence
			TErr = (abs(self.T - T_last)).max()
			phiErr = (abs(self.phi - phi_last)).max()

			# kill statement when parameters won't allow solution to converge
			if iter_k > self.ITER_KILL:
				raise Exception(f"solution not converging,\n\t iterations = {iter_k}\n\t T error ={TErr}\n\t phi "
				                f"error = {phiErr}")
			# check for errors
			self._error_check()

			iter_k += 1
			Tll = T_last.copy()
			T_last, phi_last = self.T.copy(), self.phi.copy()

		# outputs here
		self._niter(iter_k)
		self.model_time += self.dt

	def solve_heat(self, nt: int = None, dt: float = None, save_progress: float = 25, final_time: float = None):
		"""Solves the 2D heat conduction equation with phase change.

		Parameters
		----------
		nt : int
			Number of total iterations

		dt : float
			Time step, s

		save_progress : int
			This determines the frequency, based on the percent the original water body is frozen, with which to save
			the model into a temporary output file. Currently this is used only if running a simulation to freeze a
			water body.

		final_time : float
			Time to run simulation to until stopping.

		Returns
		----------
		None
		"""
		if (nt is None and final_time is None):
			raise Exception("Please choose the number of time steps or the final time")

		if dt is None and self.ADAPT_DT:
			old_of = self.dt * self.outputs.output_frequency
			self._set_dt(self.CFL)
			self.outputs.output_frequency = int(old_of / self.dt)
		else:
			pass

		if (nt is None and final_time is not None):
			nt = int(np.ceil(final_time / self.dt))
		elif (nt is not None and final_time is None):
			final_time = np.ceil(nt * self.dt)

		V0 = self.phi_initial.sum()
		t0 = self.model_time
		n0 = int(0) if t0 == 0 else int(t0 / self.dt)
		if self.verbose and n0 == 0:
			self._print_all_opts(nt)
			print(f" Starting simulation for {final_time} s ({final_time / 3.154e7} yr)")

		n = n0
		while (n < nt) or (self.model_time < t0 + final_time):
			start_time = _timer_.perf_counter()
			if self.ADAPT_DT:
				self._set_dt(self.CFL)
				self.outputs.output_frequency = int(old_of / self.dt + 1)
			self.time_step()
			try:  # save outputs
				self.outputs.get_results(self, n)
			except AttributeError:  # no outputs chosen
				pass

			if self.FREEZE_STOP:
				if (self.phi[self.geom] == 0).all():
					print(f"Instrusion frozen at {self.model_time:0.04f} s")
					print(f"  run time: {self.run_time} s")
					return

			# save progress, in case of a long run time, can restart with saved model!
			if isinstance(save_progress, (int, float)):
				pct_frozen = 100 * (1 - self.phi.sum() / V0)
				if pct_frozen % save_progress <= 1e-3:
					# if save_progress - 1e-3 <= pct_frozen <= save_progress + 1e-3:
					if self.verbose: print(f">> Saving progress at {pct_frozen}%")
					filename = self.outputs.tmp_data_directory + f"md_{self.outputs.tmp_data_file_name.split('tmp_data_')[1]}" \
					           + ".pkl"
					with open(filename, 'wb') as output:
						pickle.dump(self, output, -1)
						output.close()
			# uf.save_data(self, "model_runID{}.pkl".format(self.outputs.tmp_data_file_name.split("runID")[1]),
			#          self.outputs.tmp_data_directory, final=0)

			self.run_time += _timer_.perf_counter() - start_time  # - self.run_time
			n = n + 1
		return
