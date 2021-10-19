import numpy as np
from scipy import optimize
from HeatSolver import HeatSolver
import constants
import SalinityConstants
from ThermophysicalProperties import ThermophysicalProperties


class ShallowWater(HeatSolver, ThermophysicalProperties):
	"""Class with methods to set up initial conditions for two-dimensional, two-phase thermal diffusion model that
	includes temperature-dependent conductivity and salinity. Includes the HeatSolver class used to solve the heat
	equation utilizing an enthalpy method (Huber et al., 2008) to account for latent heat from phase change as well
	as a parameterization for a saline system."""

	def __init__(
			self, w, D, dx=None, dz=None, nx=None, nz=None, kT=True,
			use_X_symmetry=True, coordinates="cartesian", verbose=False):
		"""Initialize the system.
		Parameters
		----------
			w : float
				Width of horizontal spatial domain, m
			D : float
				Thickness of shell, length of vertical spatial domain, m
			dx : float
				horizontal spatial step size, m
			dz : float
				vertical spatial step size, m
			nx : int
				Number of grid points in the x-direction
			nz : int
				Number of grid points in the z-direction
			kT : bool
				Choose whether to use temperature-dependent thermal conductivity.
				Default = True, temperature-dependent, k=ac/T (Petrenko, Klinger, etc.)
			use_X_symmetry : bool
				assume the system is symmetric about the center of the intrusion
				* NOTE: Must use 'NoFlux' boundary condition for sides if using this
			coordinates : str
				Choose the cooridnate system assumption. Currently cartesian and polar (in the z-r plane) is
				implemented.
				Options: "xz"/"cartesian", "zr"/"cylindrical"
			verbose : bool
				Prints logs during simulation and some function calls. May make simulations slightly slower.
				Default = False.
		Usage
		----------
			Ice Shell is 40 km thick and 40 km wide at a spatial discretization of 50 m.
				>>> model = IceSystem(40e3, 40e3, 50, 50)
			Ice shell is 40 km thick and 40 km wide with a 200x100 grid
				>>> model = IceSystem(40e3, 40e3, nx=200, nz=100)

			See README
		"""
		super().__init__()
		# Create constants for simulations
		self.constants = constants.constants()
		self.model_time, self.run_time = 0, 0
		# generally a stable time step for most simulations, likely too small for many runs
		#  derived from 1D CFL condition: dt < min(dx,dz)**2 / max(k/rho/cp) / 6
		self.dt = self.constants.styr / 52
		# save coordinate system used
		self.coords = coordinates
		self.w = w
		self.D = D
		self.kT = kT  # k(T) I/O
		self.symmetric = use_X_symmetry
		self.verbose = verbose
		self.issalt = False

		if nx is None and nz is None:
			self.dx, self.dz = dx, dz
			self.nx, self.nz = int(self.w / self.dx + 1), int(self.D / self.dz + 1)
			if self.verbose:
				print(f" dx = {dx}m => nx = {self.nx}\n dz = {dz}m => nz = {self.nz}")
		elif dx is None and dz is None:
			self.nx, self.nz = nx, nz
			self.dx, self.dz = self.w / (self.nx - 1), self.D / (self.nz - 1)
			if self.verbose:
				print(f"  nx = {nx} => dx = {self.dx}m\n  nz = {nz} => dz = {self.dz}m")
		else:
			raise Exception("Error: Must choose either BOTH nx AND nz, or dx AND dz")

		self._init_grids()

	def _init_grids(self):
		"""Class method to initialize grids given the choices in initiating the object"""
		# Create spatial grid
		if self.symmetric:  # assuming horizontally symmetric heat loss about x=0
			self.w = self.w / 2
			self.nx = int(self.w / self.dx + 1)
			if self.verbose:
				print(f"  using horizontal symmetry, creating {self.nz, self.nx} grid")

			self.X, self.Z = np.meshgrid(np.array([i * self.dx for i in range(self.nx)], dtype=float),
			                             np.array([j * self.dz for j in range(self.nz)], dtype=float))

		elif self.symmetric is False:  # x domain centered on 0
			self.X, self.Z = np.meshgrid(np.array([-self.w / 2 + i * self.dx for i in range(self.nx)], dtype=float),
			                             np.array([j * self.dz for j in range(self.nz)], dtype=float))

		# Create scalar field grids
		self.T = self.constants.Tm * np.ones((self.nz, self.nx), dtype=float)  # initialize domain at one temperature
		self.S = np.zeros((self.nz, self.nx), dtype=float)  # initialize domain with no salt
		self.phi = np.zeros((self.nz, self.nx), dtype=float)  # initialize domain as ice

	def tidal_heating(self, phi=None, T=None, Tm=None):
		"""Temperature-dependent internal tidal heating for an incompressible Maxwell body applied to an ice shell as
		given by Mitri & Showman (2005).

		Parameters
		----------
		phi: float, numpy.ndarray
			Volumetric liquid fraction
		T: float, numpy.ndarray
			Temperature of ice, K
		Tm:
			Melting temperature, K

		Returns
		----------
		out: float, numpy.ndarray
			Internal heating due to tides, W / m^2
		"""
		if phi is None:
			phi = self.phi
		if T is None:
			T = self.T
		if Tm is None:
			Tm = self.Tm
		self.viscosity = self.volume_average(phi, self.constants.visc0w, self.ice_viscosity(T, Tm))
		return (self.constants.eps0 ** 2 * self.constants.omega ** 2 * self.viscosity) \
		       / (2 + 2 * self.constants.omega ** 2 * self.viscosity ** 2 / (self.constants.G ** 2))

	@staticmethod
	def volume_average(phi, water_prop, ice_prop):
		"""Volume average water and ice properties based on the volumetric liquid fraction. Generally used for
		thermophysical properties. Used in several places during simulations so collected into one function for
		readability.

		Parameters
		----------
		phi: float, numpy.ndarray
			Volumetric liquid fraction
		water_prop: float, numpy.ndarray
			A physical property of water, e.g. thermal conductivity
		ice_prop: float, numpy.ndarray
			A physical property of ice, e.g. thermal conductivity

		Returns
		----------
		out : float, numpy.ndarray
			Volume averaged physical property

		Usage
		----------
			>>> density = model.volume_average(model.phi, model.constants.rho_w, model.constants.rho_i)
		"""
		return phi * water_prop + (1 - phi) * ice_prop

	def k(self, phi=None, T=None, S=None):
		"""Volume averaged thermal conductivity. Used in several places during simulations.

		Parameters
		----------
		phi: float, numpy.ndarray
			Volumetric liquid fraction
		T: float, numpy.ndarray
			Temperature, K
		S: float, numpy.ndarray
			Salinity, ppt

		Returns
		----------
		out : float, numpy.ndarray
			Volume averaged thermal conductivity
		"""
		if phi is None:
			phi = self.phi
		if T is None:
			T = self.T
		if S is None:
			S = self.S
		return self.volume_average(phi, self.kw(T, S), self.ki(T, S))

	def rho(self, phi=None, T=None, S=None):
		"""Volume averaged density.

		Parameters
		----------
		phi: float, numpy.ndarray
			Volumetric liquid fraction
		T: float, numpy.ndarray
			Temperature, K
		S: float, numpy.ndarray
			Salinity, ppt

		Returns
		----------
		out : float, numpy.ndarray
			Volume averaged density, kg/m^3
		"""
		if phi is None:
			phi = self.phi
		if T is None:
			T = self.T
		if S is None:
			S = self.S
		return self.volume_average(phi, self.rhow(T, S), self.rhoi(T, S))

	def cp(self, phi=None, T=None, S=None):
		"""Volume averaged specific heat capacity.

		Parameters
		----------
		phi: float, numpy.ndarray
			Volumetric liquid fraction
		T: float, numpy.ndarray
			Temperature, K
		S: float, numpy.ndarray
			Salinity, ppt

		Returns
		----------
		out : float, numpy.ndarray
			Volume averaged specific heat capacity, J/m^3/K
		"""
		if phi is None:
			phi = self.phi
		if T is None:
			T = self.T
		if S is None:
			S = self.S
		return self.volume_average(phi, self.cpw(T, S), self.cpi(T, S))

	def rhoc(self, phi=None, T=None, S=None):
		"""Volume averaged volumetric heat capacity. Used in several places during simulations.

		Parameters
		----------
		phi: float, numpy.ndarray
			Volumetric liquid fraction
		T: float, numpy.ndarray
			Temperature, K
		S: float, numpy.ndarray
			Salinity, ppt

		Returns
		----------
		out : float, numpy.ndarray
			Volume averaged volumetric heat capacity, J/m^3/K
		"""
		if phi is None:
			phi = self.phi
		if T is None:
			T = self.T
		if S is None:
			S = self.S
		return self.volume_average(phi, self.rhow(T, S) * self.cpw(T, S), self.rhoi(T, S) * self.cpi(T, S))

	def _save_initials(self):
		"""
		Saves the initial values for each independent variable that evolves over time. Generally internal use only.
		"""
		self.T_initial = self.T.copy()
		self.phi_initial = self.phi.copy()
		self.S_initial = self.S.copy()
		if self.verbose: print(f"Initial conditions saved.")

	def init_T(self, Tsurf, Tbot, profile="non-linear", real_D=None):
		"""Initialize an equilibrium thermal profile of the brittle portion of an ice shell of D thickness, with surface
		temperature Tsurf and basal temperature Tbot.
		Parameters
		----------
		Tsurf: float, int
			Surface temperature of brittle ice shell, K
		Tbot: float, int
			Temperature at bottom of brittle ice shell, K
		profile: str, numpy.ndarray
			The kind of profile wanted.
			Options
				- "non-linear" (Default): Tsurf*(Tbot/Tsurf)**(Z/D) - equilibrium profile assumes a that ice thermal
				conductivity is inversely proportional to the temperature (k_i ~ 1/T)
				- "linear": (Tbot - Tsurf) * Z/D + Tsurf - equilibrium profile assumes constant ice thermal conductivity
				- "stefan": T(z>=dz) = Tm, T(z=0) = Tsurf - assumes domain is all liquid except for first row of grid
				- numpy.ndarray([]) : profile generated by user, this will be the background profile.
		 real_D: float
			Used only if using a portion of a much larger ice shell.

			For instance, an ice shell of > 50 km may be too large to simulate, but if thermal evolution of interest
			happens in the upper 1 km, we set real_D = 50e3 and it will give an accurate profile for the first 1 km.

		Returns
		----------
		out : numpy.ndarray
			Temperature distribution in ice shell, K

		Usage
		----------
		An shell with 100 K surface temperature and 273 K basal temperature assuming a non-linear ice thermal
		conductivity
			>>> model.init_T(100, 273)
		"""
		self.Tsurf = Tsurf
		self.Tbot = Tbot
		self.Tm = self.constants.Tm * np.ones(self.T.shape, dtype=float)

		if isinstance(profile, str):
			if profile == "non-linear":
				if real_D != None:
					self.Tbot = Tsurf * (Tbot / Tsurf) ** (self.D / real_D)
				self.T = Tsurf * (self.Tbot / Tsurf) ** (self.Z / self.D)

			elif profile == "linear":
				if real_D != None:
					self.Tbot = (Tbot - Tsurf) * (self.D / real_D) + Tsurf
				self.T = (Tbot - Tsurf) * abs(self.Z / self.D) + Tsurf

			elif profile == "stefan":
				self.T[0, :] = Tsurf
				self.T[1:, :] = Tbot
				self.phi[1:, :] = 1
				profile += " plus domain all water"

			if self.verbose:
				print(f"init_T(Tsurf = {self.Tsurf:0.03f}, Tbot={self.Tbot:0.03f})")
				print(f"\t Temperature profile initialized to {profile}")
		else:
			self.T = profile
			if self.verbose: print("init_T: Custom profile implemented")

		self.Tedge = self.T[:, 0]
		self._save_initials()
		self._set_dt(self.CFL)

	def _set_geometry(self, geometry, **kwargs):
		"""
		Sets and saves geometry of intrusion (sill, fracture, etc.). Functionally only used by init_intrusion function
		Parameters
		----------
		geometry: str
			String descriptive of the assumed sill geometry. Chosen in init_intrusion

		Returns
		---------
		None

		"""
		if isinstance(geometry, str):
			# adjust so center is at x=dx rather than on the boundary
			h = (self.thickness + self.dz) / 2
			if geometry == "ellipse":
				# Since we define depth of emplacement as the depth of the upper edge of the intrusion, the center will
				# be at location thickness / 2 + depth
				center = self.thickness / 2 + self.depth
				try:
					if self.symmetric: _R = self.X - self.dx
				except AttributeError:
					_R = self.X
				# Set and save geometry of ellipse with specified geometry
				self.geom = np.where((_R / self.radius) ** 2 + ((self.Z - center) / h) ** 2 <= 1)

			elif geometry in ["box", "sheet"]:
				try:
					if self.symmetric: radius = self.radius + self.dx
				except AttributeError:
					radius = radius
				self.geom = np.where(np.logical_and(self.X <= radius,
				                                    np.logical_and(self.depth <= self.Z,
				                                                   self.Z <= self.depth + self.thickness)
				                                    )
				                     )

			elif geometry in ["schmidt2011", "chaos"]:
				ir = kwargs["inner_radius"]
				m = (self.D - self.depth) / (self.radius - ir)
				b = self.depth - m * ir
				if self.symmetric:
					self.geom = np.where(np.logical_and(self.Z >= m * self.X + b,  # angled edge!
					                                    np.logical_and(self.X <= self.radius + self.dx,
					                                                   # horizontal extent
					                                                   np.logical_and(self.depth <= self.Z,
					                                                                  self.Z <= self.depth + h))
					                                    )
					                     )
				else:
					raise Exception("This is not implemented for non-symmetric systems yet :(")
				self.thickness /= 2.

			elif geometry == "fracture":
				m = self.fracture_height / self.fracture_width
				b = self.D - self.fracture_height
				try:
					if self.symmetric:
						self.geom = np.where(self.Z >= m * (self.X + self.dx) + b)
				except AttributeError:
					self.geom = np.where(np.logical_and(self.Z >= m * self.X + b,
					                                    self.Z <= -m * self.X + b)
					                     )

			if self.verbose:
				print(f"Intrusion geometry set as {geometry}")
				if geometry in ["ellipse", "box", "sheet", "chaos", "schmidt2011"]:
					print(f"\t  depth: {self.depth:0.01f}m\n\t  thickness: {self.thickness:0.01f}m \n\t  radius: "
					      f"{self.radius:0.01f}m")
				else:
					print(f"\t fracture width: {self.fracture_width:0.01}m\n\t fracture height: "
					      f"{self.fracture_height:0.01f}m")

		else:
			self.geom = geometry
			if self.verbose: print("set_geometry: Custom geometry implemented")

	def init_intrusion(self, depth, radius, thickness=None, phi=1, geometry="ellipse", T=None, **kwargs):
		"""
		Initialize the intrusion geometry.

		Parameters
		----------
		depth: float
			Depth of emplacement, defined as the depth of the upper edge of the intrusion, m
			Note: if using for fractures (i.e. geometry = "fracture"), depth is now the fracture_height!
		radius: float
			Radius of intrusion, m
			Note: if using for fracture, radius is now the fracture_width (or half-width)
		thickness: float
			Thickness of intrusion, m
		phi: float
		    Liquid fraction of intrusion. Would not reccommend below 1 (which is default) but is possible.
		geometry: str
			The assumed geometry of the sill.
			Current options: "ellipse", "sheet" or "box", "fracture"
		T : float
			Assumed temperature of the intrusion, K. Defaults to melting temperature.

		Returns
		----------
		out :

		Usage
		----------
		A 1000 m thick, sheet-like sill at 500 m below the surface and 2000 m wide
		>>> model.init_intrusion(500, 2000, 1000, geometry="sheet")
		"""
		if phi < 0 or phi > 1:
			raise Exception("Liquid fraction must be between 1 and 0")

		self.intrusion_T = self.constants.Tm if T is None else T
		if geometry in ["ellipse", "box", "sheet", "chaos", "schmidt2011"]:
			if thickness == None:
				raise Exception("init_intrusion(): Must choose a sill thickness")

			if geometry in ["chaos", "schmidt2011"] and "inner_radius" not in kwargs:
				raise Exception(f"init_intrusion():Please choose the inner radius for this geometry.\n   e.g.,"
				                f"init_intrusion(depth={depth}, radius={radius}, thickness={thickness}, "
				                f"geometry={geometry}, inner_radius=3e3)")

			self.depth = depth
			self.thickness = thickness
			self.radius = radius

		elif geometry == "fracture":
			self.fracture_width = radius
			self.fracture_height = depth

		self._set_geometry(geometry, **kwargs)
		self.T[self.geom] = self.intrusion_T
		self.phi[self.geom] = phi
		self._set_dt(self.CFL)
		self._save_initials()

	def _set_dt(self, CFL=1 / 24):
		"""Applies the Neumann analysis so that the simulation is stable
		       ∆t = CFL * min(∆x, ∆z)^2 / max(diffusivity)
		where diffusivity = conductivity / density / specific heat. We want to minimize ∆t, thus
		      max(diffusivity) = min(density * specific heat) / max(conductivity)
		CFL is the Courant-Fredrichs-Lewy condition, a factor to help with stabilization
		"""
		if self.kT:
			self.dt = CFL * min([self.dz, self.dx]) ** 2 * self.rhoc().min() / (
				self._conductivity_average(self.k().max(), self.k()[self.k() < self.k().max()].max()))
		else:
			self.dt = CFL * min([self.dz, self.dx]) ** 2 * self.rhoc().min() / self.k().max()

	def init_porosity(self, pmax=0.2, t=3.154e14):
		"""Initialize the void/fracture porosity of Europa's ice shell as caused by tidal fracturing after one million
		years, as given in the supplemental material of Nimmo et al.,2003 "On the origin of band topography" .

		Parameters
		----------
		pmax: float, int
			Maximum void/fracture porosity, what's expected to be at the surface. Should be between 0 <= pmax < 1.
		t: float, int
			Time to reach steady-state porosity evolution (defaulted to one million years, do not recommend
			changing), seconds

		Returns
		----------
		out : numpy.ndarray
			Porosity distribution in the ice shell

		Usage
		----------
		Assume an ice shell with maximum porosity of 10% at the surface
			>>> model.init_porosity(0.1)
		"""
		tau = self.ice_viscosity(self.T, self.Tm) / self.constants.rho_i / self.constants.g / self.Z
		self.porosity = pmax * np.exp(-t / tau)
		self._set_dt(self.CFL)

	###
	# Define a bunch of useful functions for salty systems, unused otherwise
	####

	# Constitutive equation fits that relate temperature gradient at time of freezing (i.e. freeze rate) to entrained
	# salt content. All are from Buffo et al, 2020a unless specified.
	@staticmethod
	def shallow_fit(dT, a, b, c, d):
		"""
		Constitutive equation for relating thermal gradient at time of freezing to entrained bulk salinity in
		the ice. Formulation derived in Buffo et al., 2020a. Constants a,b,c,d are given for a particular composition in SalinityConstants.py

		This specific formulation is for high thermal gradients (defined by the intersection of shallow_fit and
		linear_fit).
		Parameters
		----------
		dT: float, numpy.ndarray
			Thermal gradient at time of freezing, K/m
		a,b,c,d: float, int
			Fit constants

		Returns
		----------
		out : float, numpy.ndarray
			Bulk salinity, ppt
		"""
		return a + b * (dT + c) * (1 - np.exp(-d / dT)) / (1 + dT)

	@staticmethod
	def linear_fit(dT, a, b):
		"""Constitutive equation for relating thermal gradient at time of freezing to entrained bulk salinity in
		the ice. Formulation derived in Buffo et al., 2020a. Constants a,b,c,d are given for a particular composition in SalinityConstants.py

		This specific formulation is for small thermal gradients (defined by the intersection of shallow_fit and
		linear_fit).
		Parameters
		----------
		dT: float, numpy.ndarray
			Thermal gradient at time of freezing, K/m
		a,b,c,d: float, int
			Fit constants

		Returns
		----------
		out : float, numpy.ndarray
			Bulk salinity, ppt
		"""
		return a + b * dT

	@staticmethod
	def new_fit(dT, a, b, c, d, f, g, h):
		"""Constitutive equation for relating thermal gradient at time of freezing to entrained bulk salinity in
		the ice. Formulation derived in Buffo et al., in review. Constants a,b,c,d,f,g,h are given for a particular
		composition in SalinityConstants.py

		Thus far this is only used for sodium chloride (NaCl) and will work for all thermal gradients. New fit
		constants and compositions may be added to SalinityConstants.py

		Parameters
		----------
		dT: float, numpy.ndarray
			Thermal gradient at time of freezing, K/m
		a,b,c,d,f,g,h : float, int
			Fit constants

		Returns
		----------
		out : float, numpy.ndarray
			Bulk salinity, ppt

		Usage
		----------
			>>> S = model.entrain_salt(dT, *model.new_fit_consts[concentration])
		"""
		return a + b * (dT + c) * (1 - g * np.exp(-h / dT)) / (d + f * dT)

	@staticmethod
	def salinity_with_depth(z, a, b, c):
		"""Method for a salinity with depth profile via Buffo et al. 2020. Note that z is depth from surface where 0 is
		the surface and +D is the total shell thickness.

		Parameters
		----------
		z: float, numpy.ndarray
			Positive depth from surface in meters
		a,b,c : float, int
		    Fit constants

		Returns
		----------
			Bulk salinity in ice with depth, K

		Usage
		----------
			>>> S = model.salinity_with_depth(model.Z, *model.depth_consts)
		"""
		return a + b / (c - z)

	def entrain_salt(self, dT, S, dTMAX=100):
		"""Function that determines the amounts of salt entrained into the ice during freezing that relies on the
		functions new_fit, or shallow_fit and linear_fit to determine this, which are dependent both on the
		concentraiton of the salt present in the liquid reservoir as well as the composition.

		Parameters
		----------
		dT : float, numpy.ndarray
			Thermal gradient during freezing of saline water, K/m
		S: float, numpy.ndarray
			Concentration of salt in the solution before freezing, ppt
		dTMAX: float
			Assumed maximum thermal gradient above which all salts in a solution are captured in the newly formed
			ice, K/m

		Returns
		----------
		out : float, numpy.ndarray
			Amounts of salt entrained into the newly formed ice

		Usage
		----------
		Assume freezing of a 100 ppt salt solution by a thermal gradient of 0.2 over 20 meters
		>>> salt_in_ice = model.entrain_salt(0.2/20, 100)
		"""
		if isinstance(dT, (int, float)):
			if dT >= dTMAX:
				return S
			else:
				if S in self.concentrations:
					if self.composition in SalinityConstants.new_fit_consts.keys():
						ans = self.new_fit(dT, *self.new_fit_consts[S])
						return S if ans > S else ans
					elif self.composition in (
							SalinityConstants.linear_consts.keys() and SalinityConstants.shallow_consts.keys()):
						ans = self.linear_fit(dT, *self.linear_consts[S]) if dT <= self.linear_shallow_roots[S] \
							else self.shallow_fit(dT, *self.shallow_consts[S])
						return S if ans > S else ans
					else:
						raise Exception("Salt not included")

				else:
					# interpolation & extrapolation steps
					## interpolation: choose th
					c_min = self.concentrations[
						S > self.concentrations].max() if S <= self.concentrations.max() else \
						self.concentrations[-2]
					c_max = self.concentrations[
						S < self.concentrations].min() if S <= self.concentrations.max() else \
						self.concentrations[-1]
					# linearly interpolate between the two concentrations at gradient dT
					m, b = np.polyfit([c_max, c_min],
					                  [self.entrain_salt(dT, c_max), self.entrain_salt(dT, c_min)],
					                  1)

					# return concentration of entrained salt
					ans = m * S + b
					return S if ans > S else ans
		else:
			return np.array([self.entrain_salt(t, s) for t, s in zip(dT, S)], dtype=float)

	@staticmethod
	def _get_salinity_consts(composition):
		"""Grabs constants for equations relating salinity to other functions throughout a simulation. Only internal
		use within init_salinity

		Parameters
		----------
		composition: str
			Composition of major salt.
			Options "NaCl", "MgSO4"

		Returns
		----------
		out : dict
			Returns 4-5 dictionaries with constants for entrainment (entrain_salt), melting temperature depression
			with concentration, and the initial salinity profile in the ice
		"""
		shallow_consts = SalinityConstants.shallow_consts[composition]
		linear_consts = SalinityConstants.linear_consts[composition]
		Tm_consts = SalinityConstants.Tm_consts[composition]
		depth_consts = SalinityConstants.depth_consts[composition]

		if composition == "NaCl":
			return shallow_consts, linear_consts, Tm_consts, depth_consts, SalinityConstants.new_fit_consts[
				composition]
		else:
			return shallow_consts, linear_consts, Tm_consts, depth_consts

	def _get_salinity_roots(self):
		"""Only used when the two old entrained salt vs temperature gradient functions are used (shallow_fit,
		linear_fit) to determine at what temperature gradient to use either fit.

		Parameters
		----------
		composition : str
			Composition of major salt in system

		Returns
		----------
		out : dict
			Dictionary with structure {concentration: root}
		"""
		linear_shallow_roots = {}
		for concentration in self.linear_consts:
			func = lambda x: self.shallow_fit(x, *self.shallow_consts[concentration]) \
			                 - self.linear_fit(x, *self.linear_consts[concentration])
			linear_shallow_roots[concentration] = optimize.root(func, 3)['x']
		return linear_shallow_roots

	def _set_salinity_values(self, composition):
		"""Only internal use in init_salinity. Assigns values for solutal contraction coefficients for density with
		concentration and depends on composition

		Parameters
		----------
		composition: str

		Returns
		----------
		None
		"""
		if composition == "MgSO4":
			self.constants.C_rho = 1.145
			self.constants.Ci_rho = 7.02441855e-01

			self.saturation_point = 174.  # ppt, saturation concentration of MgSO4 in water (Pillay et al., 2005)
			self.constants.rho_s = 2660.  # kg/m^3, density of anhydrous MgSO4

			## Only used if heat_from_precipitation is used
			# Assuming that the bulk of MgSO4 precipitated is hydrates and epsomite (MgSO4 * 7H2O), then the heat
			# created from precipitating salt out of solution is equal to the enthalpy of formation of epsomite (
			# Grevel et al., 2012).
			self.enthalpy_of_formation = 13750.0e3  # J/kg for epsomite

		elif composition == "NaCl":
			self.constants.C_rho = 0.8644
			self.constants.Ci_rho = 6.94487270e-01

			self.saturation_point = 232.  # ppt, saturation concentration of NaCl in water
			self.constants.rho_s = 2160.  # kg/m^3, density of NaCl

			self.enthalpy_of_formation = 0.  # J/kg

		elif composition == "HCN":
			self.constants.C_rho = 0
			self.constants.Ci_rho = 0
			self.saturation_point = 1000
			self.constants.rho_s = 0
			self.enthalpy_of_formation = 0

	def _get_interpolator(self):
		import scipy.interpolate as intp
		temp_grads = np.linspace(0, 110, 1000)
		entrained_salts = np.zeros((len(self.concentrations), len(temp_grads)))
		for i, c in enumerate(self.concentrations):
			entrained_salts[i] = self.entrain_salt(temp_grads, [c] * len(temp_grads))
		func = intp.interp2d(temp_grads, self.concentrations, entrained_salts,
		                     kind='cubic', bounds_error=False, fill_value=None)
		return func

	def init_salinity(
			self, composition, concentration, *, rejection_cutoff=0.25, shell=True,
			in_situ=False, T_match=True, salinate=True, use_interpolator=False, **kwargs):
		"""Initialize salts to be included in a simulation.
		Parameters
		----------
		composition : str
			The (dominant) composition of the salt to be included. All properties of this must be present in the
			SalinityConstants file.
			Current options in the SalinityConstants file: "MgSO4", "NaCl"

		concentration : int, float
			The assumed concentration of the ocean at the current epoch. This currently is implemented only to reflect
			the concentrations from the SalinityConstants file (i.e. 35 ppt for NaCl, 100 ppt for MgSO4, etc.)

		rejection_cutoff : float
			The maximal amount a cell can be frozen before not accepting any  new salt. e.g., rejection_cutoff = 1 only
			allows fully liquid cells (phi = 1) to accept rejected salt, while rejection_cutoff = 0.1 allows cells
			with phi >= 0.1 to accept rejected salt.

		shell : bool
			Whether to include the background salinity profile as determined by the composition and concentration of
			the ocean in SalinityConstants (Buffo et al., 2020; 2021) (the salinity_with_depth function). Defaults to True

		in_situ : bool
			Whether the intrusion is assumed to be melted in situ. Generally only applicable if the shell option is
			on as it will average the total salinity of the cells assumed to be melted into the intrusion

		T_match : bool
			Whether to match the basal temperature boundary to the assumed melting temperature of the ocean,
			which depends on the composition and concentration. Defaults to True.
			Used specifically for when the brittle ice shell is assumed to be above a convecting layer

		salinate : bool
			Whether the liquid portion should salinate over time. Defaults to True.
			Used specifically for fracture simulations where the mixing time with the ocean/liquid reservoir is assumed
			to be much smaller than the freezing tiem.

		Returns
		----------
		None
		"""

		self.issalt = True

		if in_situ == True:
			shell = True

		self.composition = composition
		self.concentration = concentration
		self.rejection_cutoff = rejection_cutoff
		self.salinate = salinate

		if "fracture_width" in self.__dict__: self.salinate = False

		self.heat_from_precipitation = True if "heat_from_precipitation" in kwargs and kwargs[
			"heat_from_precipitation"] == 1 else False

		if composition == "NaCl":
			self.shallow_consts, self.linear_consts, self.Tm_consts, self.depth_consts, self.new_fit_consts = \
				self._get_salinity_consts(composition)
		else:
			self.shallow_consts, self.linear_consts, self.Tm_consts, self.depth_consts = \
				self._get_salinity_consts(composition)
			self.linear_shallow_roots = self._get_salinity_roots()

		self._set_salinity_values(composition)
		# This should be a temporary fit as it's not generalized
		self.concentrations = np.sort([conc for conc in (self.new_fit_consts if composition == "NaCl" else
		                                                 self.shallow_consts)])

		if T_match:
			self.Tbot = self.Tm_func(concentration, *self.Tm_consts)
			if self.verbose:
				print(f"\t Readjusting temperature profile to fit {concentration} ppt {composition} ocean (assuming "
				      f"non-linear profile)")
			self.init_T(Tsurf=self.Tsurf, Tbot=self.Tbot)

		if shell:
			if self.verbose: print("  Adding salts into background ice")
			self.S = self.salinity_with_depth(self.Z, *self.depth_consts[self.concentration])
			# because the above equation from Buffo et al., 2020 is not well known for depths < 10 m, it will predict
			# very high salinities, over the parent liquid concentration. Here, I force it to entrain all salt where
			# it wants to concentrate more than physically possible.
			self.S[self.S > concentration] = concentration * self.rejection_cutoff
			self.S[self.S < 0] = concentration * self.rejection_cutoff

			if not in_situ:  # for water emplaced in a salty shell via injection
				try:
					self.S[self.geom] = concentration
				except AttributeError:
					pass

			else:  # for water body emplaced via the in situ melting of the ice shell
				# calculate total salt in shell where the the intrusion is located
				try:
					if self.verbose: print(f"\t In situ melting of ice shell: Redistributing salts")
					total_salt_in_ice = self.S[self.geom].sum()
					self.S[self.geom] = total_salt_in_ice / self.geom[1].shape[0]
					# check for errors
					print(self.S[self.geom].sum() / total_salt_in_ice)
					if self.S[self.geom].sum() / total_salt_in_ice < 1.0 - 1e-15 or \
							self.S[self.geom].sum() / total_salt_in_ice > 1.0 + 1e-15:
						raise Exception("Problem with salt redistribution ?")
					if self.verbose:
						print(f"\t  New intrusion salinity {self.S[self.geom][0]} ppt {self.composition}")

				except AttributeError:
					pass
		else:
			try:
				self.S[self.geom] = self.concentration
			except AttributeError:
				pass
		try:
			self.T[self.geom] = self.Tm_func(self.S[self.geom], *self.Tm_consts)
		except AttributeError:
			pass
		self.Tm = self.Tm_func(self.S, *self.Tm_consts)

		if use_interpolator:
			self.entrain_salt = self._get_interpolator()

		self.total_salt = [self.S.sum()]
		self.removed_salt = []

		self._save_initials()
		self._set_dt(self.CFL)
