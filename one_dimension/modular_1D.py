# Testing different inheritence structure
# using 1D
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
	def __init__(self):
		self.PLOTTING = 1


class MechanicalSolver:
	def __init__(self):
		self.mechanical_stuff = 1


# Not sure how Michaut & Manga (2014) even discretize their equation (32)
# maybe also just be a part of the HeatSolver class
'''
class HeatSolver_MM2014:
	TS = 273.15 - 1
	TL = 273.15
	freezestop = 1
	w = 0.5

	def __init__(self):
		self.dummy_var = 1

	def update_liquid_fraction_MM(self):
		idx1 = np.where(self.T < self.TL)
		self.phi[idx1] = np.exp(-( (self.T[idx1] - self.TL) / self.w) ** 2)
		idx2 = np.where(self.T > self.TL)
		self.phi[idx2] = 1

	def update_vol_avg_MM(self, phi_last):
		self.update_liquid_fraction_MM()
		LS_idx = np.where((self.T >= self.TS) & (self.T <= self.TL))
		#self.rhoc = self.phi * self.constants.cp_w * self.constants.rho_w + \
		#            (1 - self.phi) * self.constants.cp_i * self.constants.rho_i

		idx1 = np.where(self.T < self.TL)
		idx2 = np.where(self.T > self.TL)

		dTHETAdT = self.phi.copy()
		dTHETAdT[idx1] = 2*(self.T[idx1] - self.TL)/(self.w**2) * np.exp(-((self.T[idx1]-self.TL)/self.w)**2)
		dTHETAdT[idx2] = 0

		self.rhoc = self.phi * self.constants.cp_w * self.constants.rho_w + \
		            (1 - self.phi) * self.constants.cp_i * self.constants.rho_i \
					- self.constants.rho_i * self.constants.Lf * dTHETAdT
		self.k = (1 - self.phi) * (self.constants.ac / self.T) + self.phi * self.constants.kw
		#self.rhoc[LS_idx] += - self.constants.rho_i ** self.constants.Lf / (
		#		self.TL - self.TS)

	def MM_solver(self, nt):
		for n in range(nt):
			TErr, phiErr = np.inf, np.inf
			iter_k = 0
			while (phiErr > 0.01) & (TErr > 0.1):
				Tn, phin = self.T.copy(), self.phi.copy()
				C = self.dt / (2 * self.rhoc[1:-1] * self.dz ** 2)
				Tz = C * ((self.k[2:] + self.k[1:-1]) * (Tn[2:] - Tn[1:-1])
				          - (self.k[:-2] + self.k[1:-1]) * (Tn[1:-1] - Tn[:-2]))
				self.T[1:-1] = Tn[1:-1] + Tz
				self.T[0] = self.Tsurf
				self.T[-1] = self.Tbot
				self.update_vol_avg_MM(phi_last=phin)
				#self.phi[0] = self.phi[-1] = 0

				TErr = np.amax(abs(self.T[1:-1] - Tn[1:-1]))
				phiErr = np.amax(abs(self.phi[1:-1] - phin[1:-1]))
				iter_k += 1

			if self.freezestop :
				sill = self.T[np.where((self.z >= self.depth) & (self.z <= self.depth + self.thickness))]
				if len(sill[sill >= self.TL]) == 0:
					print('frozen')
					self.freeze_time = n * self.dt
					print('sill frozen at {0:0.04f}s'.format(self.dt * n))
					return self.freeze_time

'''


class HeatSolver:
	'''
	Options:
		tidalheat -- binary; turns on/off viscosity-dependent tidal heating from Mitri & Showman (2005), default = 0
		Ttol -- convergence tolerance for temperature, default = 0.1 K
		phitol -- convergence tolerance for liquid fraction, default = 0.01
		latentheat -- binary; default use Huber et al. (2008), else use Michaut & Manga (2016) method, default = 1
		freezestop -- binary; stop when sill is frozen, default = 0
	'''

	# off and on options
	tidalheat = 0  # turns off or on tidalheating component
	Ttol = 0.1  # temperature tolerance
	phitol = 0.01  # liquid fraction tolerance
	latentheat = 1  # choose enthalpy method to use
	freezestop = 0  # stop simulation upon total solidification of sill
	rayleigh_number = 0  #
	water_height = 0  #
	upper_freeze_front = 1
	lower_freeze_front = 1
	salinity_profile_time = 1

	def __init__(self):
		self.hot_stuff = 1

	class outputs:
		timer = []
		h = []
		Ra = []
		ff_low, ff_high = [], []
		S_profile = []

	def set_BC(self, top=None, bottom=None):
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
		'''
		self.topBC = top
		self.botBC = bottom

	def update_salinity(self, phi_last):
		if self.issalt:
			new_ice = np.where((phi_last != 0) & (self.phi == 0))
			if len(new_ice[0]) > 0:
				S_old = self.S.copy()
				dT = (self.T[new_ice[0] + 1] - self.T[new_ice[0] - 1]) / (self.dz)
				if dT > 0:
					self.S[new_ice] = self.entrained_S(dT, S_old[new_ice])
					# redistribute rejected salt
					res_salt = sum(S_old[new_ice] - self.S[new_ice])
				elif dT < 0:
					self.S[new_ice] = S_old[new_ice]
					res_salt = 0

				still_water_here = np.where(self.phi >= 0.25)
				if (len(self.S[still_water_here]) <= 1) or (self.S[still_water_here].all() >= self.saturation_point):
					print('one cell of water left, breaking')
					print('or its saturated')
					return 1

				elif (len(self.S[still_water_here]) > 1):
					self.S[still_water_here] = self.S[still_water_here] + res_salt / len(self.S[still_water_here])
					self.total_salt.append(sum(self.S))
					return 0

	def update_liquid_fraction(self, phi_last):
		if self.issalt:
			self.Tm = self.Tm_func(self.S)
			Hs = self.cp_i * self.Tm
			En = self.cp_i * self.T + self.constants.Lf * phi_last
		else:
			Hs = self.cp_i * self.Tm
			En = self.cp_i * self.T + self.constants.Lf * phi_last
		self.phi[En < Hs] = 0.
		self.phi[En > Hs + self.constants.Lf] = 1.
		idx = np.where((Hs <= En) & (En <= Hs + self.constants.Lf))
		if (self.cpT is not None) and (type(self.Tm) != type(self.T)):
			self.phi[idx] = (En[idx] - Hs) / self.constants.Lf
		else:
			self.phi[idx] = (En[idx] - Hs[idx]) / self.constants.Lf

	def update_vol_avg(self):
		if self.kT is not None:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * (self.constants.ac / self.T) + self.phi * self.constants.kw

		if self.cpT is None:
			self.cp_i = 185. + 7.037 * self.T

		if self.issalt is True:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * (self.constants.rho_w + self.a_rho * self.S + self.b_rho * self.T) * \
			            self.constants.cp_w
		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w

	def update_Q(self, phi_last):
		self.LF = self.constants.rho_i * self.constants.Lf * (self.phi[1:-1] - phi_last[1:-1])
		self.Q = -self.LF / self.rhoc[1:-1]

	def apply_BC(self, Tn):
		# Apply top boundary condition
		if self.topBC == None:
			self.T[0] = self.Tsurf
		elif self.topBC == 'Radiative':
			c = self.dt / (2 * self.rhoc[0] * self.dz ** 2)
			Trad = c * ((self.k[1] + self.k[0]) * (Tn[1] - Tn[0])
			            - 2 * self.dz * self.constants.emiss * self.constants.stfblt * (Tn[0] - self.Tsurf) ** 4)
			self.T[0] = Tn[0] + Trad + self.Q[0]
		elif self.topBC == 'NoFlux':
			c = self.dt / (2 * self.rhoc[0] * self.dz ** 2)
			self.T[0] = Tn[0] + c * (self.k[1] + self.k[0]) * (Tn[1] - Tn[0]) + self.Q[0]

		# Apply bottom boundary condition
		if self.botBC == None:
			self.T[-1] = self.Tbot
		if self.botBC == 'NoFlux':  # Not sure why this isn't working, but it isnt :(
			c = self.dt / (2 * self.rhoc[-1] * self.dz ** 2)
			self.T[-2] = Tn[-2] + c * (self.k[-2] + self.k[-3]) * (Tn[-2] - Tn[-3]) + self.Q[-2]

	def get_outputs(self, n, OF=1):
		'''
		Determine which outputs to track in time series
		'''
		sill = np.where(self.phi > 0)[0]
		if n % OF == 0:
			self.outputs.timer.append(n * self.dt)
			if self.rayleigh_number:
				dT = self.T[sill[-1]] - self.T[sill[0]]
				density = (1 - self.phi[sill]) * (
							self.constants.rho_w + self.a_rho * self.S[sill] + self.b_rho * self.T[
						sill])
				density = np.mean(density)

				D = max(self.z[sill]) - min(self.z[sill])
				k = np.mean(self.k[sill])
				Ra = density * self.constants.expans * self.constants.g * dT * D ** 3 / (k * self.constants.visc0w)
				self.outputs.Ra.append(Ra)
			if self.lower_freeze_front:
				self.outputs.ff_low.append(max(self.z[sill]))
			if self.upper_freeze_front:
				self.outputs.ff_high.append(min(self.z[sill]))
			if self.water_height:
				self.outputs.h.append(max(self.z[sill]) - min(self.z[sill]))
			if self.salinity_profile_time:
				self.outputs.S_profile.append(self.S.copy())

	class stefan:
		def solution(self, t, T1, T0):
			'''
			Analytic solution to Stefan problem for validation. Input is time
			'''
			from scipy import optimize
			from scipy.special import erf
			from numpy import linspace
			if self.Tsurf > self.Tbot:
				kappa = self.constants.kw / (self.constants.cp_w * self.constants.rho_w)
				Stf = self.constants.cp_w * (T1 - T0) / self.constants.Lf
			elif self.Tsurf < self.Tbot:
				kappa = self.constants.ki / (self.constants.cp_i * self.constants.rho_i)
				Stf = self.constants.cp_i * (T0 - T1) / self.constants.Lf
			lam = optimize.root(lambda x: x * np.exp(x ** 2) * erf(x) - Stf / np.sqrt(np.pi), 1)['x'][0]

			self.stefan_zm = 2 * lam * np.sqrt(kappa * t)
			self.stefan_zm_func = lambda time: 2 * lam * np.sqrt(kappa * time)
			self.stefan_zm_const = 2 * lam * np.sqrt(kappa)
			self.stefan_z = linspace(0, self.stefan_zm)
			self.stefan_T = T1 - (T1 - T0) * erf(self.stefan_z / (2 * np.sqrt(kappa * t))) / erf(lam)

		def compare(self, dt):
			'''
			Compare simulation freezing front propagation zm(t) with stefan solution
			Parameters:
					dt : float
						time step
			'''
			self.dt = dt
			self._time_ = [0]
			self.freeze_front = [0]
			n = 1
			while self.freeze_front[-1] < 0.9 * self.Lz:
				TErr, phiErr = np.inf, np.inf
				iter_k = 0
				while (TErr > self.Ttol) & (phiErr > self.phitol):
					Tn, phin = self.T.copy(), self.phi.copy()
					self.update_liquid_fraction(phi_last=phin)
					self.update_salinity(phi_last=phin)
					self.update_vol_avg()
					C = self.dt / (2 * self.rhoc[1:-1] * self.dz ** 2)
					Tz = C * ((self.k[2:] + self.k[1:-1]) * (Tn[2:] - Tn[1:-1])
					          - (self.k[:-2] + self.k[1:-1]) * (Tn[1:-1] - Tn[:-2]))
					self.update_Q(phi_last=phin)
					self.T[1:-1] = Tn[1:-1] + Tz + self.Q
					self.apply_BC(Tn)

					TErr = np.amax(abs(self.T[1:-1] - Tn[1:-1]))
					phiErr = np.amax(abs(self.phi[1:-1] - phin[1:-1]))
					iter_k += 1

				if (self.phi == 0).any():
					idx = np.max(np.where(self.phi == 0))
					if self.z[idx] != self.freeze_front[-1]:
						self.freeze_front.append(self.z[idx])
						self._time_.append(n * self.dt)
				else:
					self.freeze_front.append(0)
				n += 1
			self.get_outputs(n)

	def solve_heat(self, nt, dt):
		self.dt = dt
		for n in range(nt):
			TErr, phiErr = np.inf, np.inf
			iter_k = 0
			while (TErr > self.Ttol) and (phiErr > self.phitol):
				Tn, phin = self.T.copy(), self.phi.copy()
				self.update_liquid_fraction(phi_last=phin)

				self.saturated = self.update_salinity(phi_last=phin)
				if self.issalt is True and self.saturated == 1:
					self.freeze_time = n * self.dt
					print('1 sill is saturated @ {}s'.format(self.freeze_time))
					break

				self.update_vol_avg()
				C = self.dt / (2 * self.rhoc[1:-1] * self.dz ** 2)
				Tz = C * ((self.k[2:] + self.k[1:-1]) * (Tn[2:] - Tn[1:-1])
				          - (self.k[:-2] + self.k[1:-1]) * (Tn[1:-1] - Tn[:-2]))
				self.update_Q(phi_last=phin)
				self.T[1:-1] = Tn[1:-1] + Tz + self.Q
				self.apply_BC(Tn)

				TErr = np.amax(abs(self.T[1:-1] - Tn[1:-1]))
				phiErr = np.amax(abs(self.phi[1:-1] - phin[1:-1]))
				iter_k += 1

			# if n%1000==0:
			#	plt.figure(1001)
			#	plt.clf()
			#	plt.title('t={:0.03f}s\n={:0.03f}yr'.format(n*dt, n*dt/self.constants.styr))
			#	plt.scatter(self.S, self.z, c=self.T, cmap='cividis',vmin=self.Tsurf, vmax=273.15)
			#	plt.colorbar()
			#	#plt.xlim(self.Tsurf, self.Tbot)
			#	plt.gca().invert_yaxis()
			#	plt.ylim(self.Lz, 0)
			#
			#	plt.figure(1002)
			#	plt.clf()
			#	plt.scatter(self.phi, self.z, c=self.S, cmap='cividis')
			#	plt.colorbar(label='temp (K)')
			#	plt.gca().invert_yaxis()
			#	plt.ylim(self.depth+self.thickness, self.depth)
			#	plt.xlim(0,1)
			#	plt.pause(0.00000000000001)

			if self.freezestop is True:
				if len(self.phi[self.phi > 0]) == 0:
					print('frozen')
					self.freeze_time = n * self.dt
					print('sill frozen at {0:0.04f}s'.format(self.dt * n))
					return self.freeze_time
			if self.issalt is True and self.saturated == 1:
				print('2 sill is saturated')
				self.freeze_time = n * self.dt
				break
		# self.get_outputs(n, OF=100)
		if self.issalt is True and self.saturated == 1:
			print('3 sill is saturated')
			self.freeze_time = n * self.dt
			return


class IceSystem(HeatSolver, MechanicalSolver, Plotter):
	def __init__(self, Lz, dz, kT=None, cpT=None, issalt=0):
		self.Lz = Lz
		self.dz = dz
		self.nz = int(Lz / dz)
		self.z = np.linspace(0, Lz, self.nz)
		self.T = np.zeros(self.nz)
		self.Tm = np.ones(self.nz)
		self.phi = np.zeros(self.nz)
		self.S = np.zeros(self.nz)
		self.issalt = issalt
		self.kT, self.cpT = kT, cpT


	class constants:
		styr = 3.14e7  # seconds in a year, s/yr
		g = 1.3  # europa surface gravity
		rho_i = 910.  # pure ice density
		rho_w = 1000.  # pure water density
		cp_i = 2.11e3  # pure ice specific heat
		cp_w = 4.19e3  # pure water specific heat
		ki = 2.3  # pure ice thermal conductivity
		kw = 0.56  # pure water thermal conductivity
		ac = 567  # ice thermal conductivity constant, ki = ac/T
		emiss = 0.97  # pure ice emissivity
		stfblt = 5.67e-8  # stefan-boltzman constant
		Tm = 273.15  # pure ice melting temperature at 1 atm
		Lf = 333.6e3  # latent heat of fusion
		visc0w = 1.3e-3
		expans = 1.6e-4  # 1/K, thermal expansivity

	def init_T(self, Tsurf, Tbot):
		'''
		Initialize temperature at top of domain
		'''
		self.Tsurf = Tsurf
		self.Tbot = Tbot
		self.T = Tsurf * (Tbot / Tsurf) ** (abs(self.z / self.Lz))
		self.Tm = self.constants.Tm * self.Tm

	def init_vol_avgs(self):
		'''
		Initialize volume averaged values over the domain, must be used if not using an intrusion
		Parameters:
			cpT :
			    choose whether to use temperature-depedent specific heat,
			    default = None, temperature-dependent, cp_i = 185 + 7*T (citation)
			kT :
			    choose whether to use temperature-dependent thermal conductivity,
			    default = None, temperature-dependent, k=ac/T (Petrenko, Klinger, etc.)
		'''
		if self.cpT is not None:
			self.cp_i = self.constants.cp_i
		else:
			self.cp_i = 185. + 7.037 * self.T

		if self.kT is not None:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * self.constants.ac / self.T + self.phi * self.constants.kw

		if self.issalt is True:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * (self.constants.rho_w + self.a_rho * self.S + self.b_rho * self.T) * \
			            self.constants.cp_w
		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w

	def init_sill(self, Tsill, depth, thickness, phi=1):
		'''
		Initialize intrusion. Will initialize volume averages
		Parameters:
			Tsill: float
				temperature of intrusion, K
			depth : float
				depth of intrusion, "top edge" of intrusion , m
			thickness : float
				thickness of intrusion, m
			phi : float
				liquid fraction of intrusion, 0-1
		'''
		if phi < 0 or phi > 1:
			raise Exception('liquid fraction must be between 0 and 1')

		self.Tsill = Tsill
		self.depth = depth
		self.thickness = thickness
		idx = np.where((self.z >= depth) & (self.z <= depth + thickness))
		self.T[idx] = Tsill
		self.phi[idx] = phi
		self.init_vol_avgs()

	def init_S(self, S=None, composition='Europa', concentration=12.3, a_rho=0.75, b_rho=-0.0375):
		'''
		Initialize salinity in the model
		-- add a way to include background salinity profile ala Buffo et. al (2019)
		'''

		self.issalt = 1  # turn on salinity for solvers
		self.a_rho = a_rho  # water density salinity coefficient
		self.b_rho = b_rho  # water density temperature coefficient
		self.saturation_point = 282.  # ppt, saturation point of water
		self.saturated = 0  # whether liquid is saturated

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
			# shallow fit, for large dT/dz
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
			# update volume averages
			self.init_vol_avgs()
			# save initial total salt, should not change over time
			self.total_salt = [sum(self.S)]

	def entrained_S(self, dT, S, composition='Europa'):
		'''
		Calculate entrained salt in a newly frozen cell with salinity S and temperature gradient dT
		Parameters:
			dT : float
				temperature gradient across cell
			S : float
				salinity (ppt) in cell before freezing
			composition : str
				'Europa' only works for now
		Returns:
			amount of bulk salt entrained in ice due to temperature gradient dT/dz
		'''
		from scipy import optimize
		if composition != 'Europa':
			raise Exception('RUn some Earth tests you dummy')

		if isinstance(dT, float) or isinstance(dT, int):
			if S in self.shallow_consts[composition].keys():
				switch_dT = optimize.root(lambda x: self.shallow_fit(x, *self.shallow_consts[composition][S]) \
				                                    - self.linear_fit(x, *self.linear_consts[composition][S]), 3)['x'][
					0]
				if dT > switch_dT:
					return self.shallow_fit(dT, *self.shallow_consts[composition][S])
				elif dT < switch_dT:
					return self.linear_fit(dT, *self.linear_consts[composition][S])

			# probably make this more general when implement more salinities of Earth
			# i.e. elif min(self.shallow_consts[composition].keys() < S < middle() ...
			# may have to assign them variables so we can do the next part

			# if not either actual tests, linearly interpolate between the two values
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
