# Author: Chase Chivers
# Last updated: 3/18/20

import numpy as np
import time as _timer_
from utility_funcs import *
import string, random, os
from scipy import optimize
from scipy.special import erf, erfc

class HeatSolver:
	"""
	Solves two-phase thermal diffusivity problem with a temperature-dependent thermal conductivity of ice in
	two-dimensions. Sources and sinks include latent heat of fusion and tidal heating
	Options:
		tidalheat -- binary; turns on/off viscosity-dependent tidal heating from Mitri & Showman (2005), default = 0
		Ttol -- convergence tolerance for temperature, default = 0.1 K
		phitol -- convergence tolerance for liquid fraction, default = 0.01
		latentheat -- 1 : use Huber et al. (2008) enthalpy method (other options coming soon?)
		freezestop -- binary; stop when intrusion is frozen, default = 0

	Usage:
		Assuming, model = IceSystem(...)
		- Turn on tidal heating component
			model.tidalheat = True

		- Change tolerances
			model.Ttol = 0.001
			model.phitol = 0.0001
			model.Stol = 0.0001
	"""
	# off and on options
	tidalheat = 0  # off/on tidalheating component
	Ttol = 0.1  # temperature tolerance
	phitol = 0.01  # liquid fraction tolerance
	Stol = 1  # salinity tolerance
	latentheat = 1  # choose enthalpy method to use
	freezestop = 0  # stop simulation upon total solidification of intrusion
	model_time = 0

	class outputs:
		"""Class structure to help define and calculate desired outputs of a simulation."""

		def __init__(self):
			self.outputs.tmp_data_directory = ''
			self.outputs.tmp_data_file_name = ''
			self.outputs.transient_results = dict()
			self.outputs.output_frequency = 0

		def choose(self, file_path='./tmp/', file_name='', all=False, T=False, phi=False, k=False, S=False, Q=False,
		           h=False, r=False, freeze_fronts=False, percent_frozen=False, output_frequency=1000, output_list=[]):
			"""
			Choose which outputs to track with time. Each variable is updated at the chosen output frequency and is
			returned in the dictionary object outputs.transient_results.
			Parameters:
				output_frequency : integer
					the frequency to report a transient result. Default is every 1000 time steps
				output_list : list
					list of strings for the outputs below. Generally used for simulation that had stopped (i.e. hit a
					wall time) without desired
				all : bool
					turns on all outputs listed below
				T, phi, k, S, Q : bool
					tracks and returns an array of temperature, liquid fraction, volume averaged thermal conductivity,
					salinity, and source/sink grids, respectively
				h : bool
					tracks the height (thickness) of the liquid intrusion over time into a 1d array
				r : bool
					tracks the radius of the liquid portion over time into a 1d array
				freeze_fronts : bool
					tracks the propagating freeze front at the top and bottom of the intrusion into a 1d array
				percent_frozen : bool
					tracks and returns a 1d list of the percent of the original intrusion that is now ice
			Usage:
				Output all options every 50 years
					model.outputs.choose(model, all=True, output_frequency=int(50 * model.constants.styr/model.dt))

				Output only temperature grids and salinity at every time step
					model.outputs.choose(model, T=True, S=True, output_frequency=1)
					--or--
					model.outputs.choose(model, output_list=['T','S'], output_frequency=1);
			"""
			to_output = {'time': True, 'T': T, 'phi': phi, 'k': k, 'S': S, 'Q': Q, 'h': h, 'freeze fronts':
				freeze_fronts, 'r': r, 'percent frozen': percent_frozen}
			if all:
				to_output = {key: True for key, value in to_output.items()}

			if len(output_list) != 0:
				to_output = {key: True for key in output_list}
				to_output['time'] = True

			self.outputs.transient_results = {key: [] for key in to_output if to_output[key] is True}
			self.outputs.outputs = self.outputs.transient_results.copy()
			self.outputs.tmp_data_directory = file_path
			self.outputs.tmp_data_file_name = file_name
			self.outputs.output_frequency = output_frequency
			self.outputs.tmp_data_file_name = 'tmp_data_runID' + ''.join(random.choice(string.digits) for _ in range(4))

		def calculate_outputs(self, n):
			"""
			Calculates the output and appends it to the list for chosen outputs. See outputs.choose() for description
			of values calculated here.
			Parameters:
				n : integer
					nth time step during simulation
			Returns:
				ans : dictionary object
					dictionary object with chosen outputs as 1d numpy arrays
			"""
			ans = {}
			for key in self.outputs.outputs:
				if key == 'time':
					ans[key] = self.model_time
				if key == 'percent frozen':
					ans[key] = 1 - (self.phi[self.geom].sum()) / len(self.geom[1])
				if key == 'r':
					tmp = np.where(self.phi > 0)
					ans[key] = self.dx * max(tmp[1])
					del tmp
				if key == 'h':
					tmp = np.where(self.phi > 0)
					ans[key] = (max(tmp[0]) - min(tmp[0])) * self.dz
					del tmp
				if key == 'freeze fronts':
					tmp = np.where(self.phi > 0)
					ans[key] = np.array([min(tmp[0]), max(tmp[0])]) * self.dz
					del tmp
				if key == 'T':
					ans[key] = self.T.copy()
				if key == 'S':
					ans[key] = self.S.copy()
				if key == 'phi':
					ans[key] = self.phi.copy()
				if key == 'k':
					ans[key] = self.k.copy()
				if key == 'Q':
					ans[key] = self.Q.copy()
			return ans

		def get_results(self, n):
			"""Calls outputs.calculate_outputs() then saves dictionary of results to file"""
			if n % self.outputs.output_frequency == 0:
				get = self.outputs.calculate_outputs(self, n)
				save_data(get, self.outputs.tmp_data_file_name + '_n={}'.format(n), self.outputs.tmp_data_directory)

		def get_all_data(self, del_files=True):
			"""Concatenates all saved outputs from outputs.get_results() and puts into a single dictionary object."""
			cwd = os.getcwd()  # find working directory
			os.chdir(self.outputs.tmp_data_directory)  # change to directory where data is being stored

			# make a list of all results files in directory
			data_list = nat_sort([data for data in os.listdir() if data.endswith('.pkl') and \
			                      self.outputs.tmp_data_file_name in data])
			# copy dictionary of desired results
			ans = self.outputs.transient_results.copy()
			# iterate over file list
			for file in data_list:
				tmp_dict = load_data(file)  # load file
				for key in self.outputs.outputs:  # iterate over desired outputs
					ans[key].append(tmp_dict[key])  # add output from result n to final file
				del tmp_dict
				if del_files: os.remove(file)

			# make everything a numpy array for easier manipulation
			ans = {key: np.asarray(value) for key, value in ans.items()}

			# go back to working directory
			os.chdir(cwd)
			return ans

	def set_boundaryconditions(self, top=True, bottom=True, sides=True, **kwargs):
		"""
			Set boundary conditions for heat solver. A bunch of options are available in case they want to be tested
			or used.
			top : bool, string
				top boundary conditions.
				default: True = Dirichlet, Ttop = Tsurf chosen earlier
				'Flux': surface loses heat to a "ghost cell" of ice
						temperature of ghost cell is based on the equilibrium temperature profile at the depth one
						spatial size "above" the domain,
						i.e.: T_(ghost_cell) = Tsurf * (Tbot/Tsurf) ** (-dz/Lz)
								for Tsurf = 110 K, Tbot = 273.15, & Lz = 5 km => T_(ghost_cell) = 109.8 K
				'Radiative': surface loses heat beyond the "background" surface temperature through blackbody
							radiation to a vacuum using Stefan-Boltzmann Law.
							Note: The smaller the time step the better this will predict the surface warming. Larger
							time steps make the surface gain heat too fast. This is especially important if upper
							domain is cold and simulation is using temperature-dependent thermal conductivity.

			bottom: bool, string
				bottom boundary condition
				default, True: Dirichlet, Tbottom = Tbot chosen earlier
				'Flux' : bottom loses heat to a "ghost cell" of ice at a constant temperature
						 temperature of ghost cell is based on equilibrium temperature profile at depth one spatial
						 size below the domain
						 i.e. T_(ghost_cell) = Tsurf * (Tbot/Tsurf) ** ((Lz+dz)/Lz)
						    for Tsurf = 110 K, Tbot = 273.15, & Lz = 5 km => T_(ghost_cell) = 273.647 K
						-> this can be helpful for using a "cheater" vertical domain so as not to have to simulate a
						whole shell
				'FluxI', 'FluxW': bottom loses heat to a "ghost cell" of ice ('FluxI') or water ('FluxW') at a chosen
								constant temperature
								ex: model.set_boundaryconditions(bottom='FluxI', botT=260);
			sides: bool, string
				Left and right boundary conditions
				True :  Dirichlet boundary condition;
						Tleft = Tright =  Tedge (see init_T)
						* NOTE: must set up domain such that anomaly is far enough away to not interact with the
						edges of domain
				'NoFlux : a 'no flux' boundary condition
							-> boundaries are forced to be the same temperature as the adjacent cells in the domain
				'RFlux' : 'NoFlux' boundary condition on the left, with a flux boundary at T(z,x=Lx,t) = Tedge(z)
						that is dL far away. Most useful when using the symmetry about x option.
						dL value must be chosen when using this option:
						ex: model.set_boundaryconditions(sides='RFlux', dL=500e3)
		"""

		self.topBC = top
		if top == 'Radiative':
			# self.Std_Flux_in = (self.k_initial[0, 1:-1] + self.k_initial[1, 1:-1]) \
			#                   * (self.T_initial[1, 1:-1] - self.T_initial[0,1:-1])
			self.std_set = 0

		self.botBC = bottom
		if bottom == 'FluxI' or bottom == 'FluxW':
			try:
				self.botT = kwargs['botT']
			except:
				raise Exception('Bottom boundary condition temperature not chosen\n\t->ex: '
				                'model.set_boundaryconditions(bottom=\'FluxI\', botT=260);')
		self.sidesBC = sides
		if sides == 'RFlux':
			try:
				self.dL = kwargs['dL']
			except:
				raise Exception('Length for right side flux not chosen\n\t->model.set_boundaryconditions('
				                'sides=\'RFlux\', dL=500e3)')

	def update_salinity(self, phi_last):
		"""
		Parameterization of salt advection and diffusion in the intrusion. See Chivers et al., 201X for full
		description of parameterization.
		"""
		if self.issalt:
			z_ni, x_ni = np.where((phi_last > 0) & (self.phi == 0))  # find where ice has just formed
			water = np.where(self.phi >= self.rejection_cutoff)  # find cells that can accept rejected salts
			vol = water[1].shape[0]  # calculate "volume" of water
			# rejected_salt = 0  # initialize amount of salt rejected, ppt
			self.wat_vol.append(self.phi.sum())
			self.removed_salt.append(0)  # start catalogue of salt removed from system
			self.mass_removed.append(0)
			self.ppt_removed.append(0)

			if len(z_ni) > 0 and vol != 0:  # iterate over cells where ice has just formed
				Sn = self.S.copy()
				for i in range(len(z_ni)):
					# save starting salinity in cell
					S_old = self.S[z_ni[i], x_ni[i]]
					# calculate thermal gradients across each cell
					if self.symmetric and x_ni[i] in [0, self.nx - 1]:
						dTx = 0
					else:
						dTx = abs(self.T[z_ni[i], x_ni[i] - 1] - self.T[z_ni[i], x_ni[i] + 1]) / (2 * self.dx)
					dTz = (self.T[z_ni[i] - 1, x_ni[i]] - self.T[z_ni[i] + 1, x_ni[i]]) / (2 * self.dz)

					# brine drainage parameterization:
					#  bottom of intrusion -> no gravity-drainage, salt stays
					if dTz > 0:
						self.S[z_ni[i], x_ni[i]] = S_old

					#  top of intrusion -> brine drains and rejects salt
					elif dTz < 0:
						# dT = np.hypot(dTx, dTz)  # gradient across the diagonal of the cell
						# dT = max(abs(dTx), abs(dTz))  # maximum value
						# dT = np.sqrt(dTx*abs(dTz))  # geometric mean
						# dT = 2/(1/dTx + 1/abs(dTz))  # harmonic mean
						dT = (dTx + abs(dTz)) / 2.  # arithmetic mean
						# salt entrained in newly formed ice determined by Buffo et al., 2019 results. (See
						# IceSystem.entrain_salt() function)
						self.S[z_ni[i], x_ni[i]] = self.entrain_salt(dT, S_old, self.composition)
						# not all salt will be entrained in ice, some will be mixed back into
					# rejected_salt += S_old - self.S[z_ni[i], x_ni[i]]

				# assume the salt is well mixed into remaining liquid solution in time step dt
				self.S[water] = self.S[water] + (Sn.sum() - self.S.sum()) / vol
				'''
				#  attempt at vectorizing the salt parameterization but it ends up being slower (?)
				#   unless entrain_salt can be rewritten without the for-loop

				S_old = self.S.copy()
				dTz = (self.T[:-2,:] - self.T[2:,:]) / (2 * self.dz)
				dTx = abs(self.T[:,:-2] - self.T[:,2:]) / (2 * self.dx)
				grad = (abs(dTx[z_ni, x_ni-1]) + abs(dTz[z_ni-1, x_ni])) / 2
				self.S[z_ni, x_ni] = self.entrain_salt(grad, self.S[z_ni, x_ni], self.composition)

				# do bottom parameterization
				new_dTz = dTz[z_ni-1, x_ni]
				loc = np.where(new_dTz > 0)[0]
				self.S[z_ni[loc],x_ni[loc]] = S_old[z_ni[loc], x_ni[loc]]

				self.S[water] = self.S[water] + (S_old.sum() - self.S.sum()) / vol
				'''
				# remove salt from system if liquid is above the saturation point
				self.removed_salt[-1] += (self.S[self.S >= self.saturation_point] - self.saturation_point).sum()
				self.mass_removed[-1] += vol * (self.S[self.geom].max() * self.C_rho + self.constants.rho_w) * \
				                         self.removed_salt[-1] / vol / 1000
				self.ppt_removed[-1] += self.removed_salt[-1] / vol

				# ensure liquid hits only the saturation concentration
				self.S[self.S > self.saturation_point] = self.saturation_point

			# check mass conservation
			total_S_new = self.S.sum() + np.asarray(self.removed_salt).sum()
			if abs(total_S_new - self.total_salt[0]) <= self.Stol:
				self.total_salt.append(total_S_new)
			else:
				self.total_salt.append(total_S_new)
				raise Exception('Mass not being conserved')

			# outdated usage, may delete at some point
			#   however, it can be used to stop simulation after liquid becomes saturated
			if (self.S[water] >= self.saturation_point).all() and water[0].sum() > 0:
				return 1
			else:
				return 0

	def update_liquid_fraction(self, phi_last):
		"""Application of Huber et al., 2008 enthalpy method. Determines volume fraction of liquid/solid in a cell."""

		# update melting temperature for enthalpy if salt is included in simulation
		if self.issalt:
			self.Tm = self.Tm_func(self.S, *self.Tm_consts[self.composition])
		# calculate new enthalpy of solid ice
		Hs = self.cp_i * self.Tm  # update entalpy of solid ice
		H = self.cp_i * self.T + self.constants.Lf * phi_last  # calculate the enthalpy in each cell
		# update liquid fraction
		self.phi[H >= Hs] = (H[H >= Hs] - Hs[H >= Hs]) / self.constants.Lf
		self.phi[H <= Hs + self.constants.Lf] = (H[H <= Hs + self.constants.Lf] - Hs[
			H <= Hs + self.constants.Lf]) / self.constants.Lf
		# all ice
		self.phi[H < Hs] = 0.
		# all water
		self.phi[H > Hs + self.constants.Lf] = 1

	def update_volume_averages(self):
		"""Updates volume averaged thermal properties."""

		if self.kT:
			self.k = (1 - self.phi) * (self.constants.ac / self.T) + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw

		if self.cpT == "GM89":
			"Use temperature-dependent specific heat for pure ice from Grimm & McSween 1989"
			self.cp_i = 185. + 7.037 * self.T
		elif self.cpT == "CG10":
			"Use temperature-dependent specific heat for pure ice from Choukroun & Grasset 2010"
			self.cp_i = 74.11 + 7.56 * self.T
		else:
			self.cp_i = self.constants.cp_i

		if self.issalt:
			self.rhoc = (1 - self.phi) * (self.constants.rho_i + self.Ci_rho * self.S) * self.cp_i + \
			            self.phi * (self.constants.rho_w + self.C_rho * self.S) * self.constants.cp_w
		elif not self.issalt:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i + \
			            self.phi * self.constants.rho_w * self.constants.cp_w

	def update_sources_sinks(self, phi_last, T_last):
		"""Updates external heat or heat-sinks during simulation."""
		self.latent_heat = self.constants.rho_i * self.constants.Lf * (
					self.phi[1:-1, 1:-1] - phi_last[1:-1, 1:-1]) / self.dt

		self.tidal_heat = 0
		if self.tidalheat:
			# ICE effective viscosity follows an Arrenhius law
			#   viscosity = reference viscosity * exp[C/Tm * (Tm/T - 1)]
			# if cell is water, just use reference viscosity for pure ice at 0 K
			self.visc = (1 - phi_last[1:-1, 1:-1]) * self.constants.visc0i \
			            * np.exp(self.constants.Qs * (self.Tm[1:-1, 1:-1] / T_last[1:-1, 1:-1] - 1) / \
			                     (self.constants.Rg * self.Tm[1:-1, 1:-1])) \
			            + phi_last[1:-1, 1:-1] * self.constants.visc0w
			self.tidal_heat = (self.constants.eps0 ** 2 * self.constants.omega ** 2 * self.visc) / (
					2 + 2 * self.constants.omega ** 2 * self.visc ** 2 / (self.constants.G ** 2))

		return self.tidal_heat - self.latent_heat

	def apply_boundary_conditions(self, T_last, k_last, rhoc_last):
		"""Applies chosen boundary conditions during simulation run."""
		# apply chosen boundary conditions at bottom of domain
		if self.botBC == True:
			self.T[-1, 1:-1] = self.TbotBC[1:-1]

		elif self.botBC == 'Flux':
			T_bot_out = self.Tsurf * (self.Tbot / self.Tsurf) ** ((self.Lz + self.dz) / self.Lz)
			c = self.dt / (2 * rhoc_last[-1, 1:-1])

			Tbotx = c / self.dx ** 2 * ((k_last[-1, 1:-1] + k_last[-1, 2:]) * (T_last[-1, 2:] - T_last[-1, 1:-1]) \
			                            - (k_last[-1, 1:-1] + k_last[-1, :-2]) * (T_last[-1, 1:-1] - T_last[-1, :-2]))
			Tbotz = c / self.dz ** 2 * (
					(k_last[-1, 1:-1] + self.constants.ac / T_bot_out) * (T_bot_out - T_last[-1, :-1]) \
					- (k_last[-1, 1:-1] + k_last[-2, 1:-1]) * (T_last[-1, 1:-1] - T_last[-2, 1:-1]))
			self.T[-1, 1:-1] = T_last[-1, 1:-1] + Tbotx + Tbotz + self.Q[-1, :] * 2 * c

		elif self.botBC == 'FluxI' or self.botBC == 'FluxW':  # constant temperature ice
			c = self.dt / (2 * rhoc_last[-1, 1:-1])

			if self.botBC == 'FluxI':
				kbot = self.constants.ac / self.botT
			elif self.botBC == 'FluxW':
				kbot = self.constants.kw

			Tbotx = c / self.dx ** 2 * ((k_last[-1, 1:-1] + k_last[-1, 2:]) * (T_last[-1, 2:] - T_last[-1, 1:-1]) \
			                            - (k_last[-1, 1:-1] + k_last[-1, :-2]) * (T_last[-1, 1:-1] - T_last[-1, :-2]))
			Tbotz = c / self.dz ** 2 * ((k_last[-1, 1:-1] + kbot) * (self.botT - T_last[-1, 1:-1]) \
			                            - (k_last[-1, 1:-1] + k_last[-2, 1:-1]) * (T_last[-1, 1:-1] - T_last[-2, 1:-1]))
			self.T[-1, 1:-1] = T_last[-1, 1:-1] + Tbotx + Tbotz + self.Q[-1, :] * 2 * c

		# apply chosen boundary conditions at top of domain
		if self.topBC == True:
			self.T[0, 1:-1] = self.TtopBC[1:-1]

		elif self.topBC == 'Flux':
			T_top_out = self.Tsurf * (self.Tbot / self.Tsurf) ** (-self.dz / self.Lz)

			if self.cpT is True:
				Cbc = rhoc_last[0, 1:-1] / (self.constants.rho_i * (185. + 2 * 7.037 * T_top_out))
			else:
				Cbc = 1
			c = self.dt / (2 * rhoc_last[0, 1:-1])
			Ttopx = c / self.dx ** 2 * ((k_last[0, 1:-1] + k_last[0, 2:]) * (T_last[0, 2:] - T_last[0, 1:-1]) \
			                            - (k_last[0, 1:-1] + k_last[0, :-2]) * (T_last[0, 1:-1] - T_last[0, :-2]))
			Ttopz = c / self.dz ** 2 * ((k_last[0, 1:-1] + k_last[1, 1:-1]) * (T_last[1, 1:-1] - T_last[0, 1:-1]) \
			                            - (k_last[0, 1:-1] + Cbc * self.constants.ac / T_top_out) * (
					                            T_last[0, 1:-1] - T_top_out))
			self.T[0, 1:-1] = T_last[0, 1:-1] + Ttopx + Ttopz + self.Q[0, :] * 2 * c

		elif self.topBC == 'Radiative':
			c = self.dt / (2 * rhoc_last[0, :])
			rad = self.dz * self.constants.stfblt * self.constants.emiss * (T_last[0, :] ** 4 - self.Tsurf ** 4)
			Ttopz = c / self.dz ** 2 * ((k_last[0, :] + k_last[1, :]) * (T_last[1, :] - T_last[0, :]) \
			                            - (self.k_initial[0, :] + self.k_initial[1, :]) * (
						                            self.T_initial[1, :] - self.Tsurf))
			Ttopx = 1 / self.dx ** 2 * ((k_last[0, 1:-1] + k_last[0, 2:]) * (T_last[0, 2:] - T_last[0, 1:-1]) \
			                            - (k_last[0, 1:-1] + k_last[0, :-2]) * (T_last[0, 1:-1] - T_last[0, :-2]))

			self.T[0, :] = T_last[0, :] + Ttopz - rad * c / self.dz ** 2
			self.T[0, 1:-1] += (Ttopx + self.Q[0, :] * 2) * self.dt / (2 * rhoc_last[0, 1:-1])
		# else:
		#	self.T[0, 1:-1] = self.Tsurf
    
		# apply chosen boundary conditions at sides of domain
		if self.sidesBC == True:
			self.T[:, 0] = self.Tedge.copy()
			self.T[:, self.nx - 1] = self.Tedge.copy()

		elif self.sidesBC == 'NoFlux':
			self.T[:, 0] = self.T[:, 1].copy()
			self.T[:, -1] = self.T[:, -2].copy()

		elif self.sidesBC == 'RFlux':
			# left side of domain uses 'NoFlux'
			self.T[:, 0] = T_last[:, 1].copy()

			# right side of domain
			c = self.dt / (2 * rhoc_last[1:-1, -1])
			TRX = c * ((k_last[1:-1, -1] + self.constants.ac / self.Tedge[1:-1]) * \
			           (self.Tedge[1:-1] - T_last[1:-1, -1]) / self.dx \
			           - (k_last[1:-1, -1] + k_last[1:-1, -2]) * (T_last[1:-1, -1] - T_last[1:-1, -2]) / self.dL)

			TRZ = c * ((k_last[1:-1, -1] + k_last[2:, -1]) * (T_last[2:, -1] - T_last[1:-1, -1]) \
			           - (k_last[1:-1, -1] + k_last[:-2, -1]) * (T_last[1:-1, -1] - T_last[:-2, -1])) / self.dz ** 2

			self.T[1:-1, -1] = T_last[1:-1, -1] + TRX + TRZ + self.Q[:, -1] * 2 * c

	def get_gradients(self, T_last):
		# constant in front of x-terms
		Cx = self.dt / (2 * self.rhoc[1:-1, 1:-1] * self.dx ** 2)
		# temperature terms in x direction
		Tx = Cx * ((self.k[1:-1, 1:-1] + self.k[1:-1, 2:]) * (T_last[1:-1, 2:] - T_last[1:-1, 1:-1]) \
		           - (self.k[1:-1, 1:-1] + self.k[1:-1, :-2]) * (T_last[1:-1, 1:-1] - T_last[1:-1, :-2]))
		# constant in front of z-terms
		Cz = self.dt / (2 * self.rhoc[1:-1, 1:-1] * self.dz ** 2)
		# temperature terms in z direction
		Tz = Cz * ((self.k[1:-1, 1:-1] + self.k[2:, 1:-1]) * (T_last[2:, 1:-1] - T_last[1:-1, 1:-1]) \
		           - (self.k[1:-1, 1:-1] + self.k[:-2, 1:-1]) * (T_last[1:-1, 1:-1] - T_last[:-2, 1:-1]))
		return Tx, Tz

	def print_all_options(self, nt):
		"""Prints options chosen for simulation."""

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
		print('\t   dt = {} s'.format(self.dt))
		print('\t Ice shell thickness: {} m'.format(self.Lz))
		print('\t Lateral domain size: {} m'.format(self.Lx))
		print('\t    dz = {} m;  dx = {} m'.format(self.dz, self.dx))
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
		if self.issalt:
			print('\t    salinity:        {}'.format(self.Stol))
		print('\t thermal properties:')
		print('\t    ki(T):    {}'.format(stringIO(self.kT)))
		print('\t    ci(T):    {}'.format(stringIO(self.cpT)))
		print('\t intrusion/salt:')
		try:
			self.geom
			print(f'\t    radius:    {self.R_int}m')
			print(f'\t    thickness: {self.thickness}m')
			print(f'\t    depth:     {self.depth}m')
		except:
			pass
		print('\t    salinity: {}'.format(stringIO(self.issalt)))
		if self.issalt:
			print(f'\t       composition:    {self.composition}')
			print(f'\t       concentration:  {self.concentration}ppt')
		print('\t other:')
		print(f'\t     stop on freeze: {stringIO(self.freezestop)}')
		print('-------------------------')
		try:
			print('Requested outputs: {}'.format(list(self.outputs.transient_results.keys())))
		except AttributeError:
			print('no outputs requested')


	def solve_heat(self, nt, dt, print_opts=True, n0=0):
		"""
		Iteratively solve heat two-dimension heat diffusion problem with temperature-dependent conductivity of ice.
		Parameters:
			nt : int
				number of time steps to take
			dt : float
				time step, s
			print_opts: bool
				whether to call print_opts() function above to print all chosen options
			n0 : float
				use if not starting simulation from nt=0, generally used for restarting a simulation (see
				restart_simulation.py)

		Usage:
			Run simulation for 1000 time steps with dt = 300 s
				model.solve_heat(nt=1000, dt=300)
		"""
		self.dt = dt
		start_time = _timer_.clock()
		self.num_iter = []
		if print_opts: self.print_all_options(nt)

		for n in range(n0, n0 + nt):
			TErr, phiErr = np.inf, np.inf
			T_last, phi_last = self.T.copy(), self.phi.copy()
			k_last, rhoc_last = self.k.copy(), self.rhoc.copy()
			iter_k = 0
			while (TErr > self.Ttol and phiErr > self.phitol):

				Tx, Tz = self.get_gradients(T_last)
				self.update_liquid_fraction(phi_last=phi_last)
				if self.issalt: self.saturated = self.update_salinity(phi_last=phi_last)
				self.update_volume_averages()
				self.Q = self.update_sources_sinks(phi_last=phi_last, T_last=T_last)

				self.T[1:-1, 1:-1] = T_last[1:-1, 1:-1] + Tx + Tz + self.Q * self.dt / rhoc_last[1:-1, 1:-1]

				self.apply_boundary_conditions(T_last, k_last, rhoc_last)

				TErr = (abs(self.T[1:-1, 1:-1] - T_last[1:-1, 1:-1])).max()
				phiErr = (abs(self.phi[1:-1, 1:-1] - phi_last[1:-1, 1:-1])).max()

				# kill statement when parameters won't allow solution to converge
				if iter_k > 2000:
					raise Exception('solution not converging')

				iter_k += 1
				T_last, phi_last = self.T.copy(), self.phi.copy()
				k_last, rhoc_last = self.k.copy(), self.rhoc.copy()
			# outputs here
			self.num_iter.append(iter_k)
			self.model_time += self.dt

			try:  # save outputs
				self.outputs.get_results(self, n=n)
			# this makes the runs incredibly slow and is really not super useful, but here if needed
			# save_data(self, 'model_runID{}.pkl'.format(self.outputs.tmp_data_file_name.split('runID')[1]),
			#          self.outputs.tmp_data_directory, final=0)
			except AttributeError:  # no outputs chosen
				pass

			if self.freezestop:  # stop if no liquid remains
				if (self.phi[self.geom] == 0).all():
					print('instrusion frozen at {0:0.04f}s'.format(self.model_time))
					self.run_time = _timer_.clock() - start_time
					return self.model_time

		# del T_last, phi_last, Tx, Tz, iter_k, TErr, phiErr

		self.run_time = _timer_.clock() - start_time

	class stefan:
		"""
		Analytical two-phase heat diffusion problem for comparison with model results.
		(See https://en.wikipedia.org/wiki/Stefan_problem)
		"""

		def solution(self, t, Ti, Tw=273.15):
			"""
			Analytical solution to the two-phase Stefan Problem for freezing
			Parameters:
				t : float
					time, s
				Ti : float
					temperature of solid ice, K
				Tw : float
					temperature of liquid (generally at freezing temperature), K
			'Returns':
				stefan.zm : float
					melting/freezing front position at time t
				stefan.zm_func : function object
					melting/freezing front position as a function of time
						Usage: t = array(0, 1e6)  # s; zm = stefan.zm_func(t)
				stefan.zm_const : float
					constant for the melting/freezing front position,i.e. 2 * lam * sqrt(kappa)
				stefan.zi : array
					array of position values from 0 to the melting/freezing front problem (0 < z < zm), for use with
					stefan.T
				stefan.Ti : array
					temperature profile of ice (Ti) at time t along position z
				stefan.zw : array
					array of position values from the freezing front position zm to the domain size (zm < z < Lz)
				stefan.Tw : array
					temperature profile of water (Tw) at time t along position z

			Usage:
					model.stefan.solution(model, t=1e6, Ti=100, Tw=273.15)
			"""
			# ice properties are constant
			Ki = self.constants.ki / (self.constants.rho_i * self.constants.cp_i)
			Kw = self.constants.kw / (self.constants.rho_w * self.constants.cp_w)
			v = np.sqrt(Ki / Kw)
			Sti = self.constants.cp_i * (self.constants.Tm - Ti) / self.constants.Lf
			Stw = self.constants.cp_w * (Tw - self.constants.Tm) / self.constants.Lf

			func = lambda x: Sti / (np.exp(x ** 2) * erf(x)) \
			                 - v * self.constants.kw * self.constants.cp_i * Stw \
			                 / (self.constants.ki * self.constants.cp_w * erfc(v * x) * np.exp(x ** 2 * v ** 2)) \
			                 - x * np.sqrt(np.pi)
			lam = optimize.root(func, 1)['x'][0]

			self.stefan.zm_const = 2 * lam * np.sqrt(Ki)
			self.stefan.zm = self.stefan.zm_const * np.sqrt(t)
			def zm_func(t):
				return 2 * lam * np.sqrt(Ki * t)
			self.stefan.zm_func = zm_func
			self.stefan.zi = np.linspace(0, self.stefan.zm, int(self.stefan.zm / self.dz))
			self.stefan.Ti = Ti + (self.constants.Tm - Ti) * erf(self.stefan.zi / np.sqrt(4 * Ki * t)) / erf(lam)
			self.stefan.zw = np.linspace(self.stefan.zm, self.Lz, int((self.Lz - self.stefan.zm) / self.dz))
			self.stefan.Tw = Tw - (Tw - self.constants.Tm) * erfc(self.stefan.zw / np.sqrt(4 * Kw * t)) / erfc(v * lam)

		def compare(self, dt, stop=0.9):
			"""
			Runs a simulation of the Stefan problem to ensure discretization is correct
			Parameters:
				dt : float
					time step, s
				stop : float
					percent of domain that is melted/frozen at which to stop the simulation
			Usage:
				model.stefan.compare(0.1, stop=0.5)
			"""
			self.dt = dt
			self.model_time = 0
			self.set_boundaryconditions(top=True, bottom=True, sides='NoFlux')
			self.outputs.get_results(self, n=0)
			n = 1
			fflast = 0
			self.num_iter = []
			tmp = np.where(self.phi > 0)
			ff = min(tmp[0]) * self.dz
			strt = _timer_.time()
			while ff < stop * self.Lz:
				TErr, phiErr = np.inf, np.inf
				T_last, phi_last = self.T.copy(), self.phi.copy()
				k_last, rhoc_last = self.k.copy(), self.rhoc.copy()
				iter_k = 0
				while (TErr > self.Ttol and phiErr > self.phitol):

					Tx, Tz = self.get_gradients(T_last)
					self.update_liquid_fraction(phi_last=phi_last)
					self.update_volume_averages()
					self.Q = self.update_sources_sinks(phi_last=phi_last, T_last=T_last)

					self.T[1:-1, 1:-1] = T_last[1:-1, 1:-1] + Tx + Tz + self.Q * self.dt / rhoc_last[1:-1, 1:-1]

					self.apply_boundary_conditions(T_last, k_last, rhoc_last)

					TErr = (abs(self.T[1:-1, 1:-1] - T_last[1:-1, 1:-1])).max()
					phiErr = (abs(self.phi[1:-1, 1:-1] - phi_last[1:-1, 1:-1])).max()

					# kill statement when parameters won't allow solution to converge
					if iter_k > 1000:
						raise Exception('solution not converging')

					iter_k += 1
					T_last, phi_last = self.T.copy(), self.phi.copy()
					k_last, rhoc_last = self.k.copy(), self.rhoc.copy()

				# outputs here
				self.outputs.get_results(self, n=n)
				self.model_time += self.dt
				self.num_iter.append(iter_k)

				n += 1
				tmp = np.where(self.phi == 0)
				ff = max(tmp[0]) * self.dz
				if ff / self.Lz in [.10, .20, .30, .40, .50, .60, .70, .80, .90]:
					if ff != fflast: print('\t {}% frozen at {}s'.format(ff / self.Lz * 100, self.model_time))
					fflast = ff
			self.run_time = _timer_.time() - strt
			self.stefan.solution(self, t=n * self.dt, Ti=self.Tsurf, Tw=self.Tbot)
