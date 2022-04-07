import numpy as np


class ThermophysicalProperties:
	"""
	Class instance to hold all of the thermophysical properties of the phases in a simulation.
	"""

	def ki_eff(self, k, p, which=2):
		"""Function for effective ice thermal conductivity as a function of porosity.

		Parameters
		----------
		k : numpy.ndarray, float
			Ice thermal conductivity, W/m/K
		p : numpy.ndarray, float
			Ice void/fracture/etc. porosity, volume fraction

		Returns
		----------
		keff : numpy.ndarray, float
			Ice thermal conductivity, W/m/K
		"""
		if which == 0:
			return k * (1 - (3 * p) / (2 + p))
		elif which == 1:
			return 2 * k * (1 - p) / (2 + p)
		elif which == 2:
			return k * (1 - p)
		elif which == 3:
			return k * (2 - 3 * p) / 2.
		elif which == 4:  # Callone 2019
			kvoid = 1e-16
			rho = (1 - p) * self.constants.rho_i
			rho_transition = 450.
			theta = 1 / (1 + np.exp(-2 * 0.02 * (rho - rho_transition)))
			kfirn = 2.107 + 0.003618 * (rho - self.constants.rho_i)
			ksnow = 0.024 - 1.2e3 - 4 * rho + 2.5e-6 * rho ** 2
			keff = (1 - theta) * k * kvoid * ksnow / self.constants.ki / kvoid + theta * k * kfirn / self.constants.ki
			return keff

	def ki(self, T, S):
		"""Function for ice thermal conductivity as a function of salinity and/or temperature.

		Porosity-dependence is also implemented by calling the ki_eff function aboe.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		ans : numpy.ndarray, float
			Ice thermal conductivity, W/m/K
		"""
		ans = self.constants.ac / T if self.kT else self.constants.ki
		return self.ki_eff(ans, self.porosity) if "porosity" in self.__dict__ else ans

	def kw(self, T, S):
		"""Function for water thermal conductivity as a function of salinity and/or temperature.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		k_w : numpy.ndarray, float
			Water thermal conductivity, W/m/K
		"""
		return self.constants.kw

	def dkidT(self, T, S):
		"""First derivative of ice thermal conductivity (assumed to be ki ~ const./T) with respect to temperature. Used
		when coordinate system is assumed to be cylindrical.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		ans : numpy.ndarray, float
			First derivative of water thermal conductivity w.r.t T, W/m/K^2
		"""
		ans = -self.constants.ac / T ** 2 if self.kT else 0
		return ans

	def dkwdT(self, T, S):
		"""First derivative of  water thermal conductivity (assumed to be kw ~ const.) with respect to temperature. Used
		when coordinate system is assumed to be cylindrical.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		d k_w / dT : numpy.ndarray, float
			First derivative of water thermal conductivity w.r.t T, W/m/K^2
		"""
		return 0.

	def cpi(self, T, S):
		"""Function for ice specific heat capacity as a function of salinity and/or temperature.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		cp_i : numpy.ndarray, float
			Ice specific heat capacity, J/kg/K
		"""
		return self.constants.cp_i * ((1 - self.p) if "p" in self.__dict__ else 1)

	def cpw(self, T, S):
		"""Function for water specific heat capacity as a function of salinity and/or temperature.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		cp_w : numpy.ndarray, float
			Water specific heat capacity as a function of salinity and/or temperature, J/kg/K
		"""
		return self.constants.cp_w

	def rhoi(self, T, S):
		"""Function for ice density as a function of salinity and/or temperature.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		rhoi : numpy.ndarray, float
			Ice density, kg/m^3
		"""
		ice_density = self.constants.rho_i + self.constants.Ci_rho * S
		if "porosity" in self.__dict__.keys():
			return (1 - self.porosity) * ice_density
		else:
			return ice_density

	def rhow(self, T, S):
		"""Function for water density as a function of salinity and/or temperature.

		Parameters
		----------
		T : numpy.ndarray, float
			Temperature, K
		S : numpy.ndarray, float
			Salinity, ppt

		Returns
		----------
		rhow : numpy.ndarray, float
			Water density, kg/m^3
		"""
		return self.constants.rho_w + self.constants.C_rho * S

	def Tm_func(self, S, a, b, c):
		"""
		Melting temperature of water based on the concentration of major salt. Assumes second order polynomial fit to
		FREEZCHEM derived melting temperature (Tm) as a function of salinity (in ppt).
		Note: This may be modified to any fit (3rd+ order polynomial, exponential, etc.), just make sure to modify
		constants in SalinityConstants.py for concentration as well as the return (i.e. instead write: return a*np.exp(
		-b*S) + c)
		:param S: float, arr
			A float/int or array of salinities in ppt
		:param a: float
			Polynomial coefficient
		:param b: float
			Polynomial constants
		:param c: float
			Polynomial constants
		:return:
			Melting temperature at salinity S
		"""
		return a * S ** 2 + b * S + c

	def ice_viscosity(self, T, Tm):
		"""Arrhenius form of temperature-dependent effective viscosity of ice.

		Parameters
		----------
		T : float, numpy.ndarrary
			Temperature in K
		Tm : float, numpy.ndarray
			Melting temperature of ice, K

		Returns
		----------
		viscosity : float, numpy.ndarray
			Ice viscosity, Pa s
		"""
		return self.constants.visc0i * np.exp(self.constants.Qs * (Tm / T - 1) / self.constants.Rg / Tm)

	def pv(self, T):
		"""Equilibrium vapor pressure of ice at temperature T (Marti & Mauersberger, 1993)

		Parameters
		----------
		T : np.ndarray
			Surface temperatures, K

		Returns
		----------
		pv : np.ndarray
			Equilibrium vapor pressure, Pa
		"""
		return 10. ** (-2663.5 / T + 12.537)

	def skin_depth(self, T, S, P):
		"""Calculates the thermal skin depth -- the depth of penetration of a temperature wave with period P.

		Generally this is only used when a "radiative" upper boundary condition is used with orbital elements.

		Parameters
		----------
		T : float
			Temperature, K
		S : float
			Salinity, ppt
		P : float
			Period of temperature wave (daily, yearly, etc.), s

		Returns
		----------
		skin_depth : float
			Thermal skin depth, m
		"""
		return (self.ki(T, S) * P / np.pi / self.rhoi(T, S) / self.cpi(T, S)) ** 0.5
