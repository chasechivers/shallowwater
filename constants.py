from dataclasses import dataclass

@dataclass
class constants:
	styr: float = 3.154e7  # s/yr, seconds in a year

	# Orbital/other parameters
	GM: float = 3.96423e-14  # G*Msun [AU^3/s^2]
	g: float = 1.32  # m/s2, Europa surface gravity
	rAU: float = 5.20442  # AU, Europa's solar distance
	# ecc : float    = 0.009       # Europa's eccentricity about Jupiter
	ecc: float = 0.049  # Jupiter's eccentricity about the sun
	obliq: float = 3.1  # Jupiter's obliquity
	lonPer: float = 0.257061  # Europa's longitude of perihelion

	# Thermal properties
	rho_i: float = 917.  # kg/m^3, pure ice density
	rho_w: float = 1000.  # kg/m^3 pure water density
	cp_i: float = 2.11e3  # J/kg/K, pure ice heat capacity
	cp_w: float = 4.19e3  # J/kg/K, pure water heat capacity
	ki: float = 2.3  # W/m/K, pure ice thermal conductivity
	kw: float = 0.56  # W/m/K, pure water thermal conductivity
	ac: float = 567.  # W/m, ice thermal conductivity constant, ki = ac/T (Klinger, 1980)
	Tm: float = 273.15  # K, pure ice melting temperature at 1 atm
	Lf: float = 333.6e3  # J/kg, latent heat of fusion of ice
	Lv: float = 2833.0e3  # J/kg, Latent heat of sublimation of H2O (Stearns and Weidner, 1993)
	expans: float = 1.6e-4  # 1/K, thermal expansivity of ice

	C_rho: float = 0.  # kg/m^3/ppt, linear density-salinity relationship for water (density = rho_w + C_rho * S)
	Ci_rho: float = 0.  # kg/m^3/ppt, linear density-salinity relationship for ice (density = rho_i + Ci_rho * S)
	rho_s: float = 0.  # kg/m^3, salt density, assigned only when salinity is used

	# Radiative properties
	# average albedo, assumes that equilibrium surface temperature at equator is 110 K
	# iterated over in model to account for equilibrium fluxes into/out of top grid and tidal heating
	albedo: float = 0.68  # Europa bond albedo (Ashkenazy, 2019)
	emiss: float = 0.94  # Europa emissivity (Ashkenazy, 2019)
	stfblt: float = 5.67e-08  # W/m^2/K^4 Stefan-Boltzmann constant
	latitude: float = 0.0
	solar_const: float = 1361.  # W/m^2, Solar constant
	solar_day: float = 3.06822e5  # s, Mean length of solar day
	m_H2O: float = 18.02e-3  # kg/mol, molar mass of water

	# Constants for viscosity dependent tidal heating
	#   from Mitri & Showman (2005)
	Qs: float = 60e3  # J/mol, activation energy of ice (Goldsby & Kohlstadt, 2001)
	Rg: float = 8.3144598  # J/K/mol, gas constant
	eps0: float = 1e-5  # maximum tidal flexing strain
	omega: float = 2.5e-5  # 1/s, tidal flexing frequency
	visc0i: float = 1e13  # Pa s, minimum reference ice viscosity at T=Tm
	visc0w: float = 1.3e-3  # Pa s, dynamic viscosity of water at 0 K

	# Mechanical properties of ice
	G: float = 3.52e9  # Pa, shear modulus/rigidity (Moore & Schubert, 2000)
	E: float = 2.66 * G  # Pa, Young's Modulus, assuming Poisson's ratio of 0.33
