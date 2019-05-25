import numpy as np


def normalize(x):
	return x / (x.max() - x.min())


def dP(fraction_frozen, radius, thickness, ri=917, rw=1000., G=3.52e9, nu=0.33, shape=1):
	'''
	Calculate the pressure in an elliptical shaped cavity in an elastic medium where the pressure created by the volume
	change due to freezing (Amorusco and Cresctini, 2009)
	Parameters:
	:param fraction_frozen:
		Volume fraction of the liquid that is now ice
	:param radius:
		Initial radius of intrusion, m
	:param thickness:
		Initital thickness of intrusion, m
	:param ri:
		Ice density
	:param rw: water density, 1000 kg/m^3
	:param G: elastic medium (ice) rigidity, Pa
	:param nu: Poisson's ratio of elastic medium (ice)
	:param shape:
		1
	:return: Pressure in cavity
	'''
	ri, rw = 917, 1000
	G, nu = 3.52e9, 0.33
	# V = 4/3. * np.pi * radius * thickness/2
	if shape == 1:
		A = fraction_frozen * rw / ri * 2 * G * (1 + nu) / (1 - 2 * nu)
		B = radius / thickness / 2 * 4 / np.pi * (1 - nu ** 2) / (1 - 2 * nu) - 3
		return A / B
	elif shape == 2:
		return fraction_frozen * rw / ri * 2 * G
	elif shape == 3:
		return 4 / 3 * fraction_frozen * rw / ri * 2 * G


def mass_of_salt(salt, radius, thickness, density, pf=0):
	'''
	Could be used to calculate the mass of salt removed at a time, or initial salt in intrusion
	Parameters:
		salt : float, 1D array
			Amount of salt in ppt
		radius : float
			Radius of initial intrusion, m
		thickness : float
			Thickness of initial intrusion, m
		density : float
			Density of water that the salt is in
		pf : float
			Percent of original liquid that is now ice, used for finding volume of water when the salt was rejected.
	Return:
		Mass of salt in grams
		
	Usage:
		- case of initial salt:
			S - 2D grid of salinity values
			beta - linear relation between density for water and salt
			Ci - initial concentration of salt in intrusion
			M_Si - initial mass of salt in intrusion
			
			M_Si = mass_of_salt(Ci, radius, thickness, rw + beta * Ci)
			
		- Case of finding the total mass of salt removed in time:
			rmv - time series of removed salt, ppt
			pf - volume fraction of original liquid that is frozen
			S_t - salinity of liquid at time of rejection, ppt
			
			mass_salt_rmv = (rmv, radius, thickness, rw + beta * S_t, pf)
			total_salt_rmv = cumsum(mass_salt_rmv)[-1]
	'''
	V = (1 - pf) * 4. / 3 * np.pi * radius * thickness / 2.
	return salt / 1000 * V * density


def height_of_salt(area, density_s, mass):
	return mass / (area * density_s)


##
'''
HOW TO FIND MASS OF SALT

initial mass of asalt = V_i * density_water+S * concentration/1000 

at the end =>
percent of salt removed = np.cumsum(md.removed_salt)[-1]/md.total_salt[0]
mass of salt removed = percent of salt removed * initial mass of salt

'''
