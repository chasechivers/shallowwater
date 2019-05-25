import dill as pickle
import re


def nat_sort(x):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(x, key=alphanum_key)


def file_namer(model, outputdir, *argv):
	# defaults chosen for file so that the file name isn't enormous...
	#  variables chosen to be model.__ can be changed to be included in file name if so desired
	defaults = {'Lz': 5e3, 'Lx': 5e3, 'dx': 10, 'dz': 10, 'dt': 3.14e7 / (24 * 4), 'kT': True, 'cpT': True,
	            'issalt': False, 'Tsurf': 0., 'Tbot': 0, 'topBC': True, 'botBC': True, 'sidesBC': True,
	            'tidalheat': False, 'nx': model.nx, 'nz': model.nz, 'Tsill': 273.15, 'depth': 0, 'R_int': 0,
	            'thickness': 0, 'freezestop': model.freezestop, 'num_iter': model.num_iter,
	            'model_time': model.model_time, 'run_time': model.run_time, 'Ttol': 0.1, 'phitol': 0.01,
	            'symmetric': 0, 'Stol': 1, 'cp_i': model.cp_i, 'tidal_heat': model.tidal_heat}
	if model.issalt:
		defaults['C_rho'] = model.C_rho
		defaults['Ci_rho'] = model.Ci_rho
		defaults['saturated'] = model.saturated
		defaults['rejection_cutoff'] = 0.25
		defaults['composition'] = 0
		defaults['concentration'] = 0
		defaults['saturation_point'] = model.saturation_point
	dict = {key: value for key, value in model.__dict__.items() if not key.startswith('__') and \
	        not callable(key) and type(value) in [str, bool, int, float] and value != defaults[key]}
	file_name = outputdir
	print('naming file!')

	def string_IO(input):
		if isinstance(input, bool):
			if input is True:
				return 'on'
			else:
				return 'off'
		else:
			return input

	if len(argv) != 0:
		print('custom naming!')
		for var in argv:
			print(var)
			if var in dict.keys():
				file_name += '_{}={}'.format(var, string_IO(dict[var]))
			else:
				file_name += var
	else:
		for key in dict.keys():
			if key in ['Lx', 'Lz', 'dx', 'dz', 'dt', 'kT', 'cpT', 'issalt', 'Tsurf', 'Tbot', 'depth', 'thickness',
			           'R_int', 'composition', 'concentration', 'topBC', 'sidesBC', 'tidalheat']:
				if isinstance(dict[key], float):
					file_name += '_{}={:0.03f}'.format(key, dict[key])
				else:
					file_name += '_{}={}'.format(key, string_IO(dict[key]))
	return file_name + '.pkl'


def save_data(data, file_name, outputdir, final=1, *argv):
	import numpy as np
	if isinstance(data, (dict, list)) or type(data).__module__ == np.__name__:
		file_name = outputdir + file_name + '.pkl'
	elif final == 1:
		file_name = file_namer(data, outputdir + file_name, *argv)
	elif final == 0:
		file_name = outputdir + file_name
	with open(file_name, 'wb') as output:
		pickle.dump(data, output, -1)
		output.close()


def load_data(file_name):
	with open(file_name, 'rb') as input:
		return pickle.load(input)
