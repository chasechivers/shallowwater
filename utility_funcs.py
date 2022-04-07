import dill as pickle
import re
import os
import numpy as np


def nat_sort(x):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(x, key=alphanum_key)


def file_namer(model, outputdir, *argv):
	"""
	Automatically name a file so that simulation choices can be gleaned from only the file name!
	:param model: IceSystem class
		IceSystem class instance
	:param outputdir: str
		Directory that file will be saved to. Should include "/" at the end!
	:param argv:
	:return:
	"""

	# Variables that would be wanted displayed in a file name
	# Listed are "defaults" that we wouldn't normally want to list them
	# Some are deliberately zero so that they are always included in the filename, i.e. thickness
	wants = {"D": 5e3, "w": model.radius * 1.25, "dx": 10, "dz": 10, "dt": model.dt, "kT": True,
	         "radius": 2.4 * model.depth, "thickness": 0, "depth": 0, "Tsurf": 0,
	         "Tbot": 0, "topBC": True, "sidesBC": "NoFlux"}

	if 'frac_width' in model.__dict__.items():
		# defaults["frac_width"] = model.frac_width
		# defaults["frac_height"] = model.frac_height
		wants["frac_width"] = model.frac_width
		wants["frac_height"] = model.frac_height

	if model.issalt:
		# defaults['C_rho'] = model.C_rho
		# defaults['Ci_rho'] = model.Ci_rho
		# defaults['rejection_cutoff'] = 0.25
		# defaults['composition'] = 0
		# defaults['concentration'] = 0
		# defaults['saturation_point'] = model.saturation_point
		wants['composition'] = 0
		wants['concentration'] = 0
	# try:
	#	defaults['heat_from_precipitation'] = model.heat_from_precipitation
	#	defaults['enthalpy_of_formation'] = model.enthalpy_of_formation
	#	defaults['salinate'] = model.salinate
	# except AttributeError:
	#	pass

	dict = {key: value for key, value in model.__dict__.items() if key in wants and value != wants[key]}
	# dict = {key: value for key, value in model.__dict__.items() if not key.startswith('__') and \
	#        not callable(key) and type(value) in [str, bool, int, float] and value != defaults[key]}

	file_name = outputdir
	if model.verbose: print(" file_namer-> naming file!")

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
				file_name += f"_{var}={dict[var]}"
			else:
				file_name += var

	else:
		for key in dict.keys():
			if isinstance(dict[key], float):
				file_name += f"_{key}={dict[key]:0.01f}"
			else:
				file_name += f"_{key}={string_IO(dict[key])}"

	return file_name + '.pkl'


def save_data(data, file_name, outputdir, final=1, *argv):
	if outputdir[-1] != "/":
		outputdir += "/"
	if isinstance(data, (dict, list)) or type(data).__module__ == np.__name__:
		file_name = outputdir + file_name + '.pkl'
	elif final == 1:
		file_name = file_namer(data, outputdir + file_name, *argv)
		print(f'Saving as {file_name}')
	elif final == 0:
		file_name = outputdir + file_name
	with open(file_name, 'wb') as output:
		pickle.dump(data, output, -1)
		output.close()


def load_data(file_name):
	try:
		with open(file_name, 'rb') as input:
			return pickle.load(input)
	except ValueError:
		import pickle5 as p5
		with open(file_name, 'rb') as input:
			return p5.load(input)
	except pickle.UnpicklingError:
		print("Not working :(")


def directory_spider(input_dir, path_pattern="", file_pattern="", maxResults=500000):
	'''
	Returns list of paths to files given an input_dir, path_pattern, and file_pattern
	'''
	file_paths = []
	if not os.path.exists(input_dir):
		raise FileNotFoundError(f"Could not find path: {input_dir}")
	for dirpath, dirnames, filenames in os.walk(input_dir):
		if re.search(path_pattern, dirpath):
			file_list = [item for item in filenames if re.search(file_pattern, item)]
			file_path_list = [os.path.join(dirpath, item) for item in file_list]
			file_paths += file_path_list
			if len(file_paths) > maxResults:
				break
	return file_paths[0:maxResults]


def untar_file(tarfilename, outdir, which_file='_'):
	import tarfile
	try:
		print('Opening file:', tarfilename)
		t = tarfile.open(tarfilename, 'r')
		print('... File opened')
	except IOError as e:
		print(e)
	else:
		if which_file == '_':
			print('Extracting all files to', outdir)
			t.extractall(outdir)
			print('Finished extracting')
			filelist = [member.name for member in t.getmembers()]
		else:
			if 'md' in which_file:
				print('Matching model file to results file in zip')
				idx = which_file.find('runID')
				NID = which_file[idx:idx + 9]
				for dirpath, dirname, filename in os.walk(outdir + '/results/'):
					filelist = [item for item in filename if NID in item]
					if len(filelist) > 0:
						print('File already extracted:\n\t', filelist)
						return filelist[0]
				extract_this = [member for member in t.getmembers() if NID in member.name]
				print('Extracting', extract_this, 'to', outdir)
				t.extractall(outdir, members=extract_this)
				if type(extract_this) is list:
					filelist = extract_this[0].name
				elif type(extract_this) is tarfile.TarInfo:
					filelist = extract_this.name
			else:
				print('Extracting files with', which_file, 'to', outdir)
				extract_this = [member for member in t.getmembers() if which_file in member.name]
				print(' ... Extracting', extract_this)
				t.extractall(outdir, members=extract_this)
				if type(extract_this) is list:
					filelist = extract_this[0].name
				elif type(extract_this) is tarfile.TarInfo:
					filelist = extract_this.name
	t.close()
	return filelist


def results_from_tar(md_filename, del_file=False):
	"""
	:param md_filename: str
		Path + file name of model file that has an associated results file ("rs_runID") with an identifier like
		"runIDWXYZ" in the string and in a subdirectory named "results"
	:param del_file: bool
		Whether to delete the untarred results file after loading. Default off
	:return: dict
		Dictionary file with structure {model property (temperature, salinity, etc): arr..} where arr can be a 3d
		array with [time, z, x] as coordinates, a 1d array of [value] with time.
	"""
	# grab the unique runID number from the model filename md_filename
	IDN = md_filename.find("runID")
	IDN = md_filename[IDN:IDN + len("runIDXXXX")]
	# get the directory that the model file is in
	data_dir = md_filename[:md_filename.find("md")]
	try:  # attempt to see if the file is already unzipped
		filename = directory_spider(data_dir + "results/")
		print("this is the filename:", filename)
		# if its not, raise error and unzip it
		if filename == []:
			raise FileNotFoundError
		# if it does exist, load the data
		print(IDN)
		print([f for f in filename if IDN in f])

		try:
			file_to_load = [f for f in filename if IDN in f][0]
			results = load_data(file_to_load)
		except IndexError:
			try:
				# file not already uncompressed, find the particular results file and uncompress it
				file_to_load = untar_file(data_dir + "results.tar.gz", outdir=data_dir, which_file=md_filename)
				results = load_data(data_dir + file_to_load)
			except FileNotFoundError:
				raise Exception("NO RESULTS FILES POSSIBLE!")

	except FileNotFoundError or IndexError:
		try:
			# file not already uncompressed, find the particular results file and uncompress it
			file_to_load = untar_file(data_dir + "results.tar.gz", outdir=data_dir, which_file=md_filename)
			results = load_data(data_dir + file_to_load)
		except FileNotFoundError:
			raise Exception("NO RESULTS FILES POSSIBLE!")
	if del_file:
		print('data_dir', data_dir)
		print('file_to_load', file_to_load)
		print(f'together: {data_dir}{file_to_load}')
		rmvfn = f'{data_dir}{file_to_load}'
		if '//' in rmvfn:
			rmvfn = file_to_load
		os.remove(rmvfn)
	return results
