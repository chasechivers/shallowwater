import dill as pickle


def file_namer(model, *argv):
	dict = {key: value for key, value in model.__dict__.items() if not key.startswith('__') and not callable(key) \
	        and type(value).__module__ is not 'numpy'}
	file_name = '../results/EuropaShell'

	def string_IO(input):
		if isinstance(input, bool):
			if input is True:
				return 'on'
			else:
				return 'off'
		else:
			return input

	for var in argv:
		if var not in dict.keys():
			file_name += '_{}'.format(var)
		else:
			file_name += '_{}={}'.format(var, string_IO(dict[var]))
	return file_name + '.pkl'


def save_model(model, *argv):
	file_name = file_namer(model, *argv)
	with open(file_name, 'wb') as output:
		pickle.dump(model, output, -1)


def load_model(file_name):
	with open(file_name, 'rb') as input:
		return pickle.load(input)
