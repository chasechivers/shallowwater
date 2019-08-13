from utility_funcs import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

sns.set(palette='colorblind', color_codes=1, style='ticks')


def slice_in_time(results, var, idx, dim='z', cmap='RdBu_r'):
	"""
	Parameters:
		results : dict
			dictionary of results from simulation
		var : str
			string of a key in results dictionary that you want to plot
		idx : int
			position that you'd want to see through time
		dim : str
			'x' or 'z', which dimension to plot through time, default 'z'
		cmap : str
			colormap to plot in
	"""
	var_t = results[var]
	time = results['time']

	if 'z' in dim:
		# space = md.Z[:,1]
		slice = var_t[:, :, idx]
		lbl = 'depth'
	elif 'x' in dim:
		# space = md.X[1,:]
		slice = var_t[:, idx, :]
		lbl = 'distance from center'

	x, time = np.meshgrid(, time)
	fig = plt.figure()
	plt.pcolormesh(slice, cmap=cmap)
	if 'z' in dim:
		plt.gca().invert_yaxis()
		plt.ylabel(lbl)
	elif 'x' in dim:
		plt.ylabel(lbl)
	plt.colorbar(label=var)
	plt.xlabel('time s')
	plt.show()

	return fig, var_t, time


class PlotModel:

	def __init__(self, model, results):
		self.model = model
		self.results = results
		self.vars = results.keys()

	def slice_in_time(self, var, idx=1, dim='z', cmap='RdBu_r'):
		"""

		:param var: string
		:param idx: int
		:param dim: string
		:param cmap:  string
		:return:
		"""
		try:
			var_t = self.results[var]
		except:
			raise Exception('No such variable in results, choose from:\n\t{}'.format(self.vars))
		fig = plt.figure()
		if 'z' in dim:
			slice = var_t[:, idx, :]
			plt.ylabel('depth, m')
			y, time = np.meshgrid(self.model.Z[:, 1], self.results['time'])
			plt.gca().invert_yaxis()
		elif 'x' in dim:
			slice = var_t[:, :, idx]
			plt.ylabel('distance from center, m')
			y, t = np.meshgrid(self.model.X[:, 1], self.results['time'])

		plt.pcolormesh(time, y, slice, cmap=cmap)
		plt.colorbar(label=var)
		plt.xlabel('time s')
		plt.show()

		return fig, var_t
