from utility_funcs import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

sns.set(palette='colorblind', color_codes=1, context='notebook', style='ticks')


class PlotModel:

	def __init__(self, model, results):
		self.model = model
		self.results = results
		self.vars = results.keys()

	def plot_2d_slice(self, time, x, var, idx):
		var = np.asarray(var)[:, :, idx]
		time, x = np.meshgrid(time, x)
		plt.figure()
		plt.pcolormesh(time, x, var)
		plt.colorbar()
		plt.show()

	def transient_video(self, *argv):
		vars = []
		for item in argv:
			if item in results.keys():
				vars.append(self.results[item])
			if 'save' in argv:
				save = True
		time = self.results['time']
		if len(vars) > 1:
			pass
		else:
			var = vars[0]
			for i in range(len(time)):
				plt.figure(1)
				plt.clf()
				plt.title('t = {:0.03f}s = {:0.03f}yr'.format(time[i], time[i] / 3.14e7))
				plt.pcolormesh(self.model.X, self.model.Z, var[i], cmap='RdBu_r')
				plt.colorbar()
				plt.gca().invert_yaxis()
				plt.pause(0.00001)
