import string, random, os
import numpy as np
from utility_funcs import *


class Outputs:
	def __init__(self, OF: int, tmpdir: str, name: str = ""):
		self.tmp_data_directory = tmpdir
		self.tmp_data_file_name = 'tmp_data_runID' + ''.join(
				random.choice(string.digits) for _ in range(4)) + f"_{name}"
		self.output_frequency = int(OF)

	def choose(self, outlist: list, issalt: bool):
		if issalt is True:
			outlist.append('S')
		self.outputs = {key: [] for key in outlist}
		self.outputs['time'] = []

	def calculate_outputs(self, model: object) -> dict:
		ans = {}
		ans["time"] = model.model_time
		for key in self.outputs.keys():
			if key in model.__dict__.keys():
				ans[key] = model.__dict__[key]
			elif key == 'percent frozen':
				ans[key] = 1 - model.phi.sum() / model.phi_initial.sum()
			elif key == 'r':
				tmp = np.where(model.phi > 0)
				ans[key] = model.dx * max(tmp[1])
				del tmp
			elif key == 'h':
				tmp = np.where(model.phi > 0)
				ans[key] = (max(tmp[0]) - min(tmp[0])) * model.dz
				del tmp
			elif key == 'freeze fronts':
				tmp = np.where(model.phi > 0)
				ans[key] = np.array([min(tmp[0]), max(tmp[0])]) * model.dz
				del tmp
			else:
				pass
		return ans

	def get_results(self, model: object, n: int):
		"""
		Calls outputs.calculate_outputs() then saves dictionary of results to file
		:parameter model: class,
		"""
		if n % self.output_frequency == 0:
			if model.verbose:
				print(f"Saving outputs at it: {n}, run time: {model.run_time / 60:0.02f} min ")
				print(f"\t time: {model.model_time:0.01f} s ({model.model_time / model.constants.styr:0.01f} yr) [time "
				      f"step: {model.dt:.01f} s]")
				print(f"\t outputs:  {list(self.outputs.keys())}")
				print(f"\t % frozen: {(1 - model.phi.sum() / model.phi_initial.sum()) * 100:0.03f}")
			get = self.calculate_outputs(model)
			save_data(get, self.tmp_data_file_name + '_n={}'.format(n), self.tmp_data_directory)

	def get_all_data(self, del_files: bool = True) -> dict:
		"""Concatenates all saved outputs from outputs.get_results() and puts into a single dictionary object."""
		cwd = os.getcwd()  # find working directory
		os.chdir(self.tmp_data_directory)  # change to directory where data is being stored

		# make a list of all results files in directory
		data_list = nat_sort([data for data in os.listdir() if data.endswith('.pkl') and \
		                      self.tmp_data_file_name in data])
		# copy dictionary of desired results
		ans = {k: v for k, v in self.outputs.items()}
		# iterate over file list
		for file in data_list:
			tmp_dict = load_data(file)  # load file
			for key in self.outputs:  # iterate over desired outputs
				ans[key].append(tmp_dict[key])  # add output from result n to final file
			del tmp_dict
			if del_files: os.remove(file)

		# make everything a numpy array for easier manipulation
		ans = {key: np.asarray(value) for key, value in ans.items()}

		# go back to working directory
		os.chdir(cwd)
		return ans

	def __repr__(self):
		st = f"temporary directory:   {self.tmp_data_directory}\n"
		st += f'file ID#:             {self.tmp_data_file_name.split("tmp_data_")[1]}\n'
		st += f'requested outputs:    {list(self.outputs.keys())}\n'
		st += f'output frequency:     {self.output_frequency} steps\n'
		return st
