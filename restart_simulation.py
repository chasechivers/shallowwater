from modular_build import IceSystem
from utility_funcs import load_data, nat_sort, save_data
import os
import numpy as np

data_dir = os.getcwd() + '/tmp/'
outputdirectory = '/nv/hp5/cchivers3/scratch/'
th = 500.0  # sill thickness
composition = 'MgSO4'
OF = 50  # years

# use thickness to identify runID for a simulation run
# results = nat_sort([res for res in os.listdir(data_dir) if str(th) in res and 'tmp_data' in res])
# runID = results[0].split('runID')[1][:4]

# or just assign the runID
runID = '9584'

# load results
print('grabbing all file names')
latest_model = [md for md in os.listdir(data_dir) if 'model' in md and runID in md]
results = nat_sort([res for res in os.listdir(data_dir) if 'runID' + runID in res and 'model' not in res])

if len(latest_model) == 1:
	print('loading from saved model: '.format(latest_model[0]))
	model = load_data(data_dir + latest_model[0])
	n0 = round(model.model_time / model.dt)
	print('   restarting at n = '.format(n0))
	latest_result = load_data(data_dir + results[-1])
	model.outputs.choose(model, output_list=list(latest_result.keys()),
	                     output_frequency=int(OF * 3.14e7 / model.dt))
	dt = model.dt

# didn't save latest model, old version
elif len(latest_model) == 0:
	print('loading from latest results')
	max_n, max_idx = 0, 0
	min_n, min_idx = 0, 0
	name = ''
	for i in range(len(results)):
		if int(results[i].split('n=')[1][:-4]) > max_n:
			max_n = int(results[i].split('n=')[1][:-4])
			max_idx = i
			name = results[i].split('n=')[0]
		if int(results[i].split('n=')[1][:-4]) == min_n:
			min_idx = i

	# get model info from latest and first results
	print('   grabbing model config from first and last results')
	res_max = load_data(data_dir + results[max_idx])
	res_min = load_data(data_dir + results[min_idx])
	dx, dz = 10., 10.
	dt = res_max['time'] / max_n
	Ti = res_min['T']
	phii = res_min['phi']
	Lz, Lx = np.shape(Ti)
	Lx, Lz = (Lx - 1) * dx * 2, (Lz - 1) * dz
	Tsurf, Tbot = Ti[0, 0], Ti[-1, -1]

	# initialize model
	model = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, cpT=False, use_X_symmetry=True)
	model.init_T(Tsurf=Tsurf, Tbot=Tbot)

	# find intrusion geometry
	w = np.where(phii > 0)
	radius = dx * (1 + w[1].max())
	depth = dz * (w[0].min() - 1)
	thickness = (dz * (w[0].max() + 1) - depth)
	model.init_intrusion(Tbot, depth, thickness, radius, geometry=w)

	# determine salinity stuff
	if model.S.any() != 0:
		print('   initializing salinity')
		Si = res_min['S']
		concentration = Si[phi == 1]

		if Si[phi == 0].any() != 0:
			shell = True
		else:
			shell = False
		if Ti[phi == 1] == Tbot:
			T_match = True
		else:
			T_match = False
		model.init_salinity(concentration=concentration, composition=composition,
		                    shell=shell, T_match=T_match)

	# update fields to latest results
	print('   updating config to latest results')
	model.T = res_max['T']
	model.phi = res_max['phi']
	model.S = res_max['S']
	if 'k' in res_max.keys():
		model.k = res_max['k']
	else:
		model.update_volume_averages()

	model.set_boundaryconditions(sides='Reflect')

	n0 = max_n
	model.model_time = res_max['time']
	model.outputs.choose(model, output_list=list(res_min.keys()),
	                     output_frequency=int(OF * 3.14e7 / dt))
	model.freezestop = 1
	model.tidalheat = 1
	del Ti, phii

# re-assign outputs
model.outputs.tmp_data_directory = data_dir
model.outputs.tmp_data_file_name = results[0].split('_n=')[0]
print('now solving; restarting from n =', n0)
model.solve_heat(nt=50000000000000000000, dt=dt, n0=n0)
print(f'solved in {model.run_time}s')
results = model.outputs.get_all_data(model)
print('saving data to ', outputdirectory)
save_data(model, 'md' + model.outputs.tmp_data_file_name.split('tmp_data_')[1], outputdirectory)
save_data(results, 'rs' + model.outputs.tmp_data_file_name.split('tmp_data_')[1], outputdirectory)
