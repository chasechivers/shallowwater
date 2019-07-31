from IceSystem import IceSystem
from utility_funcs import load_data, nat_sort, save_data
import os
import numpy as np

data_dir = '/Users/chasechivers/Desktop/'  # os.getcwd() + '/tmp/'
outputdirectory = '/tmp'  # '/nv/hp5/cchivers3/scratch/'
OF = 50  # years
composition = 'MgSO4'
topBC = True
bottomBC = True
rt = 72 * 60 * 60

# assign the runID
runID = '2271'

# or use thickness to identify runID for a simulation run
# results = nat_sort([res for res in os.listdir(data_dir) if str(th) in res and 'tmp_data' in res])
# runID = results[0].split('runID')[1][:4]

# load results
print('grabbing all file names')
results = nat_sort([res for res in os.listdir(data_dir) if 'runID' + runID in res and 'model' not in res])

# didn't save latest model, old version
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
radius = dx * w[1].max()
depth = dz * w[0].min()
thickness = dz * w[0].max() - depth
model.init_intrusion(Tbot, depth, thickness, radius, geometry=w)

# determine salinity stuff
if res_min['S'].any() != 0:
	print('   initializing salinity')
	Si = res_min['S']
	concentration = Si[phii == 1][0]

	if Si[0, 0] > 0:
		shell = True
	else:
		shell = False

	if Ti[int((depth + thickness / 2) / dz), 1] == Tbot:
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

model.set_boundaryconditions(sides='Reflect', top=topBC, bottom=bottomBC)

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

if rt > 0:
	model.run_time += rt

print(f'solved in {model.run_time}s')

print('grabbing results')
model.outputs.transient_results = model.outputs.get_all_data(model)
print('saving data to ', out_dir)
print('   saving model')
save_data(model, 'md_{}'.format(model.outputs.tmp_data_file_name.split('tmp_data_')[1]), out_dir)
print('   saving results')
save_data(model.outputs.transient_results, 'rs_{}'.format(model.outputs.tmp_data_file_name.split('tmp_data_')[1]),
          out_dir)
