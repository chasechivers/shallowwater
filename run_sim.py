from IceSystem import IceSystem
from utility_funcs import *

tmp_dir = './tmp/'
out_dir = '/Volumes/Samsung USB/sill results/'

# thin shell
Lz = 5e3
# mid shell
# Lz = 10e3
# convect shell
# Lz = 5e3
Lx = 6e3
dz = dx = 10
dt = 3.14e7 / 48

Tsurf = 50.
Tbot = 273.15

d = 1e3
h = 500.
R = 2.4 * d
comp = 'MgSO4'
conc = 100

OF = 50.  # years, output frequency

md = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, use_X_symmetry=True)
md.init_T(Tsurf=Tsurf, Tbot=Tbot)
md.init_intrusion(T=Tbot, depth=h, thickness=h, radius=R)
md.init_salinity(concentration=100.)
md.set_boundaryconditions(sides='Reflect')
md.outputs.choose(md, all=True, output_frequency=int(OF * 3.14e7 / dt))
md.outputs.tmp_data_directory = tmp_dir
md.outputs.tmp_data_file_name += '_fresh2_' + str(h) + '_' + str(d)
md.tidalheat = 1
md.freezestop = 1

md.solve_heat(nt=500, dt=dt)
md.outputs.transient_results = md.outputs.get_all_data(md)
print('saving data to ', out_dir)
save_data(md, 'md' + md.outputs.tmp_data_file_name.split('tmp_data_')[1], out_dir)
save_data(md.outputs.transient_results, 'rs' + md.outputs.tmp_data_file_name.split('tmp_data_')[1], out_dir)
