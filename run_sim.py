# Example script

from IceSystem import IceSystem
from utility_funcs import *

tmp_dir = './tmp/'  # directory to store temporary data files
out_dir = './results'  # directory to store final results

# thin shell
Lz = 5e3  # m, shell thickness
Lx = 6e3  # m, horizontal domain size
dz = dx = 10  # m
dt = 3.14e7 / 48  # s

Tsurf = 50.  # K
Tbot = 273.15  # K

uniquename = 'salt1'

d = 0.5e3  # m, depth of emplacement
h = 250.  # m, thickness
R = 2.4e3  # m, radius
comp = 'MgSO4'  # composition of salt
conc = 100  # ppt, initial concentration

OF = 50.  # years, output frequency

# initialize system
md = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, use_X_symmetry=True)
md.init_T(Tsurf=Tsurf, Tbot=Tbot)
md.init_intrusion(T=Tbot, depth=h, thickness=h, radius=R)
md.init_salinity(concentration=conc)
md.set_boundaryconditions(sides='NoFlux')

md.outputs.choose(md, all=True, output_frequency=int(OF * md.constants.styr / dt))
md.outputs.tmp_data_directory = tmp_dir
md.outputs.tmp_data_file_name += '{}_{}_{}'.format(uniquename, h, d)  # unique name for temporary data
# file

md.tidalheat = 1  # turn on tidal heating
md.freezestop = 1  # stop when intrusion is frozen

md.solve_heat(nt=5000000000000000, dt=dt)
md.outputs.transient_results = md.outputs.get_all_data(md)  # compile results
print('saving data to ', out_dir)
# save model in output directory
save_data(md, 'md' + md.outputs.tmp_data_file_name.split('tmp_data_')[1], out_dir)
# save results dictionary structure in output directory
save_data(md.outputs.transient_results, 'rs' + md.outputs.tmp_data_file_name.split('tmp_data_')[1], out_dir)
