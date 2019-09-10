import sys, getopt
from utility_funcs import *


def main(argv):
	from IceSystem import IceSystem
	import os
	cwd = os.getcwd()
	dir = '/tmp/'
	outputdirectory = '/nv/hp5/cchivers3/scratch/'
	Lz = 5e3
	Lx = 6e3
	dz = dx = 10
	dt = 3.154e7 / 52
	Tsurf = 110
	Tbot = 273.15
	outputfreq = 50
	rj = 0.25
	shell, salty = False, False
	composition, concentration = 0, 0
	names = ''
	cpT = False
	geometry = 'ellipse'
	Ttol, phitol, Stol = 0.1, 0.01, 1
	X_symmetry = True
	T_match = True
	topBC, botBC = True, True
	sidesBC = 'NoFlux'
	optstr = 'command_line.py  -> options (see IceSystem/HeatSolver documentation (or the above) for ' \
	         'defaults/options)\n' \
	         'Computational Properties, \n' \
	         '  --Lz, --Lx \t : depth and width sizes, m (float)\n' \
	         '  --dz, --dx \t : vertical and horizontal step size, m  (int)\n' \
	         '  --dt       \t : time step size, s (float)\n' \
	         '  --x-symm   \t : exploit symmetry about x-axis, (binary)' \
	         '  --Ttol     \t : temperature tolerance, K (float)\n' \
	         '  --phitol   \t : liquid fraction tolerance, (float)\n' \
	         '  --OF       \t : output frequency for results, /s (float)\n' \
	         '  --names    \t : output data file name\n' \
	         '  --topBC    \t : top boundary condition\n' \
	         '  --botBC    \t : bottom boundary condition\n' \
	         '  --sidesBC  \t : boundary condition on left and right sides\n' \
	         'Initial properties, \n' \
	         '  --Tsurf, --Tbot \t : surface and bottom of shell temperatures, K (float) \n' \
	         '  --profile       \t : equilibrium temperature profile (string)\n' \
	         '  --cpT           \t : temperature-dependence of specific heat, (binary)\n' \
	         'Sill properties,\n' \
	         '  --depth   \t : depth of upper edge of sill, m (float)\n' \
	         '  --thick    \t : thickness of sill, m (float)\n' \
	         '  --radius   \t : radius of sill, m (float)\n' \
	         '  --geometry \t : sill geometry, (string)\n' \
	         'Salinity options, \n' \
	         '  --salt             \t : turn salinity on (1), necessary if using salinity, (binary)\n' \
	         '  --composition      \t : composition of salt, (string)\n' \
	         '  --concentration    \t : initital concentration of salt, ppt (float)\n' \
	         '  --shell            \t : if 1, shell will have a salinity-depth profile according to composition, \n' \
	         '  --T-match          \t : if 1, the bottom will match the salinity of the sill or shell (ocean below)' \
	         'concentration, (binary)\n' \
	         '  --rejection-cutoff \t : liquid fraction cut-off for brine drainage parameterization, (float)\n' \
	         '  --in-situ-melt     \t : NOTE: shell option must be on. determines sill salinity by amount of salt ' \
	         'in location of intrusion, (binary)\n' \
	         '  --Stol             \t : salinity tolerance, ppt (float)'

	try:
		opts, args = getopt.getopt(argv, 'ho:v',
		                           ['Lz=', 'Lx=', 'dz=', 'dx=', 'dt=', 'Tsurf=', 'Tbot=', 'depth=', 'thick=',
		                            'radius=', 'composition=', 'concentration=', 'OF=', 'salt=', 'shell=',
		                            'rejectioncutoff=', 'names=', 'in-situ-melt=', 'cpT=', 'geometry=',
		                            'Ttol=', 'phitol=', 'Stol=', 'x-symm=', 'T-match=', 'topBC=', 'botBC=', 'sidesBC=',
		                            'dL='])
	except getopt.GetoptError:
		print(optstr)
		sys.exit(2)
	print(opts, args)
	for opt, arg in opts:
		if opt == '-h':
			print(optstr)
			sys.exit()
		elif opt in ('--Lz', '--Lz='):
			Lz = float(arg)
		elif opt in ('--Lx', '--Lx='):
			Lx = float(arg)
		elif opt in ('--dz', '--dz='):
			dz = float(arg)
		elif opt in ('--dx', '--dx='):
			dx = float(arg)
		elif opt in ('--dt', '--dt='):
			dt = float(arg)
		elif opt in ('--Tsurf', '--Tsurf='):
			Tsurf = float(arg)
		elif opt in ('--Tbot', '--Tbot='):
			Tbot = float(arg)
		elif opt in ('--profile', '--profile='):
			profile = arg
		elif opt in ('--depth', '--depth='):
			depth = float(arg)
		elif opt in ('--thick', '--thick='):
			thickness = float(arg)
		elif opt in ('--radius', '--radius='):
			radius = float(arg)
		elif opt in ('--salt', '--salt='):
			salty = True
		elif opt in ('--composition', '--composition='):
			composition = arg
		elif opt in ('--concentration', '--concentration='):
			concentration = float(arg)
		elif opt in ('--OF', '--OF='):
			outputfreq = float(arg)
		elif opt in ('--rejectioncutoff', '--rejectioncutoff='):
			rj = float(arg)
		elif opt in ('--shell', '--shell='):
			shell = True
		elif opt in ('--in-situ-melt', '--in-situ-melt='):
			in_situ = True
		elif opt in ('--names', '--names='):
			names = arg
		elif opt in ('--cpT', '--cpT='):
			cpT = bool(arg)
		elif opt in ('--geometry', '--geometry='):
			geometry = arg
		elif opt in ('--Ttol=', '--Ttol'):
			Ttol = arg
		elif opt in ('--phitol=', '--phitol'):
			phitol = arg
		elif opt in ('--Stol=', '--Stol'):
			Stol = arg
		elif opt in ('--x-symm=', '--x-symm'):
			X_symmetry = bool(arg)
		elif opt in ('--T-match=', '--T-match'):
			T_match = bool(arg)
		elif opt in ('--topBC=', '--topBC'):
			topBC = arg
		elif opt in ('--botBC=', '--botBC'):
			botBC = arg
		elif opt in ('--sidesBC=', '--sidesBC'):
			sidesBC = arg
		elif opt in ('--dL=', '--dL'):
			dL = float(arg)
		else:
			print(opt, arg)
			assert False, 'unhandled option'

	model = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, cpT=cpT, use_X_symmetry=X_symmetry)
	model.init_T(Tsurf=Tsurf, Tbot=Tbot)
	model.init_intrusion(Tbot, depth, thickness, radius, geometry=geometry)
	if salty:
		model.init_salinity(concentration=concentration, composition=composition, rejection_cutoff=rj, shell=shell,
		                    T_match=T_match)
	if sidesBC == 'RFlux':
		model.set_boundaryconditions(sides=sidesBC, top=topBC, bottom=botBC, dL=dL)
	else:
		model.set_boundaryconditions(sides=sidesBC, top=topBC, bottom=botBC)

	dtmax = min(dx, dz) ** 2 / (3 * model.k.max() ** 2 / (model.rhoc.min()))
	if dt > 0.10 * dtmax:
		print("--changing dt to meet max value\n  old dt = {}s \n  new dt = {}s".format(dt, 0.1 * dtmax))
		dt = 0.1 * dtmax

	model.outputs.choose(model, all=True, output_frequency=int(outputfreq * 3.154e7 / dt))
	model.outputs.tmp_data_directory = cwd + dir
	model.outputs.tmp_data_file_name = '{}_{}_{}_{}'.format(model.outputs.tmp_data_file_name, names, thickness, depth)

	print(model.outputs.tmp_data_file_name)
	model.tidalheat = 1
	model.freezestop = 1
	model.Ttol, model.phitol, model.Stol = Ttol, phitol, Stol

	model.solve_heat(nt=5000000000000000000000000000, dt=dt)
	print('  solved in {} s'.format(model.run_time))
	model.outputs.transient_results = model.outputs.get_all_data(model)
	print('saving data to ', outputdirectory)
	print('   saving model')
	save_data(model, 'md_{}'.format(model.outputs.tmp_data_file_name.split('tmp_data_')[1]), outputdirectory)
	print('   saving results')
	save_data(model.outputs.transient_results, 'rs_{}'.format(model.outputs.tmp_data_file_name.split('tmp_data_')[1]),
	          outputdirectory)


if __name__ == '__main__':
	main(sys.argv[1:])
