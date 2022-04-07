from argparse import ArgumentParser
from utility_funcs import *
import sys
import os
from ShallowWater import ShallowWater as Model


def main(argv):
	cwd = os.getcwd()

	parser = ArgumentParser()

	############################
	# NUMERICAL SETTINGS
	############################
	parser.add_argument("--D", "-D",
	                    type=float,
	                    default=10e3,
	                    help="brittle ice thickness, m (10 km default)")
	parser.add_argument("--w", "-w",
	                    type=float,
	                    default=None,
	                    help="horizontal domain width, m (2.5 * Radius default)")
	parser.add_argument("--dz", "-dz",
	                    type=float,
	                    default=10.,
	                    help="vertical step size, m (10 m default)")
	parser.add_argument("--dx", "-dx",
	                    type=float,
	                    default=10.,
	                    help="horizontal step size, m (10 m default)")
	parser.add_argument("--nz", "-nz",
	                    type=int,
	                    default=None,
	                    help="number of grid points in z-direction, (None default, defaults to 10 m grid step sizes)")
	parser.add_argument("--nx", "-nx",
	                    type=int,
	                    default=None,
	                    help="number of grid points in x-direction, (None default, defaults to 10 m grid step sizes)")
	parser.add_argument("--coordinates", "-coordinates", "--coords", "-coords",
	                    type=str,
	                    default="zx",
	                    help="coordinate system used for simulation. Options are either cartesian (default) or "
	                         "cylindrical",
	                    choices=["cartesian", "xz", "cyl", "cylindrical", "zr", "rz"])
	parser.add_argument("--dt", "-dt",
	                    default="CFL",
	                    help="time step size, s (CFL condition default, dt ~ min(dx,dz)^2 * rho * Cp / k_max / 12")
	parser.add_argument("--adapt_dt", "-adapt_dt",
	                    default=True,
	                    help="set whether time step will adapt during simulation")
	parser.add_argument("--nt", "-nt",
	                    type=int,
	                    default=None,  # int(200e3 * 3.154e7 / (3.154e7 / 52)),
	                    help="number of time steps (200,000 years default). mutually exclusive with --simtime")
	parser.add_argument("--simtime", "-st",
	                    type=float,
	                    default=200e3 * 3.154e7,
	                    help="total time in seconds to run a simulation (200,000 years default). "
	                         "mutually exclusive with --nt")
	parser.add_argument("--freeze_stop", "-fs",
	                    type=bool,
	                    default=True,
	                    help="set whether to stop when all water is frozen (default on)")
	parser.add_argument("--x-symm", "-x-symm",
	                    type=bool,
	                    default=True,
	                    help="exploit single sill symmetry about x-axis (bool) (default True)")
	parser.add_argument("--Ttol", "-Ttol",
	                    type=float,
	                    default=0.1,
	                    help="temperature tolerance, K (0.1 K default, absolute error)")
	parser.add_argument("--phitol", "-phitol",
	                    type=float,
	                    default=0.01,
	                    help="liquid fraction tolerance (0.01 default, absolute error)")
	parser.add_argument("--topBC", "-topBC",
	                    default=True,
	                    help="top boundary condition (Dirichlet default)",
	                    choices=[True, "Radiative"])
	# extra keyword arguments for top boundary conditions
	parser.add_argument("--use_orbits", "-use_orbits",
	                    type=bool,
	                    default=False,
	                    help="set whether radiative boundary condition ")

	parser.add_argument("--botBC", "-botBC",
	                    default=True,
	                    help="bottom boundary condition (Dirichlet default)",
	                    choices=[True, "ConstFlux", "IceFlux", "OceanFlux"])
	# extra keyword arguments for bottom boundary conditions
	parser.add_argument("--qbot", "-qb",
	                    type=float,
	                    default=0.05, # W/m^2
	                    help="choose bottom boundary flux when botBC='ConstFlux' [W/m^2]")
	parser.add_argument("--Tbotbc", "-Tbbc",
	                     type=float,
	                     default=273.15,
	                     help="choose 'ghost cell row' temperature for botBC='IceFlux' or "
	                          "botBC='OceanFlux' [K]")

	parser.add_argument("--sidesBC", "-sidesBC",
	                    default="NoFlux",
	                    help="vertical walls boundary conditions ('NoFlux' default)",
	                    choices=[True, "NoFlux", "RFlux", "LNoFlux"])
	# extra keyword arguments for side boundary conditions
	parser.add_argument("--dL", "-dL",
	                    type=float,
	                    default=1e3,
	                    help="distance right vertical wall from simulation for 'RFlux' boundary condition")


	############################
	# OUTPUT SETTINGS
	############################
	parser.add_argument("--names", "-names",
	                    type=str,
	                    help="output data file name (string)",
	                    default=None)
	parser.add_argument("--outlist", "-outlist",
	                    type=list,
	                    default=["T", "phi", "Q"],
	                    help="arrays to output, list (default: ['T','phi','Q', 'time']")
	parser.add_argument("--OF", "-OF",
	                    type=float,
	                    default=1000,
	                    help="output frequency for results, /yr (1000 /yr default)")
	parser.add_argument("--tmpdir", "-tmpdir",
	                    type=str,
	                    default="/tmp/",
	                    help="directory for outputing temporary files")
	parser.add_argument("--outdir", "-outdir",
	                    type=str,
	                    default="./results/",
	                    help="directory for outputing results files")
	parser.add_argument("--v", "-v",
	                    type=bool,
	                    default=1,
	                    help="print options to screen. default on (-v 1)",
	                    choices=[1, 0])

	#############################
	# INITIAL PHYSICAL PROPERTIES
	#############################
	parser.add_argument("--Tsurf", "-Tsurf",
	                    type=float,
	                    default=110.0,
	                    help="ice shell surface temperature, K (110 K default)")
	parser.add_argument("--Tbot", "-Tbot",
	                    type=float,
	                    default=273.15,
	                    help="ice shell basal temperature, K (273.15 K default)")
	parser.add_argument("--kconst", "-kconst",
	                    type=float,
	                    default=567.0,
	                    help="constant for temperature-dependent conductivity, W/m (567 W/m default)")
	parser.add_argument("--tidalheat", "-tidalheat",
	                    type=bool,
	                    default=True,
	                    help="tidal heating (default on)",
	                    choices=[0, 1])

	############################
	# SILL/WATER INTRUION PROPS
	############################
	parser.add_argument("--depth", "-d",
	                    type=float,
	                    default=1e3,
	                    help="depth of upper surface of sill, m (1 km default)")
	parser.add_argument("--thickness", "-H",
	                    type=float,
	                    default=0.5e3,
	                    help="thickness of sill, m (500 m default)")
	parser.add_argument("--radius", "-r",
	                    type=float,
	                    default=None,
	                    help="radius of sill, m (default = 2.4 * depth)")
	parser.add_argument("--geometry", "-geometry",
	                    type=str,
	                    default="ellipse",
	                    help="assumed sill geometry (default 'ellipse')",
	                    choices=["ellipse", "box", "chaos"])
	parser.add_argument("--inner_radius", "-ir",
	                    type=float,
	                    default=0.2,
	                    help="inside radius for the 'chaos' geometry, set default to (1 - inside_radius)*radius")

	############################
	# SALINITY/SALT OPTIONS
	############################
	parser.add_argument("--salt", "-S",
	                    type=bool,
	                    default=0,
	                    help="turn salinity on (1), necessary if using salinity (default off 0)")
	parser.add_argument("--composition", "-comp",
	                    type=str,
	                    default="MgSO4",
	                    help="composition of salt (default 'MgSO4')",
	                    choices=["MgSO4", "NaCl"])
	parser.add_argument("--concentration", "-concentration",
	                    type=float,
	                    default=12.3,
	                    help="concentration of salt, ppt (12.3 ppt default)")
	parser.add_argument("--shell", "-shell",
	                    type=bool,
	                    default=True,
	                    help="shell salinity-depth profile, 1 is on (default on: True)")
	parser.add_argument("--Tmatch", "-Tmatch",
	                    type=bool,
	                    default=True,
	                    help="bottom will match the salinity of the sill or shell (ocean below) concentration, "
	                         "1 is on (default on unless '--Tbot' is called)")
	parser.add_argument("--rejection-cutoff", "-rj",
	                    type=float,
	                    default=0.25,
	                    help="liquid fraction cut-off for brine drainage parameterization (0.25 default)")
	parser.add_argument("--in-situ-melt", "-in-situ-melt",
	                    type=bool,
	                    default=False,
	                    help="determines sill salinity by amount of salt " \
	                         "in location of intrusion. (default off False) NOTE: shell option must be on ('--shell 1')")
	parser.add_argument("--Stol", "-Stol",
	                    type=float,
	                    default=1,
	                    help="salinity tolerance, ppt (1 ppt default)")
	parser.add_argument("--load", "-load",
	                    type=str,
	                    default=None,
	                    help="Load a file to continue running simulation. Requires path and filename, e.g. --load "
	                         "/Users/user/directory/md_runIDXXXX_.pkl")

	args = parser.parse_args()
	if args.load != None:
		print(args.load)
		print(">>> Loading file for simulation")
		print(f"\t ...loading {args.load}")
		model = load_data(args.load)
		print(f"\t File loaded")
		model._print_all_opts(int(args.simtime / model.dt) if args.nt is None else args.nt)
		model.FREEZE_STOP = args.freeze_stop
		model.solve_heat(dt=model.dt,
		                 nt=args.nt if args.nt is not None else None,
		                 final_time=args.simtime if args.nt is None else None,
		                 save_progress=10 if (1 - model.phi.sum() / model.phi_initial.sum()) > 0.75 else 25)

	else:
		if args.radius == None: args.radius = 2.4 * args.depth

		if args.nx != None and args.nz != None:
			args.dx, args.dz = None, None
		model = Model(w=args.radius * 2.5 if args.w == None else args.w, D=args.D,
		              dx=args.dx, dz=args.dz,
		              nx=args.nx, nz=args.nz,
		              use_X_symmetry=args.x_symm,
		              kT=True if args.kconst > 0 else False,
		              verbose=args.v,
		              coordinates=args.coordinates)

		model.init_T(Tsurf=args.Tsurf, Tbot=args.Tbot)
		if args.inner_radius == 0.2:
			args.inner_radius = (1 - args.inner_radius) * args.radius
		model.init_intrusion(depth=args.depth, thickness=args.thickness, radius=args.radius, geometry=args.geometry,
		                     inner_radius=args.inner_radius)

		if args.salt is True:
			print(args.Tmatch)
			model.init_salinity(composition=args.composition, concentration=args.concentration,
			                    rejection_cutoff=args.rejection_cutoff, shell=args.shell,
			                    T_match=False if args.Tbot != 273.15 else args.Tmatch,
			                    in_situ=args.in_situ_melt)
			args.outlist.append("S")
		else:
			pass

		# set the boundary conditions
		model.set_boundaryconditions(sides=args.sidesBC, dL=args.dL,
		                             top=args.topBC, use_orbits=args.use_orbits,
		                             bottom=args.botBC, qbot=args.qbot, Tbot=args.Tbotbc)

		print(args.dt, type(args.dt))

		if args.dt == "min":
			args.dt = 3.154e7 / 52
		elif args.dt == "CFL":
			args.dt = model.dt
		else:
			args.dt = float(args.dt)
		print(args.dt, type(args.dt))
		# create unique name for simulation outputs
		if args.names is None:
			args.names = "" + (args.composition if args.salt is True else "fr") + str(int(model.depth / 1e3)) + "_" \
			             + ("eq" if args.Tsurf >= 90 else "") + ("pole" if args.Tsurf < 90 else "") + "_" \
			             + str(int(args.thickness)) + "_" \
			             + str(int(args.depth))
		# set outputs
		model.set_outputs(output_frequency=int(args.OF * 3.154e7 / args.dt),
		                  outlist=args.outlist,
		                  tmp_dir=cwd + args.tmpdir,
		                  tmp_file_name=args.names)

		# set options
		model.TIDAL_HEAT = args.tidalheat
		model.FREEZE_STOP = args.freeze_stop
		model.ADAPT_DT = args.adapt_dt
		model.TTOL = args.Ttol
		model.PHITOL = args.phitol
		model.STOL = args.Stol
		model.constants.ac = args.kconst

		# start simulation
		model.solve_heat(dt=args.dt, nt=int(args.simtime / args.dt) if args.nt == None else args.nt)

		print('  Solved in {} s'.format(model.run_time))
		results = model.outputs.get_all_data(model)
		print('Saving data to ', args.outdir)
		print('   Saving model')
		save_data(model, 'md_{}'.format(model.outputs.tmp_data_file_name.split('tmp_data_')[1]), args.outdir)
		print('   Saving results to separate file))
		save_data(results,
		          'rs_{}'.format(model.outputs.tmp_data_file_name.split('tmp_data_')[1]),
		          args.outdir)


if __name__ == '__main__':
	main(sys.argv[1:])
