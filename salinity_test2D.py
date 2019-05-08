from one_dimension.modular_1D import IceSystem
import seaborn as sns

sns.set(palette='colorblind', style='whitegrid', context='notebook')

Lz = 5e3
Lx = 10e3
dz = 10
dt = 3.14e7 / (24)
Ttop = 50
Tbot = 273.15
depth = 1e3
thick = 1e3

ice = IceSystem(Lz, dz)
ice.init_T(Ttop, Tbot)
ice.init_sill(Tbot, depth, thick)
ice.init_S(concentration=12.3)
ice.set_BC()

ice = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, cpT=False, kT=False)
ice.init_T(Tsurf=Tsurf, Tbot=Tbot)
ice.init_intrusion(T=300, depth=depth, thickness=thick, radius=R)
ice.init_
ice.set_boundayconditions(sides='NoFlux')
ice.solve_heat(nt=100000, dt=dt)
