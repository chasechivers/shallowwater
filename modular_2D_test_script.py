from modular_build import IceSystem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='colorblind', style='ticks', context='notebook')

Lz = 10e3
Lx = 1e3
dx = dz = 50
dt = 3.14e7
nt = 10000

Tsurf = 50.
Tbot = 273.15

thick = 0.5e3
depth = 0.25e3
R = 3e3

ice = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, cpT=False, kT=False)
ice.init_T(Tsurf=Tsurf, Tbot=Tbot)
ice.init_intrusion(T=300, depth=depth, thickness=thick, radius=R)
ice.init_volume_averages()
ice.set_boundayconditions(top='Radiative', sides='Reflect')
ice.solve_heat(nt=nt, dt=dt)

plt.figure()
plt.pcolor(ice.X, ice.Z, ice.T - ice.T_initial)
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

from scipymport root, curve_fit
