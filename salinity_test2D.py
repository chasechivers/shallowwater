from modular_build import IceSystem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='colorblind', style='ticks', context='notebook')

Lz = 5e3
Lx = 6e3
dz = dx = 10
dt = 3.14e7 / (24 * 4)
Tsurf = 50
Tbot = 273.15
depth = 1e3
thick = 0.5e3
R = 2.4 * depth

ice = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz)
ice.init_T(Tsurf=Tsurf, Tbot=Tbot)
ice.init_intrusion(T=273.15, depth=depth, thickness=thick, radius=R)
ice.init_salinity(concentration=12.3)
ice.set_boundaryconditions(sides='Reflect')
ice.solve_heat(nt=50000, dt=dt)

plt.figure()
plt.pcolor(ice.X, ice.Z, np.log(ice.S), cmap='cividis')
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

t = np.linspace(0, len(ice.total_salt), len(ice.total_salt))
plt.figure()
plt.plot(t, ice.total_salt[0] - ice.total_salt)
plt.show()

plt.figure()
plt.plot(ice.S[:, 350], ice.Z[:, 350], 'k')
plt.plot(ice.S[:, 125], ice.Z[:, 0], 'r')
plt.xlabel('ppt')
plt.ylabel('depth, m')
plt.gca().invert_yaxis()
plt.show()
