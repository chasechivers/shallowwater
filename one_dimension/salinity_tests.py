from one_dimension.modular_1D import IceSystem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='colorblind', style='ticks', context='notebook')

Lz = 10e3
dz = 10
dt = 3.14e7 / (24 * 2)
Ttop = 110
Tbot = 273.15
depth = 1e3
thick = 1e3
nt = 500000


ice = IceSystem(Lz, dz)
ice.init_T(Ttop, Tbot)
ice.init_sill(Tbot, depth, thick)
ice.init_S(concentration=33)
ice.set_BC()

T_i, phi_i, k_i, rhoc_i, S_i = ice.T.copy(), ice.phi.copy(), ice.k.copy(), ice.rhoc.copy(), ice.S.copy()

ice.solve_heat(nt=nt, dt=dt)

z = ice.z / 1e3

fig, ax = plt.subplots(2, 2, sharey='all')
ax[0, 0].plot(T_i, z, linewidth=3, color='k')
ax[0, 0].plot(ice.T, z)
ax[0, 0].set_ylabel('km')
ax[0, 0].set_title('K')
ax[0, 0].set_ylim(0, Lz / 1e3)
ax[0, 0].invert_yaxis()

ax[0, 1].plot(T_i / (np.ones(len(z)) * 273.15), z, linewidth=3, color='k')
ax[0, 1].plot(ice.T / ice.Tm, z)
ax[0, 1].set_ylabel('km')
ax[0, 1].set_title('T/Tm')

ax[1, 0].plot(phi_i, z, linewidth=3, color='k')
ax[1, 0].plot(ice.phi, z, '.-')
ax[1, 0].set_xlabel('liquid fraction')
ax[1, 0].set_ylabel('km')

ax[1, 1].plot(S_i, z, linewidth=3, color='k')
ax[1, 1].plot(ice.S, z, '.-')
ax[1, 1].set_xlabel('ppt')

plt.show()

z, time = np.meshgrid(ice.z, ice.outputs.timer)
S = np.zeros((len(ice.outputs.timer), len(ice.z)))

for i in range(len(ice.outputs.timer)):
	S[i, :] = ice.outputs.S_profile[i]

plt.figure()
plt.pcolormesh(time / 3.14e7, z, np.log10(S), cmap='jet')
plt.colorbar()
plt.plot(time[:, 0] / 3.14e7, ice.outputs.ff_low, 'k--')
plt.plot(time[:, 0] / 3.14e7, ice.outputs.ff_high, 'k--')

plt.gca().invert_yaxis()
plt.ylim(depth + thick, depth)
plt.show()
