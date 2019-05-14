from modular_build import IceSystem
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, curve_fit
from scipy.special import erf

Lz = 1.
space_steps = np.array([0.005, 0.001])
Lxs = 3 * space_steps
dts = np.array([1, 0.5, 0.1, 0.05, 0.01])

Curves = []
C = []
Err = []

Tsurf = 50.
Tbot = 273.15
dt = dts[-3]
for i, dz in enumerate(space_steps):
	print('dz =', dz)
	ice = IceSystem(Lx=3 * dz, Lz=Lz, dx=dz, dz=dz, cpT=False, kT=False)
	ice.init_T(Tsurf=Tsurf, Tbot=Tbot)
	ice.T[1:, :] = 273.15
	ice.phi[:, :] = 1.
	ice.constants.rho_w = ice.constants.rho_i
	ice.constants.cp_w = ice.constants.cp_i
	ice.constants.kw = ice.constants.ki
	ice.init_volume_averages()
	ice.set_boundaryconditions(sides='Reflect')
	ice.stefan_compare(dt)

	Curves.append(np.asarray(ice.freeze_front))
	C.append(curve_fit(lambda x, C: C * np.sqrt(x), np.asarray(ice._time_), Curves[i])[0])

	time = np.asarray(ice._time_)
	norm_time = time / (max(time) - min(time))
	Err.append(np.asarray(Curves[i] - ice.stefan_zm_func(time)))
	plt.figure(1)
	plt.clf()
	plt.plot(norm_time, Curves[i] - ice.stefan_zm_func(time), linewidth=2)

	plt.figure(2)
	plt.plot(norm_time, Curves[i], linewidth=2)
	plt.pause(0.000001)

time = np.asarray(ice._time_)
norm_time = time / (max(time) - min(time))

plt.figure()
plt.plot(norm_time, ice.stefan_zm_func(time), color='k', linewidth=4, label='analytical')
plt.plot(norm_time, Curves[0], linewidth=3, label='dz ={}'.format(space_steps[0]))
plt.plot(norm_time, Curves[1], linewidth=3, label='dz ={}'.format(space_steps[1]))
plt.plot(norm_time, Curves[2], linewidth=3, label='dz ={}'.format(space_steps[2]))
plt.plot(norm_time, Curves[3], linewidth=3, label='dz ={}'.format(space_steps[3]))
plt.xlabel('normalized time')
plt.ylabel('freezing front position')
plt.legend()
plt.show()

plt.figure()
plt.plot(norm_time, Curves[0] - ice.stefan_zm_func(time), linewidth=2, label='dt ={}'.format(space_steps[0]))
plt.plot(norm_time, Curves[1] - ice.stefan_zm_func(time), linewidth=2, label='dt ={}'.format(space_steps[1]))
# plt.plot(norm_time, Curves[2]-ice.stefan_zm_func(time), linewidth=2,label='dt ={}'.format(space_steps[2]))
# plt.plot(norm_time, Curves[3]-ice.stefan_zm_func(time),linewidth=2, label='dt ={}'.format(space_steps[3]))
plt.xlabel('normalized time')
plt.ylabel('error in freezing front position')
plt.legend()
plt.show()
