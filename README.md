# freeze_out
Two-dimensional thermal diffusion with salinity for liquid water in Europa's ice shell

## Model
See Chivers et al., 20xx, Journal for complete model description.

# Usage
## Simple system
Set up a pure ice shell that is 5 km thick (`Lz`) and 10 km wide (`Lx`) at 10 m steps (`dz, dx`), with a surface 
temperature of 110 K and bottom temperature of 273.15 K with a linear equilibrium temperature profile
```python
from IceSystem import IceSystem

verticaldomain = 5e3 # m
horizontaldomain = 10e3 # m
spatial_size = 10 # m
surface_temp = 110. # K
bot_temp = 273.15 # K

model = IceSystem(Lx=horizontaldomain, Lz=verticaldomain, dx=spatial_size, dz=spatial_size)
model.init_T(Tsurf=surface_temp, Tbot=bot_temp, profile='linear')
```

Visualize initial temperature profile
```python
import matplotlib.pyplot as plt
plt.pcolormesh(model.X, model.Y, model.T)
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()
```

Solve heat diffusion in system for 1000 years using a 3e5 s time step, tracking temperature and liquid fraction every
 10 years
```python
time_step = 3e5  # s
output_freq = 25 * model.constants.styr  # s
final_time = 1000 * model.constants.styr / dt  # s

model.set_boundaryconditions()  # all Dirichlet boundary conditions
model.outputs.choose(model, output_list=['T','phi'], output_frequency=int(output_freq/dt))

model.solve_heat(nt=final_time, dt=time_step)  # start simulation
model.outputs.get_all_data(model)  # get all ouputs
```

## Saline intrusion
Set up a 10 km thick (`Lz`) pure ice shell with a saline intrusion of 34 ppt NaCl at 2 km depth that is 250 m thick and 
10 kmin  diameter. 
The horizontal domain (`Lx`) must be at least somewhat larger than the size of the intrusion, so we choose a 13 km wide 
domain. Using a surface temperature of 50 K and 10 m spatial steps (`dx, dz`)
```python
from IceSystem import IceSystem

Lz = 5e3  # m, vertical domain size
Lx = 13e3  # m, horizontal domain size
dx = dz = 10  # m, spatial step size
Tsurf = 50  # K, surface temperature

d = 2e3  # m, intrusion depth
h = 250  # m, intrusion thickness
R = 5e3  # m, intrusion radius

model.init_T(Tsurf=Tsurf, Tbot=273.15)  # initialize temperature profile
model.init_intrusion(T=273.15, depth=d, thickness=h, radius=R)
model.init_salinity(concentration=34, composition='NaCl', T_match=True)
```
Notice that the temperature at the bottom of the domain nor the intrusion temperature has not been explicitly chosen 
but assigned
 in the `init_T` 
call. The `T_match` option in `init_salinity` will automatically change the bottom and intrusion liquid temperature to 
match the melting temperature for a given concentration and composition. It will return a message similarly to this,
```
--Pure shell; adjusting temperature profile: Tsurf = 50.0, Tbot = 271.703382899752
init_T(Tsurf = 50.0, Tbot = 271.703382899752
	 Temperature profile initialized to non-linear
--Updating intrusion temperature to reflect initial salinity, Tint = 271.703382899752
```

Visualize initial salinity profile
```python
plt.pcolormesh(model.X, model.Y, model.S)
plt.colorbar('ppt NaCl')
plt.gca().invert_yaxis()
plt.show()
```

Run simulation until liquid freezes using a 3e5 s time step (`dt`) and track everything every 50 years. Since we 
exploited the system's symmetry, we must use the `sides='Reflect'` boundary condition
```python
dt = 3e5  # s
OF = 50 * model.constants.styr  # s, ouput frequency

model.outputs.choose(model, all=True, output_frequency=int(OF/dt))
model.set_boundaryconditions(sides='Reflect')

model.freezestop = 1
model.solve_heat(nt=5000000000000000, dt=time_step)
```

Compare initial and final salinity profiles through the middle of the intrusion
```python
plt.plot(model.S_initial[:,1], model.Z[:,1],label='initial')
plt.plot(model.S[:,1], model.Z[:,1], label='final')
plt.xlabel('ppt NaCl')
plt.ylabel('Depth, m')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
```

# Package dependencies
Packages used here are generally in the standard library or in standard usage [SciPy](https://www.scipy.org/), 
[NumPy](https://www.numpy.org/), and [matplotlib](https://matplotlib.org/) for plotting. 

Outside of these, the [dill](https://pypi.org/project/dill/) package is used in `utility_funcs.py` for saving results.

## Cython 
[Cython](https://cython.org/) can be used to <b>drastically</b> speed up simulations and is <b>highly recommended</b>.
 To use Cython, rename 
`IceSystem.py` and 
`HeatSolver.py` to `IceSystem.pyx` and `HeatSolver.pyx`, then in terminal enter the command
```
$ python cysetup.py build_ext --inplace
```
Then any future call to import the `IceSystem` or `HeatSolver` modules will use the C-wrapped version