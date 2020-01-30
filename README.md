# sill_freeze
Thermal evolution of fresh and saline liquid water in Europa's ice shell

## Model
Two-dimensional, two-phase conduction problem described by the conservation of heat that uses an iterative enthalpy method (Huber et al., 2008) to account for latent heat from phase change with a temperature-dependent conductivity of ice

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbar%7B%5Crho%20C_p%7D%20T_t%20%3D%20%5Cnabla%20%28%20%5Cbar%7Bk%7D%20%5Cnabla%20T%20%29%20-%20%5Crho_i%20L_f%20%5Cphi_t%20%2B%20Q&bc=Transparent&fc=Black&im=png&fs=18&ff=txfonts&edit=0" align="center" border="0" alt="\bar{\rho C_p} T_t = \nabla ( \bar{k} \nabla T ) - \rho_i L_f \phi_t + Q" width="312" height="33" />

We use an explicit finite difference method that conserves flux to numerically solve the above. Model is fully
 described in ().

# Package dependencies
Packages used here are generally in the standard library or in standard usage: [SciPy](https://www.scipy.org/), 
[NumPy](https://www.numpy.org/), and [matplotlib](https://matplotlib.org/) for plotting. 

Outside of these, the [dill](https://pypi.org/project/dill/) package is used in `utility_funcs.py` for saving results.

These can be installed by using pip in your shell through the command
```
$ pip install -r requirements.txt
```

## Cython 
[Cython](https://cython.org/) can be used to <b>drastically</b> speed up simulations and is <b>highly recommended</b>.
 To use Cython, rename 
`IceSystem.py` and 
`HeatSolver.py` to `IceSystem.pyx` and `HeatSolver.pyx`, then in terminal enter the command
```
$ python cysetup.py build_ext --inplace
```
Then any future call to import the `IceSystem` or `HeatSolver` modules will use the C-wrapped version. 
