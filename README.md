# sill_freeze
Thermal evolution of fresh and saline liquid water in Europa's ice shell

## Model
Two-dimensional, two-phase conduction problem described by the conservation of heat that uses an iterative enthalpy method (Huber et al., 2008) to account for latent heat from phase change with a temperature-dependent conductivity of ice

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbar%7B%5Crho%20C_p%7D%20%7B%5Cpartial%20T%20%5Cover%20%5Cpartial%20t%7D%20%3D%20%5Cnabla%5Cleft%28%5Cbar%7Bk%7D%5Cnabla%20T%7D%5Cright%29%20-%20%5Crho_i%20L_f%20%7B%5Cpartial%20%5Cphi%20%5Cover%20%5Cpartial%20t%7D%20%2B%20Q&bc=Transparent&fc=Black&im=png&fs=18&ff=txfonts&edit=0" align="center" border="0" alt="\bar{\rho C_p} {\partial T \over \partial t} = \nabla\left(\bar{k}\nabla T}\right) - \rho_i L_f {\partial \phi \over \partial t} + Q" width="346" height="54" />

We use an explicit finite difference method that conserves flux to numerically solve the above. Model is fully
 described in [Chivers et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JE006692).

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
