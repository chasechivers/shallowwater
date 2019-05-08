# freeze_out
two-dimensional thermal diffusion with salinity for liquid water in Europa's ice shell

## Model
See Chivers et al., 20xx, Journal 


## Usage
Set up a pure ice ice shell that is 5 km deep and 10 km wide at 10 m steps, with a surface temperature of 110 K and 
bottom 
temperature of 273.15 K with a linear equilibrium temperature profile
```
from modular_build import IceSystem

verticaldomain = 5e3 # m
horizontaldomain = 10e3 # m
spatial_size = 10 # m
surface_temp = 110. # K
bot_temp = 273.15 # K

model = IceSystem(Lx=horizontaldomain, Lz=verticaldomain, dx=spatial_size, dz=spatial_size)
model.init_T(Tsurf=surface_temp, Tbot=bot_temp, profile='linear')
```

Visualize initial temperature profile
```
import matplotlib.pyplot as plt
plt.pcolormesh(model.X, model.Y, model.T)
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()
```