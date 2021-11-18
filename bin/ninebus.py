# Sample script: runs IEEE 9 bus sytem and obtains sensitivities w.r.t load composition.

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.dynamics import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf

# runtime parameters
zfault = 0.2 # perturbation fault
dt = 1.0/(120.0) # integration step in seconds

# load static file
psys = load_psse(raw_filename="../data/ieee9_v33.raw")

# add dynamics
add_dyr(psys, "../data/ieee9bus.dyr")

# add fault and create initial data structures
psys.add_busfault(7, zfault, 1.0)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

# set up parameters
pnom = np.array([0.5, 0.5, 0.5])
psys.set_load_parameters(pnom)

# run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
# Second-order sensitivity computation is a bit slow at this time.
res = integrate_system(psys, verbose=True, comp_sens=True, dt=dt, tend=10.0)

# plot generator speeds
bus_idx = psys.genspeed_idx_set()

for bus in bus_idx:
    label = "generator at bus %d" % (bus)
    plt.plot(res["tvec"], res["history"][bus,:], label=label)
plt.legend()
plt.show()

# select variable $\omega$ index
bus = bus_idx[0]

# plot sensitivities \frac{d\omega}{d \alpha_i} for \alpha = 1, 2, 3.
for i in range(len(pnom)):
    label = "Sensitivity of $\omega$ w.r.t parameter %d." % (i + 1)
    plt.plot(res["tvec"], res["history_u"][bus,i, :], label=label)
plt.legend()
plt.show()

