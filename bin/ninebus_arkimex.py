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

tend = 5.0
tend = 0.3
ton = 0.3

# Experiment 1: no faut is applied
results = integrate_system(psys, verbose=True, comp_sens=False,
                       petsc=True, dt=dt, tend=tend, ton=ton, arkimex=False)
results2 = integrate_system(psys, verbose=True, comp_sens=False,
                          petsc=True, dt=dt, tend=tend, ton=ton, arkimex=True)

# plot generator speeds
bus_idx = psys.genspeed_idx_set()

fig, axs = plt.subplots(3, 1)
for i, bus in enumerate(bus_idx):
    axs[i].plot(results["tvec"], results["history"][bus,:], label="Theta")
    axs[i].plot(results2["tvec"], results2["history"][bus,:], label="Arkimex")
    axs[i].legend()
plt.show()
