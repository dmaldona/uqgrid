# Sample script: runs New England 39 bus case and obtains sensitivities w.r.t load composition.

import sys
sys.path.append("..")

import numpy as np
from uqgrid.psysdef import Psystem
from uqgrid.dynamics import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
import matplotlib.pyplot as plt


print("Test PETSc adjoint")
# runtime parameters
zfault = 0.5
dt = 1.0/(120.0) # integration step in seconds

# load static file
psys = load_psse(raw_filename="../data/2bus_33.raw")

# add dynamics
add_dyr(psys, "../data/GENROU.dyr")

# add fault and create initial data structures
psys.add_busfault(1, zfault, 0.01)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

# set up parameters
print("Number of loads (parameters): %d" % (psys.nloads))
pmax = np.ones(psys.nloads)
pmin = np.zeros(psys.nloads)
tend = 5.0
ton = 0.1
toff = 0.2

pnom = pmin + 0.5*(pmax - pmin)
psys.set_load_parameters(pnom)

# Experiment 1: no faut is applied
results = integrate_system(psys, verbose=True, comp_sens=False,
                       petsc=True, dt=dt, tend=tend, ton=ton, arkimex=False)
results2 = integrate_system(psys, verbose=True, comp_sens=False,
                          petsc=True, dt=dt, tend=tend, ton=ton, arkimex=True)

# plot generator speeds
bus_idx = psys.genspeed_idx_set()

fig, axs = plt.subplots(4, 1)
axs[0].plot(results["tvec"], results["history"][bus_idx[0],:], label="Theta")
axs[0].plot(results2["tvec"], results2["history"][bus_idx[0],:], label="Arkimex")
axs[0].legend()
axs[1].plot(results["tvec"], results["history"][bus_idx[0] + 1,:], label="Theta")
axs[1].plot(results2["tvec"], results2["history"][bus_idx[0] + 1,:], label="Arkimex")
axs[1].legend()
vmags = psys.busmag_idx_set()
vangs = psys.busang_idx_set()

axs[2].plot(results["tvec"], results["history"][vmags[0],:], label="Theta")
axs[2].plot(results2["tvec"], results2["history"][vmags[0],:], label="Arkimex")
axs[2].legend()

axs[3].plot(results["tvec"], results["history"][vangs[0],:], label="Theta")
axs[3].plot(results2["tvec"], results2["history"][vangs[0],:], label="Arkimex")
axs[3].legend()

plt.show()
