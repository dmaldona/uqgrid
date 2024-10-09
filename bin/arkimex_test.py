# Sample script: runs IEEE 9 bus sytem and obtains sensitivities w.r.t load composition.

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.dynamics import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
from time import time

# runtime parameters
zfault = 0.02 # perturbation fault
dt = 1.0/(120.0) # integration step in seconds

# load static file
#psys = load_psse(raw_filename="../data/ieee9_v33.raw")
psys = load_psse(raw_filename="../data/IEEE39_v33.raw")

# add dynamics
#add_dyr(psys, "../data/ieee9bus.dyr")
add_dyr(psys, "../data/IEEE39.dyr")

# add fault and create initial data structures
psys.add_busfault(7, zfault, 1.0)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

tend = 3.0
ton = 0.3

# Experiment 1: no faut is applied
begin = time()
results = integrate_system(psys, verbose=False, comp_sens=False,
                       petsc=True, dt=dt, tend=tend, ton=ton, arkimex=False)
print("Elapsed time for Theta: ", time()-begin)

begin = time()
results2 = integrate_system(psys, verbose=False, comp_sens=False,
                          petsc=True, dt=dt, tend=tend, ton=ton, arkimex=True)
print("Elapsed time for Arkimex: ", time()-begin)

# plot generator speeds
bus_idx = psys.genspeed_idx_set()

ngen = len(bus_idx)

fig, axs = plt.subplots(ngen, 1)
for i, bus in enumerate(bus_idx):
    axs[i].plot(results["tvec"], results["history"][bus,:], label="Theta")
    axs[i].plot(results2["tvec"], results2["history"][bus,:], label="Arkimex")
    axs[i].legend()
plt.show()
