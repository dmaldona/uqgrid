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
zfault = 0.5 # perturbation fault
dt = 1.0/(120.0) # integration step in seconds

# load static file
#psys = load_psse(raw_filename="../data/ieee9_v33.raw")
psys = load_psse(raw_filename="../data/IEEE39_v33.raw")
#psys = load_psse(raw_filename="../data/2bus_33.raw")

# add dynamics
#add_dyr(psys, "../data/ieee9bus.dyr")
add_dyr(psys, "../data/IEEE39.dyr")
#add_dyr(psys, "../data/GENROU.dyr")

# add fault and create initial data structures
psys.add_busfault(1, zfault, 1.0)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

tend = 0.5
ton = 0.05

# OPTIONS

THETA = True
ARKIMEX = True
PLOT_RESULTS = (THETA and ARKIMEX) and True


if THETA:
    begin = time()
    results = integrate_system(psys, verbose=False, comp_sens=False,
                           petsc=True, dt=dt, tend=tend, ton=ton, arkimex=False)
    print("Elapsed time for Theta: ", time()-begin)

if ARKIMEX:
    begin = time()
    results2 = integrate_system(psys, verbose=False, comp_sens=False,
                              petsc=True, dt=dt, tend=tend, ton=ton, arkimex=True)
    print("Elapsed time for Arkimex: ", time()-begin)

if PLOT_RESULTS:
    # plot generator speeds
    bus_idx = psys.genspeed_idx_set()

    ngen = len(bus_idx)

    if ngen == 1:
        fig, axs = plt.subplots(1, 1)
        axs.plot(results["tvec"], results["history"][bus_idx[0],:], label="Theta")
        axs.plot(results2["tvec"], results2["history"][bus_idx[0],:], label="Arkimex")
        axs.legend()
    else:
        fig, axs = plt.subplots(ngen, 1)
        for i, bus in enumerate(bus_idx):
            axs[i].plot(results["tvec"], results["history"][bus,:], label="Theta")
            axs[i].plot(results2["tvec"], results2["history"][bus,:], label="Arkimex")
            axs[i].legend()
    plt.show()
