# Sample script: runs New England 39 bus case and obtains sensitivities w.r.t load composition.

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.dynamics import integrate_system
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf

# runtime parameters
zfault = 99999999999999999.0 # perturbation fault
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
tend = 10
ton = 0.1
toff = 0.2


print("Time horizon: %g secs" % (tend))

eps = 1e-5

pnom = pmin + 0.5*(pmax - pmin)
psys.set_load_parameters(pnom)
log = {}
tvec, history, history_u, history_v, history_m = integrate_system(psys,
        verbose=False, comp_sens=True, dt=dt, tend=tend, petsc=True, log=log, ton=ton, toff=toff)

psys.set_load_parameters(pnom + eps)
log2 = {}
tvec, history, history_u, history_v, history_m = integrate_system(psys,
        verbose=False, comp_sens=True, dt=dt, tend=tend, petsc=True, log=log2, ton=ton, toff=toff)

psys.set_load_parameters(pnom - eps)
log3 = {}
tvec, history, history_u, history_v, history_m = integrate_system(psys,
        verbose=False, comp_sens=True, dt=dt, tend=tend, petsc=True, log=log3, ton=ton, toff=toff)

fd_grad = (log2["cost"] - log3["cost"])/(2*(pnom[0] + eps - pnom[0]))

print("PETSc Adjoint (mu): ", log["v_mu"]) 
print("Forward differences: ", (log2["cost"] - log["cost"])/eps)

print("Centered differences: ", fd_grad)
print("Relative error (mu, centered) ", (log["v_mu"] - fd_grad)/fd_grad)

plt.plot(history[4,:])
plt.show()
