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
dt = 1.0/(120.0) # integration step in seconds
zfault = 0.03

# load static file
#psys = load_psse(raw_filename="../data/IEEE39_v33.raw")
#psys = load_psse(raw_filename="../data/ieee9_v33.raw")
psys = load_psse(raw_filename="../data/ACTIVSg200.raw")

# add dynamics
#add_dyr(psys, "../data/IEEE39.dyr")
#add_dyr(psys, "../data/ieee9bus.dyr")
add_dyr(psys, "../data/ACTIVSg200.dyr")

# add fault and create initial data structures
psys.add_busfault(1, zfault, 0.01)
psys.createYbusComplex()
v, Sinj = runpf(psys, verbose=True)

# set up parameters
print("Number of loads (parameters): %d" % (psys.nloads))
pmax = np.ones(psys.nloads)
pmin = np.zeros(psys.nloads)

pnom = pmin + 0.5*(pmax - pmin)
psys.set_load_parameters(pnom)
    
alg_size = psys.num_dof_alg
dif_size = psys.num_dof_dif
pow_size = 2*psys.nbuses  # power balance equations
sys_size = alg_size + dif_size + 2*psys.nbuses

print(alg_size, dif_size, pow_size, sys_size)
tend = 5.0
# run mode. Note: compute_sens= True will return First and Second-Order local sensitivities.
# Second-order sensitivity computation is a bit slow at this time.
results = integrate_system(psys, verbose=True, comp_sens=True, dt=dt, tend=tend, petsc=True, power_injection=False)
print("PETSc Adjoint (mu): ", results["v_mu"]) 

# plot generator speeds
bus_idx = psys.genspeed_idx_set()

for bus in bus_idx:
    label = "generator at bus %d" % (bus)
    plt.plot(results["tvec"], results["history"][bus,:], label=label)
plt.legend()
plt.show()
