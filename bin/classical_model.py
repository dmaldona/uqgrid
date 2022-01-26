# Implement classical multimachine model as described in Fouad's book.

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf

def compute_pmec(pmec, vmag, vang, yred):
    print(yred)
    for i in range(len(pmec)):
        for j in range(len(pmec)):
            if i == j:
                pmec[i] += vmag[i]*vmag[i]*np.real(yred[i, i])
            else:
                pmec[i] += vmag[i]*vmag[j]*(np.imag(yred[i, j])*np.sin(vang[i] - vang[j]) +
                        np.real(yred[i, j])*np.cos(vang[i] - vang[j]))

# load static file
#psys = load_psse(raw_filename="../data/ieee9_v33.raw")
psys = load_psse(raw_filename="../data/2bus_33.raw")

# add dynamics
#add_dyr(psys, "../data/ieee9bus.dyr")
add_dyr(psys, "../data/GENROU.dyr")

# Run power flow
psys.createYbusComplex()
v, s_inj = runpf(psys, verbose=True)
s_load = psys.get_loadvec()

# Create some data structures.
ngen = psys.ngens

# Retrieve admittance matrix
ymat = np.copy(psys.ybus)


# We assume loads are constant admittance.
for load in psys.loads:
    vmag = v[2*load.bus]
    yload = -load.pload/vmag**2 + 1j*(load.qload/vmag**2)

    ymat[load.bus, load.bus] -= yload

# Create augmented voltage vector
vmag = np.zeros(ngen)
vang = np.zeros(ngen)

# Create a new, extended admittance matrix:
ybus_aug = np.zeros((ngen + ymat.shape[0], ngen + ymat.shape[0]), dtype=complex)

# insert existing admittance matrix
ybus_aug[ngen:, ngen:] = np.copy(ymat)

# NOTE: This wont work when multiple generators in same bus
for i, gen in enumerate(psys.gendyn):
    vm = v[2*gen.bus]
    va = v[2*gen.bus + 1]

    pi = s_inj[2*gen.bus] - s_load[2*gen.bus]
    qi = s_inj[2*gen.bus + 1] - s_load[2*gen.bus + 1]
    
    xdp = gen.x_dp
    egen = (vm + qi*xdp/vm) + 1j*(pi*xdp/vm)

    vmag[i] = np.abs(egen)
    vang[i] = np.angle(egen) + va

    # add new branches in augmented impedance matrix
    yint = 1/(1j*xdp)
    ybus_aug[i, i] += yint
    ybus_aug[i, ngen + gen.bus] -= yint
    ybus_aug[ngen + gen.bus, i] -= yint
    ybus_aug[ngen + gen.bus, ngen + gen.bus] += yint



# Compute reduced admittance matrix
# Actually i can refactor this and not compute ybus_aug
ynn = ybus_aug[:ngen, :ngen]
ynr = ybus_aug[:ngen, ngen:]
yrn = ybus_aug[ngen:, :ngen]
yrr = ybus_aug[ngen:, ngen:]

yred = (ynn - np.dot(ynr, np.dot(np.linalg.inv(yrr), yrn)))

## Compute mechanical power vector
pmec = np.zeros(ngen)
compute_pmec(pmec, vmag, vang, yred)

caca = np.imag(yred)

print(pmec)
