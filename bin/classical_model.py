# Implement classical multimachine model as described in Fouad's book.

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf

np.__config__.show()

# load static file
psys = load_psse(raw_filename="../data/ieee9_v33.raw")

# add dynamics
add_dyr(psys, "../data/ieee9bus.dyr")

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
    yload = load.pload/vmag**2 - 1j*(load.qload/vmag**2)
    
    ymat[load.bus, load.bus] += yload

# Create augmented voltage vector
vmag = np.zeros(ngen)
vang = np.zeros(ngen)

# Create a new, extended admittance matrix:
ybus_aug = np.zeros((ngen + ymat.shape[0], ngen + ymat.shape[0]), dtype=np.complex)

# insert existing admittance matrix
ybus_aug[ngen:, ngen:] = np.copy(ymat)

# NOTE: This wont work when multiple generators in same bus
for i, gen in enumerate(psys.gendyn):
    vm = v[2*gen.bus]
    va = v[2*gen.bus + 1]

    pi = s_inj[2*gen.bus] - s_load[2*gen.bus]
    qi = s_inj[2*gen.bus + 1] - s_load[2*gen.bus + 1]
    
    xdp = gen.x_dp

    egen = (vm + qi*xdp/vm) + 1j*(pi/vm)

    vmag[i] = np.abs(egen)
    vang[i] = np.angle(egen) + va

    print("Generator bus voltage. Vmag: %g. Vang: %g." % (vm, va))
    print("Generator internal voltage. Emag: %g. Eang: %g." % (vmag[i], vang[i]))

    # add new branches in augmented impedance matrix
    yint = 1/(1j*xdp)
    ybus_aug[i, i] += yint
    ybus_aug[i, ngen + gen.bus] -= yint
    ybus_aug[ngen + gen.bus, i] -= yint


# Compute reduced admittance matrix
# Actually i can refactor this and not compute ybus_aug
ynn = ybus_aug[:ngen, :ngen]
ynr = ybus_aug[:ngen, ngen:]
yrn = ybus_aug[ngen:, :ngen]
yrr = ybus_aug[ngen:, ngen:]

print(ynn)
print(yrr)

print(yrr.shape)
print(yrn.shape)


yrr_inv = np.linalg.inv(yrr)
yrryrn = np.dot(yrr_inv, yrn)

#yred = (ynn - np.dot(ynr, np.dot(np.linalg.inv(yrr), yrn)))
#print(yred)

