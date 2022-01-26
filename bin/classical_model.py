# Implement classical multimachine model as described in Fouad's book.

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from uqgrid.psysdef import Psystem
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf

def compute_pelec(pelec, vmag, vang, yred):
    nbus = pelec.size
    pelec.fill(0.0)
    for i in range(nbus):
        for j in range(nbus):
            if i == j:
                pelec[i] += vmag[i]*vmag[i]*np.real(yred[i, i])
            else:
                pelec[i] += vmag[i]*vmag[j]*(np.imag(yred[i, j])*np.sin(vang[i] - vang[j]) +
                        np.real(yred[i, j])*np.cos(vang[i] - vang[j]))

def classic_resfun(t, x, v, yred, pmec, H, D):
    ngen = v.size
    F = np.zeros(2*ngen)
    pelec = np.zeros(ngen)
    w = x[:ngen]
    delta = x[ngen:]
    compute_pelec(pelec, v, delta, yred)
    for i in range(ngen):
        F[i] = (1.0/2.0*H[i])*(pmec[i] - pelec[i] - D[i]*w[i])
        F[ngen + i] = w[i] - 1.0
    return F

# load static file
psys = load_psse(raw_filename="../data/ieee9_v33.raw")
#psys = load_psse(raw_filename="../data/2bus_33.raw")

# add dynamics
add_dyr(psys, "../data/ieee9bus.dyr")
#add_dyr(psys, "../data/GENROU.dyr")

# Run power flow
psys.createYbusComplex()
v, s_inj = runpf(psys, verbose=True)
s_load = psys.get_loadvec()

# Create some data structures.
ngen = psys.ngens
gen_inertia = np.zeros(ngen)
gen_damping = np.zeros(ngen)

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

    # inertia and damping
    gen_inertia[i] = gen.H
    gen_damping[i] = gen.D

gen_damping[:] = 0.01*gen_inertia[:]

# Compute reduced admittance matrix
# Actually i can refactor this and not compute ybus_aug
ynn = ybus_aug[:ngen, :ngen]
ynr = ybus_aug[:ngen, ngen:]
yrn = ybus_aug[ngen:, :ngen]
yrr = ybus_aug[ngen:, ngen:]

# Kron reduction
yred = (ynn - np.dot(ynr, np.dot(np.linalg.inv(yrr), yrn)))

## Determine initial state
w = np.ones(ngen) # ws = 1
delta = vang

## Compute mechanical power vector
## If we have damping, we would do:
##      pmec = pelec + D*wi
pmec = np.zeros(ngen)
compute_pelec(pmec, vmag, vang, yred)
for i in range(ngen):
    pmec[i] += gen_damping[i]*w[i]

x0 = np.hstack((w, delta))
F = classic_resfun(None, x0, vmag, yred, pmec, gen_inertia, gen_damping)
print(F)

x0[0] += 0.1



## ODE Integrator
step_size = 1.0/60.0

sol = solve_ivp(classic_resfun, (0.0, 2.0), x0, dense_output=True,
        args=(vmag, yred, pmec, gen_inertia, gen_damping,),
        max_step=step_size)

for i in range(ngen):
    plt.plot(sol.t, sol.y[i, :])

plt.show()
