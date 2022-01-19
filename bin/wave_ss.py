# Wave small signal

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from uqgrid.psysdef import Psystem, GenGENROU
from uqgrid.dynamics import integrate_system, initialize_system, preallocate_jacobian, residual_jacobian
from uqgrid.pflow import runpf


def system_perturb(bus_inertia=None):

	psys = Psystem()

	busp = np.zeros(5)
	if bus_inertia is not None:
		busp[bus_inertia] += 1.0
	print(busp)

	psys.add_bus(1, bus_type = 3)
	psys.add_bus(2, bus_type = 2)
	psys.add_bus(3, bus_type = 2)
	psys.add_bus(4, bus_type = 2)
	psys.add_bus(5, bus_type = 2)

	psys.buses[0].set_vinit(1.04000, (np.pi/180.0)*0.0)
	psys.buses[1].set_vinit(1.02500, (np.pi/180.0)*9.6926)
	psys.buses[2].set_vinit(1.02500, (np.pi/180.0)*4.8812)
	psys.buses[3].set_vinit(0.99574, (np.pi/180.0)*-2.3060)
	psys.buses[4].set_vinit(0.95068, (np.pi/180.0)*-4.1382)

	psys.add_branch(0, 1, 0.0000, 0.05)
	psys.add_branch(1, 2, 0.0000, 0.05)
	psys.add_branch(2, 3, 0.0000, 0.05)
	psys.add_branch(3, 4, 0.0000, 0.05)

	psys.add_gen(0, 1.0, 0.5, mbase=100.0)
	psys.add_gen(1, 1.0, 0.5, mbase=100.0)
	psys.add_gen(2, 1.0, 0.5, mbase=100.0)
	psys.add_gen(3, 1.0, 0.5, mbase=100.0)
	psys.add_gen(4, 1.0, 0.5, mbase=100.0)

	psys.add_load(0, 0, 1.0, -0.5)
	psys.add_load(1, 0, 1.0, -0.5)
	psys.add_load(2, 0, 1.0, -0.5)
	psys.add_load(3, 0, 1.0, -0.5)
	psys.add_load(4, 0, 1.0, -0.5)
	psys.assemble()

	psys.add_gen_dynamics(psys.gens[0],
	    GenGENROU(0, 1.575, 1.512, 0.291, 0.39, 0.1733,
	    0.0787, 3.38 + busp[0], 0.0, 6.1, 1.0, 0.05, 0.15))

	psys.add_gen_dynamics(psys.gens[1],
	    GenGENROU(0, 1.575, 1.512, 0.291, 0.39, 0.1733,
	    0.0787, 3.38 + busp[1], 0.0, 6.1, 1.0, 0.05, 0.15))

	psys.add_gen_dynamics(psys.gens[2],
	    GenGENROU(0, 1.575, 1.512, 0.291, 0.39, 0.1733,
	    0.0787, 3.38 + busp[2], 0.0, 6.1, 1.0, 0.05, 0.15))

	psys.add_gen_dynamics(psys.gens[3],
	    GenGENROU(0, 1.575, 1.512, 0.291, 0.39, 0.1733,
	    0.0787, 3.38 + busp[3], 0.0, 6.1, 1.0, 0.05, 0.15))

	psys.add_gen_dynamics(psys.gens[4],
	    GenGENROU(0, 1.575, 1.512, 0.291, 0.39, 0.1733,
	    0.0787, 3.38 + busp[4], 0.0, 6.1, 1.0, 0.05, 0.15))

	return psys

def return_max_eigv(psys):

	psys.createYbusComplex()
	volt, Pinj = runpf(psys, verbose=False)
	z0, theta = initialize_system(volt, Pinj, psys)
	system_size = z0.shape[0]
	J = preallocate_jacobian(psys)
	F = np.zeros(system_size)
	residual_jacobian(J, z0, theta, psys)

	w, v = np.linalg.eig(J.todense())
	
	return (np.max(np.real(w)))


for i in range(5):
	psys = system_perturb(i)
	mval = return_max_eigv(psys)

	print("Location %d. Max real eigv: %g." % (i, mval))
