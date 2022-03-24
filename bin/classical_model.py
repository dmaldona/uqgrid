import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from uqgrid.psysdef import Psystem, GenGENROU
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
from uqgrid.classical import *


def system_perturb(nbus=5):

    psys = Psystem()

    busp = np.zeros(nbus)

    psys.add_bus(1, bus_type = 3)
    for i in range(nbus-1):
        psys.add_bus(2 + i, bus_type = 2)

    for i in range(nbus):
        psys.buses[i].set_vinit(1.00000, (np.pi/180.0)*0.0)

    for i in range(nbus - 1):
        psys.add_branch(i, i + 1, 0.0000, 1.0)

    for i in range(nbus):
        psys.add_gen(i, 100.0, 0.0, mbase=100.0)
        psys.add_load(i, 0, 100.0, 0.0)
    psys.assemble()

    for i in range(nbus):
        psys.add_gen_dynamics(psys.gens[i],
            GenGENROU(0, 1.575, 1.512, 0.0000291, 0.39, 0.1733,
            0.0787, 0.5, 0.0, 6.1, 1.0, 0.05, 0.15))

    return psys


if __name__ == "__main__":
    psys = load_psse(raw_filename="../data/ieee9_v33.raw")
    #psys = load_psse(raw_filename="../data/2bus_33.raw")
    #psys = load_psse(raw_filename="../data/IEEE39_v33.raw")

    add_dyr(psys, "../data/ieee9bus.dyr")
    #add_dyr(psys, "../data/GENROU.dyr")
    #add_dyr(psys, "../data/IEEE39.dyr")
    
    #psys = system_perturb(2)

    x0, vmag, yred, pmec, genH, genD = reduced_system(psys)

    # perturb
    x = np.copy(x0)
    #x[0] -= 0.03

    ## ODE Integrator
    step_size = 1.0/60.0
    tend = 11.5
    sol = solve_ivp(classic_resfun, (0.0, tend), x, dense_output=True,
            args=(vmag, yred, pmec, genH, genD,),
        max_step=step_size)

    ## Linearized system
    J = classic_jacobian(0, x0, vmag, yred, pmec, genH, genD)
    dx = x - x0
    sol2 = solve_ivp(classic_resfun_lin, (0.0, tend), dx, dense_output=True,
            args=(J,), max_step=step_size)

    # Plots
    ngen = pmec.size
    colors = [plt.cm.tab20(i) for i in range(50)]
    fig = plt.figure(figsize=(10, 8))
    for i in range(ngen):
        plt.plot(sol.t, sol.y[ngen + i, :], color=colors[i])
        plt.plot(sol2.t, sol2.y[ngen + i, :] + x0[ngen + i], color=colors[i], linestyle='--')
    plt.xlabel("Time (sec)")
    plt.ylabel("Rotor frequency dev. (p.u)")
    plt.show()


    # eigenvalue stuff
    l, u = np.linalg.eig(J)

    u0 = np.imag(u[1])
    sol = solve_ivp(classic_resfun, (0.0, tend), x0 + u0, dense_output=True,
            args=(vmag, yred, pmec, genH, genD,),
        max_step=step_size)
    sol2 = solve_ivp(classic_resfun_lin, (0.0, tend), u0, dense_output=True,
            args=(J,), max_step=step_size)
    
    fig = plt.figure(figsize=(10, 8))
    for i in range(ngen):
        plt.plot(sol.t, sol.y[ngen + i, :], color=colors[i])
        plt.plot(sol2.t, sol2.y[ngen + i, :] + x0[ngen + i], color=colors[i], linestyle='--')
    plt.title("Eigenresponse")
    plt.xlabel("Time (sec)")
    plt.ylabel("Rotor frequency dev. (p.u)")
    plt.show()
