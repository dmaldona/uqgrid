import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from uqgrid.psysdef import Psystem, GenGENROU
from uqgrid.parse import load_psse, add_dyr
from uqgrid.pflow import runpf
from numba import jit, njit

WS = 377.0

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

@njit
def compute_pelec(pelec, vmag, vang, yred):
    """Compute power injection"""
    nbus = pelec.size
    pelec.fill(0.0)
    for i in range(nbus):
        for j in range(nbus):
            if i == j:
                pelec[i] += vmag[i]*vmag[i]*np.real(yred[i, i])
            else:
                pelec[i] += vmag[i]*vmag[j]*(np.imag(yred[i, j])*np.sin(vang[i] - vang[j]) +
                        np.real(yred[i, j])*np.cos(vang[i] - vang[j]))

def classic_resfun(t, x, v, yred, pmec2, H, D):
    """ Classical model described in Anderson and Fouad"""
    
    ngen = v.size
    F = np.zeros(2*ngen)
    pelec = np.zeros(ngen)
    w = x[:ngen]
    delta = x[ngen:]
    compute_pelec(pelec, v, delta, yred)
    for i in range(ngen):
        F[i] = (1.0/(2.0*H[i]))*(pmec[i] - pelec[i] - D[i]*w[i])
        F[ngen + i] = w[i] - 1.0
    return F

def classic_resfun_lin(t, x, J):
    return np.dot(J, x)

def classic_jacobian(t, x, v, yred, pmec, H, D):
    """ Jacobian matrix of the classical model """
    ngen = v.size
    J = np.zeros((2*ngen, 2*ngen))
    vang = x[ngen:]
    for i in range(ngen):
        ## df1/dw
        J[i, i] = -D[i]/(2*H[i])
        ## df2/dw
        J[ngen + i, i] = 1

        for j in range(ngen):
            if i != j:
                J[i, ngen + j] = -v[i]*v[j]*(np.real(yred[i, j])*np.sin(vang[i] - vang[j]) -
                        np.imag(yred[i, j])*np.cos(vang[i] - vang[j]))
                J[i, ngen + j] = (1/(2*H[i]))*J[i, ngen +j]
                J[i, ngen + i] += v[i]*v[j]*(np.real(yred[i, j])*np.sin(vang[i] - vang[j]) -
                        np.imag(yred[i, j])*np.cos(vang[i] - vang[j]))
        J[i, ngen + i] = (1/(2*H[i]))*J[i, ngen + i]

    return J


def reduced_system(psys):
    """ Compute reduced system via Kron reduction. In this system, we only have
        generator buses and we assume generator-behind-reactance (constant voltage)
        hypothesis

        Returns:
        + x0 initial steady-state conditions
        + vmag voltage magnitues
        + pmec mechanical power (electrical + frictional damping)
        + gen_inertia generator inertia
        + gen_damping generator damping

    """
    
    # Run power flow
    psys.createYbusComplex()
    v, s_inj = runpf(psys, verbose=False)
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
        #gen_damping[i] = 0.1*gen.H

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
    pmec = np.zeros(ngen)
    compute_pelec(pmec, vmag, vang, yred)
    for i in range(ngen):
        pmec[i] += gen_damping[i]*w[i]

    x0 = np.hstack((w, delta))

    return x0, vmag, yred, pmec, gen_inertia, gen_damping

if __name__ == "__main__":
    #psys = load_psse(raw_filename="../data/ieee9_v33.raw")
    #psys = load_psse(raw_filename="../data/2bus_33.raw")
    #psys = load_psse(raw_filename="../data/IEEE39_v33.raw")

    #add_dyr(psys, "../data/ieee9bus.dyr")
    #add_dyr(psys, "../data/GENROU.dyr")
    #add_dyr(psys, "../data/IEEE39.dyr")
    
    psys = system_perturb(2)

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

    u0 = np.real(u[0])
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
