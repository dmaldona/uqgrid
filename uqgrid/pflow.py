import numpy as np
import cmath
import numba
from numba import jit

from .psysdef import Psystem
from scipy import optimize

# Notes: this implementation of power flow is awful. But I cannot spend too much time here.
# Matpower has a cool implementation. But they use rectangular coordinates and their own
# N-R implementation. If I want to use a SCIPY solver I need to write the problem
# in a canonical form.

@jit(nopython=True, cache=True)
def resfun(F, x, vmag, vang, Pinj, Qinj, ybus, bus_type, PQ_idx, PQV_idx, graph_mat):

    # The first step is to susbtitute back the vmag and vang unknown variables
    # from x to 'vmag' and 'vang'. It might seem confusing to mix in the same
    # vector unknown variables and parameters. However, this makes writing the 
    # equations cleaner.

    nPQ = np.sum(bus_type == 1)
    nbus = len(bus_type)

    for i in range(nbus):

        if PQ_idx[i] >= 0:
            vmag[i] = x[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = x[nPQ + PQV_idx[i]]

    # Form the residual. Currently, for each bus we iterate all the n buses and
    # compute equations if ymat[i, j] is non-zero.
    # This is of course a waste of time. We need to have data structures that 
    # list the buses connected to one bus (e.g. a dictionary)

    for fr in range(nbus):
        if PQ_idx[fr] >= 0:
            F[PQ_idx[fr]] -= Qinj[fr]
            
            # self contribution
            bij = ybus[fr, fr].imag
            F[PQ_idx[fr]] -= vmag[fr]*vmag[fr]*bij

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                
                gij = ybus[fr, to].real
                bij = ybus[fr, to].imag

                angleij = vang[fr] - vang[to]

                F[PQ_idx[fr]] += vmag[fr]*vmag[to]*(gij*np.sin(angleij)
                    - bij*np.cos(angleij))

        if PQV_idx[fr] >= 0:
            F[nPQ + PQV_idx[fr]] -= Pinj[fr]
            
            # self contribution
            gij = ybus[fr, fr].real
            F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[fr]*gij

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                
                gij = ybus[fr, to].real
                bij = ybus[fr, to].imag

                angleij = vang[fr] - vang[to]
                
                F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[to]*(gij*np.cos(angleij)
                    + bij*np.sin(angleij))

    return F

def resfun_wrapper(x, vmag, vang, Pinj, Qinj, ybus, bus_type, PQ_idx, PQV_idx, graph_mat):
    F = np.zeros(len(x))
    resfun(F, x, vmag, vang, Pinj, Qinj, ybus, bus_type, PQ_idx, PQV_idx, graph_mat)
    return F

# power injection 
def compute_pinj(vmag, vang, Pinj, Qinj, ybus):

    nbus = len(vmag)

    for i in range(nbus):

        Pinj[i] = 0.0
        Qinj[i] = 0.0

        for j in range(nbus):
            gij = ybus[i, j].real
            bij = ybus[i, j].imag

            angleij = vang[i] - vang[j]

            Pinj[i] += vmag[i]*vmag[j]*(gij*np.cos(angleij)
                + bij*np.sin(angleij))

            Qinj[i] += vmag[i]*vmag[j]*(gij*np.sin(angleij)
                - bij*np.cos(angleij))

@jit(nopython=True, cache=True)
def compute_pinj_alt(v, Sinj, ybus_mat, graph_mat, nbus):
    """ Same as above but v and Sinj alternate """

    for fr_bus in range(nbus):

        Sinj[2*fr_bus] = 0.0 # P
        Sinj[2*fr_bus + 1] = 0.0 # Q
        
        vmag_i = v[2*fr_bus]
        vang_i = v[2*fr_bus + 1]
        angleij = 0.0
        
        gij = ybus_mat[fr_bus, 0].real
        bij = ybus_mat[fr_bus, 0].imag
            
        Sinj[2*fr_bus] += vmag_i*vmag_i*(gij*np.cos(angleij)
            + bij*np.sin(angleij))

        Sinj[2*fr_bus + 1] += vmag_i*vmag_i*(gij*np.sin(angleij)
            - bij*np.cos(angleij))
        
        for j in range(graph_mat[fr_bus, 0]):

            to_bus = graph_mat[fr_bus, j + 1]

            gij = ybus_mat[fr_bus, j + 1].real
            bij = ybus_mat[fr_bus, j + 1].imag

            vmag_j = v[2*to_bus]
            vang_j = v[2*to_bus + 1]

            angleij = vang_i - vang_j

            Sinj[2*fr_bus] += vmag_i*vmag_j*(gij*np.cos(angleij)
                + bij*np.sin(angleij))

            Sinj[2*fr_bus + 1] += vmag_i*vmag_j*(gij*np.sin(angleij)
                - bij*np.cos(angleij))
            

def runpf(psys, verbose=False):

    # Slack  (1) variables: p, q. parameters: vmag, vang. 
    # PV gen (2) variables: q, vang. parameters: P, vmag.
    # PQ load (3) variables: vmag, vang. parameters: P, Q.

    # We create vectors
    # vmag: voltage magnitude vector (buses 1 to n)
    # vang: voltage angle vector (buses 1 to n)
    # x0: vector of unknowns

    bus_type = np.zeros(psys.nbuses)
    vmag = np.zeros(psys.nbuses, dtype=float)
    vang = np.zeros(psys.nbuses, dtype=float)
    Pinj = np.zeros(psys.nbuses, dtype=float)
    Qinj = np.zeros(psys.nbuses, dtype=float)

    for i in range(psys.nbuses):

        vmag[i] = psys.buses[i].v0m
        vang[i] = psys.buses[i].v0a
        bus_type[i] = psys.buses[i].type

    for gen in psys.gens:
        Pinj[gen.bus] += gen.psch
        Qinj[gen.bus] += gen.qsch

    for load in psys.loads:
        Pinj[load.bus] -= load.pload
        Qinj[load.bus] += load.qload

    nslack = np.sum(bus_type == 3)
    nPV = np.sum(bus_type == 2)
    nPQ = np.sum(bus_type == 1)

    if verbose: print("Solving power flow with nslack: %d, nPV: %d, nPQ: %d" % (
        nslack, nPV, nPQ))

    x0 = np.zeros(2*nPQ + nPV)
    # indexing for PQ buses

    PQ_bus = np.where(bus_type == 1, 1, 0)
    PQ_idx = (np.where(PQ_bus == 1, np.cumsum(PQ_bus), PQ_bus) - 1)

    # indexing for PQ and PV buses
    PQV_bus = (np.where(bus_type == 1, 1, 0) +
        np.where(bus_type == 2, 1, 0))
    PQV_idx = (np.where(PQV_bus == 1, np.cumsum(PQV_bus), PQV_bus) - 1)

    # these index sets are used to build the x0 vector.
    #         [vmag] ~ 1 ... nPQ
    #   x0 =  [vang] ~ 1 ... nPV

    for i in range(psys.nbuses):

        if PQ_idx[i] >= 0:
            x0[PQ_idx[i]] = psys.buses[i].v0m

        if PQV_idx[i] >= 0:
            x0[nPQ + PQV_idx[i]] = psys.buses[i].v0a

    # pack data structures

    sol, info, ier, msg = optimize.fsolve(resfun_wrapper, x0, args = (vmag, vang, Pinj, Qinj, 
        psys.ybus, bus_type, PQ_idx, PQV_idx, psys.graph_mat), full_output=True, epsfcn=1e-10)

    if ier == 1:
        if verbose: print("Power flow converged.")
    else:
        print(msg)
        raise("Power flow solution did not converge")


    # retrieve voltage magnitudes and angles
    for i in range(psys.nbuses):

        if PQ_idx[i] >= 0:
            vmag[i] = sol[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = sol[nPQ + PQV_idx[i]]

    # retrieve power injections
    compute_pinj(vmag, vang, Pinj, Qinj, psys.ybus)

    # we will return a vetor v and pinj such that
    # v = [vmag1, vang1, vmag2, vang2, ...]
    # Sinj = [pinj1, qinj, pinj2, qinj2, ...]
    v = np.array([vmag, vang]).T.flatten()
    Sinj = np.array([Pinj, Qinj]).T.flatten()

    return v, Sinj
