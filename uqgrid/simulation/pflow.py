import numpy as np
from numba import jit
from scipy.sparse import csr_matrix
from scipy import optimize
from scipy.optimize.nonlin import nonlin_solve
from uqgrid.core.psydef import Psystem


@jit(nopython=True, cache=True)
def resfun(F, x, vmag, vang, Pinj, Qinj, ybus_mat,
           bus_type, PQ_idx, PQV_idx, graph_mat):

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

    for fr in range(nbus):
        if PQ_idx[fr] >= 0:
            F[PQ_idx[fr]] -= Qinj[fr]

            # self contribution
            bij = ybus_mat[fr, 0].imag
            F[PQ_idx[fr]] -= vmag[fr]*vmag[fr]*bij

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]

                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag

                angleij = vang[fr] - vang[to]

                F[PQ_idx[fr]] += vmag[fr]*vmag[to]*(gij*np.sin(angleij)
                                 - bij*np.cos(angleij))

        if PQV_idx[fr] >= 0:
            F[nPQ + PQV_idx[fr]] -= Pinj[fr]

            # self contribution
            gij = ybus_mat[fr, 0].real
            F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[fr]*gij

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]

                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag

                angleij = vang[fr] - vang[to]

                F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[to]*(gij*np.cos(angleij)
                                        + bij*np.sin(angleij))
    return F


def resfun_wrapper(x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type,
                   PQ_idx, PQV_idx, graph_mat):
    F = np.zeros(len(x))
    resfun(F, x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type,
           PQ_idx, PQV_idx, graph_mat)
    return F


@jit(nopython=True, cache=True)
def compute_jac_nnz(graph_mat, PQ_idx, PQV_idx):
    nnz = 0
    for i in range(graph_mat.shape[0]):
        if PQ_idx[i] >= 0:
            nnz += 4
        elif PQV_idx[i] >= 0:
            nnz += 1
        else:
            continue

        for j in range(graph_mat[i, 0]):
            to = graph_mat[i, j + 1]
            if PQ_idx[to] >= 0 and PQ_idx[i] >= 0:
                nnz += 4
            elif PQ_idx[to] >= 0 and PQV_idx[i] >= 0:
                nnz += 2
            elif PQV_idx[to] >= 0 and PQ_idx[i] >= 0:
                nnz += 2
            elif PQV_idx[to] >= 0 and PQV_idx[i] >= 0:
                nnz += 1
    return nnz

@jit(nopython=True, cache=True)
def fill_jacobian(x, vmag, vang, Pinj, Qinj, ybus_mat,
        bus_type, PQ_idx, PQV_idx, graph_mat,
        row, col, val):
    ptr = 0
    nPQ = np.sum(bus_type == 1)
    nbus = len(bus_type)

    for i in range(nbus):
        if PQ_idx[i] >= 0:
            vmag[i] = x[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = x[nPQ + PQV_idx[i]]

    for (fr, elem) in enumerate(graph_mat):
        if PQ_idx[fr] >= 0:
            # self contribution
            vmag_fr_idx = PQ_idx[fr]
            vang_fr_idx = nPQ + PQV_idx[fr]

            bij = ybus_mat[fr, 0].imag

            accum_self_vmag = -2*vmag[fr]*bij
            accum_self_vang = 0.0

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag

                angleij = vang[fr] - vang[to]

                accum_self_vmag += vmag[to]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                accum_self_vang += vmag[fr]*vmag[to]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))

                if PQV_idx[to] >= 0:
                    vang_to_idx = nPQ + PQV_idx[to]
                    row[ptr] = PQ_idx[fr]
                    col[ptr] = vang_to_idx
                    val[ptr] = vmag[fr]*vmag[to]*(-gij*np.cos(angleij)
                        - bij*np.sin(angleij))
                    ptr += 1

                if PQ_idx[to] >= 0:
                    vmag_to_idx = PQ_idx[to]
                    row[ptr] = PQ_idx[fr]
                    col[ptr] = vmag_to_idx
                    val[ptr] = vmag[fr]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                    ptr += 1

            row[ptr] = PQ_idx[fr]
            col[ptr] = vmag_fr_idx
            val[ptr] = accum_self_vmag
            ptr += 1

            row[ptr] = PQ_idx[fr]
            col[ptr] = vang_fr_idx
            val[ptr] = accum_self_vang
            ptr += 1

        if PQV_idx[fr] >= 0:
            # self contribution
            gij = ybus_mat[fr, 0].real
            #F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[fr]*gij

            bij = ybus_mat[fr, 0].imag

            accum_self_vmag = 2*vmag[fr]*gij
            accum_self_vang = 0.0

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag
                angleij = vang[fr] - vang[to]
                accum_self_vmag += vmag[to]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))
                accum_self_vang += vmag[fr]*vmag[to]*(-gij*np.sin(angleij)
                        + bij*np.cos(angleij))

                if PQV_idx[to] >= 0:
                    vang_to_idx = nPQ + PQV_idx[to]
                    row[ptr] = nPQ + PQV_idx[fr]
                    col[ptr] = vang_to_idx
                    val[ptr] = vmag[fr]*vmag[to]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                    ptr += 1

                if PQ_idx[to] >= 0:
                    vmag_to_idx = PQ_idx[to]
                    row[ptr] = nPQ + PQV_idx[fr]
                    col[ptr] = vmag_to_idx
                    val[ptr] = vmag[fr]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))
                    ptr += 1

            if PQ_idx[fr] >= 0:
                vmag_fr_idx = PQ_idx[fr]
                row[ptr] = nPQ + PQV_idx[fr]
                col[ptr] = vmag_fr_idx
                val[ptr] = accum_self_vmag
                ptr += 1

            vang_fr_idx = nPQ + PQV_idx[fr]
            row[ptr] = nPQ + PQV_idx[fr]
            col[ptr] = vang_fr_idx
            val[ptr] = accum_self_vang
            ptr += 1

def jac_wrapper(x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type, PQ_idx, PQV_idx, graph_mat):
    nnz = compute_jac_nnz(graph_mat, PQ_idx, PQV_idx)
    row = np.zeros(nnz)
    col = np.zeros(nnz)
    val = np.zeros(nnz)
    fill_jacobian(x, vmag, vang, Pinj, Qinj, ybus_mat,
        bus_type, PQ_idx, PQV_idx, graph_mat,
        row, col, val)
    J =  csr_matrix((val, (row, col)), shape=(x.shape[0], x.shape[0]))
    return J

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
    fun = lambda x : resfun_wrapper(x, vmag, vang, Pinj, Qinj, psys.ybus_mat, bus_type,
            PQ_idx, PQV_idx, psys.graph_mat)
    jac = lambda x : jac_wrapper(x, vmag, vang, Pinj, Qinj, psys.ybus_mat, bus_type,
            PQ_idx, PQV_idx, psys.graph_mat)

    # https://github.com/scipy/scipy/blob/main/scipy/optimize/_nonlin.py#L116
    # The only solver in SciPy that allowed me to pass a sparse Jacobian
    sol, info = nonlin_solve(fun, x0, jacobian=jac, full_output=True, f_tol=1e-9)

    if info["success"]:
        if verbose: print("Power flow converged.")
    else:
        print(info["message"])
        raise Exception("Power flow solution did not converge")

    # retrieve voltage magnitudes and angles
    for i in range(psys.nbuses):
        if PQ_idx[i] >= 0:
            vmag[i] = sol[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = sol[nPQ + PQV_idx[i]]

    # we will return a vector v and pinj such that
    # v = [vmag1, vang1, vmag2, vang2, ...]
    # Sinj = [pinj1, qinj, pinj2, qinj2, ...]
    v = np.array([vmag, vang]).T.flatten()
    Sinj = np.zeros(len(v))
    compute_pinj_alt(v, Sinj, psys.ybus_mat, psys.graph_mat, psys.nbuses)

    return v, Sinj
