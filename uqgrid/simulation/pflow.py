import logging
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix
from scipy import optimize
import sys
try:
    from scipy.optimize._nonlin import nonlin_solve # For newer SciPy
except ImportError:
    from scipy.optimize.nonlin import nonlin_solve # Fallback for older SciPy
from uqgrid.core.psydef import Psystem, Bus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional: PETSC4py
try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ImportError:
    petsc4py = None
    logger.warning("PETSc4py not available. Some functionality will not be available.")

class PowerFlowSolution:
    def __init__(self, num_buses, num_gens):
        # Voltages
        self.v_magnitudes = np.zeros(num_buses)
        self.v_angles = np.zeros(num_buses) # in radians

        # Original flat voltage vector (Vm, Va, Vm, Va, ...)
        self.v_vector = np.zeros(2 * num_buses)
        # Original power injection vector (P, Q, P, Q, ...)
        self.s_inj_vector = np.zeros(2 * num_buses)

        # Updated generator setpoints (for slack and PV buses)
        # Store as dictionaries mapping generator original index to value
        self.gen_psch = np.zeros(num_gens)
        self.gen_qsch = np.zeros(num_gens)

    def __str__(self):
        return (f"PowerFlowSolution:\n"
                f"  V_magnitudes: {self.v_magnitudes[:3]}... ({len(self.v_magnitudes)} buses)\n"
                f"  V_angles: {self.v_angles[:3]}... ({len(self.v_angles)} buses)\n"
                f"  Gen Ps Psch entries: {len(self.gen_psch)}\n"
                f"  Gen Qs Qsch entries: {len(self.gen_qsch)}")

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


def _jac_with_structure(x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type, PQ_idx, PQV_idx,
        graph_mat, indptr, indices, data):
    fill_jacobian_data(x, vmag, vang, Pinj, Qinj, ybus_mat,
        bus_type, PQ_idx, PQV_idx, graph_mat, indptr, data)
    return csr_matrix((data, indices, indptr), shape=(x.shape[0], x.shape[0]), copy=False)


if petsc4py:
    class PFlowPetsc(object):
        def __init__(self, vmag, vang, Pinj, Qinj, ybus_mat, bus_type, PQ_idx, PQV_idx, graph_mat,
                     indptr=None, indices=None, data=None):
            self.vmag = vmag
            self.vang = vang
            self.Pinj = Pinj
            self.Qinj = Qinj
            self.ybus_mat = ybus_mat
            self.bus_type = bus_type
            self.PQ_idx = PQ_idx
            self.PQV_idx = PQV_idx
            self.graph_mat = graph_mat
            self.indptr = indptr
            self.indices = indices
            self.data = data

        def evalFunction(self, snes, x, f):
            start, end = x.getOwnershipRange()
            xx = np.array(x[start:end])
            ff = resfun_wrapper(xx, self.vmag, self.vang, self.Pinj, self.Qinj,
                                self.ybus_mat, self.bus_type, self.PQ_idx, self.PQV_idx,
                                self.graph_mat)
            f.setArray(ff)
            f.assemble()

        def evalJacobian(self, snes, x, J, P):
            start, end = x.getOwnershipRange()
            xx = np.array(x[start:end])
            fill_jacobian_data(
                xx, self.vmag, self.vang, self.Pinj, self.Qinj, self.ybus_mat,
                self.bus_type, self.PQ_idx, self.PQV_idx, self.graph_mat,
                self.indptr, self.data
            )
            P.setValuesCSR(self.indptr, self.indices, self.data)
            P.assemble()
            if J != P:
                J.assemble()
            return True


@jit(nopython=True, cache=True)
def compute_row_counts(graph_mat, PQ_idx, PQV_idx):
    nPQ = np.sum(PQ_idx >= 0)
    nPQV = np.sum(PQV_idx >= 0)
    nrows = nPQ + nPQV
    row_counts = np.zeros(nrows, dtype=np.int64)

    for fr in range(graph_mat.shape[0]):
        if PQ_idx[fr] >= 0:
            row = PQ_idx[fr]
            # neighbor contributions
            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                if PQV_idx[to] >= 0:
                    row_counts[row] += 1
                if PQ_idx[to] >= 0:
                    row_counts[row] += 1
            # self vmag + self vang
            row_counts[row] += 2

        if PQV_idx[fr] >= 0:
            row = nPQ + PQV_idx[fr]
            # neighbor contributions
            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                if PQV_idx[to] >= 0:
                    row_counts[row] += 1
                if PQ_idx[to] >= 0:
                    row_counts[row] += 1
            # self vmag only for PQ buses
            if PQ_idx[fr] >= 0:
                row_counts[row] += 1
            # self vang
            row_counts[row] += 1

    return row_counts


@jit(nopython=True, cache=True)
def build_jacobian_structure(graph_mat, PQ_idx, PQV_idx):
    row_counts = compute_row_counts(graph_mat, PQ_idx, PQV_idx)
    nrows = row_counts.shape[0]

    indptr = np.zeros(nrows + 1, dtype=np.int64)
    for i in range(nrows):
        indptr[i + 1] = indptr[i] + row_counts[i]

    nnz = indptr[-1]
    indices = np.empty(nnz, dtype=np.int64)

    # fill indices in the same order as value computation
    row_ptrs = indptr.copy()
    nPQ = np.sum(PQ_idx >= 0)

    for fr in range(graph_mat.shape[0]):
        if PQ_idx[fr] >= 0:
            row = PQ_idx[fr]
            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                if PQV_idx[to] >= 0:
                    indices[row_ptrs[row]] = nPQ + PQV_idx[to]
                    row_ptrs[row] += 1
                if PQ_idx[to] >= 0:
                    indices[row_ptrs[row]] = PQ_idx[to]
                    row_ptrs[row] += 1
            indices[row_ptrs[row]] = PQ_idx[fr]
            row_ptrs[row] += 1
            indices[row_ptrs[row]] = nPQ + PQV_idx[fr]
            row_ptrs[row] += 1

        if PQV_idx[fr] >= 0:
            row = nPQ + PQV_idx[fr]
            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                if PQV_idx[to] >= 0:
                    indices[row_ptrs[row]] = nPQ + PQV_idx[to]
                    row_ptrs[row] += 1
                if PQ_idx[to] >= 0:
                    indices[row_ptrs[row]] = PQ_idx[to]
                    row_ptrs[row] += 1
            if PQ_idx[fr] >= 0:
                indices[row_ptrs[row]] = PQ_idx[fr]
                row_ptrs[row] += 1
            indices[row_ptrs[row]] = nPQ + PQV_idx[fr]
            row_ptrs[row] += 1

    return indptr, indices


@jit(nopython=True, cache=True)
def fill_jacobian_data(x, vmag, vang, Pinj, Qinj, ybus_mat,
        bus_type, PQ_idx, PQV_idx, graph_mat,
        indptr, data):
    nPQ = np.sum(bus_type == 1)
    nbus = len(bus_type)

    for i in range(nbus):
        if PQ_idx[i] >= 0:
            vmag[i] = x[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = x[nPQ + PQV_idx[i]]

    row_ptrs = indptr.copy()

    for fr in range(nbus):
        if PQ_idx[fr] >= 0:
            row = PQ_idx[fr]
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
                    data[row_ptrs[row]] = vmag[fr]*vmag[to]*(-gij*np.cos(angleij)
                        - bij*np.sin(angleij))
                    row_ptrs[row] += 1

                if PQ_idx[to] >= 0:
                    data[row_ptrs[row]] = vmag[fr]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                    row_ptrs[row] += 1

            data[row_ptrs[row]] = accum_self_vmag
            row_ptrs[row] += 1
            data[row_ptrs[row]] = accum_self_vang
            row_ptrs[row] += 1

        if PQV_idx[fr] >= 0:
            row = nPQ + PQV_idx[fr]
            gij = ybus_mat[fr, 0].real
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
                    data[row_ptrs[row]] = vmag[fr]*vmag[to]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                    row_ptrs[row] += 1

                if PQ_idx[to] >= 0:
                    data[row_ptrs[row]] = vmag[fr]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))
                    row_ptrs[row] += 1

            if PQ_idx[fr] >= 0:
                data[row_ptrs[row]] = accum_self_vmag
                row_ptrs[row] += 1
            data[row_ptrs[row]] = accum_self_vang
            row_ptrs[row] += 1

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

def runpf(psys, verbose=False, f_tol=1e-9, maxiter=None, use_krylov=False,
          use_petsc=False, petsc_jacobian=False, petsc_options_prefix="pf_",
          debug=False):

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

    nslack = np.sum(bus_type == Bus.SLACK)
    nPV = np.sum(bus_type == Bus.PV)
    nPQ = np.sum(bus_type == Bus.PQ)

    if verbose: print("Solving power flow with nslack: %d, nPV: %d, nPQ: %d" % (
        nslack, nPV, nPQ))

    x0 = np.zeros(2*nPQ + nPV)

    # indexing for PQ buses
    PQ_bus = np.where(bus_type == Bus.PQ, 1, 0)
    PQ_idx = (np.where(PQ_bus == 1, np.cumsum(PQ_bus), PQ_bus) - 1)

    # indexing for PQ and PV buses
    PQV_bus = (np.where(bus_type == Bus.PQ, 1, 0) +
        np.where(bus_type == Bus.PV, 1, 0))
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
    indptr, indices = build_jacobian_structure(psys.graph_mat, PQ_idx, PQV_idx)
    jac_data = np.empty(indices.shape[0], dtype=np.float64)
    jac = lambda x : _jac_with_structure(
        x, vmag, vang, Pinj, Qinj, psys.ybus_mat, bus_type,
        PQ_idx, PQV_idx, psys.graph_mat, indptr, indices, jac_data
    )

    # https://github.com/scipy/scipy/blob/main/scipy/optimize/_nonlin.py#L116
    # The only solver in SciPy that allowed me to pass a sparse Jacobian
    if verbose or debug:
        initial_residual = fun(x0)
        print(f"[Power Flow] Initial residual norm: {np.linalg.norm(initial_residual):.6e}")
        if debug:
            eq_bus = np.full_like(initial_residual, -1, dtype=int)
            eq_kind = np.full(initial_residual.shape[0], "", dtype=object)
            int2ext = None
            if hasattr(psys, "ext2int"):
                int2ext = {v: k for k, v in psys.ext2int.items()}
            for i in range(psys.nbuses):
                if PQ_idx[i] >= 0:
                    eq_bus[PQ_idx[i]] = i
                    eq_kind[PQ_idx[i]] = "Q"
                if PQV_idx[i] >= 0:
                    eq_bus[nPQ + PQV_idx[i]] = i
                    eq_kind[nPQ + PQV_idx[i]] = "P"

            finite_mask = np.isfinite(initial_residual)
            if not finite_mask.all():
                bad_idx = np.where(~finite_mask)[0]
                print(f"[Power Flow] Non-finite residual entries: {bad_idx.size}")
                for k in bad_idx[:10]:
                    bus = eq_bus[k]
                    ext = int2ext.get(bus, bus) if int2ext else bus
                    print(f"  eq[{k}] bus={bus} ext={ext} type={eq_kind[k]} val={initial_residual[k]}")

            abs_res = np.abs(initial_residual)
            top = np.argsort(-abs_res)[:10]
            print("[Power Flow] Top residual entries:")
            for k in top:
                bus = eq_bus[k]
                ext = int2ext.get(bus, bus) if int2ext else bus
                print(f"  eq[{k}] bus={bus} ext={ext} type={eq_kind[k]} |F|={abs_res[k]:.6e}")
    
    if use_petsc:
        if not petsc4py:
            raise RuntimeError("PETSc requested but petsc4py is not available.")
        n = x0.shape[0]
        solver = PFlowPetsc(vmag, vang, Pinj, Qinj, psys.ybus_mat, bus_type,
                            PQ_idx, PQV_idx, psys.graph_mat, indptr, indices, jac_data)

        indptr_p = indptr.astype(PETSc.IntType, copy=False)
        indices_p = indices.astype(PETSc.IntType, copy=False)
        solver.indptr = indptr_p
        solver.indices = indices_p

        x = PETSc.Vec().createWithArray(x0, comm=PETSc.COMM_WORLD)
        f = x.duplicate()
        snes = PETSc.SNES().create(comm=PETSc.COMM_WORLD)
        snes.setFunction(solver.evalFunction, f)

        J = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr_p, indices_p, jac_data),
                                  comm=PETSc.COMM_WORLD)
        if petsc_jacobian:
            snes.setJacobian(solver.evalJacobian, J, J)
        else:
            snes.setJacobian(None, J, J)

        if petsc_options_prefix:
            snes.setOptionsPrefix(petsc_options_prefix)
        snes.setFromOptions()
        snes.solve(None, x)

        sol = np.array(x.getArray())
        info = {"success": snes.getConvergedReason() > 0,
                "message": snes.getConvergedReason()}
    elif use_krylov:
        sol, info = nonlin_solve(fun, x0, jacobian="krylov", full_output=True,
                                 f_tol=f_tol, maxiter=maxiter)
    else:
        sol, info = nonlin_solve(fun, x0, jacobian=jac, full_output=True,
                                 f_tol=f_tol, maxiter=maxiter)
    
    if verbose:
        final_residual = fun(sol)
        print(f"[Power Flow] Final residual norm: {np.linalg.norm(final_residual):.6e}")

    if isinstance(info, dict):
        success = info.get("success", False)
        message = info.get("message", "Unknown failure")
    else:
        success = False
        message = str(info)

    if success:
        if verbose: print("Power flow converged.")
    else:
        print(message)
        raise Exception("Power flow solution did not converge")
    
    # Create PowerFlowSolution object to store results
    pf_solution = PowerFlowSolution(psys.nbuses, psys.ngens)

    # retrieve voltage magnitudes and angles
    for i in range(psys.nbuses):
        if PQ_idx[i] >= 0:
            vmag[i] = sol[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = sol[nPQ + PQV_idx[i]]

    pf_solution.v_magnitudes = np.copy(vmag)
    pf_solution.v_angles = np.copy(vang)

    # Populate the flat v_vector in pf_solution
    pf_solution.v_vector = np.array([pf_solution.v_magnitudes, pf_solution.v_angles]).T.flatten()

    # Compute power injections (sinj_solved) based on the solved voltages
    sinj_solved = np.zeros(len(pf_solution.v_vector))
    compute_pinj_alt(pf_solution.v_vector, sinj_solved, psys.ybus_mat, psys.graph_mat, psys.nbuses)
    pf_solution.s_inj_vector = sinj_solved

    bus_to_gen = psys.create_bus_to_gen_map()

    # now we compute a vector of generations by substracting the load to sinj
    sgen_dispatch = np.copy(sinj_solved)

    for load in psys.loads:
        sgen_dispatch[2*load.bus] += load.pload
        sgen_dispatch[2*load.bus + 1] -= load.qload
    
    for (bus_idx_internal, bus_obj) in enumerate(psys.buses):
        # For all generators, copy their original Psch unless they are on a slack bus
        # For all generators, copy their original Qsch unless they are on a slack or PV bus
        # This ensures gens on PQ buses retain their input Psch/Qsch.
        for gen_orig_idx in bus_to_gen[bus_idx_internal]:
            # Start by assuming original values from psys.gens
            # The psys.gens objects are read-only in this function's context.
            pf_solution.gen_psch[gen_orig_idx] = psys.gens[gen_orig_idx].psch
            pf_solution.gen_qsch[gen_orig_idx] = psys.gens[gen_orig_idx].qsch

        ngen_on_bus = len(bus_to_gen[bus_idx_internal]) #
        if bus_obj.type == Bus.PV: # PV bus
            if ngen_on_bus == 0: #
                raise ValueError(f"PV bus {bus_obj.id} with no generator")
            q_to_dispatch = sgen_dispatch[2*bus_idx_internal + 1] / ngen_on_bus
            for gen_orig_idx in bus_to_gen[bus_idx_internal]:
                # Psch is fixed from input for PV bus gens (already set above)
                pf_solution.gen_qsch[gen_orig_idx] = q_to_dispatch
        elif bus_obj.type == Bus.SLACK: # Slack bus
            if ngen_on_bus == 0:
                raise ValueError(f"Slack bus {bus_obj.id} with no generator")
            p_to_dispatch = sgen_dispatch[2*bus_idx_internal] / ngen_on_bus
            q_to_dispatch = sgen_dispatch[2*bus_idx_internal + 1] / ngen_on_bus
            for gen_orig_idx in bus_to_gen[bus_idx_internal]:
                pf_solution.gen_psch[gen_orig_idx] = p_to_dispatch
                pf_solution.gen_qsch[gen_orig_idx] = q_to_dispatch

    return pf_solution
