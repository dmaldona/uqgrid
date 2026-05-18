import numpy as np
from numba import jit
from uqgrid.utils.tools import csr_add_row, csr_set_row
from uqgrid.core.base_models import Governor
from scipy import optimize


@jit(nopython=True, cache=True)
def ieesgo_resdiff(F, z, theta, idxs, w_idx, T1, T2, T3, T4, T5, T6, K1, K2, K3):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[2]

    PF0 = z[dp]
    PLL = z[dp + 1]
    TP1 = z[dp + 2]
    TP2 = z[dp + 3]
    TP3 = z[dp + 4]
    p_m = z[ap]
    w = z[w_idx]
    pref = theta[pp + 9]

    F[dp] = (1.0 / T1) * (K1 * w - PF0)
    F[dp + 1] = (1.0 / T3) * ((1.0 - (T2 / T3)) * PF0 - PLL)
    SatP = pref - (T2 / T3) * PF0 - PLL
    F[dp + 2] = (1.0 / T4) * (SatP - TP1)
    F[dp + 3] = (1.0 / T5) * (K2 * TP1 - TP2)
    F[dp + 4] = (1.0 / T6) * (K3 * TP2 - TP3)
    F[ap] = TP1 * (1 - K2) + TP2 * (1 - K3) + TP3 - p_m


@jit(nopython=True, cache=True)
def ieesgo_jac(data, indptr, indices, idxs, w_idx, T1, T2, T3, T4, T5, T6, K1, K2, K3):
    dp = idxs[0]
    ap = idxs[1]

    col = np.empty(4, dtype=np.int64)
    val = np.empty(4, dtype=np.float64)

    row = dp
    if w_idx < dp:
        col[0] = w_idx
        val[0] = K1 / T1
        col[1] = dp
        val[1] = -1.0 / T1
    else:
        col[0] = dp
        val[0] = -1.0 / T1
        col[1] = w_idx
        val[1] = K1 / T1
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = dp + 1
    col[0] = dp
    val[0] = (1.0 - T2 / T3) / T3
    col[1] = dp + 1
    val[1] = -1.0 / T3
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = dp + 2
    col[0] = dp
    val[0] = -T2 / (T3 * T4)
    col[1] = dp + 1
    val[1] = -1.0 / T4
    col[2] = dp + 2
    val[2] = -1.0 / T4
    csr_set_row(data, indptr, indices, 3, row, col, val)

    row = dp + 3
    col[0] = dp + 2
    val[0] = K2 / T5
    col[1] = dp + 3
    val[1] = -1.0 / T5
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = dp + 4
    col[0] = dp + 3
    val[0] = K3 / T6
    col[1] = dp + 4
    val[1] = -1.0 / T6
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = ap
    col[0] = dp + 2
    val[0] = -K2 + 1
    col[1] = dp + 3
    val[1] = -K3 + 1
    col[2] = dp + 4
    val[2] = 1.0
    col[3] = ap
    val[3] = -1.0
    csr_set_row(data, indptr, indices, 4, row, col, val)


class GovIEESGO(Governor):
    def __init__(self, id_tag, T1, T2, T3, T4, T5, T6, K1, K2, K3):

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3

        # control variable
        self.pref = None

        parameter_list = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'K1', 'K2', 'K3', 'pref']
        state_list = ['PF0', 'PLL', 'TP1', 'TP2', 'TP3', 'p_m']

        Governor.__init__(self, id_tag, 6, 5, 1, len(parameter_list), state_list)

    def residualFinit(self, x, v, theta, p0, q0, w):

        F = np.zeros(self.initdim)

        PF0 = x[0]
        PLL = x[1]
        TP1 = x[2]
        TP2 = x[3]
        TP3 = x[4]
        pref = x[5]

        T1 = self.T1
        T2 = self.T2
        T3 = self.T3
        T4 = self.T4
        T5 = self.T5
        T6 = self.T6
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3

        F[0] = (1.0/T1)*(K1*w - PF0)
        F[1] = (1/T3)*((1.0 - (T2/T3))*PF0 - PLL)

        SatP = pref - (T2/T3)*PF0 - PLL

        F[2] = (1/T4)*(SatP - TP1)
        F[3] = (1/T5)*(K2*TP1 - TP2)
        F[4] = (1/T6)*(K3*TP2 - TP3)
        F[5] = TP1*(1 - K2) + TP2*(1 - K3) + TP3 - self.p_m0

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        w = x[self.w_idx]

        x0 = np.ones(self.initdim)
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, p, q, w),
            method='krylov',
            options={
                'xtol': 1e-8,
                'disp': False
            })

        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 5] = sol.x[0:5]
        y[self.alg_ptr:self.alg_ptr + 1] = self.p_m0
        self.pref = sol.x[5]

        return None

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.T1
        theta[idx + 1] = self.T2
        theta[idx + 2] = self.T3
        theta[idx + 3] = self.T4
        theta[idx + 4] = self.T5
        theta[idx + 5] = self.T6
        theta[idx + 6] = self.K1
        theta[idx + 7] = self.K2
        theta[idx + 8] = self.K3
        theta[idx + 9] = self.pref

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        ieesgo_resdiff(
            F, z, theta, idxs, self.w_idx,
            self.T1, self.T2, self.T3, self.T4, self.T5, self.T6,
            self.K1, self.K2, self.K3,
        )
        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None
    
    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):

        coord = []

        dp = idxs[0]
        ap = idxs[1]

        # these are INDEXES
        PF0 = dp
        PLL = dp + 1
        TP1 = dp + 2
        TP2 = dp + 3
        TP3 = dp + 4
        p_m = ap

        w = self.w_idx

        # first row
        row = dp
        cols = [w, PF0]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [PF0, PLL]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [PF0, PLL, TP1]
        coord.append([row, cols])

        # fourth row
        row = dp + 3
        cols = [TP1, TP2]
        coord.append([row, cols])

        # fifth row
        row = dp + 4
        cols = [TP2, TP3]
        coord.append([row, cols])

        row = ap
        cols = [TP1, TP2, TP3, p_m]
        coord.append([row, cols])

        return coord

    def preallocate_hessian(self, h_nnz, idxs, psys):
        # Function is linear
        pass

    def residual_hess(self, HESS, z, v, theta, idxs):
        pass

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        ieesgo_jac(
            J.data, J.indptr, J.indices, idxs, self.w_idx,
            self.T1, self.T2, self.T3, self.T4, self.T5, self.T6,
            self.K1, self.K2, self.K3,
        )

if __name__ == "__main__":
    import sympy as sp
    from sympy.printing.pycode import pycode

    T1, T2, T3, T4, T5, T6, K1, K2, K3 = sp.symbols("T1, T2, T3, T4, T5, T6, K1, K2, K3")
    PF0, PLL, TP1, TP2, TP3, p_m = sp.symbols("PF0, PLL, TP1, TP2, TP3, p_m")
    w, pref = sp.symbols("w, p_ref")

    # RESIDUAL
    F0 = (1.0 / T1) * (K1 * w - PF0)
    F1 = (1.0 / T3) * ((1.0 - (T2 / T3)) * PF0 - PLL)

    SatP = pref - (T2 / T3) * PF0 - PLL

    F2 = (1.0 / T4) * (SatP - TP1)
    F3 = (1.0 / T5) * (K2 * TP1 - TP2)
    F4 = (1.0 / T6) * (K3 * TP2 - TP3)

    F5 = TP1 * (1 - K2) + TP2 * (1 - K3) + TP3 - p_m

    FF = [F1, F1, F2, F3, F4, F5]
    state_vars = [PF0, PLL, TP1, TP2, TP3, p_m]
    state_name = ["PF0", "PLL", "TP1", "TP2", "TP3", "p_m"]
    nvars = len(state_vars)

    print("HESSIAN CALCULATION")
    for m in range(len(FF)):
        print("### HESSIAN OF F%d ###\n" % (m))
        for i in range(nvars):
            differential_var = []
            differential_val = []
            for j in range(nvars):
                differential = sp.diff(FF[m], state_vars[i], state_vars[j])
                if (differential.is_zero is None) or (differential.is_zero is False):
                    differential_var.append(state_name[j])
                    differential_val.append(str(differential))

            if len(differential_var) > 0:
                print("row = %s_idx" % (state_name[i]))
                for k in range(len(differential_var)):
                    print("col[%d] = %s_idx" % (k, differential_var[k]))
                    print("val[%d] = %s" % (k, differential_val[k]))
                print(
                    "csr_set_row(H%d.data, H%d.indptr, H%d.indices, %d, row, col, val)\n"
                    % (m, m, m, len(differential_var))
                )
