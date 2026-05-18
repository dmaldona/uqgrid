import numpy as np
from numba import jit
from uqgrid.core.base_models import Governor
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def tgov1_resdiff(F, z, theta, idxs, w_idx, R, T1, T2, T3, DT):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[2]

    x1 = z[dp]
    x2 = z[dp + 1]
    p_m = z[ap]
    w = z[w_idx]
    pref = theta[pp + 7]

    t2_over_t3 = T2 / T3
    F[dp] = (-x1 + (1.0 - t2_over_t3) * x2) / T3
    F[dp + 1] = ((pref - w) / R - x2) / T1
    F[ap] = x1 + t2_over_t3 * x2 - DT * w - p_m


@jit(nopython=True, cache=True)
def tgov1_jac(data, indptr, indices, idxs, w_idx, R, T1, T2, T3, DT):
    dp = idxs[0]
    ap = idxs[1]
    t2_over_t3 = T2 / T3

    col = np.empty(4, dtype=np.int64)
    val = np.empty(4, dtype=np.float64)

    row = dp
    col[0] = dp
    val[0] = -1.0 / T3
    col[1] = dp + 1
    val[1] = (1.0 - t2_over_t3) / T3
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = dp + 1
    if w_idx < dp + 1:
        col[0] = w_idx
        val[0] = -1.0 / (R * T1)
        col[1] = dp + 1
        val[1] = -1.0 / T1
    else:
        col[0] = dp + 1
        val[0] = -1.0 / T1
        col[1] = w_idx
        val[1] = -1.0 / (R * T1)
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = ap
    if w_idx < dp:
        col[0] = w_idx
        val[0] = -DT
        col[1] = dp
        val[1] = 1.0
        col[2] = dp + 1
        val[2] = t2_over_t3
        col[3] = ap
        val[3] = -1.0
    elif w_idx < ap:
        col[0] = dp
        val[0] = 1.0
        col[1] = dp + 1
        val[1] = t2_over_t3
        col[2] = w_idx
        val[2] = -DT
        col[3] = ap
        val[3] = -1.0
    else:
        col[0] = dp
        val[0] = 1.0
        col[1] = dp + 1
        val[1] = t2_over_t3
        col[2] = ap
        val[2] = -1.0
        col[3] = w_idx
        val[3] = -DT
    csr_set_row(data, indptr, indices, 4, row, col, val)


class GovTGOV1(Governor):
    """
    TGOV1 turbine governor model (PSS/E).

    Parameters follow ordering:
    R, T1, VMAX, VMIN, T2, T3, DT.
    """

    def __init__(self, id_tag, R, T1, VMAX, VMIN, T2, T3, DT):
        if R == 0.0:
            raise ValueError("TGOV1 R must be non-zero.")
        if T1 == 0.0 or T3 == 0.0:
            raise ValueError("TGOV1 T1 and T3 must be non-zero.")

        self.R = R
        self.T1 = T1
        self.VMAX = VMAX
        self.VMIN = VMIN
        self.T2 = T2
        self.T3 = T3
        self.DT = DT

        self.pref = None

        parameter_list = ["R", "T1", "VMAX", "VMIN", "T2", "T3", "DT", "pref"]
        state_list = ["x1", "x2", "p_m"]

        Governor.__init__(self, id_tag, 3, 2, 1, len(parameter_list), state_list)

    def initialize(self, vm, va, p, q, x, y, psys):
        if self.p_m0 is None:
            raise ValueError("TGOV1 requires p_m0 from the generator before initialization.")

        x2 = self.p_m0
        x1 = (1.0 - self.T2 / self.T3) * x2

        self.pref = self.R * self.p_m0
        x[self.dif_ptr] = x1
        x[self.dif_ptr + 1] = x2
        y[self.alg_ptr] = self.p_m0
        self.initialized = True
        return None

    def initialize_theta(self, theta):
        idx = self.par_ptr
        theta[idx] = self.R
        theta[idx + 1] = self.T1
        theta[idx + 2] = self.VMAX
        theta[idx + 3] = self.VMIN
        theta[idx + 4] = self.T2
        theta[idx + 5] = self.T3
        theta[idx + 6] = self.DT
        theta[idx + 7] = self.pref

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        tgov1_resdiff(
            F, z, theta, idxs, self.w_idx,
            self.R, self.T1, self.T2, self.T3, self.DT,
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

        x1 = dp
        x2 = dp + 1
        p_m = ap
        w = self.w_idx

        self._jac_cols_x1 = sorted([x1, x2])
        self._jac_cols_x2 = sorted([x2, w])
        self._jac_cols_pm = sorted([x1, x2, w, p_m])

        coord.append([dp, self._jac_cols_x1])
        coord.append([dp + 1, self._jac_cols_x2])
        coord.append([ap, self._jac_cols_pm])
        return coord

    def preallocate_hessian(self, h_nnz, idxs, psys):
        pass

    def residual_hess(self, HESS, z, v, theta, idxs):
        pass

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        tgov1_jac(
            J.data, J.indptr, J.indices, idxs, self.w_idx,
            self.R, self.T1, self.T2, self.T3, self.DT,
        )
