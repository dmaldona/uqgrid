import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Governor
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def gast_resdiff(F, z, theta, idxs, w_idx):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[2]

    x1 = z[dp]
    x2 = z[dp + 1]
    x3 = z[dp + 2]
    p_m = z[ap]
    w = z[w_idx]
    R = theta[pp]
    T1 = theta[pp + 1]
    T2 = theta[pp + 2]
    T3 = theta[pp + 3]
    AT = theta[pp + 4]
    KT = theta[pp + 5]
    DT = theta[pp + 8]
    pref = theta[pp + 10]

    demand = (pref - w) / R
    temperature = AT + KT * (AT - x3)
    if demand < temperature:
        u = demand
    else:
        u = temperature

    F[dp] = (u - x1) / T1
    F[dp + 1] = (x1 - x2) / T2
    F[dp + 2] = (x2 - x3) / T3
    F[ap] = x2 - DT * w - p_m


@jit(nopython=True, cache=True)
def _set_sorted_row(data, indptr, indices, row, col, val, nvalues):
    for i in range(1, nvalues):
        current_col = col[i]
        current_val = val[i]
        j = i - 1
        while j >= 0 and col[j] > current_col:
            col[j + 1] = col[j]
            val[j + 1] = val[j]
            j -= 1
        col[j + 1] = current_col
        val[j + 1] = current_val
    csr_set_row(data, indptr, indices, nvalues, row, col, val)


@jit(nopython=True, cache=True)
def gast_jac(data, indptr, indices, z, theta, idxs, w_idx):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[3]

    x3 = z[dp + 2]
    w = z[w_idx]
    R = theta[pp]
    T1 = theta[pp + 1]
    T2 = theta[pp + 2]
    T3 = theta[pp + 3]
    AT = theta[pp + 4]
    KT = theta[pp + 5]
    DT = theta[pp + 8]
    pref = theta[pp + 10]
    demand = (pref - w) / R
    temperature = AT + KT * (AT - x3)

    col = np.empty(3, dtype=np.int64)
    val = np.empty(3, dtype=np.float64)

    col[0] = dp
    val[0] = -1.0 / T1
    col[1] = dp + 2
    val[1] = 0.0
    col[2] = w_idx
    val[2] = 0.0
    if demand < temperature:
        val[2] = -1.0 / (R * T1)
    else:
        val[1] = -KT / T1
    _set_sorted_row(data, indptr, indices, dp, col, val, 3)

    col[0] = dp
    val[0] = 1.0 / T2
    col[1] = dp + 1
    val[1] = -1.0 / T2
    _set_sorted_row(data, indptr, indices, dp + 1, col, val, 2)

    col[0] = dp + 1
    val[0] = 1.0 / T3
    col[1] = dp + 2
    val[1] = -1.0 / T3
    _set_sorted_row(data, indptr, indices, dp + 2, col, val, 2)

    col[0] = dp + 1
    val[0] = 1.0
    col[1] = ap
    val[1] = -1.0
    col[2] = w_idx
    val[2] = -DT
    _set_sorted_row(data, indptr, indices, ap, col, val, 3)


class GovGAST(Governor):
    """GAST gas-turbine governor model."""

    bounded_state_metadata = (
        BoundedStateMetadata(
            state_name="x1",
            state_offset=0,
            lower_parameter_offset=7,
            upper_parameter_offset=6,
            enabled_parameter_offset=9,
            device_type="GAST",
        ),
    )

    def __init__(
        self, id_tag, R, T1, T2, T3, AT, KT, VMAX, VMIN, DT,
        enable_limits=False,
    ):
        if R == 0.0:
            raise ValueError("GAST R must be non-zero.")
        if T1 <= 0.0 or T2 <= 0.0 or T3 <= 0.0:
            raise ValueError("GAST time constants must be positive.")
        if VMIN >= VMAX:
            raise ValueError("GAST VMIN must be less than VMAX.")

        self.R = R
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.AT = AT
        self.KT = KT
        self.VMAX = VMAX
        self.VMIN = VMIN
        self.DT = DT
        self.enable_limits = enable_limits
        self.pref = None

        parameter_list = [
            "R", "T1", "T2", "T3", "AT", "KT", "VMAX", "VMIN", "DT",
            "enable_limits", "pref",
        ]
        state_list = ["x1", "x2", "x3", "p_m"]
        Governor.__init__(self, id_tag, 4, 3, 1, len(parameter_list), state_list)

    def initialize(self, vm, va, p, q, x, y, psys):
        if self.p_m0 is None:
            raise ValueError("GAST requires p_m0 from the generator before initialization.")

        pref = self.R * self.p_m0
        demand = (pref - x[self.w_idx]) / self.R
        temperature = self.AT + self.KT * (self.AT - self.p_m0)
        u = demand if demand < temperature else temperature
        if not np.isclose(u, self.p_m0, rtol=1e-10, atol=1e-12):
            raise ValueError("GAST has no stationary selector branch at p_m0.")

        self.pref = pref
        x[self.dif_ptr:self.dif_ptr + 3] = self.p_m0
        y[self.alg_ptr] = self.p_m0
        self.initialized = True
        return None

    def initialize_theta(self, theta):
        idx = self.par_ptr
        theta[idx] = self.R
        theta[idx + 1] = self.T1
        theta[idx + 2] = self.T2
        theta[idx + 3] = self.T3
        theta[idx + 4] = self.AT
        theta[idx + 5] = self.KT
        theta[idx + 6] = self.VMAX
        theta[idx + 7] = self.VMIN
        theta[idx + 8] = self.DT
        theta[idx + 9] = float(self.enable_limits)
        theta[idx + 10] = self.pref

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        gast_resdiff(
            F, z, theta, idxs, self.w_idx,
        )
        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp = idxs[0]
        ap = idxs[1]
        return [
            [dp, sorted([dp, dp + 2, self.w_idx])],
            [dp + 1, [dp, dp + 1]],
            [dp + 2, [dp + 1, dp + 2]],
            [ap, sorted([dp + 1, ap, self.w_idx])],
        ]

    def preallocate_hessian(self, h_nnz, idxs, psys):
        raise NotImplementedError("GAST selector Hessians are not implemented.")

    def residual_hess(self, HESS, z, v, theta, idxs):
        raise NotImplementedError("GAST selector Hessians are not implemented.")

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        gast_jac(
            J.data, J.indptr, J.indices, z, theta, idxs, self.w_idx,
        )
