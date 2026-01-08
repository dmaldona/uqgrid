import numpy as np
from uqgrid.core.base_models import Governor
from uqgrid.utils.tools import csr_set_row


def _ordered_vals(cols, mapping):
    return np.array([mapping[c] for c in cols], dtype=np.float64)


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

        parameter_list = ["R", "T1", "VMAX", "VMIN", "T2", "T3", "DT"]
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

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        dp = idxs[0]
        ap = idxs[1]

        x1 = z[dp]
        x2 = z[dp + 1]
        p_m = z[ap]
        w = z[self.w_idx]

        t2_over_t3 = self.T2 / self.T3
        dx1 = (-x1 + (1.0 - t2_over_t3) * x2) / self.T3
        dx2 = ((self.pref - w) / self.R - x2) / self.T1

        if (x2 >= self.VMAX and dx2 > 0.0) or (x2 <= self.VMIN and dx2 < 0.0):
            dx2 = 0.0

        F[dp] = dx1
        F[dp + 1] = dx2
        F[ap] = x1 + t2_over_t3 * x2 - self.DT * w - p_m
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
        dp = idxs[0]
        ap = idxs[1]

        x1 = z[dp]
        x2 = z[dp + 1]
        w = z[self.w_idx]

        t2_over_t3 = self.T2 / self.T3
        dx2 = ((self.pref - w) / self.R - x2) / self.T1
        limited = (x2 >= self.VMAX and dx2 > 0.0) or (x2 <= self.VMIN and dx2 < 0.0)

        # row for x1
        row = dp
        cols = np.array(self._jac_cols_x1, dtype=np.int32)
        col_map = {
            dp: -1.0 / self.T3,
            dp + 1: (1.0 - t2_over_t3) / self.T3,
        }
        vals = _ordered_vals(cols, col_map)
        csr_set_row(J.data, J.indptr, J.indices, len(cols), row, cols, vals)

        # row for x2
        row = dp + 1
        cols = np.array(self._jac_cols_x2, dtype=np.int32)
        if limited:
            col_map = {dp + 1: 0.0, self.w_idx: 0.0}
        else:
            col_map = {dp + 1: -1.0 / self.T1, self.w_idx: -1.0 / (self.R * self.T1)}
        vals = _ordered_vals(cols, col_map)
        csr_set_row(J.data, J.indptr, J.indices, len(cols), row, cols, vals)

        # algebraic p_m row
        row = ap
        cols = np.array(self._jac_cols_pm, dtype=np.int32)
        col_map = {dp: 1.0, dp + 1: t2_over_t3, self.w_idx: -self.DT, ap: -1.0}
        vals = _ordered_vals(cols, col_map)
        csr_set_row(J.data, J.indptr, J.indices, len(cols), row, cols, vals)
