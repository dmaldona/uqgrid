import numpy as np
from numba import jit

from uqgrid.core.base_models import Stabilizer
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def _ieeest_signals(z, idxs, w_idx, par):
    dp = idxs[0]

    f1_x = z[dp]
    f1_y = z[dp + 1]
    f1_dx = (z[w_idx] - f1_y - par[0] * f1_x) / par[1]

    f2_x1 = z[dp + 2]
    f2_x2 = z[dp + 3]
    if par[3] == 0.0:
        f2_dx1 = 0.0
        f2_dx2 = 0.0
        f2_y = f1_y + par[4] * f1_x + par[5] * f1_dx
    else:
        f2_dx1 = (f1_y - f2_x2 - par[2] * f2_x1) / par[3]
        f2_dx2 = f2_x1
        f2_y = f2_x2 + par[4] * f2_x1 + par[5] * f2_dx1

    ll1_x = z[dp + 4]
    if par[7] == 0.0:
        ll1_dx = 0.0
        ll1_y = f2_y
    else:
        ll1_dx = (f2_y - ll1_x) / par[7]
        ll1_y = ll1_x + par[6] * ll1_dx

    ll2_x = z[dp + 5]
    if par[9] == 0.0:
        ll2_dx = 0.0
        ll2_y = ll1_y
    else:
        ll2_dx = (ll1_y - ll2_x) / par[9]
        ll2_y = ll2_x + par[8] * ll2_dx

    wash_x = z[dp + 6]
    wash_input = par[12] * ll2_y
    wash_dx = (wash_input - wash_x) / par[11]
    wash_y = par[10] * wash_dx if par[10] > 0.0 else wash_x

    return (
        f1_dx, f1_x, f2_dx1, f2_dx2, ll1_dx, ll2_dx, wash_dx, wash_y
    )


@jit(nopython=True, cache=True)
def ieeest_resdiff(F, z, v, idxs, w_idx, bus, power_injection, par):
    dp = idxs[0]
    ap = idxs[1]
    signals = _ieeest_signals(z, idxs, w_idx, par)
    for offset in range(7):
        F[dp + offset] = signals[offset]

    if power_injection:
        vm = v[2 * bus]
    else:
        vr = v[2 * bus]
        vi = v[2 * bus + 1]
        vm = np.sqrt(vr * vr + vi * vi)

    output = signals[7]
    if output >= par[13]:
        output = par[13]
    elif output <= par[14]:
        output = par[14]
    if vm >= par[15] or vm <= par[16]:
        output = 0.0
    F[ap] = output - z[ap]


@jit(nopython=True, cache=True)
def _set_ieeest_row(data, indptr, indices, row, dp, w_idx, derivatives, output_col=-1):
    count = 8 + int(output_col >= 0)
    columns = np.empty(9, dtype=np.int64)
    values = np.empty(9, dtype=np.float64)
    for offset in range(7):
        columns[offset] = dp + offset
        values[offset] = derivatives[offset]
    columns[7] = w_idx
    values[7] = derivatives[7]
    if output_col >= 0:
        columns[8] = output_col
        values[8] = -1.0

    for left in range(1, count):
        column = columns[left]
        value = values[left]
        right = left - 1
        while right >= 0 and columns[right] > column:
            columns[right + 1] = columns[right]
            values[right + 1] = values[right]
            right -= 1
        columns[right + 1] = column
        values[right + 1] = value
    csr_set_row(data, indptr, indices, count, row, columns, values)


@jit(nopython=True, cache=True)
def ieeest_jac(data, indptr, indices, z, v, idxs, w_idx, bus, power_injection, par):
    dp = idxs[0]
    ap = idxs[1]
    deriv = np.zeros((8, 8), dtype=np.float64)

    deriv[0, 0] = -par[0] / par[1]
    deriv[0, 1] = -1.0 / par[1]
    deriv[0, 7] = 1.0 / par[1]
    deriv[1, 0] = 1.0
    f1_y = np.zeros(8, dtype=np.float64)
    f1_y[1] = 1.0

    if par[3] == 0.0:
        f2_y = f1_y + par[4] * deriv[1] + par[5] * deriv[0]
    else:
        deriv[2] = f1_y / par[3]
        deriv[2, 2] -= par[2] / par[3]
        deriv[2, 3] -= 1.0 / par[3]
        deriv[3, 2] = 1.0
        f2_y = deriv[3].copy()
        f2_y[2] += par[4]
        f2_y += par[5] * deriv[2]

    if par[7] == 0.0:
        ll1_y = f2_y.copy()
    else:
        deriv[4] = f2_y / par[7]
        deriv[4, 4] -= 1.0 / par[7]
        ll1_y = np.zeros(8, dtype=np.float64)
        ll1_y[4] = 1.0
        ll1_y += par[6] * deriv[4]

    if par[9] == 0.0:
        ll2_y = ll1_y.copy()
    else:
        deriv[5] = ll1_y / par[9]
        deriv[5, 5] -= 1.0 / par[9]
        ll2_y = np.zeros(8, dtype=np.float64)
        ll2_y[5] = 1.0
        ll2_y += par[8] * deriv[5]

    deriv[6] = par[12] * ll2_y / par[11]
    deriv[6, 6] -= 1.0 / par[11]
    if par[10] > 0.0:
        output_deriv = par[10] * deriv[6]
    else:
        output_deriv = np.zeros(8, dtype=np.float64)
        output_deriv[6] = 1.0

    signals = _ieeest_signals(z, idxs, w_idx, par)
    raw_output = signals[7]
    if power_injection:
        vm = v[2 * bus]
    else:
        vr = v[2 * bus]
        vi = v[2 * bus + 1]
        vm = np.sqrt(vr * vr + vi * vi)
    if raw_output >= par[13] or raw_output <= par[14] or vm >= par[15] or vm <= par[16]:
        output_deriv[:] = 0.0

    for offset in range(7):
        _set_ieeest_row(data, indptr, indices, dp + offset, dp, w_idx, deriv[offset])
    _set_ieeest_row(data, indptr, indices, ap, dp, w_idx, output_deriv, ap)


class PssIEEEST(Stabilizer):
    """IEEEST stabilizer for the rotor-speed mode used by the target cases."""

    def __init__(
        self, id_tag, MODE, BUSR, A1, A2, A3, A4, A5, A6,
        T1, T2, T3, T4, T5, T6, KS, LSMAX, LSMIN, VCU, VCL,
    ):
        if int(MODE) != MODE or int(MODE) != 1:
            raise ValueError(f"IEEEST MODE {MODE:g} is unsupported; supported modes: 1.")
        if int(BUSR) != 0:
            raise ValueError("IEEEST remote-bus sensing is unsupported; BUSR must be 0.")
        if A2 <= 0.0:
            raise ValueError("IEEEST A2 must be positive.")
        if A4 < 0.0 or (A4 == 0.0 and A3 != 0.0):
            raise ValueError(
                "IEEEST requires A4 > 0, or A3 = A4 = 0 for denominator bypass."
            )
        if T2 < 0.0 or (T2 == 0.0 and T1 != 0.0):
            raise ValueError("IEEEST requires T2 > 0, or T1 = T2 = 0 for bypass.")
        if T4 < 0.0 or (T4 == 0.0 and T3 != 0.0):
            raise ValueError("IEEEST requires T4 > 0, or T3 = T4 = 0 for bypass.")
        if T6 <= 0.0:
            raise ValueError("IEEEST T6 must be positive.")
        if LSMIN >= LSMAX:
            raise ValueError("IEEEST LSMIN must be less than LSMAX.")

        self.MODE = int(MODE)
        self.BUSR = int(BUSR)
        self.A1, self.A2, self.A3 = A1, A2, A3
        self.A4, self.A5, self.A6 = A4, A5, A6
        self.T1, self.T2, self.T3 = T1, T2, T3
        self.T4, self.T5, self.T6 = T4, T5, T6
        self.KS = KS
        self.LSMAX, self.LSMIN = LSMAX, LSMIN
        self.VCU = 999.0 if VCU == 0.0 else VCU
        self.VCL = -999.0 if VCL == 0.0 else VCL
        if self.VCL >= self.VCU:
            raise ValueError("IEEEST VCL must be less than VCU.")

        parameter_list = [
            "A1", "A2", "A3", "A4", "A5", "A6", "T1", "T2", "T3",
            "T4", "T5", "T6", "KS", "LSMAX", "LSMIN", "VCU", "VCL",
        ]
        state_list = [
            "f1_x", "f1_y", "f2_x1", "f2_x2", "ll1_x", "ll2_x", "wash_x",
            "v_pss",
        ]
        super().__init__(id_tag, 8, 7, 1, len(parameter_list), state_list)

    @property
    def _parameters(self):
        return np.array([
            self.A1, self.A2, self.A3, self.A4, self.A5, self.A6,
            self.T1, self.T2, self.T3, self.T4, self.T5, self.T6,
            self.KS, self.LSMAX, self.LSMIN, self.VCU, self.VCL,
        ])

    def initialize(self, vm, va, p, q, x, y, psys):
        x[self.dif_ptr:self.dif_ptr + self.dif_dim] = 0.0
        y[self.alg_ptr] = 0.0
        self.initialized = True

    def initialize_theta(self, theta):
        theta[self.par_ptr:self.par_ptr + self.par_dim] = self._parameters

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        ieeest_resdiff(
            F, z, v, idxs, self.w_idx, self.bus, power_injection,
            theta[self.par_ptr:self.par_ptr + self.par_dim],
        )

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp = int(idxs[0])
        ap = int(idxs[1])
        columns = sorted((*range(dp, dp + 7), self.w_idx))
        coordinates = [[dp + offset, columns] for offset in range(7)]
        coordinates.append([ap, sorted((*columns, ap))])
        return coordinates

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        ieeest_jac(
            J.data, J.indptr, J.indices, z, v, idxs, self.w_idx, self.bus,
            power_injection, theta[self.par_ptr:self.par_ptr + self.par_dim],
        )
