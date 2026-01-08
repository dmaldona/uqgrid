import numpy as np
from uqgrid.core.base_models import Exciter
from uqgrid.utils.tools import csr_set_row


def _ordered_vals(cols, mapping):
    return np.array([mapping[c] for c in cols], dtype=np.float64)


class ExcSEXS(Exciter):
    """
    Simplified Excitation System (SEXS) with continuous limiter.

    Parameters are:
    TA_TB, TB, K, TE, Emin, Emax.
    """

    def __init__(self, id_tag, TA_TB, TB, K, TE, Emin, Emax):
        self.TA_TB = TA_TB
        self.TB = TB
        self.K = K
        self.TE = TE
        self.Emin = Emin
        self.Emax = Emax

        # control variables
        self.vref = None
        self.efd_idx = 1

        parameter_list = ['TA_TB', 'TB', 'K', 'TE', 'Emin', 'Emax']
        state_list = ['x1', 'e_fd']

        Exciter.__init__(self, id_tag, 2, 2, 0, len(parameter_list), state_list)

    def initialize(self, vm, va, p, q, x, y, psys):
        # Initial conditions per TSOPF:
        # Vref = Efd/K + Vm, x1 = (1 - TA_TB) * (Vref - Vm)
        e_fd = self.e_fd0
        if self.K == 0:
            raise ValueError("SEXS K must be non-zero.")
        vref = e_fd / self.K + vm
        x1 = (1.0 - self.TA_TB) * (vref - vm)

        self.vref = vref
        x[self.dif_ptr] = x1
        x[self.dif_ptr + 1] = e_fd
        self.initialized = True
        return None

    def initialize_theta(self, theta):
        idx = self.par_ptr
        theta[idx] = self.TA_TB
        theta[idx + 1] = self.TB
        theta[idx + 2] = self.K
        theta[idx + 3] = self.TE
        theta[idx + 4] = self.Emin
        theta[idx + 5] = self.Emax

    def _vm_from_v(self, v, power_injection):
        if power_injection:
            vm = v[2 * self.bus]
        else:
            vr = v[2 * self.bus]
            vi = v[2 * self.bus + 1]
            vm = np.sqrt(vr * vr + vi * vi)
            if vm == 0.0:
                vm = 1e-12
        return vm

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        dp = idxs[0]

        # states
        x1 = z[dp]
        e_fd = z[dp + 1]

        vm = self._vm_from_v(v, power_injection)
        vref = self.vref

        # lead-lag output
        y1 = x1 + self.TA_TB * (vref - vm)

        # dynamics
        dx1 = (-x1 + (1.0 - self.TA_TB) * (vref - vm)) / self.TB
        dedt = (-e_fd + self.K * y1) / self.TE

        # continuous limiter: freeze at bounds when pushing further
        if (e_fd >= self.Emax and dedt > 0.0) or (e_fd <= self.Emin and dedt < 0.0):
            dedt = 0.0

        F[dp] = dx1
        F[dp + 1] = dedt
        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        coord = []
        dp = idxs[0]
        dev = idxs[2]

        x1 = dp
        e_fd = dp + 1

        if power_injection:
            vm = dev + 2 * self.bus
        else:
            vr = dev + 2 * self.bus
            vi = dev + 2 * self.bus + 1

        # row for x1
        if power_injection:
            self._jac_cols_x1 = sorted([x1, vm])
            coord.append([dp, self._jac_cols_x1])
        else:
            self._jac_cols_x1 = sorted([x1, vr, vi])
            coord.append([dp, self._jac_cols_x1])

        # row for e_fd
        if power_injection:
            self._jac_cols_efd = sorted([x1, e_fd, vm])
            coord.append([dp + 1, self._jac_cols_efd])
        else:
            self._jac_cols_efd = sorted([x1, e_fd, vr, vi])
            coord.append([dp + 1, self._jac_cols_efd])

        return coord

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        dp = idxs[0]
        dev = idxs[2]

        x1 = z[dp]
        e_fd = z[dp + 1]

        if power_injection:
            vm = v[2 * self.bus]
        else:
            vr = v[2 * self.bus]
            vi = v[2 * self.bus + 1]
            vm = np.sqrt(vr * vr + vi * vi)
            if vm == 0.0:
                vm = 1e-12
            dvm_dvr = vr / vm
            dvm_dvi = vi / vm

        vref = self.vref

        y1 = x1 + self.TA_TB * (vref - vm)
        dedt = (-e_fd + self.K * y1) / self.TE
        limited = (e_fd >= self.Emax and dedt > 0.0) or (e_fd <= self.Emin and dedt < 0.0)

        # Row for x1
        row = dp
        if power_injection:
            col_map = {
                dp: -1.0 / self.TB,
                dev + 2 * self.bus: -(1.0 - self.TA_TB) / self.TB,
            }
        else:
            col_map = {
                dp: -1.0 / self.TB,
                dev + 2 * self.bus: -(1.0 - self.TA_TB) * dvm_dvr / self.TB,
                dev + 2 * self.bus + 1: -(1.0 - self.TA_TB) * dvm_dvi / self.TB,
            }
        cols = np.array(self._jac_cols_x1, dtype=np.int32)
        vals = _ordered_vals(cols, col_map)
        csr_set_row(J.data, J.indptr, J.indices, len(cols), row, cols, vals)

        # Row for e_fd
        row = dp + 1
        if not limited:
            if power_injection:
                col_map = {
                    dp: self.K / self.TE,
                    dp + 1: -1.0 / self.TE,
                    dev + 2 * self.bus: -(self.K * self.TA_TB) / self.TE,
                }
            else:
                col_map = {
                    dp: self.K / self.TE,
                    dp + 1: -1.0 / self.TE,
                    dev + 2 * self.bus: -(self.K * self.TA_TB) * dvm_dvr / self.TE,
                    dev + 2 * self.bus + 1: -(self.K * self.TA_TB) * dvm_dvi / self.TE,
                }
            cols = np.array(self._jac_cols_efd, dtype=np.int32)
            vals = _ordered_vals(cols, col_map)
            csr_set_row(J.data, J.indptr, J.indices, len(cols), row, cols, vals)
