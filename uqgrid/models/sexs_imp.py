import numpy as np
from numba import jit
from uqgrid.core.base_models import Exciter
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def sexs_resdiff(F, z, v, theta, idxs, power_injection, bus, TA_TB, TB, K, TE):
    dp = idxs[0]
    pp = idxs[2]

    x1 = z[dp]
    e_fd = z[dp + 1]

    if power_injection:
        vm = v[2 * bus]
    else:
        vr = v[2 * bus]
        vi = v[2 * bus + 1]
        vm = np.sqrt(vr * vr + vi * vi)
        if vm == 0.0:
            vm = 1e-12

    vref = theta[pp + 6]
    y1 = x1 + TA_TB * (vref - vm)

    F[dp] = (-x1 + (1.0 - TA_TB) * (vref - vm)) / TB
    F[dp + 1] = (-e_fd + K * y1) / TE


@jit(nopython=True, cache=True)
def sexs_jac(data, indptr, indices, v, idxs, power_injection, bus, TA_TB, TB, K, TE):
    dp = idxs[0]
    dev = idxs[2]

    x1_idx = dp
    e_fd_idx = dp + 1

    if power_injection:
        vm_idx = dev + 2 * bus
        vm = v[2 * bus]
        dvm_dvr = 1.0
        dvm_dvi = 0.0
        n_x1 = 2
        n_efd = 3
    else:
        vr_idx = dev + 2 * bus
        vi_idx = dev + 2 * bus + 1
        vr = v[2 * bus]
        vi = v[2 * bus + 1]
        vm = np.sqrt(vr * vr + vi * vi)
        if vm == 0.0:
            vm = 1e-12
        dvm_dvr = vr / vm
        dvm_dvi = vi / vm
        n_x1 = 3
        n_efd = 4

    row = dp
    col = np.empty(4, dtype=np.int64)
    val = np.empty(4, dtype=np.float64)

    if power_injection:
        col[0] = x1_idx
        val[0] = -1.0 / TB
        col[1] = vm_idx
        val[1] = -(1.0 - TA_TB) / TB
    else:
        col[0] = x1_idx
        val[0] = -1.0 / TB
        col[1] = vr_idx
        val[1] = -(1.0 - TA_TB) * dvm_dvr / TB
        col[2] = vi_idx
        val[2] = -(1.0 - TA_TB) * dvm_dvi / TB
    csr_set_row(data, indptr, indices, n_x1, row, col, val)

    row = dp + 1
    if power_injection:
        col[0] = x1_idx
        val[0] = K / TE
        col[1] = e_fd_idx
        val[1] = -1.0 / TE
        col[2] = vm_idx
        val[2] = -(K * TA_TB) / TE
    else:
        col[0] = x1_idx
        val[0] = K / TE
        col[1] = e_fd_idx
        val[1] = -1.0 / TE
        col[2] = vr_idx
        val[2] = -(K * TA_TB) * dvm_dvr / TE
        col[3] = vi_idx
        val[3] = -(K * TA_TB) * dvm_dvi / TE
    csr_set_row(data, indptr, indices, n_efd, row, col, val)


class ExcSEXS(Exciter):
    """
    Simplified Excitation System (SEXS).

    Parameters are:
    TA_TB, TB, K, TE, Emin, Emax. Limits are parsed but ignored.
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

        parameter_list = ['TA_TB', 'TB', 'K', 'TE', 'Emin', 'Emax', 'vref']
        state_list = ['x1', 'e_fd']

        Exciter.__init__(self, id_tag, 2, 2, 0, len(parameter_list), state_list)

    def initialize(self, vm, va, p, q, x, y, psys):
        # Initial conditions:
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
        theta[idx + 6] = self.vref

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
        sexs_resdiff(
            F, z, v, theta, idxs, power_injection, self.bus,
            self.TA_TB, self.TB, self.K, self.TE,
        )
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
        sexs_jac(
            J.data, J.indptr, J.indices, v, idxs, power_injection, self.bus,
            self.TA_TB, self.TB, self.K, self.TE,
        )
