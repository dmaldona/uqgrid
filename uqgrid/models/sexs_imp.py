import numpy as np
from uqgrid.core.base_models import Exciter


class ExcSEXS(Exciter):
    """
    Simplified Excitation System (SEXS) with continuous limiter.

    Parameters are consistent with TSOPF's SEXS model:
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
            coord.append([dp, [x1, vm]])
        else:
            coord.append([dp, [x1, vr, vi]])

        # row for e_fd
        if power_injection:
            coord.append([dp + 1, [x1, e_fd, vm]])
        else:
            coord.append([dp + 1, [x1, e_fd, vr, vi]])

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
        cols = np.zeros(4, dtype=np.int32)
        vals = np.zeros(4, dtype=np.float64)
        cols[0] = dp
        vals[0] = -1.0 / self.TB
        ncols = 1

        if power_injection:
            cols[ncols] = dev + 2 * self.bus
            vals[ncols] = -(1.0 - self.TA_TB) / self.TB
            ncols += 1
        else:
            cols[ncols] = dev + 2 * self.bus
            vals[ncols] = -(1.0 - self.TA_TB) * dvm_dvr / self.TB
            ncols += 1
            cols[ncols] = dev + 2 * self.bus + 1
            vals[ncols] = -(1.0 - self.TA_TB) * dvm_dvi / self.TB
            ncols += 1

        from uqgrid.utils.tools import csr_set_row
        order = np.argsort(cols[:ncols])
        csr_set_row(J.data, J.indptr, J.indices, ncols, row, cols[order], vals[order])

        # Row for e_fd
        row = dp + 1
        cols = np.zeros(5, dtype=np.int32)
        vals = np.zeros(5, dtype=np.float64)
        ncols = 0

        if not limited:
            cols[ncols] = dp
            vals[ncols] = self.K / self.TE
            ncols += 1

            cols[ncols] = dp + 1
            vals[ncols] = -1.0 / self.TE
            ncols += 1

            if power_injection:
                cols[ncols] = dev + 2 * self.bus
                vals[ncols] = -(self.K * self.TA_TB) / self.TE
                ncols += 1
            else:
                cols[ncols] = dev + 2 * self.bus
                vals[ncols] = -(self.K * self.TA_TB) * dvm_dvr / self.TE
                ncols += 1
                cols[ncols] = dev + 2 * self.bus + 1
                vals[ncols] = -(self.K * self.TA_TB) * dvm_dvi / self.TE
                ncols += 1
        else:
            # If limited, keep the row zero (no dynamics).
            ncols = 0

        if ncols > 0:
            order = np.argsort(cols[:ncols])
            csr_set_row(J.data, J.indptr, J.indices, ncols, row, cols[order], vals[order])

