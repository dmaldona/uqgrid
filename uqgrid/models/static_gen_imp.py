import numpy as np
from numba import jit

from uqgrid.core.base_models import DeviceModel
from uqgrid.utils.tools import csr_add_row, csr_set_row

BUS_PV = 2
BUS_SLACK = 3


@jit(nopython=True, cache=True)
def _add_static_gen_current(F, v, p, q, bus):
    vr = v[2 * bus]
    vi = v[2 * bus + 1]
    vm2 = max(vr * vr + vi * vi, 0.2)

    F[2 * bus] += (p * vr + q * vi) / vm2
    F[2 * bus + 1] += (p * vi - q * vr) / vm2


@jit(nopython=True, cache=True)
def _add_static_gen_current_jacobian(J_data, J_ptr, J_idx, dev, v, p, q, bus):
    vr = v[2 * bus]
    vi = v[2 * bus + 1]
    vm2_raw = vr * vr + vi * vi
    vm2 = max(vm2_raw, 0.2)
    ir_num = p * vr + q * vi
    ii_num = p * vi - q * vr
    vm4 = vm2 * vm2

    if vm2_raw > 0.2:
        dir_dvr = p / vm2 - 2.0 * vr * ir_num / vm4
        dir_dvi = q / vm2 - 2.0 * vi * ir_num / vm4
        dii_dvr = -q / vm2 - 2.0 * vr * ii_num / vm4
        dii_dvi = p / vm2 - 2.0 * vi * ii_num / vm4
    else:
        dir_dvr = p / vm2
        dir_dvi = q / vm2
        dii_dvr = -q / vm2
        dii_dvi = p / vm2

    col = np.zeros(2, dtype=np.int64)
    val = np.zeros(2)
    col[0] = dev + 2 * bus
    col[1] = dev + 2 * bus + 1

    row = dev + 2 * bus
    val[0] = dir_dvr
    val[1] = dir_dvi
    csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

    row = dev + 2 * bus + 1
    val[0] = dii_dvr
    val[1] = dii_dvi
    csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)


class StaticGenerator(DeviceModel):
    """Aggregated static generator retained when a bus has no machine model."""

    def __init__(self, bus, gen_idxs, bus_type, vset, aset, limits):
        if bus_type == BUS_PV:
            alg_dim = 1
        elif bus_type == BUS_SLACK:
            alg_dim = 2
        else:
            alg_dim = 0

        DeviceModel.__init__(self, 0, alg_dim, 4, f"static@{bus}", "static_generator")
        self.bus = bus
        self.gen_idxs = tuple(gen_idxs)
        self.bus_type = bus_type
        self.vset = vset
        self.aset = aset
        self.pmin, self.pmax, self.qmin, self.qmax = limits
        self.enable_limits = False
        self.p0 = 0.0
        self.q0 = 0.0

    def initialize(self, vm, va, p, q, x, y, psys):
        self.p0 = p
        self.q0 = q
        if self.bus_type == BUS_PV:
            y[self.alg_ptr] = q
        elif self.bus_type == BUS_SLACK:
            y[self.alg_ptr] = p
            y[self.alg_ptr + 1] = q

    def initialize_theta(self, theta):
        theta[self.par_ptr] = self.p0
        theta[self.par_ptr + 1] = self.q0
        theta[self.par_ptr + 2] = self.vset
        theta[self.par_ptr + 3] = self.aset

    def preallocate_jacobian(self, idxs, psys, power_injection):
        if power_injection:
            raise NotImplementedError("Static generators require current-injection mode")

        ap = idxs[1]
        dev = idxs[2]
        bus_vr = dev + 2 * self.bus
        bus_vi = bus_vr + 1
        rows = []

        if self.bus_type == BUS_PV:
            rows.extend([
                [ap, [bus_vr, bus_vi]],
                [bus_vr, [ap]],
                [bus_vi, [ap]],
            ])
        elif self.bus_type == BUS_SLACK:
            rows.extend([
                [ap, [bus_vr]],
                [ap + 1, [bus_vi]],
                [bus_vr, [ap, ap + 1]],
                [bus_vi, [ap, ap + 1]],
            ])
        return rows

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        if power_injection:
            raise NotImplementedError("Static generators require current-injection mode")

        ap = idxs[1]
        pp = idxs[2]
        vr = v[2 * self.bus]
        vi = v[2 * self.bus + 1]
        if self.bus_type == BUS_PV:
            F[ap] += vr * vr + vi * vi - theta[pp + 2] ** 2
        elif self.bus_type == BUS_SLACK:
            aset = theta[pp + 3]
            vset = theta[pp + 2]
            F[ap] += vr - vset * np.cos(aset)
            F[ap + 1] += vi - vset * np.sin(aset)

    def residual_pinj(self, F, z, v, theta, idxs):
        raise NotImplementedError("Static generators require current-injection mode")

    def residual_cinj(self, F, z, v, theta, idxs):
        ap = idxs[1]
        pp = idxs[2]
        p, q = self._power(z, theta, ap, pp)
        _add_static_gen_current(F, v, p, q, self.bus)

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        if power_injection:
            raise NotImplementedError("Static generators require current-injection mode")

        ap = idxs[1]
        dev = idxs[2]
        pp = idxs[3]
        p, q = self._power(z, theta, ap, pp)
        _add_static_gen_current_jacobian(J.data, J.indptr, J.indices, dev, v, p, q, self.bus)

        vr = v[2 * self.bus]
        vi = v[2 * self.bus + 1]
        bus_vr = dev + 2 * self.bus
        bus_vi = bus_vr + 1
        col = np.zeros(2, dtype=np.int64)
        val = np.zeros(2)

        if self.bus_type == BUS_PV:
            col[0] = bus_vr
            col[1] = bus_vi
            val[0] = 2.0 * vr
            val[1] = 2.0 * vi
            csr_set_row(J.data, J.indptr, J.indices, 2, ap, col, val)
            self._add_power_jacobian(J, dev, ap, vr, vi, p, q, include_p=False)
        elif self.bus_type == BUS_SLACK:
            col[0] = bus_vr
            val[0] = 1.0
            csr_set_row(J.data, J.indptr, J.indices, 1, ap, col, val)
            col[0] = bus_vi
            csr_set_row(J.data, J.indptr, J.indices, 1, ap + 1, col, val)
            self._add_power_jacobian(J, dev, ap, vr, vi, p, q, include_p=True)

    def _power(self, z, theta, ap, pp):
        if self.bus_type == BUS_PV:
            return theta[pp], z[ap]
        if self.bus_type == BUS_SLACK:
            return z[ap], z[ap + 1]
        return theta[pp], theta[pp + 1]

    def _add_power_jacobian(self, J, dev, ap, vr, vi, p, q, include_p):
        vm2 = max(vr * vr + vi * vi, 0.2)
        row = dev + 2 * self.bus
        col = np.zeros(2, dtype=np.int64)
        val = np.zeros(2)

        if include_p:
            col[0] = ap
            col[1] = ap + 1
            val[0] = vr / vm2
            val[1] = vi / vm2
            csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

            row += 1
            val[0] = vi / vm2
            val[1] = -vr / vm2
            csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)
        else:
            col[0] = ap
            val[0] = vi / vm2
            csr_set_row(J.data, J.indptr, J.indices, 1, row, col, val)

            row += 1
            val[0] = -vr / vm2
            csr_set_row(J.data, J.indptr, J.indices, 1, row, col, val)
