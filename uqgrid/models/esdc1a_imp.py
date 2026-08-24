import numpy as np
from numba import jit
from uqgrid.utils.tools import csr_set_row
from uqgrid.core.base_models import Exciter


@jit(nopython=True, cache=True)
def esdc1a_sat(e_fd, sat_a, sat_b):
    if sat_b == 0.0 or e_fd <= sat_a:
        return 0.0
    return sat_b * (e_fd - sat_a) ** 2.0


@jit(nopython=True, cache=True)
def esdc1a_dsat(e_fd, sat_a, sat_b):
    if sat_b == 0.0 or e_fd <= sat_a:
        return 0.0
    return 2.0 * sat_b * (e_fd - sat_a)


@jit(nopython=True, cache=True)
def esdc1a_resdiff(
    F, z, v, theta, idxs, bus, Ka, Ta, Kf, Tf, Ke, Te, sat_a, sat_b,
    pss_input=0.0,
):
    dp = idxs[0]
    pp = idxs[2]

    vref = theta[pp + 18]
    vr1 = z[dp]
    vr2 = z[dp + 1]
    e_fd = z[dp + 2]

    vr = v[2 * bus]
    vi = v[2 * bus + 1]
    vm = np.sqrt(vr * vr + vi * vi)
    if vm == 0.0:
        vm = 1e-12

    F[dp] = (Ka * (vref - vm + pss_input - vr2 - (Kf / Tf) * e_fd) - vr1) / Ta
    F[dp + 1] = -((Kf / Tf) * e_fd + vr2) / Tf
    F[dp + 2] = (vr1 - Ke * e_fd - esdc1a_sat(e_fd, sat_a, sat_b)) / Te


@jit(nopython=True, cache=True)
def esdc1a_jac(
    data, indptr, indices, z, v, idxs, bus, Ka, Ta, Kf, Tf, Ke, Te,
    sat_a, sat_b, pss_input_idx=-1,
):
    dp = idxs[0]
    dev = idxs[2]

    vr1_idx = dp
    vr2_idx = dp + 1
    e_fd_idx = dp + 2
    e_fd = z[e_fd_idx]

    vr_idx = dev + 2 * bus
    vi_idx = dev + 2 * bus + 1
    vr = v[2 * bus]
    vi = v[2 * bus + 1]
    vm = np.sqrt(vr * vr + vi * vi)
    if vm == 0.0:
        vm = 1e-12
    dvm_dvr = vr / vm
    dvm_dvi = vi / vm

    col = np.empty(6, dtype=np.int64)
    val = np.empty(6, dtype=np.float64)

    row = dp
    col[0] = vr1_idx
    val[0] = -1.0 / Ta
    col[1] = vr2_idx
    val[1] = -Ka / Ta
    col[2] = e_fd_idx
    val[2] = -Ka * Kf / (Ta * Tf)
    nvalues = 5
    if pss_input_idx >= 0:
        col[3] = pss_input_idx
        val[3] = Ka / Ta
        col[4] = vr_idx
        val[4] = (-Ka / Ta) * dvm_dvr
        col[5] = vi_idx
        val[5] = (-Ka / Ta) * dvm_dvi
        nvalues = 6
    else:
        col[3] = vr_idx
        val[3] = (-Ka / Ta) * dvm_dvr
        col[4] = vi_idx
        val[4] = (-Ka / Ta) * dvm_dvi
    csr_set_row(data, indptr, indices, nvalues, row, col, val)

    row = dp + 1
    col[0] = vr2_idx
    val[0] = -1.0 / Tf
    col[1] = e_fd_idx
    val[1] = -Kf / (Tf * Tf)
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = dp + 2
    col[0] = vr1_idx
    val[0] = 1.0 / Te
    col[1] = e_fd_idx
    val[1] = -(Ke + esdc1a_dsat(e_fd, sat_a, sat_b)) / Te
    csr_set_row(data, indptr, indices, 2, row, col, val)


def esdc1a_sat_coefficients(E1, SE1, E2, SE2):
    if E1 <= 0.0 or E2 <= 0.0 or SE1 <= 0.0 or SE2 <= 0.0:
        return 0.0, 0.0
    a = np.sqrt(SE1 * E1 / (SE2 * E2))
    if a == 1.0:
        return 0.0, 0.0
    sat_a = E2 - (E1 - E2) / (a - 1.0)
    sat_b = SE2 * E2 * (a - 1.0) ** 2.0 / (E1 - E2) ** 2.0
    return sat_a, sat_b


class ExcESDC1A(Exciter):
    def __init__(
        self, id_tag, Ka, Ta, Kf, Tf, Ke, Te, Tr, E1, SE1, E2, SE2,
        Tb=0.0, Tc=0.0, Vrmax=0.0, Vrmin=0.0, Sw=0.0,
    ):

        self.Ka = Ka
        self.Ta = Ta
        self.Tb = Tb
        self.Tc = Tc
        self.Kf = Kf
        self.Tf = Tf
        self.Ke = Ke
        self.Te = Te
        self.Tr = Tr
        self.Vrmax = Vrmax
        self.Vrmin = Vrmin
        self.Sw = Sw
        self.E1 = E1
        self.SE1 = SE1
        self.E2 = E2
        self.SE2 = SE2
        self.sat_a, self.sat_b = esdc1a_sat_coefficients(E1, SE1, E2, SE2)

        # control variables
        self.vref = None
        self.efd_idx = 2

        parameter_list = [
            'Ka', 'Ta', 'Tb', 'Tc', 'Kf', 'Tf', 'Ke', 'Te', 'Tr',
            'Vrmax', 'Vrmin', 'Sw', 'E1', 'SE1', 'E2', 'SE2',
            'sat_a', 'sat_b', 'vref'
        ]
        state_list = ['vr1', 'vr2', 'e_fd']

        Exciter.__init__(self, id_tag, 3, 3, 0, len(parameter_list), state_list)

    def residualFinit(self, x, v, theta, p0, q0):

        F = np.zeros(self.initdim)

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        sat_a = self.sat_a
        sat_b = self.sat_b
        e_fd = self.e_fd0

        vr1 = x[0]
        vr2 = x[1]
        vref = x[2]

        F[0] = (Ka*(vref - v - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
        F[1] = -((Kf/Tf)*e_fd + vr2)/Tf
        F[2] = (vr1 - Ke*e_fd - self._sat(e_fd, sat_a, sat_b))/Te

        return F

    def initialize(self, vm, va, p, q, x, y, psys):
        e_fd = self.e_fd0
        vr2 = -(self.Kf / self.Tf) * e_fd
        vr1 = self.Ke * e_fd + self._sat(e_fd, self.sat_a, self.sat_b)
        vref = vm + vr1 / self.Ka
        self.initialized = True
        x[self.dif_ptr] = vr1
        x[self.dif_ptr + 1] = vr2
        x[self.dif_ptr + 2] = e_fd
        self.vref = vref
        return None

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.Ka
        theta[idx + 1] = self.Ta
        theta[idx + 2] = self.Tb
        theta[idx + 3] = self.Tc
        theta[idx + 4] = self.Kf
        theta[idx + 5] = self.Tf
        theta[idx + 6] = self.Ke
        theta[idx + 7] = self.Te
        theta[idx + 8] = self.Tr
        theta[idx + 9] = self.Vrmax
        theta[idx + 10] = self.Vrmin
        theta[idx + 11] = self.Sw
        theta[idx + 12] = self.E1
        theta[idx + 13] = self.SE1
        theta[idx + 14] = self.E2
        theta[idx + 15] = self.SE2
        theta[idx + 16] = self.sat_a
        theta[idx + 17] = self.sat_b
        theta[idx + 18] = self.vref

    @staticmethod
    def _sat(e_fd, sat_a, sat_b):
        if sat_b == 0.0 or e_fd <= sat_a:
            return 0.0
        return sat_b * (e_fd - sat_a) ** 2.0

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss_input = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        esdc1a_resdiff(
            F, z, v, theta, idxs, self.bus,
            self.Ka, self.Ta, self.Kf, self.Tf, self.Ke, self.Te,
            self.sat_a, self.sat_b, pss_input,
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
        dev = idxs[2]

        # these are INDEXES
        vr1 = dp
        vr2 = dp + 1
        e_fd = dp + 2

        vr = dev + 2*self.bus
        vi = dev + 2*self.bus + 1

        # first row
        row = dp
        cols = [vr1, vr2, e_fd, vr, vi]
        if self.pss_input_idx >= 0:
            cols.append(self.pss_input_idx)
        cols.sort()
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [vr2, e_fd]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [vr1, e_fd]
        coord.append([row, cols])

        return coord

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        esdc1a_jac(
            J.data, J.indptr, J.indices, z, v, idxs, self.bus,
            self.Ka, self.Ta, self.Kf, self.Tf, self.Ke, self.Te,
            self.sat_a, self.sat_b, self.pss_input_idx,
        )
