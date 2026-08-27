import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Exciter
from uqgrid.models.esdc1a_imp import esdc1a_dsat, esdc1a_sat, esdc1a_sat_coefficients
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def ieeet1_resdiff(
    F, z, v, theta, idxs, bus, Tr, Ka, Ta, Ke, Te, Kf, Tf, sat_a, sat_b,
    has_sensor, power_injection, pss_input=0.0,
):
    dp = idxs[0]
    pp = idxs[2]
    vr_offset = 1 if has_sensor else 0

    vr = z[dp + vr_offset]
    e_fd = z[dp + vr_offset + 1]
    washout_state = z[dp + vr_offset + 2]

    v_real = v[2 * bus]
    v_imag = v[2 * bus + 1]
    vm = v_real if power_injection else np.sqrt(v_real * v_real + v_imag * v_imag)
    if vm == 0.0:
        vm = 1e-12

    if has_sensor:
        sensed_voltage = z[dp]
        F[dp] = (vm - sensed_voltage) / Tr
    else:
        sensed_voltage = vm

    vref = theta[pp + 18]
    stabilizing_feedback = (Kf / Tf) * (e_fd - washout_state)
    F[dp + vr_offset] = (
        Ka * (vref - sensed_voltage + pss_input - stabilizing_feedback) - vr
    ) / Ta
    F[dp + vr_offset + 1] = (
        vr - Ke * e_fd - esdc1a_sat(e_fd, sat_a, sat_b)
    ) / Te
    F[dp + vr_offset + 2] = (e_fd - washout_state) / Tf


@jit(nopython=True, cache=True)
def ieeet1_jac(
    data, indptr, indices, z, v, idxs, bus, Tr, Ka, Ta, Ke, Te, Kf, Tf,
    sat_a, sat_b, has_sensor, power_injection, pss_input_idx=-1,
):
    dp = idxs[0]
    dev = idxs[2]
    vr_offset = 1 if has_sensor else 0
    vr_idx = dp + vr_offset
    e_fd_idx = vr_idx + 1
    washout_idx = vr_idx + 2

    v_real_idx = dev + 2 * bus
    v_imag_idx = v_real_idx + 1
    v_real = v[2 * bus]
    v_imag = v[2 * bus + 1]
    vm = v_real if power_injection else np.sqrt(v_real * v_real + v_imag * v_imag)
    if vm == 0.0:
        vm = 1e-12
    dvm_dreal = 1.0 if power_injection else v_real / vm
    dvm_dimag = 0.0 if power_injection else v_imag / vm

    col = np.empty(7, dtype=np.int64)
    val = np.empty(7, dtype=np.float64)

    if has_sensor:
        col[0] = dp
        val[0] = -1.0 / Tr
        col[1] = v_real_idx
        val[1] = dvm_dreal / Tr
        col[2] = v_imag_idx
        val[2] = dvm_dimag / Tr
        csr_set_row(data, indptr, indices, 3, dp, col, val)

    row = vr_idx
    nvalues = 0
    if has_sensor:
        col[nvalues] = dp
        val[nvalues] = -Ka / Ta
        nvalues += 1
    col[nvalues] = vr_idx
    val[nvalues] = -1.0 / Ta
    nvalues += 1
    col[nvalues] = e_fd_idx
    val[nvalues] = -Ka * Kf / (Ta * Tf)
    nvalues += 1
    col[nvalues] = washout_idx
    val[nvalues] = Ka * Kf / (Ta * Tf)
    nvalues += 1
    if pss_input_idx >= 0:
        col[nvalues] = pss_input_idx
        val[nvalues] = Ka / Ta
        nvalues += 1
    if not has_sensor:
        col[nvalues] = v_real_idx
        val[nvalues] = -Ka * dvm_dreal / Ta
        nvalues += 1
        col[nvalues] = v_imag_idx
        val[nvalues] = -Ka * dvm_dimag / Ta
        nvalues += 1
    csr_set_row(data, indptr, indices, nvalues, row, col, val)

    row = e_fd_idx
    col[0] = vr_idx
    val[0] = 1.0 / Te
    col[1] = e_fd_idx
    val[1] = -(Ke + esdc1a_dsat(z[e_fd_idx], sat_a, sat_b)) / Te
    csr_set_row(data, indptr, indices, 2, row, col, val)

    row = washout_idx
    col[0] = e_fd_idx
    val[0] = 1.0 / Tf
    col[1] = washout_idx
    val[1] = -1.0 / Tf
    csr_set_row(data, indptr, indices, 2, row, col, val)


class ExcIEEET1(Exciter):
    """IEEE Type 1 excitation system."""

    def __init__(
        self, id_tag, Tr, Ka, Ta, Vrmax, Vrmin, Ke, Te, Kf, Tf, Switch,
        E1, SE1, E2, SE2,
    ):
        if Ta <= 0.0 or Te <= 0.0 or Tf <= 0.0 or Tr < 0.0:
            raise ValueError("IEEET1 time constants require TR >= 0 and TA, TE, TF > 0.")
        if Ka == 0.0:
            raise ValueError("IEEET1 KA must be non-zero.")

        self.Tr = Tr
        self.Ka = Ka
        self.Ta = Ta
        self.Vrmax = Vrmax
        self.Vrmin = Vrmin
        self.Ke = Ke
        self.Te = Te
        self.Kf = Kf
        self.Tf = Tf
        self.Switch = Switch
        self.E1 = E1
        self.SE1 = SE1
        self.E2 = E2
        self.SE2 = SE2
        self.sat_a, self.sat_b = esdc1a_sat_coefficients(E1, SE1, E2, SE2)
        self.effective_vrmax = 999.0 if Vrmax == 0.0 else Vrmax
        if not self.Vrmin < self.effective_vrmax:
            raise ValueError("IEEET1 VRMIN must be less than the effective VRMAX.")

        self.has_sensor = Tr > 0.0
        self.vr_idx = 1 if self.has_sensor else 0
        self.efd_idx = self.vr_idx + 1
        self.washout_idx = self.vr_idx + 2
        state_list = (["sensed_voltage"] if self.has_sensor else []) + [
            "vr", "e_fd", "washout_state"
        ]
        parameter_list = [
            "Tr", "Ka", "Ta", "Vrmax", "Vrmin", "Ke", "Te", "Kf", "Tf",
            "Switch", "E1", "SE1", "E2", "SE2", "sat_a", "sat_b",
            "effective_vrmax", "enable_limits", "vref",
        ]
        self.enable_limits = True
        self.bounded_state_metadata = (
            BoundedStateMetadata(
                "Vr", self.vr_idx, 4, 16, 17, "IEEET1"
            ),
        )
        Exciter.__init__(
            self, id_tag, len(state_list), len(state_list), 0,
            len(parameter_list), state_list,
        )

    def initialize(self, vm, va, p, q, x, y, psys):
        e_fd = self.e_fd0
        vr = self.Ke * e_fd + esdc1a_sat(e_fd, self.sat_a, self.sat_b)
        if vr < self.Vrmin:
            raise ValueError(
                f"IEEET1 initial regulator output {vr} is outside "
                f"[{self.Vrmin}, {self.effective_vrmax}]."
            )
        self.effective_vrmax = max(self.effective_vrmax, vr)
        if self.has_sensor:
            x[self.dif_ptr] = vm
        x[self.dif_ptr + self.vr_idx] = vr
        x[self.dif_ptr + self.efd_idx] = e_fd
        x[self.dif_ptr + self.washout_idx] = e_fd
        self.vref = vm + vr / self.Ka
        self.initialized = True

    def initialize_theta(self, theta):
        values = (
            self.Tr, self.Ka, self.Ta, self.Vrmax, self.Vrmin, self.Ke, self.Te,
            self.Kf, self.Tf, self.Switch, self.E1, self.SE1, self.E2, self.SE2,
            self.sat_a, self.sat_b, self.effective_vrmax,
            float(self.enable_limits), self.vref,
        )
        theta[self.par_ptr:self.par_ptr + len(values)] = values

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss_input = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        ieeet1_resdiff(
            F, z, v, theta, idxs, self.bus, self.Tr, self.Ka, self.Ta,
            self.Ke, self.Te, self.Kf, self.Tf, self.sat_a, self.sat_b,
            self.has_sensor, power_injection, pss_input,
        )

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp = idxs[0]
        dev = idxs[2]
        vr = dp + self.vr_idx
        e_fd = dp + self.efd_idx
        washout = dp + self.washout_idx
        v_real = dev + 2 * self.bus
        v_imag = v_real + 1
        coords = []
        if self.has_sensor:
            coords.append([dp, sorted([dp, v_real, v_imag])])
        regulator_cols = [vr, e_fd, washout]
        if self.has_sensor:
            regulator_cols.append(dp)
        else:
            regulator_cols.extend([v_real, v_imag])
        if self.pss_input_idx >= 0:
            regulator_cols.append(self.pss_input_idx)
        coords.append([vr, sorted(regulator_cols)])
        coords.append([e_fd, sorted([vr, e_fd])])
        coords.append([washout, sorted([e_fd, washout])])
        return coords

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        ieeet1_jac(
            J.data, J.indptr, J.indices, z, v, idxs, self.bus, self.Tr,
            self.Ka, self.Ta, self.Ke, self.Te, self.Kf, self.Tf,
            self.sat_a, self.sat_b, self.has_sensor, power_injection,
            self.pss_input_idx,
        )
