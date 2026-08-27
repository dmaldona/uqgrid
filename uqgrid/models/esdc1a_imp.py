import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Exciter
from uqgrid.utils.tools import csr_set_row


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
def _set_row(data, indptr, indices, nvalues, row, columns, values):
    for i in range(1, nvalues):
        column = columns[i]
        value = values[i]
        j = i - 1
        while j >= 0 and columns[j] > column:
            columns[j + 1] = columns[j]
            values[j + 1] = values[j]
            j -= 1
        columns[j + 1] = column
        values[j + 1] = value
    csr_set_row(data, indptr, indices, nvalues, row, columns, values)


@jit(nopython=True, cache=True)
def esdc1a_resdiff(
    F, z, v, theta, idxs, bus, power_injection, pss_input,
    Tr, Ka, Ta, Tb, Tc, Ke, Te, Kf, Tf, sat_a, sat_b,
):
    dp = idxs[0]
    pp = idxs[2]
    vt, ll, vr, e_fd, wf = z[dp:dp + 5]

    v_real = v[2 * bus]
    v_imag = v[2 * bus + 1]
    vm = v_real if power_injection else np.sqrt(v_real * v_real + v_imag * v_imag)
    if vm == 0.0:
        vm = 1e-12

    sensed = vm if Tr == 0.0 else vt
    washout = (Kf / Tf) * (e_fd - wf)
    error = theta[pp + 19] - sensed - washout + pss_input
    ll_output = error if Tb == 0.0 else ll + (Tc / Tb) * (error - ll)

    F[dp] = -vt if Tr == 0.0 else (vm - vt) / Tr
    F[dp + 1] = -ll if Tb == 0.0 else (error - ll) / Tb
    F[dp + 2] = (Ka * ll_output - vr) / Ta
    F[dp + 3] = (vr - Ke * e_fd - esdc1a_sat(e_fd, sat_a, sat_b)) / Te
    F[dp + 4] = (e_fd - wf) / Tf


@jit(nopython=True, cache=True)
def esdc1a_jac(
    data, indptr, indices, z, v, idxs, bus, power_injection, pss_input_idx,
    Tr, Ka, Ta, Tb, Tc, Ke, Te, Kf, Tf, sat_a, sat_b,
):
    dp = idxs[0]
    dev = idxs[2]
    e_fd = z[dp + 3]
    v_real = v[2 * bus]
    v_imag = v[2 * bus + 1]
    vm = v_real if power_injection else np.sqrt(v_real * v_real + v_imag * v_imag)
    if vm == 0.0:
        vm = 1e-12
    dvm_real = 1.0 if power_injection else v_real / vm
    dvm_imag = 0.0 if power_injection else v_imag / vm
    v_real_idx = dev + 2 * bus
    v_imag_idx = v_real_idx + 1

    cols = np.empty(8, dtype=np.int64)
    vals = np.empty(8, dtype=np.float64)

    cols[0], vals[0] = dp, (-1.0 if Tr == 0.0 else -1.0 / Tr)
    n = 1
    if Tr != 0.0:
        cols[n], vals[n] = v_real_idx, dvm_real / Tr
        n += 1
        if not power_injection:
            cols[n], vals[n] = v_imag_idx, dvm_imag / Tr
            n += 1
    _set_row(data, indptr, indices, n, dp, cols, vals)

    if Tb == 0.0:
        cols[0], vals[0] = dp + 1, -1.0
        _set_row(data, indptr, indices, 1, dp + 1, cols, vals)
    else:
        cols[0], vals[0] = dp + 1, -1.0 / Tb
        n = 1
        if Tr != 0.0:
            cols[n], vals[n] = dp, -1.0 / Tb
            n += 1
        else:
            cols[n], vals[n] = v_real_idx, -dvm_real / Tb
            n += 1
            if not power_injection:
                cols[n], vals[n] = v_imag_idx, -dvm_imag / Tb
                n += 1
        cols[n], vals[n] = dp + 3, -Kf / (Tb * Tf)
        n += 1
        cols[n], vals[n] = dp + 4, Kf / (Tb * Tf)
        n += 1
        if pss_input_idx >= 0:
            cols[n], vals[n] = pss_input_idx, 1.0 / Tb
            n += 1
        _set_row(data, indptr, indices, n, dp + 1, cols, vals)

    lead = 1.0 if Tb == 0.0 else Tc / Tb
    cols[0], vals[0] = dp + 2, -1.0 / Ta
    n = 1
    if Tb != 0.0:
        cols[n], vals[n] = dp + 1, Ka * (1.0 - lead) / Ta
        n += 1
    error_gain = Ka * lead / Ta
    if Tr != 0.0:
        cols[n], vals[n] = dp, -error_gain
        n += 1
    else:
        cols[n], vals[n] = v_real_idx, -error_gain * dvm_real
        n += 1
        if not power_injection:
            cols[n], vals[n] = v_imag_idx, -error_gain * dvm_imag
            n += 1
    cols[n], vals[n] = dp + 3, -error_gain * Kf / Tf
    n += 1
    cols[n], vals[n] = dp + 4, error_gain * Kf / Tf
    n += 1
    if pss_input_idx >= 0:
        cols[n], vals[n] = pss_input_idx, error_gain
        n += 1
    _set_row(data, indptr, indices, n, dp + 2, cols, vals)

    cols[0], vals[0] = dp + 2, 1.0 / Te
    cols[1], vals[1] = dp + 3, -(Ke + esdc1a_dsat(e_fd, sat_a, sat_b)) / Te
    _set_row(data, indptr, indices, 2, dp + 3, cols, vals)

    cols[0], vals[0] = dp + 3, 1.0 / Tf
    cols[1], vals[1] = dp + 4, -1.0 / Tf
    _set_row(data, indptr, indices, 2, dp + 4, cols, vals)


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
    """Direct-current commutator exciter with constant regulator bounds."""

    bound_scale = None
    device_type = "ESDC1A"

    def __init__(
        self, id_tag, Ka, Ta, Kf, Tf, Ke, Te, Tr, E1, SE1, E2, SE2,
        Tb=0.0, Tc=0.0, Vrmax=0.0, Vrmin=0.0, Sw=0.0,
        enable_limits=True, adjust_initial_limits=False,
    ):
        if Tr < 0.0 or Tb < 0.0 or Ta <= 0.0 or Te <= 0.0 or Tf <= 0.0:
            raise ValueError("ESDC1A requires TR, TB >= 0 and TA, TE, TF > 0.")
        if Tb == 0.0 and Tc != 0.0:
            raise ValueError("ESDC1A requires TC = 0 when TB = 0.")
        if Ka == 0.0:
            raise ValueError("ESDC1A KA must be non-zero.")

        self.Tr, self.Ka, self.Ta = Tr, Ka, Ta
        self.Tb, self.Tc = Tb, Tc
        self.Vrmax, self.Vrmin = Vrmax, Vrmin
        self.Vrmin_original = Vrmin
        self.Ke, self.Te, self.Kf, self.Tf = Ke, Te, Kf, Tf
        self.Sw = Sw
        self.E1, self.SE1, self.E2, self.SE2 = E1, SE1, E2, SE2
        self.sat_a, self.sat_b = esdc1a_sat_coefficients(E1, SE1, E2, SE2)
        self.effective_vrmax = 999.0 if Vrmax == 0.0 else Vrmax
        if Vrmin >= self.effective_vrmax:
            raise ValueError("ESDC1A VRMIN must be less than the effective VRMAX.")
        self.enable_limits = enable_limits
        self.adjust_initial_limits = adjust_initial_limits
        self.limit_initialization_diagnostics = None
        self.vref = None
        self.efd_idx = 3
        self.bounded_state_metadata = (
            BoundedStateMetadata(
                "VR", 2, 10, 9, 18, self.device_type,
                bound_scale=self.bound_scale,
            ),
        )
        parameter_list = [
            "Ka", "Ta", "Tb", "Tc", "Kf", "Tf", "Ke", "Te", "Tr",
            "effective_vrmax", "Vrmin", "Sw", "E1", "SE1", "E2", "SE2",
            "sat_a", "sat_b", "enable_limits", "vref",
        ]
        state_list = ["vt", "ll", "vr", "e_fd", "wf"]
        super().__init__(id_tag, 5, 5, 0, len(parameter_list), state_list)

    def initialize(self, vm, va, p, q, x, y, psys):
        e_fd = self.e_fd0
        vr = self.Ke * e_fd + esdc1a_sat(e_fd, self.sat_a, self.sat_b)
        scale = vm if self.bound_scale == "terminal_voltage" else 1.0
        initial_coefficient = vr / scale
        source_upper = 999.0 if self.Vrmax == 0.0 else self.Vrmax
        outside_bounds = initial_coefficient < self.Vrmin_original or initial_coefficient > source_upper
        if self.enable_limits and outside_bounds and not self.adjust_initial_limits:
            raise ValueError(
                f"{self.device_type} initial regulator state is outside its enabled bounds."
            )
        self.Vrmin = (
            min(self.Vrmin_original, initial_coefficient)
            if self.enable_limits and self.adjust_initial_limits
            else self.Vrmin_original
        )
        self.effective_vrmax = (
            max(source_upper, initial_coefficient)
            if self.enable_limits and self.adjust_initial_limits
            else source_upper
        )
        self.limit_initialization_diagnostics = {
            "initial_regulator_coefficient": float(initial_coefficient),
            "source_VRMIN": float(self.Vrmin_original),
            "source_VRMAX": float(source_upper),
            "effective_VRMIN": float(self.Vrmin),
            "effective_VRMAX": float(self.effective_vrmax),
            "bounds_adjusted": bool(
                self.Vrmin != self.Vrmin_original
                or self.effective_vrmax != source_upper
            ),
            "adjust_initial_limits": bool(self.adjust_initial_limits),
        }
        error = vr / self.Ka
        x[self.dif_ptr:self.dif_ptr + 5] = (
            vm if self.Tr != 0.0 else 0.0,
            error if self.Tb != 0.0 else 0.0,
            vr,
            e_fd,
            e_fd,
        )
        self.vref = vm + error
        self.initialized = True

    def initialize_theta(self, theta):
        values = (
            self.Ka, self.Ta, self.Tb, self.Tc, self.Kf, self.Tf, self.Ke,
            self.Te, self.Tr, self.effective_vrmax, self.Vrmin, self.Sw,
            self.E1, self.SE1, self.E2, self.SE2, self.sat_a, self.sat_b,
            float(self.enable_limits), self.vref,
        )
        theta[self.par_ptr:self.par_ptr + len(values)] = values

    def residualFinit(self, x, v, theta, p0, q0):
        return np.zeros(self.initdim)

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        esdc1a_resdiff(
            F, z, v, theta, idxs, self.bus, power_injection, pss,
            self.Tr, self.Ka, self.Ta, self.Tb, self.Tc, self.Ke, self.Te,
            self.Kf, self.Tf, self.sat_a, self.sat_b,
        )

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp, dev = idxs[0], idxs[2]
        v_real = dev + 2 * self.bus
        v_imag = v_real + 1
        sensor = [dp]
        if self.Tr != 0.0:
            sensor.extend([v_real, v_imag])
        ll = [dp + 1]
        if self.Tb != 0.0:
            ll.extend([dp if self.Tr != 0.0 else v_real, dp + 3, dp + 4])
            if self.Tr == 0.0:
                ll.append(v_imag)
            if self.pss_input_idx >= 0:
                ll.append(self.pss_input_idx)
        regulator = [dp + 2, dp + 3, dp + 4]
        if self.Tb != 0.0:
            regulator.append(dp + 1)
        regulator.append(dp if self.Tr != 0.0 else v_real)
        if self.Tr == 0.0:
            regulator.append(v_imag)
        if self.pss_input_idx >= 0:
            regulator.append(self.pss_input_idx)
        if self.bound_scale == "terminal_voltage":
            regulator.extend([v_real, v_imag])
        return [
            [dp, sorted(set(sensor))],
            [dp + 1, sorted(set(ll))],
            [dp + 2, sorted(set(regulator))],
            [dp + 3, [dp + 2, dp + 3]],
            [dp + 4, [dp + 3, dp + 4]],
        ]

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        esdc1a_jac(
            J.data, J.indptr, J.indices, z, v, idxs, self.bus,
            power_injection, self.pss_input_idx, self.Tr, self.Ka, self.Ta,
            self.Tb, self.Tc, self.Ke, self.Te, self.Kf, self.Tf,
            self.sat_a, self.sat_b,
        )
