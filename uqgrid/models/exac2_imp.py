import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Exciter
from uqgrid.models.esdc1a_imp import esdc1a_sat_coefficients
from uqgrid.models.genrou_imp import sat_coefficients
from uqgrid.utils.tools import csr_set_row


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
def _rectifier(value):
    if value <= 0.0:
        return 1.0, 0.0
    if value <= 0.433:
        return 1.0 - 0.577 * value, -0.577
    if value <= 0.75:
        root = np.sqrt(0.75 - value * value)
        return root, -value / root
    if value <= 1.0:
        return 1.732 * (1.0 - value), -1.732
    return 0.0, 0.0


@jit(nopython=True, cache=True)
def _machine_field(z, gen_dp, gen_ap, xd, xdp, xqp, xddp, xl, sat_a, sat_b):
    eqp = z[gen_dp]
    edp = z[gen_dp + 1]
    phi1 = z[gen_dp + 2]
    phi2 = z[gen_dp + 3]
    id_ = z[gen_ap + 3]

    dd = xdp - xl
    dq = xqp - xl
    ad = (xddp - xl) / dd
    bd = (xdp - xddp) / dd
    aq = (xddp - xl) / dq
    bq = (xqp - xddp) / dq
    psi_d = ad * eqp + bd * phi1
    psi_q = -aq * edp + bq * phi2
    psi = np.sqrt(psi_d * psi_d + psi_q * psi_q)

    se = 0.0
    dse_dpsi = 0.0
    if sat_b != 0.0 and psi > sat_a and psi != 0.0:
        se = sat_b * (psi - sat_a) ** 2 / psi
        dse_dpsi = sat_b * (1.0 - sat_a * sat_a / (psi * psi))

    k = xdp - xddp
    n = -eqp + id_ * dd + phi1
    current = id_ - k * n / (dd * dd)
    xad = eqp + (xd - xdp) * current + se * psi_d

    dpsi = np.empty(4)
    if psi == 0.0:
        dpsi[:] = 0.0
    else:
        dpsi[0] = psi_d * ad / psi
        dpsi[1] = -psi_q * aq / psi
        dpsi[2] = psi_d * bd / psi
        dpsi[3] = psi_q * bq / psi

    deriv = np.empty(5)
    flux_deriv = np.array((ad, 0.0, bd, 0.0))
    current_deriv = np.array((k / (dd * dd), 0.0, -k / (dd * dd), 0.0))
    for i in range(4):
        deriv[i] = (
            (1.0 if i == 0 else 0.0)
            + (xd - xdp) * current_deriv[i]
            + se * flux_deriv[i]
            + psi_d * dse_dpsi * dpsi[i]
        )
    deriv[4] = (xd - xdp) * (1.0 - k / dd)
    return xad, deriv


@jit(nopython=True, cache=True)
def _signals(
    z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
    TR, TB, TC, KA, VAMAX, VAMIN, KB, VRMAX, VRMIN, KL, KH,
    KF, TF, KC, KD, KE, VLRx, sat_a, sat_b,
    xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp = idxs[0]
    vt, ll, va, ve, wf = z[dp:dp + 5]
    if power_injection:
        vm = v[2 * bus]
    else:
        vr = v[2 * bus]
        vi = v[2 * bus + 1]
        vm = np.sqrt(vr * vr + vi * vi)
        if vm == 0.0:
            vm = 1e-12

    xad, dxad = _machine_field(
        z, gen_dp, gen_ap, xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b
    )
    se = 0.0
    dse = 0.0
    if sat_b != 0.0 and ve > sat_a:
        se = sat_b * (ve - sat_a) ** 2
        dse = 2.0 * sat_b * (ve - sat_a)
    vfe = KE * ve + se + KD * xad
    dvfe_dve = KE + dse

    washout = 0.0 if TF == 0.0 or KF == 0.0 else KF * (vfe - wf) / TF
    sensed = vm if TR == 0.0 else vt
    vi_error = vref - sensed - washout + pss_input
    ll_out = ll + (TC / TB) * (vi_error - ll)
    va_raw = KA * ll_out - va

    vha = va - KH * vfe
    vl = KL * (VLRx - vfe)
    gate_high = vha <= vl
    gate = vha if gate_high else vl
    vr_raw = KB * gate
    if vr_raw <= VRMIN:
        vr_out = VRMIN
        dvr_dgate = 0.0
    elif vr_raw >= VRMAX:
        vr_out = VRMAX
        dvr_dgate = 0.0
    else:
        vr_out = vr_raw
        dvr_dgate = KB

    ratio = KC * xad / ve
    fex, dfex = _rectifier(ratio)
    efd = ve * fex
    return (
        vm, xad, dxad, vfe, dvfe_dve, washout, vi_error, ll_out,
        va_raw, gate_high, dvr_dgate, vr_out, ratio, fex, dfex, efd,
    )


@jit(nopython=True, cache=True)
def exac2_resdiff(
    F, z, v, theta, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
    TR, TB, TC, KA, TA, VAMAX, VAMIN, KB, VRMAX, VRMIN, TE, KL, KH,
    KF, TF, KC, KD, KE, VLRx, sat_a, sat_b,
    xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp = idxs[0]
    ap = idxs[1]
    vt, ll, _, ve, wf = z[dp:dp + 5]
    signals = _signals(
        z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
        TR, TB, TC, KA, VAMAX, VAMIN, KB, VRMAX, VRMIN, KL, KH,
        KF, TF, KC, KD, KE, VLRx, sat_a, sat_b,
        xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
    )
    vm, _, _, vfe, _, _, vi_error, _, va_raw, _, _, vr_out, _, _, _, efd = signals
    F[dp] = -vt if TR == 0.0 else (vm - vt) / TR
    F[dp + 1] = (vi_error - ll) / TB
    F[dp + 2] = va_raw / TA
    F[dp + 3] = (vr_out - vfe) / TE
    F[dp + 4] = -wf if TF == 0.0 or KF == 0.0 else (vfe - wf) / TF
    F[ap] = efd - z[ap]


@jit(nopython=True, cache=True)
def exac2_jac(
    data, indptr, indices, z, v, idxs, bus, power_injection, pss_input_idx, vref,
    gen_dp, gen_ap, TR, TB, TC, KA, TA, VAMAX, VAMIN, KB, VRMAX, VRMIN,
    TE, KL, KH, KF, TF, KC, KD, KE, VLRx, sat_a, sat_b,
    xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp, ap, dev = idxs[0], idxs[1], idxs[2]
    pss_input = z[pss_input_idx] if pss_input_idx >= 0 else 0.0
    signals = _signals(
        z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
        TR, TB, TC, KA, VAMAX, VAMIN, KB, VRMAX, VRMIN, KL, KH,
        KF, TF, KC, KD, KE, VLRx, sat_a, sat_b,
        xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
    )
    vm, _, dxad, _, dvfe_dve, _, _, _, _, gate_high, dvr_dgate, _, ratio, fex, dfex, _ = signals
    ve = z[dp + 3]
    wash_gain = 0.0 if TF == 0.0 or KF == 0.0 else KF / TF
    lead = TC / TB

    machine_cols = np.array((gen_dp, gen_dp + 1, gen_dp + 2, gen_dp + 3, gen_ap + 3))
    cols = np.empty(14, dtype=np.int64)
    vals = np.empty(14)
    vr_col = dev + 2 * bus
    vi_col = vr_col + 1
    if power_injection:
        dvm_r, dvm_i = 1.0, 0.0
    else:
        dvm_r = v[2 * bus] / vm
        dvm_i = v[2 * bus + 1] / vm

    cols[0], vals[0] = dp, (-1.0 if TR == 0.0 else -1.0 / TR)
    n = 1
    if TR != 0.0:
        cols[n], vals[n] = vr_col, dvm_r / TR
        n += 1
        if not power_injection:
            cols[n], vals[n] = vi_col, dvm_i / TR
            n += 1
    _set_row(data, indptr, indices, n, dp, cols, vals)

    sensed_vt = TR != 0.0
    cols[0], vals[0] = dp + 1, -1.0 / TB
    n = 1
    if sensed_vt:
        cols[n], vals[n] = dp, -1.0 / TB
        n += 1
    else:
        cols[n], vals[n] = vr_col, -dvm_r / TB
        n += 1
        if not power_injection:
            cols[n], vals[n] = vi_col, -dvm_i / TB
            n += 1
    cols[n], vals[n] = dp + 3, -wash_gain * dvfe_dve / TB
    n += 1
    cols[n], vals[n] = dp + 4, wash_gain / TB
    n += 1
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], -wash_gain * KD * dxad[i] / TB
        n += 1
    if pss_input_idx >= 0:
        cols[n], vals[n] = pss_input_idx, 1.0 / TB
        n += 1
    _set_row(data, indptr, indices, n, dp + 1, cols, vals)

    cols[0], vals[0] = dp + 2, -1.0 / TA
    cols[1], vals[1] = dp + 1, KA * (1.0 - lead) / TA
    n = 2
    error_gain = KA * lead / TA
    if sensed_vt:
        cols[n], vals[n] = dp, -error_gain
        n += 1
    else:
        cols[n], vals[n] = vr_col, -error_gain * dvm_r
        n += 1
        if not power_injection:
            cols[n], vals[n] = vi_col, -error_gain * dvm_i
            n += 1
    cols[n], vals[n] = dp + 3, -error_gain * wash_gain * dvfe_dve
    n += 1
    cols[n], vals[n] = dp + 4, error_gain * wash_gain
    n += 1
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], -error_gain * wash_gain * KD * dxad[i]
        n += 1
    if pss_input_idx >= 0:
        cols[n], vals[n] = pss_input_idx, error_gain
        n += 1
    _set_row(data, indptr, indices, n, dp + 2, cols, vals)

    gate_vfe_gain = -KH if gate_high else -KL
    vr_vfe_gain = dvr_dgate * gate_vfe_gain
    cols[0], vals[0] = dp + 2, dvr_dgate / TE if gate_high else 0.0
    cols[1], vals[1] = dp + 3, (vr_vfe_gain - 1.0) * dvfe_dve / TE
    n = 2
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], (vr_vfe_gain - 1.0) * KD * dxad[i] / TE
        n += 1
    _set_row(data, indptr, indices, n, dp + 3, cols, vals)

    if TF == 0.0 or KF == 0.0:
        cols[0], vals[0] = dp + 4, -1.0
        _set_row(data, indptr, indices, 1, dp + 4, cols, vals)
    else:
        cols[0], vals[0] = dp + 3, dvfe_dve / TF
        cols[1], vals[1] = dp + 4, -1.0 / TF
        n = 2
        for i in range(5):
            cols[n], vals[n] = machine_cols[i], KD * dxad[i] / TF
            n += 1
        _set_row(data, indptr, indices, n, dp + 4, cols, vals)

    doutput_dve = fex - dfex * ratio
    doutput_dxad = dfex * KC
    cols[0], vals[0] = dp + 3, doutput_dve
    cols[1], vals[1] = ap, -1.0
    n = 2
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], doutput_dxad * dxad[i]
        n += 1
    _set_row(data, indptr, indices, n, ap, cols, vals)


class ExcEXAC2(Exciter):
    output_is_algebraic = True
    bounded_state_metadata = (
        BoundedStateMetadata("VA", 2, 6, 5, 27, "EXAC2"),
    )

    def __init__(
        self, id_tag, generator, TR, TB, TC, KA, TA, VAMAX, VAMIN, KB,
        VRMAX, VRMIN, TE, KL, KH, KF, TF, KC, KD, KE, VLR, E1, SE1, E2, SE2,
    ):
        if TB <= 0.0 or TA <= 0.0 or TE <= 0.0 or KB == 0.0 or KL == 0.0:
            raise ValueError("EXAC2 requires positive TB, TA, TE and non-zero KB, KL.")
        self.generator = generator
        names = (
            "TR", "TB", "TC", "KA", "TA", "VAMAX", "VAMIN", "KB",
            "VRMAX", "VRMIN", "TE", "KL", "KH", "KF", "TF", "KC",
            "KD", "KE", "VLR", "E1", "SE1", "E2", "SE2",
        )
        values = (TR, TB, TC, KA, TA, VAMAX, VAMIN, KB, VRMAX, VRMIN, TE,
                  KL, KH, KF, TF, KC, KD, KE, VLR, E1, SE1, E2, SE2)
        for name, value in zip(names, values):
            setattr(self, name, value)
        self.sat_a, self.sat_b = esdc1a_sat_coefficients(E1, SE1, E2, SE2)
        self.vref = None
        self.VLRx = None
        self.efd_idx = 0
        parameter_list = list(names) + ["sat_a", "sat_b", "VLRx", "vref", "enable_limits"]
        state_list = ["vt", "ll", "va", "ve", "wf", "e_fd"]
        super().__init__(id_tag, 5, 5, 1, len(parameter_list), state_list)

    def _generator_args(self):
        gen = self.generator
        gen_sat_a, gen_sat_b = sat_coefficients(gen.S1, gen.S2)
        return (
            gen.dif_ptr, gen.alg_ptr + self._dif_size, gen.x_d, gen.x_dp,
            gen.x_qp, gen.x_ddp, gen.xl, gen_sat_a, gen_sat_b,
        )

    def initialize(self, vm, va, p, q, x, y, psys):
        self._dif_size = psys.num_dof_dif
        efd0 = self.e_fd0
        xad = efd0
        ve = max(abs(efd0), 0.1)
        for _ in range(30):
            ratio = self.KC * xad / ve
            fex, dfex = _rectifier(ratio)
            error = ve * fex - efd0
            derivative = fex - dfex * ratio
            if abs(error) < 1e-12:
                break
            if derivative == 0.0:
                raise ValueError("EXAC2 initialization has no rectifier solution.")
            ve -= error / derivative
            if ve <= 0.0:
                ve = 0.5 * max(abs(efd0), 0.1)
        if abs(ve * _rectifier(self.KC * xad / ve)[0] - efd0) > 1e-8:
            raise ValueError("EXAC2 initialization failed to solve the rectifier equation.")

        se = 0.0 if self.sat_b == 0.0 or ve <= self.sat_a else self.sat_b * (ve - self.sat_a) ** 2
        vfe = self.KE * ve + se + self.KD * xad
        low_gate_threshold = vfe + vfe / (self.KB * self.KL)
        self.VLRx = max(self.VLR, low_gate_threshold)
        if self.VLR >= low_gate_threshold:
            regulator = self.KH * vfe + vfe / self.KB
        else:
            regulator = self.KL * vfe + vfe / self.KB
        self.VAMAX = max(self.VAMAX, regulator)
        self.VAMIN = min(self.VAMIN, regulator)
        vi = regulator / self.KA
        self.vref = vm + vi
        x[self.dif_ptr:self.dif_ptr + 5] = (vm if self.TR != 0.0 else 0.0, vi, regulator, ve, vfe if self.TF != 0.0 and self.KF != 0.0 else 0.0)
        y[self.alg_ptr] = efd0
        self.initialized = True

    def initialize_theta(self, theta):
        values = (self.TR, self.TB, self.TC, self.KA, self.TA, self.VAMAX,
                  self.VAMIN, self.KB, self.VRMAX, self.VRMIN, self.TE, self.KL,
                  self.KH, self.KF, self.TF, self.KC, self.KD, self.KE, self.VLR,
                  self.E1, self.SE1, self.E2, self.SE2, self.sat_a, self.sat_b,
                  self.VLRx, self.vref, 1.0)
        theta[self.par_ptr:self.par_ptr + len(values)] = values

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        gen_dp, gen_ap, *machine = self._generator_args()
        exac2_resdiff(F, z, v, theta, idxs, self.bus, power_injection, pss,
                     self.vref,
                     gen_dp, gen_ap, self.TR, self.TB, self.TC, self.KA,
                     self.TA, self.VAMAX, self.VAMIN, self.KB, self.VRMAX,
                     self.VRMIN, self.TE, self.KL, self.KH, self.KF, self.TF,
                     self.KC, self.KD, self.KE, self.VLRx, self.sat_a, self.sat_b,
                     *machine)

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp, ap, dev = idxs[0], idxs[1], idxs[2]
        gen_dp = self.generator.dif_ptr
        gen_ap = psys.num_dof_dif + self.generator.alg_ptr
        machine = [gen_dp, gen_dp + 1, gen_dp + 2, gen_dp + 3, gen_ap + 3]
        vr, vi = dev + 2 * self.bus, dev + 2 * self.bus + 1
        rows = []
        rows.append([dp, sorted([dp] + ([] if self.TR == 0.0 else [vr] + ([] if power_injection else [vi])))])
        common = [dp + 1, dp + 3, dp + 4] + machine
        common += [dp] if self.TR != 0.0 else [vr] + ([] if power_injection else [vi])
        if self.pss_input_idx >= 0:
            common.append(self.pss_input_idx)
        rows.append([dp + 1, sorted(set(common))])
        rows.append([dp + 2, sorted(set(common + [dp + 2]))])
        rows.append([dp + 3, sorted([dp + 2, dp + 3] + machine)])
        rows.append([dp + 4, sorted([dp + 3, dp + 4] + machine)])
        rows.append([ap, sorted([dp + 3, ap] + machine)])
        return rows

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        gen_dp, gen_ap, *machine = self._generator_args()
        exac2_jac(J.data, J.indptr, J.indices, z, v, idxs, self.bus,
                  power_injection, self.pss_input_idx, self.vref, gen_dp, gen_ap,
                  self.TR, self.TB, self.TC, self.KA, self.TA, self.VAMAX,
                  self.VAMIN, self.KB, self.VRMAX, self.VRMIN, self.TE, self.KL,
                  self.KH, self.KF, self.TF, self.KC, self.KD, self.KE,
                  self.VLRx, self.sat_a, self.sat_b, *machine)
