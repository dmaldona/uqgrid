import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Exciter
from uqgrid.models.esdc1a_imp import esdc1a_sat_coefficients
from uqgrid.models.exac2_imp import _machine_field, _rectifier, _set_row
from uqgrid.models.genrou_imp import sat_coefficients


@jit(nopython=True, cache=True)
def _signals(
    z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
    TR, TB, TC, KA, KF, TF, KC, KD, KE, sat_a, sat_b,
    xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp = idxs[0]
    vt, ll, vr, ve, wf = z[dp:dp + 5]
    if power_injection:
        vm = v[2 * bus]
    else:
        real = v[2 * bus]
        imag = v[2 * bus + 1]
        vm = np.sqrt(real * real + imag * imag)
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
    error = vref - sensed - washout + pss_input
    ll_out = error if TB == 0.0 else ll + (TC / TB) * (error - ll)
    regulator_raw = KA * ll_out - vr

    ratio = KC * xad / ve
    fex, dfex = _rectifier(ratio)
    efd = ve * fex
    return (
        vm, dxad, vfe, dvfe_dve, error, ll_out, regulator_raw,
        ratio, fex, dfex, efd,
    )


@jit(nopython=True, cache=True)
def exac1_resdiff(
    F, z, v, theta, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
    TR, TB, TC, KA, TA, VRMAX, VRMIN, TE, KF, TF, KC, KD, KE, sat_a, sat_b,
    xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp, ap = idxs[0], idxs[1]
    vt, ll, vr, _, wf = z[dp:dp + 5]
    signals = _signals(
        z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
        TR, TB, TC, KA, KF, TF, KC, KD, KE, sat_a, sat_b,
        xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
    )
    vm, _, vfe, _, error, _, regulator_raw, _, _, _, efd = signals
    F[dp] = -vt if TR == 0.0 else (vm - vt) / TR
    F[dp + 1] = error - ll if TB == 0.0 else (error - ll) / TB
    F[dp + 2] = regulator_raw / TA
    F[dp + 3] = (vr - vfe) / TE
    F[dp + 4] = -wf if TF == 0.0 or KF == 0.0 else (vfe - wf) / TF
    F[ap] = efd - z[ap]


@jit(nopython=True, cache=True)
def exac1_jac(
    data, indptr, indices, z, v, idxs, bus, power_injection, pss_input_idx,
    vref, gen_dp, gen_ap, TR, TB, TC, KA, TA, VRMAX, VRMIN, TE, KF, TF,
    KC, KD, KE, sat_a, sat_b, xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp, ap, dev = idxs[0], idxs[1], idxs[2]
    pss_input = z[pss_input_idx] if pss_input_idx >= 0 else 0.0
    signals = _signals(
        z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
        TR, TB, TC, KA, KF, TF, KC, KD, KE, sat_a, sat_b,
        xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
    )
    vm, dxad, _, dvfe_dve, _, _, _, ratio, fex, dfex, _ = signals
    ve = z[dp + 3]
    wash_gain = 0.0 if TF == 0.0 or KF == 0.0 else KF / TF
    lead = 1.0 if TB == 0.0 else TC / TB
    ll_gain = 0.0 if TB == 0.0 else 1.0 - lead
    ll_scale = 1.0 if TB == 0.0 else 1.0 / TB

    machine_cols = np.array((gen_dp, gen_dp + 1, gen_dp + 2, gen_dp + 3, gen_ap + 3))
    cols = np.empty(14, dtype=np.int64)
    vals = np.empty(14)
    real_col = dev + 2 * bus
    imag_col = real_col + 1
    if power_injection:
        dvm_real, dvm_imag = 1.0, 0.0
    else:
        dvm_real = v[2 * bus] / vm
        dvm_imag = v[2 * bus + 1] / vm

    cols[0], vals[0] = dp, -1.0 if TR == 0.0 else -1.0 / TR
    n = 1
    if TR != 0.0:
        cols[n], vals[n] = real_col, dvm_real / TR
        n += 1
        if not power_injection:
            cols[n], vals[n] = imag_col, dvm_imag / TR
            n += 1
    _set_row(data, indptr, indices, n, dp, cols, vals)

    cols[0], vals[0] = dp + 1, -ll_scale
    n = 1
    if TR != 0.0:
        cols[n], vals[n] = dp, -ll_scale
        n += 1
    else:
        cols[n], vals[n] = real_col, -ll_scale * dvm_real
        n += 1
        if not power_injection:
            cols[n], vals[n] = imag_col, -ll_scale * dvm_imag
            n += 1
    cols[n], vals[n] = dp + 3, -ll_scale * wash_gain * dvfe_dve
    n += 1
    cols[n], vals[n] = dp + 4, ll_scale * wash_gain
    n += 1
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], -ll_scale * wash_gain * KD * dxad[i]
        n += 1
    if pss_input_idx >= 0:
        cols[n], vals[n] = pss_input_idx, ll_scale
        n += 1
    _set_row(data, indptr, indices, n, dp + 1, cols, vals)

    cols[0], vals[0] = dp + 2, -1.0 / TA
    n = 1
    if ll_gain != 0.0:
        cols[n], vals[n] = dp + 1, KA * ll_gain / TA
        n += 1
    error_gain = KA * lead / TA
    if TR != 0.0:
        cols[n], vals[n] = dp, -error_gain
        n += 1
    else:
        cols[n], vals[n] = real_col, -error_gain * dvm_real
        n += 1
        if not power_injection:
            cols[n], vals[n] = imag_col, -error_gain * dvm_imag
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

    cols[0], vals[0] = dp + 2, 1.0 / TE
    cols[1], vals[1] = dp + 3, -dvfe_dve / TE
    n = 2
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], -KD * dxad[i] / TE
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

    cols[0], vals[0] = dp + 3, fex - dfex * ratio
    cols[1], vals[1] = ap, -1.0
    n = 2
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], dfex * KC * dxad[i]
        n += 1
    _set_row(data, indptr, indices, n, ap, cols, vals)


class ExcEXAC1(Exciter):
    output_is_algebraic = True
    bounded_state_metadata = (
        BoundedStateMetadata("VR", 2, 6, 5, 20, "EXAC1"),
    )

    def __init__(
        self, id_tag, generator, TR, TB, TC, KA, TA, VRMAX, VRMIN, TE,
        KF, TF, KC, KD, KE, E1, SE1, E2, SE2,
    ):
        if TA <= 0.0 or TE <= 0.0:
            raise ValueError("EXAC1 requires positive TA and TE.")
        if TB < 0.0 or (TB == 0.0 and TC != 0.0):
            raise ValueError("EXAC1 requires non-negative TB and TC=0 when TB=0.")
        self.generator = generator
        names = (
            "TR", "TB", "TC", "KA", "TA", "VRMAX", "VRMIN", "TE",
            "KF", "TF", "KC", "KD", "KE", "E1", "SE1", "E2", "SE2",
        )
        values = (TR, TB, TC, KA, TA, VRMAX, VRMIN, TE, KF, TF, KC, KD,
                  KE, E1, SE1, E2, SE2)
        for name, value in zip(names, values):
            setattr(self, name, value)
        self.sat_a, self.sat_b = esdc1a_sat_coefficients(E1, SE1, E2, SE2)
        self.vref = None
        self.efd_idx = 0
        parameter_list = list(names) + ["sat_a", "sat_b", "vref", "enable_limits"]
        state_list = ["vt", "ll", "vr", "ve", "wf", "e_fd"]
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
                raise ValueError("EXAC1 initialization has no rectifier solution.")
            ve -= error / derivative
            if ve <= 0.0:
                ve = 0.5 * max(abs(efd0), 0.1)
        if abs(ve * _rectifier(self.KC * xad / ve)[0] - efd0) > 1e-8:
            raise ValueError("EXAC1 initialization failed to solve the rectifier equation.")

        se = 0.0 if self.sat_b == 0.0 or ve <= self.sat_a else self.sat_b * (ve - self.sat_a) ** 2
        vfe = self.KE * ve + se + self.KD * xad
        if not self.VRMIN <= vfe <= self.VRMAX:
            raise ValueError("EXAC1 initial regulator state is outside VRMIN/VRMAX.")
        error = vfe / self.KA
        self.vref = vm + error
        x[self.dif_ptr:self.dif_ptr + 5] = (
            vm if self.TR != 0.0 else 0.0,
            error,
            vfe,
            ve,
            vfe if self.TF != 0.0 and self.KF != 0.0 else 0.0,
        )
        y[self.alg_ptr] = efd0
        self.initialized = True

    def initialize_theta(self, theta):
        values = (
            self.TR, self.TB, self.TC, self.KA, self.TA, self.VRMAX,
            self.VRMIN, self.TE, self.KF, self.TF, self.KC, self.KD, self.KE,
            self.E1, self.SE1, self.E2, self.SE2, self.sat_a, self.sat_b,
            self.vref, 1.0,
        )
        theta[self.par_ptr:self.par_ptr + len(values)] = values

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        gen_dp, gen_ap, *machine = self._generator_args()
        exac1_resdiff(
            F, z, v, theta, idxs, self.bus, power_injection, pss, self.vref,
            gen_dp, gen_ap, self.TR, self.TB, self.TC, self.KA, self.TA,
            self.VRMAX, self.VRMIN, self.TE, self.KF, self.TF, self.KC,
            self.KD, self.KE, self.sat_a, self.sat_b, *machine,
        )

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp, ap, dev = idxs[0], idxs[1], idxs[2]
        gen_dp = self.generator.dif_ptr
        gen_ap = psys.num_dof_dif + self.generator.alg_ptr
        machine = [gen_dp, gen_dp + 1, gen_dp + 2, gen_dp + 3, gen_ap + 3]
        real, imag = dev + 2 * self.bus, dev + 2 * self.bus + 1
        sensed = [dp] if self.TR != 0.0 else [real] + ([] if power_injection else [imag])
        common = [dp + 3, dp + 4] + machine + sensed
        if self.pss_input_idx >= 0:
            common.append(self.pss_input_idx)
        regulator = common + [dp + 2]
        if self.TB != 0.0 and self.TC != self.TB:
            regulator.append(dp + 1)
        return [
            [dp, sorted([dp] + ([] if self.TR == 0.0 else [real] + ([] if power_injection else [imag])))],
            [dp + 1, sorted(set(common + [dp + 1]))],
            [dp + 2, sorted(set(regulator))],
            [dp + 3, sorted([dp + 2, dp + 3] + machine)],
            [dp + 4, sorted([dp + 3, dp + 4] + machine)],
            [ap, sorted([dp + 3, ap] + machine)],
        ]

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        gen_dp, gen_ap, *machine = self._generator_args()
        exac1_jac(
            J.data, J.indptr, J.indices, z, v, idxs, self.bus,
            power_injection, self.pss_input_idx, self.vref, gen_dp, gen_ap,
            self.TR, self.TB, self.TC, self.KA, self.TA, self.VRMAX,
            self.VRMIN, self.TE, self.KF, self.TF, self.KC, self.KD, self.KE,
            self.sat_a, self.sat_b, *machine,
        )
