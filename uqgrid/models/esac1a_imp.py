import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Exciter
from uqgrid.models.esdc1a_imp import esdc1a_sat_coefficients
from uqgrid.models.exac2_imp import _machine_field, _rectifier, _set_row
from uqgrid.models.genrou_imp import sat_coefficients


@jit(nopython=True, cache=True)
def _regulator_limit(value, lower, upper):
    if value <= lower:
        return lower, 0.0
    if value >= upper:
        return upper, 0.0
    return value, 1.0


@jit(nopython=True, cache=True)
def _signals(
    z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
    TR, TB, TC, KA, TE, VRMIN, VRMAX, KC, KD, KE, KF, TF,
    sat_a, sat_b, xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
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

    washout = KF * (vfe - wf) / TF
    sensed = vm if TR == 0.0 else vt
    error = vref - sensed - washout + pss_input
    if TB == 0.0:
        ll_out = error
    else:
        ll_out = ll + (TC / TB) * (error - ll)

    vr, dvr_dva = _regulator_limit(va, VRMIN, VRMAX)

    ratio = KC * xad / ve
    fex, dfex = _rectifier(ratio)
    efd = ve * fex
    return (
        vm, dxad, vfe, dvfe_dve, error, ll_out, ratio, fex, dfex, efd,
        vr, dvr_dva,
    )


@jit(nopython=True, cache=True)
def esac1a_resdiff(
    F, z, v, theta, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
    TR, TB, TC, KA, TA, TE, VRMIN, VRMAX, KC, KD, KE, KF, TF,
    sat_a, sat_b, xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp, ap = idxs[0], idxs[1]
    vt, ll, va, _, wf = z[dp:dp + 5]
    signals = _signals(
        z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
        TR, TB, TC, KA, TE, VRMIN, VRMAX, KC, KD, KE, KF, TF,
        sat_a, sat_b, xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
    )
    vm, _, vfe, _, error, ll_out, _, _, _, efd, vr, _ = signals
    F[dp] = -vt if TR == 0.0 else (vm - vt) / TR
    F[dp + 1] = -ll if TB == 0.0 else (error - ll) / TB
    F[dp + 2] = (KA * ll_out - va) / TA
    F[dp + 3] = (vr - vfe) / TE
    F[dp + 4] = (vfe - wf) / TF
    F[ap] = efd - z[ap]


@jit(nopython=True, cache=True)
def esac1a_jac(
    data, indptr, indices, z, v, idxs, bus, power_injection, pss_input_idx,
    vref, gen_dp, gen_ap, TR, TB, TC, KA, TA, TE, VRMIN, VRMAX,
    KC, KD, KE, KF, TF, sat_a, sat_b,
    xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
):
    dp, ap, dev = idxs[0], idxs[1], idxs[2]
    pss_input = z[pss_input_idx] if pss_input_idx >= 0 else 0.0
    signals = _signals(
        z, v, idxs, bus, power_injection, pss_input, vref, gen_dp, gen_ap,
        TR, TB, TC, KA, TE, VRMIN, VRMAX, KC, KD, KE, KF, TF,
        sat_a, sat_b, xd, xdp, xqp, xddp, xl, gen_sat_a, gen_sat_b,
    )
    vm, dxad, _, dvfe_dve, _, _, ratio, fex, dfex, _, _, dvr_dva = signals
    machine_cols = np.array((gen_dp, gen_dp + 1, gen_dp + 2, gen_dp + 3, gen_ap + 3))
    vr_col = dev + 2 * bus
    vi_col = vr_col + 1
    if power_injection:
        dvm_r, dvm_i = 1.0, 0.0
    else:
        dvm_r = v[2 * bus] / vm
        dvm_i = v[2 * bus + 1] / vm

    cols = np.empty(14, dtype=np.int64)
    vals = np.empty(14)
    cols[0], vals[0] = dp, (-1.0 if TR == 0.0 else -1.0 / TR)
    n = 1
    if TR != 0.0:
        cols[n], vals[n] = vr_col, dvm_r / TR
        n += 1
        if not power_injection:
            cols[n], vals[n] = vi_col, dvm_i / TR
            n += 1
    _set_row(data, indptr, indices, n, dp, cols, vals)

    wash_gain = KF / TF
    sensed_vt = TR != 0.0
    if TB == 0.0:
        cols[0], vals[0] = dp + 1, -1.0
        _set_row(data, indptr, indices, 1, dp + 1, cols, vals)
    else:
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

    lead = 1.0 if TB == 0.0 else TC / TB
    cols[0], vals[0] = dp + 2, -1.0 / TA
    n = 1
    if TB != 0.0:
        cols[n], vals[n] = dp + 1, KA * (1.0 - lead) / TA
        n += 1
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

    cols[0], vals[0] = dp + 2, dvr_dva / TE
    cols[1], vals[1] = dp + 3, -dvfe_dve / TE
    n = 2
    for i in range(5):
        cols[n], vals[n] = machine_cols[i], -KD * dxad[i] / TE
        n += 1
    _set_row(data, indptr, indices, n, dp + 3, cols, vals)

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


class ExcESAC1A(Exciter):
    """Alternating-current commutator excitation system."""

    output_is_algebraic = True
    bounded_state_metadata = (
        BoundedStateMetadata("VA", 2, 4, 3, 23, "ESAC1A"),
    )

    def __init__(
        self, id_tag, generator, TR, TB, TC, VAMAX, VAMIN, KA, TA,
        VRMAX, VRMIN, TE, E1, SE1, E2, SE2, KC, KD, KE, KF, TF,
    ):
        if TR < 0.0 or TB < 0.0 or TA <= 0.0 or TE <= 0.0 or TF <= 0.0:
            raise ValueError("ESAC1A requires TR, TB >= 0 and TA, TE, TF > 0.")
        if TB == 0.0 and TC != 0.0:
            raise ValueError("ESAC1A requires TC = 0 when TB = 0.")
        if KA == 0.0:
            raise ValueError("ESAC1A KA must be non-zero.")
        if VAMIN >= VAMAX:
            raise ValueError("ESAC1A VAMIN must be less than VAMAX.")
        effective_vrmax = 999.0 if VRMAX == 0.0 else VRMAX
        if VRMIN >= effective_vrmax:
            raise ValueError("ESAC1A VRMIN must be less than the effective VRMAX.")

        self.generator = generator
        names = (
            "TR", "TB", "TC", "VAMAX", "VAMIN", "KA", "TA", "VRMAX",
            "VRMIN", "TE", "E1", "SE1", "E2", "SE2", "KC", "KD", "KE",
            "KF", "TF",
        )
        values = (
            TR, TB, TC, VAMAX, VAMIN, KA, TA, VRMAX, VRMIN, TE,
            E1, SE1, E2, SE2, KC, KD, KE, KF, TF,
        )
        for name, value in zip(names, values):
            setattr(self, name, value)
        self.sat_a, self.sat_b = esdc1a_sat_coefficients(E1, SE1, E2, SE2)
        self.effective_vrmax = effective_vrmax
        self.enable_limits = True
        self.vref = None
        self.efd_idx = 0
        parameter_list = list(names) + [
            "sat_a", "sat_b", "effective_vrmax", "vref", "enable_limits",
        ]
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
            if abs(error) < 1e-12:
                break
            derivative = fex - dfex * ratio
            if derivative == 0.0:
                raise ValueError("ESAC1A initialization has no rectifier solution.")
            ve -= error / derivative
            if ve <= 0.0:
                ve = 0.5 * max(abs(efd0), 0.1)
        if abs(ve * _rectifier(self.KC * xad / ve)[0] - efd0) > 1e-8:
            raise ValueError("ESAC1A initialization failed to solve the rectifier equation.")

        se = 0.0 if self.sat_b == 0.0 or ve <= self.sat_a else self.sat_b * (ve - self.sat_a) ** 2
        vfe = self.KE * ve + se + self.KD * xad
        if not self.VAMIN <= vfe <= self.VAMAX:
            raise ValueError("ESAC1A initial regulator state is outside VAMIN/VAMAX.")
        if not self.VRMIN <= vfe <= self.effective_vrmax:
            raise ValueError(
                "ESAC1A initial regulator signal is outside effective VRMIN/VRMAX."
            )
        error = vfe / self.KA
        self.vref = vm + error
        x[self.dif_ptr:self.dif_ptr + 5] = (
            vm if self.TR != 0.0 else 0.0,
            error if self.TB != 0.0 else 0.0,
            vfe,
            ve,
            vfe,
        )
        y[self.alg_ptr] = efd0
        self.initialized = True

    def initialize_theta(self, theta):
        values = (
            self.TR, self.TB, self.TC, self.VAMAX, self.VAMIN, self.KA,
            self.TA, self.VRMAX, self.VRMIN, self.TE, self.E1, self.SE1,
            self.E2, self.SE2, self.KC, self.KD, self.KE, self.KF, self.TF,
            self.sat_a, self.sat_b, self.effective_vrmax, self.vref,
            float(self.enable_limits),
        )
        theta[self.par_ptr:self.par_ptr + len(values)] = values

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        gen_dp, gen_ap, *machine = self._generator_args()
        esac1a_resdiff(
            F, z, v, theta, idxs, self.bus, power_injection, pss, self.vref,
            gen_dp, gen_ap, self.TR, self.TB, self.TC, self.KA, self.TA,
            self.TE, self.VRMIN, self.effective_vrmax, self.KC, self.KD,
            self.KE, self.KF, self.TF, self.sat_a, self.sat_b, *machine,
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
        vr, vi = dev + 2 * self.bus, dev + 2 * self.bus + 1
        rows = [[dp, sorted([dp] + ([] if self.TR == 0.0 else [vr] + ([] if power_injection else [vi])))]]
        if self.TB == 0.0:
            rows.append([dp + 1, [dp + 1]])
        else:
            common = [dp + 1, dp + 3, dp + 4] + machine
            common += [dp] if self.TR != 0.0 else [vr] + ([] if power_injection else [vi])
            if self.pss_input_idx >= 0:
                common.append(self.pss_input_idx)
            rows.append([dp + 1, sorted(set(common))])
        regulator = [dp + 2, dp + 3, dp + 4] + machine
        if self.TB != 0.0:
            regulator.append(dp + 1)
        regulator += [dp] if self.TR != 0.0 else [vr] + ([] if power_injection else [vi])
        if self.pss_input_idx >= 0:
            regulator.append(self.pss_input_idx)
        rows.append([dp + 2, sorted(set(regulator))])
        rows.append([dp + 3, sorted([dp + 2, dp + 3] + machine)])
        rows.append([dp + 4, sorted([dp + 3, dp + 4] + machine)])
        rows.append([ap, sorted([dp + 3, ap] + machine)])
        return rows

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        gen_dp, gen_ap, *machine = self._generator_args()
        esac1a_jac(
            J.data, J.indptr, J.indices, z, v, idxs, self.bus,
            power_injection, self.pss_input_idx, self.vref, gen_dp, gen_ap,
            self.TR, self.TB, self.TC, self.KA, self.TA,
            self.TE, self.VRMIN, self.effective_vrmax, self.KC, self.KD,
            self.KE, self.KF, self.TF, self.sat_a, self.sat_b, *machine,
        )
