import numpy as np
from numba import jit

from uqgrid.core.base_models import Exciter
from uqgrid.models.genrou_imp import sat_coefficients
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def esst4b_rectifier(current_ratio):
    """Return the commutation factor and its slope."""
    if current_ratio <= 0.0:
        return 1.0, 0.0
    if current_ratio <= 0.433:
        return 1.0 - 0.577 * current_ratio, -0.577
    if current_ratio <= 0.75:
        root = np.sqrt(max(0.75 - current_ratio * current_ratio, 0.0))
        return (root, -current_ratio / root) if root > 0.0 else (0.0, 0.0)
    if current_ratio <= 1.0:
        return 1.732 * (1.0 - current_ratio), -1.732
    return 0.0, 0.0


@jit(nopython=True, cache=True)
def _limited(value, lower, upper):
    if value <= lower:
        return lower, 0.0
    if value >= upper:
        return upper, 0.0
    return value, 1.0


@jit(nopython=True, cache=True)
def _antiwindup(limited_output, raw_derivative, lower, upper):
    if limited_output >= upper and raw_derivative > 0.0:
        return 0.0, 0.0
    if limited_output <= lower and raw_derivative < 0.0:
        return 0.0, 0.0
    return raw_derivative, 1.0


@jit(nopython=True, cache=True)
def _signals(z, v, idxs, bus, gen_dp, gen_ap, par, machine, power_injection, pss_input):
    dp, ap = idxs[0], idxs[1]
    TR, KPR, _, VRMAX, VRMIN, TA, KPM, _, VMMAX, VMMIN, KG, KP, KI, VBMAX, KC, XL, THETAP, vref, VGMAX = par
    v_sensed, xi_r, v_lag, xi_m = z[dp:dp + 4]
    e_fd = z[ap]
    vr, vi = v[2 * bus], v[2 * bus + 1]
    v_terminal = max(vr if power_injection else np.sqrt(vr * vr + vi * vi), 1e-12)
    sensed = v_sensed if TR > 0.0 else v_terminal
    error = vref - sensed + pss_input
    regulator_raw = KPR * error + xi_r
    regulator, _ = _limited(regulator_raw, VRMIN, VRMAX)
    lag_output = v_lag if TA > 0.0 else regulator
    feedback = min(KG * e_fd, VGMAX)
    inner_error = lag_output - feedback
    inner_raw = KPM * inner_error + xi_m
    inner, _ = _limited(inner_raw, VMMIN, VMMAX)

    x_d, x_dp, x_qp, x_ddp, xl, sat_a, sat_b = machine
    e_qp, e_dp, phi_1d, phi_2q = z[gen_dp:gen_dp + 4]
    v_q, v_d, i_q, i_d = z[gen_ap:gen_ap + 4]
    den = x_dp - xl
    qden = x_qp - xl
    psi_d = (x_ddp - xl) / den * e_qp + (x_dp - x_ddp) / den * phi_1d
    psi_q = -(x_ddp - xl) / qden * e_dp + (x_qp - x_ddp) / qden * phi_2q
    psi = np.sqrt(psi_d * psi_d + psi_q * psi_q)
    saturation = 0.0
    if sat_b != 0.0 and psi > sat_a and psi > 0.0:
        saturation = sat_b * (psi - sat_a) ** 2.0 / psi
    correction = i_d - (x_dp - x_ddp) * (-e_qp + i_d * den + phi_1d) / (den * den)
    field_current = e_qp + (x_d - x_dp) * correction + saturation * psi_d

    angle = THETAP * np.pi / 180.0
    kr, ki = KP * np.cos(angle), KP * np.sin(angle)
    br, bi = KI + XL * kr, XL * ki
    potential_r = kr * v_d - ki * v_q - bi * i_d - br * i_q
    potential_i = ki * v_d + kr * v_q + br * i_d - bi * i_q
    potential = max(np.sqrt(potential_r * potential_r + potential_i * potential_i), 1e-12)
    factor, _ = esst4b_rectifier(KC * field_current / potential)
    source_voltage = min(potential * factor, VBMAX)
    return (v_terminal, error, regulator_raw, regulator, lag_output,
            feedback, inner_error, inner_raw, inner, source_voltage)


@jit(nopython=True, cache=True)
def esst4b_resdiff(
    F, z, v, idxs, bus, gen_dp, gen_ap, par, machine, power_injection,
    pss_input=0.0,
):
    dp, ap = idxs[0], idxs[1]
    TR, _, KIR, _, _, TA, _, KIM = par[:8]
    v_sensed, _, v_lag, _ = z[dp:dp + 4]
    values = _signals(
        z, v, idxs, bus, gen_dp, gen_ap, par, machine, power_injection, pss_input
    )
    vt, error, _, regulator, _, _, inner_error, _, inner, source = values
    outer_derivative, _ = _antiwindup(
        regulator, KIR * error, par[4], par[3]
    )
    inner_derivative, _ = _antiwindup(
        inner, KIM * inner_error, par[9], par[8]
    )
    F[dp] = (vt - v_sensed) / TR if TR > 0.0 else 0.0
    F[dp + 1] = outer_derivative
    F[dp + 2] = (regulator - v_lag) / TA if TA > 0.0 else 0.0
    F[dp + 3] = inner_derivative
    F[ap] = source * inner - z[ap]


@jit(nopython=True, cache=True)
def esst4b_jac(
    data, indptr, indices, z, v, idxs, bus, gen_dp, gen_ap, par, machine,
    power_injection, pss_idx=-1,
):
    dp, ap, dev = idxs[0], idxs[1], idxs[2]
    TR, KPR, KIR, VRMAX, VRMIN, TA, KPM, KIM, VMMAX, VMMIN, KG, KP, KI, VBMAX, KC, XL, THETAP, _, VGMAX = par
    n = 16
    columns = np.empty(n, dtype=np.int64)
    for j in range(4):
        columns[j] = gen_dp + j
        columns[4 + j] = gen_ap + j
        columns[8 + j] = dp + j
    columns[12] = ap
    columns[13] = pss_idx
    columns[14] = dev + 2 * bus
    columns[15] = dev + 2 * bus + 1

    dvt = np.zeros(n)
    vr, vi = v[2 * bus], v[2 * bus + 1]
    vt = max(vr if power_injection else np.sqrt(vr * vr + vi * vi), 1e-12)
    if power_injection:
        dvt[14] = 1.0
    elif vr * vr + vi * vi > 0.0:
        dvt[14], dvt[15] = vr / vt, vi / vt
    dsensed = np.zeros(n)
    if TR > 0.0:
        dsensed[8] = 1.0
    else:
        dsensed[:] = dvt
    derror = -dsensed
    if pss_idx >= 0:
        derror[13] = 1.0
    draw_r = KPR * derror
    draw_r[9] += 1.0
    outer_error = (
        par[17] - (z[dp] if TR > 0.0 else vt)
        + (z[pss_idx] if pss_idx >= 0 else 0.0)
    )
    raw_r = KPR * outer_error + z[dp + 1]
    regulator, slope_r = _limited(raw_r, VRMIN, VRMAX)
    dregulator = slope_r * draw_r
    dlag = np.zeros(n)
    if TA > 0.0:
        dlag[10] = 1.0
    else:
        dlag[:] = dregulator
    e_fd = z[ap]
    dfeedback = np.zeros(n)
    feedback = min(KG * e_fd, VGMAX)
    if KG * e_fd <= VGMAX:
        dfeedback[12] = KG
    dinner_error = dlag - dfeedback
    inner_error = (z[dp + 2] if TA > 0.0 else regulator) - feedback
    draw_m = KPM * dinner_error
    draw_m[11] += 1.0
    raw_m = KPM * inner_error + z[dp + 3]
    inner, slope_m = _limited(raw_m, VMMIN, VMMAX)
    dinner = slope_m * draw_m
    _, outer_aw_slope = _antiwindup(
        regulator, KIR * outer_error, VRMIN, VRMAX
    )
    _, inner_aw_slope = _antiwindup(
        inner, KIM * inner_error, VMMIN, VMMAX
    )

    x_d, x_dp, x_qp, x_ddp, xl, sat_a, sat_b = machine
    e_qp, e_dp, phi_1d, phi_2q = z[gen_dp:gen_dp + 4]
    den = x_dp - xl
    qden = x_qp - xl
    ad, bd = (x_ddp - xl) / den, (x_dp - x_ddp) / den
    aq, bq = (x_ddp - xl) / qden, (x_qp - x_ddp) / qden
    psi_d, psi_q = ad * e_qp + bd * phi_1d, -aq * e_dp + bq * phi_2q
    psi = np.sqrt(psi_d * psi_d + psi_q * psi_q)
    dpsi_d = np.zeros(n)
    dpsi_d[0], dpsi_d[2] = ad, bd
    dpsi_q = np.zeros(n)
    dpsi_q[1], dpsi_q[3] = -aq, bq
    dpsi = np.zeros(n)
    if psi > 0.0:
        dpsi = (psi_d * dpsi_d + psi_q * dpsi_q) / psi
    saturation, dsat_dpsi = 0.0, 0.0
    if sat_b != 0.0 and psi > sat_a and psi > 0.0:
        gap = psi - sat_a
        saturation = sat_b * gap * gap / psi
        dsat_dpsi = sat_b * (2.0 * gap * psi - gap * gap) / (psi * psi)
    dcorrection = np.zeros(n)
    dcorrection[0] = (x_dp - x_ddp) / (den * den)
    dcorrection[2] = -(x_dp - x_ddp) / (den * den)
    dcorrection[7] = 1.0 - (x_dp - x_ddp) / den
    dfield = (x_d - x_dp) * dcorrection + saturation * dpsi_d + psi_d * dsat_dpsi * dpsi
    dfield[0] += 1.0
    correction = z[gen_ap + 3] - (x_dp - x_ddp) * (-e_qp + z[gen_ap + 3] * den + phi_1d) / (den * den)
    field = e_qp + (x_d - x_dp) * correction + saturation * psi_d

    angle = THETAP * np.pi / 180.0
    kr, ki = KP * np.cos(angle), KP * np.sin(angle)
    br, bi = KI + XL * kr, XL * ki
    v_q, v_d, i_q, i_d = z[gen_ap:gen_ap + 4]
    pr = kr * v_d - ki * v_q - bi * i_d - br * i_q
    pi = ki * v_d + kr * v_q + br * i_d - bi * i_q
    potential = max(np.sqrt(pr * pr + pi * pi), 1e-12)
    dpr, dpi = np.zeros(n), np.zeros(n)
    dpr[4], dpr[5], dpr[6], dpr[7] = -ki, kr, -br, -bi
    dpi[4], dpi[5], dpi[6], dpi[7] = kr, ki, -bi, br
    dpotential = (pr * dpr + pi * dpi) / potential
    ratio = KC * field / potential
    dratio = KC * (dfield / potential - field * dpotential / (potential * potential))
    factor, dfactor = esst4b_rectifier(ratio)
    source_raw = potential * factor
    dsource = factor * dpotential + potential * dfactor * dratio
    source = min(source_raw, VBMAX)
    if source_raw >= VBMAX:
        dsource[:] = 0.0

    rows = np.empty((5, n))
    rows[:] = 0.0
    if TR > 0.0:
        rows[0] = (dvt - dsensed) / TR
    rows[1] = outer_aw_slope * KIR * derror
    if TA > 0.0:
        rows[2] = (dregulator - dlag) / TA
    rows[3] = inner_aw_slope * KIM * dinner_error
    rows[4] = source * dinner + inner * dsource
    rows[4, 12] -= 1.0

    used = 16 if pss_idx >= 0 else 15
    if pss_idx < 0:
        columns[13] = columns[14]
        columns[14] = columns[15]
        for i in range(5):
            rows[i, 13] = rows[i, 14]
            rows[i, 14] = rows[i, 15]
    for i in range(1, used):
        column = columns[i]
        values = rows[:, i].copy()
        j = i - 1
        while j >= 0 and columns[j] > column:
            columns[j + 1] = columns[j]
            rows[:, j + 1] = rows[:, j]
            j -= 1
        columns[j + 1] = column
        rows[:, j + 1] = values
    row_ids = (dp, dp + 1, dp + 2, dp + 3, ap)
    for i in range(5):
        csr_set_row(data, indptr, indices, used, row_ids[i], columns, rows[i])


class ExcESST4B(Exciter):
    """ESST4B static excitation system."""

    output_is_algebraic = True

    def __init__(self, id_tag, generator, *parameters):
        if len(parameters) != 17:
            raise ValueError("ESST4B requires 17 parameters.")
        self.generator = generator
        names = ("TR", "KPR", "KIR", "VRMAX", "VRMIN", "TA", "KPM", "KIM",
                 "VMMAX", "VMMIN", "KG", "KP", "KI", "VBMAX", "KC", "XL", "THETAP")
        for name, value in zip(names, parameters):
            setattr(self, name, value)
        self.VGMAX = 20.0
        self.vref = None
        self.efd_idx = 0
        self._par = np.empty(19)
        self._machine = np.empty(7)
        self.VMMAX_original = self.VMMAX
        self.limit_initialization_diagnostics = None
        Exciter.__init__(self, id_tag, 0, 4, 1, 19,
                         ["v_sensed", "xi_r", "v_lag", "xi_m", "e_fd"])

    def _refresh_parameters(self):
        self._par[:] = (self.TR, self.KPR, self.KIR, self.VRMAX, self.VRMIN, self.TA,
                        self.KPM, self.KIM, self.VMMAX, self.VMMIN, self.KG, self.KP,
                        self.KI, self.VBMAX, self.KC, self.XL, self.THETAP, self.vref,
                        self.VGMAX)
        gen = self.generator
        sat_a, sat_b = sat_coefficients(gen.S1, gen.S2)
        self._machine[:] = (
            gen.x_d, gen.x_dp, gen.x_qp, gen.x_ddp, gen.xl, sat_a, sat_b
        )

    def initialize(self, vm, va, p, q, x, y, psys):
        if self.VRMIN >= self.VRMAX or self.VMMIN >= self.VMMAX:
            raise ValueError("ESST4B regulator limits must be strictly ordered.")
        gen = self.generator
        v_q, v_d, i_q, i_d = y[gen.alg_ptr:gen.alg_ptr + 4]
        kpc = self.KP * np.exp(1j * np.radians(self.THETAP))
        potential = abs(kpc * (v_d + 1j * v_q) + 1j * (self.KI + kpc * self.XL) * (i_d + 1j * i_q))
        if potential <= 0.0:
            raise ValueError("ESST4B potential-source voltage must be positive at initialization.")
        factor, _ = esst4b_rectifier(self.KC * self.e_fd0 / potential)
        source = min(potential * factor, self.VBMAX)
        if source <= 0.0:
            raise ValueError("ESST4B rectifier output must be positive at initialization.")
        inner = self.e_fd0 / source
        feedback = min(self.KG * self.e_fd0, self.VGMAX)
        if inner < self.VMMIN or not self.VRMIN <= feedback <= self.VRMAX:
            raise ValueError("ESST4B initial regulator output is outside its limits.")
        self.VMMAX = max(self.VMMAX_original, inner)
        self.limit_initialization_diagnostics = {
            "initial_inner_output": float(inner),
            "source_VMMAX": float(self.VMMAX_original),
            "effective_VMMAX": float(self.VMMAX),
            "upper_bound_adjusted": bool(self.VMMAX != self.VMMAX_original),
        }
        self.vref = vm
        x[self.dif_ptr:self.dif_ptr + 4] = (vm, feedback, feedback, inner)
        y[self.alg_ptr] = self.e_fd0
        self._refresh_parameters()
        self.initialized = True

    def initialize_theta(self, theta):
        self._refresh_parameters()
        theta[self.par_ptr:self.par_ptr + 19] = self._par

    def _gen_ap(self, idxs):
        return idxs[1] - self.alg_ptr + self.generator.alg_ptr

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        pss = z[self.pss_input_idx] if self.pss_input_idx >= 0 else 0.0
        esst4b_resdiff(F, z, v, idxs, self.bus, self.generator.dif_ptr,
                       self._gen_ap(idxs), theta[self.par_ptr:self.par_ptr + 19],
                       self._machine, power_injection, pss)

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def _columns(self, idxs):
        columns = list(range(self.generator.dif_ptr, self.generator.dif_ptr + 4))
        columns += list(range(self._gen_ap(idxs), self._gen_ap(idxs) + 4))
        columns += list(range(idxs[0], idxs[0] + 4)) + [idxs[1]]
        if self.pss_input_idx >= 0:
            columns.append(self.pss_input_idx)
        columns += [idxs[2] + 2 * self.bus, idxs[2] + 2 * self.bus + 1]
        return sorted(set(columns))

    def preallocate_jacobian(self, idxs, psys, power_injection):
        columns = self._columns(idxs)
        return [[row, columns] for row in range(idxs[0], idxs[0] + 4)] + [[idxs[1], columns]]

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        esst4b_jac(J.data, J.indptr, J.indices, z, v, idxs, self.bus,
                   self.generator.dif_ptr, self._gen_ap(idxs),
                    theta[self.par_ptr:self.par_ptr + 19], self._machine,
                    power_injection, self.pss_input_idx)
