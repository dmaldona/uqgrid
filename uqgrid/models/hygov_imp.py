import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Governor


@jit(nopython=True, cache=True)
def hygov_resdiff(
    F, z, theta, idxs, w_idx
):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[2]

    LG = z[dp]
    gtpos = z[dp + 1]
    g = z[dp + 2]
    q = z[dp + 3]
    p_m = z[ap]
    w = z[w_idx]
    R = theta[pp]
    r = theta[pp + 1]
    Tr = theta[pp + 2]
    Tf = theta[pp + 3]
    Tg = theta[pp + 4]
    VELM = theta[pp + 5]
    Tw = theta[pp + 8]
    At = theta[pp + 9]
    DT = theta[pp + 10]
    qNL = theta[pp + 11]
    g_floor = theta[pp + 12]
    enable_limits = theta[pp + 13] != 0.0
    pref = theta[pp + 14]

    filter_rate = (pref - w - R * gtpos - LG) / Tf
    raw_gate_rate = (LG + Tr * filter_rate) / (r * Tr)
    if enable_limits and raw_gate_rate <= -VELM:
        gtpos_rate = -VELM
    elif enable_limits and raw_gate_rate >= VELM:
        gtpos_rate = VELM
    else:
        gtpos_rate = raw_gate_rate

    if g <= g_floor:
        g_eff = g_floor
    else:
        g_eff = g
    h = q * q / (g_eff * g_eff)

    F[dp] = filter_rate
    F[dp + 1] = gtpos_rate
    F[dp + 2] = (gtpos - g) / Tg
    F[dp + 3] = (1.0 - h) / Tw
    F[ap] = At * h * (q - qNL) - DT * w * g - p_m


@jit(nopython=True, cache=True)
def hygov_jac(
    data, indptr, indices, z, theta, idxs, w_idx
):
    dp = idxs[0]
    ap = idxs[1]

    pp = idxs[3]
    R = theta[pp]
    r = theta[pp + 1]
    Tr = theta[pp + 2]
    Tf = theta[pp + 3]
    Tg = theta[pp + 4]
    VELM = theta[pp + 5]
    Tw = theta[pp + 8]
    At = theta[pp + 9]
    DT = theta[pp + 10]
    qNL = theta[pp + 11]
    g_floor = theta[pp + 12]
    enable_limits = theta[pp + 13] != 0.0

    LG = z[dp]
    g = z[dp + 2]
    q = z[dp + 3]
    w = z[w_idx]

    gtpos = z[dp + 1]
    filter_rate = (theta[pp + 14] - w - R * gtpos - LG) / Tf
    raw_gate_rate = (LG + Tr * filter_rate) / (r * Tr)
    if enable_limits and (raw_gate_rate <= -VELM or raw_gate_rate >= VELM):
        dgtpos_rate_dLG = 0.0
        dgtpos_rate_dgtpos = 0.0
        dgtpos_rate_dw = 0.0
    else:
        dgtpos_rate_dLG = (1.0 / Tr - 1.0 / Tf) / r
        dgtpos_rate_dgtpos = -R / (r * Tf)
        dgtpos_rate_dw = -1.0 / (r * Tf)

    if g <= g_floor:
        g_eff = g_floor
        dh_dg = 0.0
    else:
        g_eff = g
        dh_dg = -2.0 * q * q / (g * g * g)
    h = q * q / (g_eff * g_eff)
    dh_dq = 2.0 * q / (g_eff * g_eff)

    for ptr in range(indptr[dp], indptr[dp + 1]):
        col = indices[ptr]
        if col == dp:
            data[ptr] = -1.0 / Tf
        elif col == dp + 1:
            data[ptr] = -R / Tf
        elif col == w_idx:
            data[ptr] = -1.0 / Tf

    for ptr in range(indptr[dp + 1], indptr[dp + 2]):
        col = indices[ptr]
        if col == dp:
            data[ptr] = dgtpos_rate_dLG
        elif col == dp + 1:
            data[ptr] = dgtpos_rate_dgtpos
        elif col == w_idx:
            data[ptr] = dgtpos_rate_dw

    for ptr in range(indptr[dp + 2], indptr[dp + 3]):
        col = indices[ptr]
        if col == dp + 1:
            data[ptr] = 1.0 / Tg
        elif col == dp + 2:
            data[ptr] = -1.0 / Tg

    for ptr in range(indptr[dp + 3], indptr[dp + 4]):
        col = indices[ptr]
        if col == dp + 2:
            data[ptr] = -dh_dg / Tw
        elif col == dp + 3:
            data[ptr] = -dh_dq / Tw

    for ptr in range(indptr[ap], indptr[ap + 1]):
        col = indices[ptr]
        if col == dp + 2:
            data[ptr] = At * dh_dg * (q - qNL) - DT * w
        elif col == dp + 3:
            data[ptr] = At * (dh_dq * (q - qNL) + h)
        elif col == w_idx:
            data[ptr] = -DT * g
        elif col == ap:
            data[ptr] = -1.0


class GovHYGOV(Governor):
    """PSS/E HYGOV hydro governor with gate position and rate limits."""

    bounded_state_metadata = (
        BoundedStateMetadata(
            state_name="gtpos",
            state_offset=1,
            lower_parameter_offset=7,
            upper_parameter_offset=6,
            enabled_parameter_offset=13,
            device_type="HYGOV",
        ),
    )

    def __init__(
        self, id_tag, R, r, Tr, Tf, Tg, VELM, GMAX, GMIN, Tw, At, DT, qNL,
        g_floor, enable_limits=False, adjust_initial_limits=False,
    ):
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError("HYGOV r must be positive.")
        if not np.isfinite(Tr) or Tr <= 0.0:
            raise ValueError("HYGOV Tr must be positive.")
        if any(not np.isfinite(value) or value <= 0.0 for value in (Tf, Tg, Tw)):
            raise ValueError("HYGOV Tf, Tg, and Tw must be positive.")
        if not np.isfinite(At) or At <= 0.0:
            raise ValueError("HYGOV At must be positive.")
        if not np.isfinite(VELM) or VELM < 0.0:
            raise ValueError("HYGOV VELM must be non-negative.")
        if not np.isfinite(GMIN) or not np.isfinite(GMAX) or GMIN >= GMAX:
            raise ValueError("HYGOV GMIN must be less than GMAX.")
        if not np.isfinite(g_floor) or g_floor <= 0.0:
            raise ValueError("HYGOV g_floor must be positive.")

        self.R = R
        self.r = r
        self.Tr = Tr
        self.Tf = Tf
        self.Tg = Tg
        self.VELM = VELM
        self.GMAX_original = GMAX
        self.GMIN_original = GMIN
        self.GMAX = GMAX
        self.GMIN = GMIN
        self.Tw = Tw
        self.At = At
        self.DT = DT
        self.qNL = qNL
        self.g_floor = g_floor
        self.enable_limits = enable_limits
        self.adjust_initial_limits = adjust_initial_limits
        self.limit_initialization_diagnostics = None
        self.p_m_idx = 0

        parameter_list = [
            "R", "r", "Tr", "Tf", "Tg", "VELM", "GMAX", "GMIN", "Tw",
            "At", "DT", "qNL", "g_floor", "enable_limits", "pref",
        ]
        state_list = ["LG", "gtpos", "g", "q", "p_m"]
        Governor.__init__(self, id_tag, 5, 4, 1, len(parameter_list), state_list)

    def initialize(self, vm, va, p, q, x, y, psys):
        if self.p_m0 is None:
            raise ValueError("HYGOV requires p_m0 from the generator before initialization.")

        q0 = self.p_m0 / self.At + self.qNL
        outside_bounds = q0 < self.GMIN_original or q0 > self.GMAX_original
        if self.enable_limits and outside_bounds and not self.adjust_initial_limits:
            raise ValueError(
                "HYGOV initial gate position is outside enabled GMIN/GMAX bounds."
            )
        if self.enable_limits and self.adjust_initial_limits:
            self.GMIN = min(self.GMIN_original, q0)
            self.GMAX = max(self.GMAX_original, q0)
        else:
            self.GMIN = self.GMIN_original
            self.GMAX = self.GMAX_original
        self.limit_initialization_diagnostics = {
            "initial_gate_position": float(q0),
            "source_GMIN": float(self.GMIN_original),
            "source_GMAX": float(self.GMAX_original),
            "effective_GMIN": float(self.GMIN),
            "effective_GMAX": float(self.GMAX),
            "bounds_adjusted": bool(
                self.GMIN != self.GMIN_original or self.GMAX != self.GMAX_original
            ),
            "adjust_initial_limits": bool(self.adjust_initial_limits),
            "initialization_policy": (
                "adjust" if self.adjust_initial_limits else "strict"
            ),
        }
        self.pref = self.R * q0

        x[self.dif_ptr] = 0.0
        x[self.dif_ptr + 1] = q0
        x[self.dif_ptr + 2] = q0
        x[self.dif_ptr + 3] = q0
        y[self.alg_ptr] = self.p_m0
        self.initialized = True
        return None

    def initialize_theta(self, theta):
        idx = self.par_ptr
        theta[idx] = self.R
        theta[idx + 1] = self.r
        theta[idx + 2] = self.Tr
        theta[idx + 3] = self.Tf
        theta[idx + 4] = self.Tg
        theta[idx + 5] = self.VELM
        theta[idx + 6] = self.GMAX
        theta[idx + 7] = self.GMIN
        theta[idx + 8] = self.Tw
        theta[idx + 9] = self.At
        theta[idx + 10] = self.DT
        theta[idx + 11] = self.qNL
        theta[idx + 12] = self.g_floor
        theta[idx + 13] = float(self.enable_limits)
        theta[idx + 14] = self.pref

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        hygov_resdiff(
            F, z, theta, idxs, self.w_idx,
        )
        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp = idxs[0]
        ap = idxs[1]
        return [
            [dp, sorted({dp, dp + 1, self.w_idx})],
            [dp + 1, sorted({dp, dp + 1, self.w_idx})],
            [dp + 2, [dp + 1, dp + 2]],
            [dp + 3, [dp + 2, dp + 3]],
            [ap, sorted({dp + 2, dp + 3, self.w_idx, ap})],
        ]

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        hygov_jac(
            J.data, J.indptr, J.indices, z, theta, idxs, self.w_idx,
        )

    def preallocate_hessian(self, h_nnz, idxs, psys):
        raise NotImplementedError("HYGOV Hessian is not implemented.")

    def residual_hess(self, HESS, z, v, theta, idxs):
        raise NotImplementedError("HYGOV Hessian is not implemented.")
