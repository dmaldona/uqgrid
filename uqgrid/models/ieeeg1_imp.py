import numpy as np
from numba import jit

from uqgrid.core.base_models import BoundedStateMetadata, Governor
from uqgrid.utils.tools import csr_set_row


@jit(nopython=True, cache=True)
def ieeeg1_resdiff(F, z, theta, idxs, w_idx, has_secondary_output):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[2]

    K = theta[pp]
    T1 = theta[pp + 1]
    T2 = theta[pp + 2]
    T3 = theta[pp + 3]
    UO = theta[pp + 4]
    UC = theta[pp + 5]
    pref = theta[pp + 28]
    enable_limits = theta[pp + 31] != 0.0

    speed_error = -z[w_idx]
    if T1 == 0.0:
        F[dp] = 0.0
        conditioned = K * speed_error
    else:
        F[dp] = (speed_error - z[dp]) / T1
        conditioned = K * (z[dp] + T2 * F[dp])

    raw_rate = (conditioned + pref - z[dp + 1]) / T3
    F[dp + 1] = min(UO, max(UC, raw_rate)) if enable_limits else raw_rate

    signal = z[dp + 1]
    stage_values = np.empty(4, dtype=np.float64)
    for stage in range(4):
        state_offset = stage + 2
        time_constant = theta[pp + 8 + stage]
        if time_constant == 0.0:
            F[dp + state_offset] = 0.0
        else:
            F[dp + state_offset] = (signal - z[dp + state_offset]) / time_constant
            signal = z[dp + state_offset]
        stage_values[stage] = signal

    hp = 0.0
    lp = 0.0
    for stage in range(4):
        hp += theta[pp + 12 + 2 * stage] * stage_values[stage]
        lp += theta[pp + 13 + 2 * stage] * stage_values[stage]
    F[ap] = hp - z[ap]
    if has_secondary_output:
        F[ap + 1] = lp - z[ap + 1]


@jit(nopython=True, cache=True)
def _set_sorted_row(data, indptr, indices, row, columns, values, count):
    for left in range(1, count):
        column = columns[left]
        value = values[left]
        right = left - 1
        while right >= 0 and columns[right] > column:
            columns[right + 1] = columns[right]
            values[right + 1] = values[right]
            right -= 1
        columns[right + 1] = column
        values[right + 1] = value
    csr_set_row(data, indptr, indices, count, row, columns, values)


@jit(nopython=True, cache=True)
def ieeeg1_jac(data, indptr, indices, z, theta, idxs, w_idx, has_secondary_output):
    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[3]
    K = theta[pp]
    T1 = theta[pp + 1]
    T2 = theta[pp + 2]
    T3 = theta[pp + 3]
    UO = theta[pp + 4]
    UC = theta[pp + 5]
    pref = theta[pp + 28]
    enable_limits = theta[pp + 31] != 0.0

    columns = np.empty(6, dtype=np.int64)
    values = np.empty(6, dtype=np.float64)

    columns[0] = dp
    columns[1] = w_idx
    if T1 == 0.0:
        values[0] = 0.0
        values[1] = 0.0
        conditioned = K * -z[w_idx]
        dconditioned_dx = 0.0
        dconditioned_dw = -K
    else:
        values[0] = -1.0 / T1
        values[1] = -1.0 / T1
        derivative = (-z[w_idx] - z[dp]) / T1
        conditioned = K * (z[dp] + T2 * derivative)
        dconditioned_dx = K * (1.0 - T2 / T1)
        dconditioned_dw = -K * T2 / T1
    _set_sorted_row(data, indptr, indices, dp, columns, values, 2)

    raw_rate = (conditioned + pref - z[dp + 1]) / T3
    columns[0] = dp
    columns[1] = dp + 1
    columns[2] = w_idx
    if enable_limits and (raw_rate <= UC or raw_rate >= UO):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
    else:
        values[0] = dconditioned_dx / T3
        values[1] = -1.0 / T3
        values[2] = dconditioned_dw / T3
    _set_sorted_row(data, indptr, indices, dp + 1, columns, values, 3)

    source = dp + 1
    stage_sources = np.empty(4, dtype=np.int64)
    for stage in range(4):
        row = dp + stage + 2
        time_constant = theta[pp + 8 + stage]
        count = stage + 2
        for item in range(count):
            columns[item] = dp + 1 + item
            values[item] = 0.0
        if time_constant != 0.0:
            values[source - (dp + 1)] = 1.0 / time_constant
            values[count - 1] = -1.0 / time_constant
            source = row
        stage_sources[stage] = source
        csr_set_row(data, indptr, indices, count, row, columns, values)

    for output in range(1 + int(has_secondary_output)):
        row = ap + output
        for item in range(5):
            columns[item] = dp + 1 + item
            values[item] = 0.0
        for stage in range(4):
            coefficient = theta[pp + 12 + 2 * stage + output]
            values[stage_sources[stage] - (dp + 1)] += coefficient
        columns[5] = row
        values[5] = -1.0
        _set_sorted_row(data, indptr, indices, row, columns, values, 6)


class GovIEEEG1(Governor):
    """IEEEG1 steam-turbine governor with fixed differential dimensions."""

    bounded_state_metadata = (
        BoundedStateMetadata(
            state_name="valve_position",
            state_offset=1,
            lower_parameter_offset=30,
            upper_parameter_offset=29,
            enabled_parameter_offset=31,
            device_type="IEEEG1",
        ),
    )

    def __init__(
        self,
        id_tag,
        BUS2,
        ID2,
        K,
        T1,
        T2,
        T3,
        UO,
        UC,
        PMAX,
        PMIN,
        T4,
        K1,
        K2,
        T5,
        K3,
        K4,
        T6,
        K5,
        K6,
        T7,
        K7,
        K8,
        enable_limits=True,
        adjust_initial_limits=False,
    ):
        if T3 <= 0.0:
            raise ValueError("IEEEG1 T3 must be positive.")
        if any(value < 0.0 for value in (T1, T4, T5, T6, T7)):
            raise ValueError("IEEEG1 time constants must be non-negative.")
        if UC > 0.0 or UO < 0.0 or UC > UO:
            raise ValueError("IEEEG1 rate limits must satisfy UC <= 0 <= UO.")
        if PMIN >= PMAX:
            raise ValueError("IEEEG1 PMIN must be less than PMAX.")

        self.BUS2 = BUS2
        self.ID2 = ID2
        self.K = K
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.UO = UO
        self.UC = UC
        self.PMAX = PMAX
        self.PMIN = PMIN
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6
        self.T7 = T7
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.K5 = K5
        self.K6 = K6
        self.K7 = K7
        self.K8 = K8
        self.enable_limits = enable_limits
        self.adjust_initial_limits = adjust_initial_limits

        coefficient_sum = sum((K1, K2, K3, K4, K5, K6, K7, K8))
        if coefficient_sum == 0.0:
            raise ValueError("IEEEG1 K1-K8 must not all be zero.")
        self.normalized_K = tuple(
            value / coefficient_sum
            for value in (K1, K2, K3, K4, K5, K6, K7, K8)
        )
        for index, value in enumerate(self.normalized_K, start=1):
            setattr(self, f"K{index}n", value)

        self.has_secondary_output = not (
            BUS2 in (0, "0", None, "") and ID2 in (0, "0", None, "")
        )
        algebraic_dimension = 2 if self.has_secondary_output else 1
        if self.has_secondary_output:
            self.secondary_output_offset = 1

        self.effective_PMIN = PMIN
        self.effective_PMAX = PMAX
        self.limit_initialization_diagnostics = None
        self.pref = None

        parameter_list = [
            "K", "T1", "T2", "T3", "UO", "UC", "PMAX", "PMIN",
            "T4", "T5", "T6", "T7", "K1n", "K2n", "K3n", "K4n",
            "K5n", "K6n", "K7n", "K8n", "K1", "K2", "K3", "K4",
            "K5", "K6", "K7", "K8", "pref", "effective_PMAX",
            "effective_PMIN", "enable_limits",
        ]
        state_list = [
            "lead_lag", "valve_position", "stage4", "stage5", "stage6", "stage7"
        ]
        if self.has_secondary_output:
            state_list.extend(("p_m", "p_m_secondary"))
        else:
            state_list.append("p_m")
        Governor.__init__(
            self, id_tag, 7, 6, algebraic_dimension, len(parameter_list), state_list
        )

    def initialize(self, vm, va, p, q, x, y, psys):
        if self.p_m0 is None:
            raise ValueError("IEEEG1 requires p_m0 before initialization.")

        hp_fraction = sum(self.normalized_K[0::2])
        if hp_fraction == 0.0:
            raise ValueError("IEEEG1 primary output fraction must be non-zero.")
        valve = self.p_m0 / hp_fraction

        if self.has_secondary_output:
            if self.p_m0_secondary is None:
                raise ValueError("IEEEG1 requires p_m0_secondary for its second output.")
            lp_fraction = sum(self.normalized_K[1::2])
            if lp_fraction == 0.0:
                raise ValueError("IEEEG1 secondary output fraction must be non-zero.")
            secondary_valve = self.p_m0_secondary / lp_fraction
            if not np.isclose(valve, secondary_valve, rtol=1e-8, atol=1e-10):
                raise ValueError("IEEEG1 initial primary and secondary powers are inconsistent.")

        outside_bounds = valve < self.PMIN or valve > self.PMAX
        if self.enable_limits and outside_bounds and not self.adjust_initial_limits:
            raise ValueError(
                "IEEEG1 initial valve position is outside enabled PMIN/PMAX bounds."
            )
        self.pref = valve
        if self.enable_limits and self.adjust_initial_limits:
            self.effective_PMIN = min(self.PMIN, valve)
            self.effective_PMAX = max(self.PMAX, valve)
        else:
            self.effective_PMIN = self.PMIN
            self.effective_PMAX = self.PMAX
        self.limit_initialization_diagnostics = {
            "initial_valve": float(valve),
            "source_PMIN": float(self.PMIN),
            "source_PMAX": float(self.PMAX),
            "effective_PMIN": float(self.effective_PMIN),
            "effective_PMAX": float(self.effective_PMAX),
            "bounds_adjusted": bool(
                self.effective_PMIN != self.PMIN or self.effective_PMAX != self.PMAX
            ),
            "adjust_initial_limits": bool(self.adjust_initial_limits),
        }

        x[self.dif_ptr:self.dif_ptr + 6] = (0.0, valve, valve, valve, valve, valve)
        y[self.alg_ptr] = self.p_m0
        if self.has_secondary_output:
            y[self.alg_ptr + 1] = self.p_m0_secondary
        self.initialized = True

    def initialize_theta(self, theta):
        values = (
            self.K, self.T1, self.T2, self.T3, self.UO, self.UC,
            self.PMAX, self.PMIN, self.T4, self.T5, self.T6, self.T7,
            *self.normalized_K,
            self.K1, self.K2, self.K3, self.K4, self.K5, self.K6, self.K7, self.K8,
            self.pref, self.effective_PMAX, self.effective_PMIN,
            float(self.enable_limits),
        )
        theta[self.par_ptr:self.par_ptr + self.par_dim] = values

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        ieeeg1_resdiff(F, z, theta, idxs, self.w_idx, self.has_secondary_output)

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        dp = int(idxs[0])
        ap = int(idxs[1])
        coordinates = [
            [dp, sorted((dp, self.w_idx))],
            [dp + 1, sorted((dp, dp + 1, self.w_idx))],
        ]
        for stage in range(4):
            coordinates.append(
                [dp + stage + 2, list(range(dp + 1, dp + stage + 3))]
            )
        output_columns = list(range(dp + 1, dp + 6))
        coordinates.append([ap, sorted((*output_columns, ap))])
        if self.has_secondary_output:
            coordinates.append([ap + 1, sorted((*output_columns, ap + 1))])
        return coordinates

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        ieeeg1_jac(
            J.data, J.indptr, J.indices, z, theta, idxs, self.w_idx,
            self.has_secondary_output,
        )

    def preallocate_hessian(self, h_nnz, idxs, psys):
        raise NotImplementedError("IEEEG1 Hessian is not implemented.")

    def residual_hess(self, HESS, z, v, theta, idxs):
        raise NotImplementedError("IEEEG1 Hessian is not implemented.")
