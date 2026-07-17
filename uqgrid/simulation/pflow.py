import numpy as np
import networkx as nx
from numba import jit
from scipy.sparse import csr_matrix
from scipy import optimize
import logging
try:
    from scipy.optimize._nonlin import nonlin_solve # For newer SciPy
except ImportError:
    from scipy.optimize.nonlin import nonlin_solve # Fallback for older SciPy
from uqgrid.core.psydef import Psystem, Bus

logger = logging.getLogger(__name__)

class PowerFlowSolution:
    def __init__(self, num_buses, num_gens):
        # Voltages
        self.v_magnitudes = np.zeros(num_buses)
        self.v_angles = np.zeros(num_buses) # in radians

        # Original flat voltage vector (Vm, Va, Vm, Va, ...)
        self.v_vector = np.zeros(2 * num_buses)
        # Original power injection vector (P, Q, P, Q, ...)
        self.s_inj_vector = np.zeros(2 * num_buses)

        # Updated generator setpoints (for slack and PV buses)
        # Store as dictionaries mapping generator original index to value
        self.gen_psch = np.zeros(num_gens)
        self.gen_qsch = np.zeros(num_gens)

        # Q-limit active-set diagnostics. These remain populated with the
        # original bus types and empty events when limit enforcement is off.
        self.bus_types = np.zeros(num_buses, dtype=np.int64)
        self.q_limit_enforced = False
        self.q_limit_iterations = 0
        self.q_limit_events = []
        self.residual_norm = None
        self.validation = None

    def __str__(self):
        return (f"PowerFlowSolution:\n"
                f"  V_magnitudes: {self.v_magnitudes[:3]}... ({len(self.v_magnitudes)} buses)\n"
                f"  V_angles: {self.v_angles[:3]}... ({len(self.v_angles)} buses)\n"
                f"  Gen Ps Psch entries: {len(self.gen_psch)}\n"
                f"  Gen Qs Qsch entries: {len(self.gen_qsch)}")


class PowerFlowValidationError(RuntimeError):
    """Raised when an enabled final operating-point validation fails."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        reasons = ", ".join(diagnostics.get("failure_reasons", []))
        super().__init__(
            "Power-flow operating-point validation failed"
            + (f": {reasons}" if reasons else "")
        )


@jit(nopython=True, cache=True)
def resfun(F, x, vmag, vang, Pinj, Qinj, ybus_mat,
           bus_type, PQ_idx, PQV_idx, graph_mat):

    # The first step is to susbtitute back the vmag and vang unknown variables
    # from x to 'vmag' and 'vang'. It might seem confusing to mix in the same
    # vector unknown variables and parameters. However, this makes writing the
    # equations cleaner.

    nPQ = np.sum(bus_type == 1)
    nbus = len(bus_type)

    for i in range(nbus):
        if PQ_idx[i] >= 0:
            vmag[i] = x[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = x[nPQ + PQV_idx[i]]

    for fr in range(nbus):
        if PQ_idx[fr] >= 0:
            F[PQ_idx[fr]] -= Qinj[fr]

            # self contribution
            bij = ybus_mat[fr, 0].imag
            F[PQ_idx[fr]] -= vmag[fr]*vmag[fr]*bij

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]

                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag

                angleij = vang[fr] - vang[to]

                F[PQ_idx[fr]] += vmag[fr]*vmag[to]*(gij*np.sin(angleij)
                                 - bij*np.cos(angleij))

        if PQV_idx[fr] >= 0:
            F[nPQ + PQV_idx[fr]] -= Pinj[fr]

            # self contribution
            gij = ybus_mat[fr, 0].real
            F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[fr]*gij

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]

                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag

                angleij = vang[fr] - vang[to]

                F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[to]*(gij*np.cos(angleij)
                                        + bij*np.sin(angleij))
    return F


def resfun_wrapper(x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type,
                   PQ_idx, PQV_idx, graph_mat):
    F = np.zeros(len(x))
    resfun(F, x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type,
           PQ_idx, PQV_idx, graph_mat)
    return F


@jit(nopython=True, cache=True)
def compute_jac_nnz(graph_mat, PQ_idx, PQV_idx):
    nnz = 0
    for i in range(graph_mat.shape[0]):
        if PQ_idx[i] >= 0:
            nnz += 4
        elif PQV_idx[i] >= 0:
            nnz += 1
        else:
            continue

        for j in range(graph_mat[i, 0]):
            to = graph_mat[i, j + 1]
            if PQ_idx[to] >= 0 and PQ_idx[i] >= 0:
                nnz += 4
            elif PQ_idx[to] >= 0 and PQV_idx[i] >= 0:
                nnz += 2
            elif PQV_idx[to] >= 0 and PQ_idx[i] >= 0:
                nnz += 2
            elif PQV_idx[to] >= 0 and PQV_idx[i] >= 0:
                nnz += 1
    return nnz

@jit(nopython=True, cache=True)
def fill_jacobian(x, vmag, vang, Pinj, Qinj, ybus_mat,
        bus_type, PQ_idx, PQV_idx, graph_mat,
        row, col, val):
    ptr = 0
    nPQ = np.sum(bus_type == 1)
    nbus = len(bus_type)

    for i in range(nbus):
        if PQ_idx[i] >= 0:
            vmag[i] = x[PQ_idx[i]]

        if PQV_idx[i] >= 0:
            vang[i] = x[nPQ + PQV_idx[i]]

    for (fr, elem) in enumerate(graph_mat):
        if PQ_idx[fr] >= 0:
            # self contribution
            vmag_fr_idx = PQ_idx[fr]
            vang_fr_idx = nPQ + PQV_idx[fr]

            bij = ybus_mat[fr, 0].imag

            accum_self_vmag = -2*vmag[fr]*bij
            accum_self_vang = 0.0

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag

                angleij = vang[fr] - vang[to]

                accum_self_vmag += vmag[to]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                accum_self_vang += vmag[fr]*vmag[to]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))

                if PQV_idx[to] >= 0:
                    vang_to_idx = nPQ + PQV_idx[to]
                    row[ptr] = PQ_idx[fr]
                    col[ptr] = vang_to_idx
                    val[ptr] = vmag[fr]*vmag[to]*(-gij*np.cos(angleij)
                        - bij*np.sin(angleij))
                    ptr += 1

                if PQ_idx[to] >= 0:
                    vmag_to_idx = PQ_idx[to]
                    row[ptr] = PQ_idx[fr]
                    col[ptr] = vmag_to_idx
                    val[ptr] = vmag[fr]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                    ptr += 1

            row[ptr] = PQ_idx[fr]
            col[ptr] = vmag_fr_idx
            val[ptr] = accum_self_vmag
            ptr += 1

            row[ptr] = PQ_idx[fr]
            col[ptr] = vang_fr_idx
            val[ptr] = accum_self_vang
            ptr += 1

        if PQV_idx[fr] >= 0:
            # self contribution
            gij = ybus_mat[fr, 0].real
            #F[nPQ + PQV_idx[fr]] += vmag[fr]*vmag[fr]*gij

            bij = ybus_mat[fr, 0].imag

            accum_self_vmag = 2*vmag[fr]*gij
            accum_self_vang = 0.0

            for j in range(graph_mat[fr, 0]):
                to = graph_mat[fr, j + 1]
                gij = ybus_mat[fr, j + 1].real
                bij = ybus_mat[fr, j + 1].imag
                angleij = vang[fr] - vang[to]
                accum_self_vmag += vmag[to]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))
                accum_self_vang += vmag[fr]*vmag[to]*(-gij*np.sin(angleij)
                        + bij*np.cos(angleij))

                if PQV_idx[to] >= 0:
                    vang_to_idx = nPQ + PQV_idx[to]
                    row[ptr] = nPQ + PQV_idx[fr]
                    col[ptr] = vang_to_idx
                    val[ptr] = vmag[fr]*vmag[to]*(gij*np.sin(angleij)
                        - bij*np.cos(angleij))
                    ptr += 1

                if PQ_idx[to] >= 0:
                    vmag_to_idx = PQ_idx[to]
                    row[ptr] = nPQ + PQV_idx[fr]
                    col[ptr] = vmag_to_idx
                    val[ptr] = vmag[fr]*(gij*np.cos(angleij)
                        + bij*np.sin(angleij))
                    ptr += 1

            if PQ_idx[fr] >= 0:
                vmag_fr_idx = PQ_idx[fr]
                row[ptr] = nPQ + PQV_idx[fr]
                col[ptr] = vmag_fr_idx
                val[ptr] = accum_self_vmag
                ptr += 1

            vang_fr_idx = nPQ + PQV_idx[fr]
            row[ptr] = nPQ + PQV_idx[fr]
            col[ptr] = vang_fr_idx
            val[ptr] = accum_self_vang
            ptr += 1

def jac_wrapper(x, vmag, vang, Pinj, Qinj, ybus_mat, bus_type, PQ_idx, PQV_idx, graph_mat):
    nnz = compute_jac_nnz(graph_mat, PQ_idx, PQV_idx)
    row = np.zeros(nnz)
    col = np.zeros(nnz)
    val = np.zeros(nnz)
    fill_jacobian(x, vmag, vang, Pinj, Qinj, ybus_mat,
        bus_type, PQ_idx, PQV_idx, graph_mat,
        row, col, val)
    J =  csr_matrix((val, (row, col)), shape=(x.shape[0], x.shape[0]))
    return J

@jit(nopython=True, cache=True)
def compute_pinj_alt(v, Sinj, ybus_mat, graph_mat, nbus):
    """ Same as above but v and Sinj alternate """

    for fr_bus in range(nbus):

        Sinj[2*fr_bus] = 0.0 # P
        Sinj[2*fr_bus + 1] = 0.0 # Q

        vmag_i = v[2*fr_bus]
        vang_i = v[2*fr_bus + 1]
        angleij = 0.0

        gij = ybus_mat[fr_bus, 0].real
        bij = ybus_mat[fr_bus, 0].imag

        Sinj[2*fr_bus] += vmag_i*vmag_i*(gij*np.cos(angleij)
            + bij*np.sin(angleij))

        Sinj[2*fr_bus + 1] += vmag_i*vmag_i*(gij*np.sin(angleij)
            - bij*np.cos(angleij))

        for j in range(graph_mat[fr_bus, 0]):

            to_bus = graph_mat[fr_bus, j + 1]

            gij = ybus_mat[fr_bus, j + 1].real
            bij = ybus_mat[fr_bus, j + 1].imag

            vmag_j = v[2*to_bus]
            vang_j = v[2*to_bus + 1]

            angleij = vang_i - vang_j

            Sinj[2*fr_bus] += vmag_i*vmag_j*(gij*np.cos(angleij)
                + bij*np.sin(angleij))

            Sinj[2*fr_bus + 1] += vmag_i*vmag_j*(gij*np.sin(angleij)
                - bij*np.cos(angleij))


def _project_reactive_dispatch(total_q, lower, upper, tolerance=1e-10):
    """Project equal Q sharing onto generator bounds while preserving the sum."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("Reactive-power lower and upper bounds must have matching shapes")
    if lower.size == 0:
        raise ValueError("Cannot dispatch reactive power at a bus with no generators")
    if np.any(lower > upper):
        raise ValueError("Reactive-power lower bound exceeds upper bound")

    lower_sum = float(np.sum(lower))
    upper_sum = float(np.sum(upper))
    if total_q < lower_sum - tolerance or total_q > upper_sum + tolerance:
        raise ValueError(
            f"Reactive-power target {total_q:.6e} is outside aggregate bounds "
            f"[{lower_sum:.6e}, {upper_sum:.6e}]"
        )

    target = float(np.clip(total_q, lower_sum, upper_sum))
    dispatch = np.zeros_like(lower)
    free = np.ones(lower.size, dtype=bool)
    remaining = target

    while np.any(free):
        share = remaining / int(np.sum(free))
        below = free & (share < lower)
        above = free & (share > upper)

        if not np.any(below) and not np.any(above):
            dispatch[free] = share
            break

        if np.any(below):
            dispatch[below] = lower[below]
            remaining -= float(np.sum(dispatch[below]))
            free[below] = False
        if np.any(above):
            dispatch[above] = upper[above]
            remaining -= float(np.sum(dispatch[above]))
            free[above] = False

    mismatch = target - float(np.sum(dispatch))
    if abs(mismatch) > tolerance:
        if mismatch > 0.0:
            adjustable = np.where(dispatch < upper - tolerance)[0]
        else:
            adjustable = np.where(dispatch > lower + tolerance)[0]
        if adjustable.size == 0:
            raise RuntimeError("Reactive-power projection failed to preserve its sum")
        dispatch[adjustable[0]] += mismatch

    return dispatch


def _power_flow_indices(bus_type):
    """Return Newton unknown indices for the supplied working bus types."""
    pq_bus = np.where(bus_type == Bus.PQ, 1, 0)
    pq_idx = np.where(pq_bus == 1, np.cumsum(pq_bus), pq_bus) - 1

    pqv_bus = (
        np.where(bus_type == Bus.PQ, 1, 0)
        + np.where(bus_type == Bus.PV, 1, 0)
    )
    pqv_idx = np.where(pqv_bus == 1, np.cumsum(pqv_bus), pqv_bus) - 1
    return pq_idx, pqv_idx


def _solve_power_flow_once(psys, bus_type, vmag, vang, Pinj, Qinj, verbose, pass_idx):
    """Solve one Newton power flow for a fixed active set of bus types."""
    nslack = int(np.sum(bus_type == Bus.SLACK))
    npv = int(np.sum(bus_type == Bus.PV))
    npq = int(np.sum(bus_type == Bus.PQ))

    if verbose:
        logger.info(
            "Solving power flow pass %d with nslack: %d, nPV: %d, nPQ: %d",
            pass_idx,
            nslack,
            npv,
            npq,
        )

    pq_idx, pqv_idx = _power_flow_indices(bus_type)
    x0 = np.zeros(2*npq + npv)

    for i in range(psys.nbuses):
        if pq_idx[i] >= 0:
            x0[pq_idx[i]] = vmag[i]
        if pqv_idx[i] >= 0:
            x0[npq + pqv_idx[i]] = vang[i]

    fun = lambda x: resfun_wrapper(
        x, vmag, vang, Pinj, Qinj, psys.ybus_mat, bus_type,
        pq_idx, pqv_idx, psys.graph_mat,
    )
    jac = lambda x: jac_wrapper(
        x, vmag, vang, Pinj, Qinj, psys.ybus_mat, bus_type,
        pq_idx, pqv_idx, psys.graph_mat,
    )

    if verbose:
        logger.info(
            "[Power Flow] Initial residual norm: %.6e",
            np.linalg.norm(fun(x0)),
        )

    sol, info = nonlin_solve(
        fun, x0, jacobian=jac, full_output=True, f_tol=1e-9,
    )

    if not info["success"]:
        logger.error(info["message"])
        raise Exception("Power flow solution did not converge")

    for i in range(psys.nbuses):
        if pq_idx[i] >= 0:
            vmag[i] = sol[pq_idx[i]]
        if pqv_idx[i] >= 0:
            vang[i] = sol[npq + pqv_idx[i]]

    residual_norm = float(np.linalg.norm(fun(sol)))
    if verbose:
        logger.info("[Power Flow] Final residual norm: %.6e", residual_norm)
        logger.info("Power flow pass %d converged.", pass_idx)

    return vmag, vang, residual_norm


def _solved_power_injections(psys, vmag, vang):
    """Return solved bus injections and implied total generator dispatch."""
    voltage = np.array([vmag, vang]).T.flatten()
    sinj = np.zeros(2*psys.nbuses)
    compute_pinj_alt(
        voltage, sinj, psys.ybus_mat, psys.graph_mat, psys.nbuses,
    )

    sgen = np.copy(sinj)
    for load in psys.loads:
        sgen[2*load.bus] += load.pload
        sgen[2*load.bus + 1] -= load.qload
    return voltage, sinj, sgen


def _reactive_injections(psys, q_gen):
    """Build the specified bus Q-injection vector for a Newton pass."""
    qinj = np.zeros(psys.nbuses, dtype=float)
    for gen_index, gen in enumerate(psys.gens):
        qinj[gen.bus] += q_gen[gen_index]
    for load in psys.loads:
        qinj[load.bus] += load.qload
    return qinj


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _generator_limit_diagnostics(psys, values, lower, upper, tolerance, top_n=10):
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower_violation = np.maximum(lower - values, 0.0)
    upper_violation = np.maximum(values - upper, 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    violating = np.where(violation > tolerance)[0]
    ordered = sorted(violating, key=lambda index: violation[index], reverse=True)

    top = []
    for gen_index in ordered[:top_n]:
        gen = psys.gens[int(gen_index)]
        bus = psys.buses[gen.bus]
        side = (
            "lower"
            if lower_violation[gen_index] > upper_violation[gen_index]
            else "upper"
        )
        top.append({
            "gen_index": int(gen_index),
            "gen_id": str(gen.idx),
            "bus_index": int(gen.bus),
            "bus_id": _json_value(bus.id),
            "value": float(values[gen_index]),
            "lower": float(lower[gen_index]),
            "upper": float(upper[gen_index]),
            "violation": float(violation[gen_index]),
            "side": side,
            "is_slack": bool(bus.type == Bus.SLACK),
        })

    return {
        "violation_count": int(violating.size),
        "violation_max": float(np.max(violation)) if violation.size else 0.0,
        "violation_total_abs": (
            float(np.sum(violation[violating])) if violating.size else 0.0
        ),
        "violation_top": top,
    }


def compute_branch_loading_diagnostics(psys, pf_solution, top_n=10):
    """Return JSON-safe branch apparent-power loading diagnostics."""
    voltages = pf_solution.v_magnitudes * np.exp(1j * pf_solution.v_angles)
    records = []

    for branch_index, branch in enumerate(psys.branches):
        rate = float(getattr(branch, "rateA", 0.0))
        if rate <= 0.0:
            continue

        tap = float(branch.tap)
        shift = float(branch.shift)
        if tap > 0.0:
            tpsh = tap * np.exp(1j * np.pi / 180.0 * shift)
        else:
            tap = 1.0
            tpsh = 1.0

        impedance = branch.r + 1j * branch.x
        if abs(impedance) <= 1e-12:
            continue
        y_series = 1.0 / impedance
        y_shunt = 1j * 0.5 * branch.sh
        v_from = voltages[branch.fr]
        v_to = voltages[branch.to]
        if not np.isfinite(v_from) or not np.isfinite(v_to):
            continue
        i_from = ((y_series + y_shunt) / (tap * tap)) * v_from - (
            y_series / np.conj(tpsh)
        ) * v_to
        i_to = -(y_series / tpsh) * v_from + (y_series + y_shunt) * v_to
        s_from_mva = abs(v_from * np.conj(i_from)) * psys.basemva
        s_to_mva = abs(v_to * np.conj(i_to)) * psys.basemva
        loading = max(s_from_mva, s_to_mva) / rate
        records.append({
            "branch_index": int(branch_index),
            "from_bus_index": int(branch.fr),
            "to_bus_index": int(branch.to),
            "from_bus_id": _json_value(psys.buses[branch.fr].id),
            "to_bus_id": _json_value(psys.buses[branch.to].id),
            "rateA_mva": rate,
            "s_from_mva": float(s_from_mva),
            "s_to_mva": float(s_to_mva),
            "loading": float(loading),
        })

    records.sort(key=lambda record: record["loading"], reverse=True)
    return {
        "available": bool(records),
        "rated_branch_count": len(records),
        "loading_max": records[0]["loading"] if records else None,
        "loading_argmax": records[0]["branch_index"] if records else None,
        "loading_above_one_count": int(
            sum(record["loading"] > 1.0 for record in records)
        ),
        "loading_top": records[:top_n],
    }


def _island_slack_diagnostics(psys):
    graph = nx.Graph()
    graph.add_nodes_from(range(psys.nbuses))
    graph.add_edges_from((branch.fr, branch.to) for branch in psys.branches)

    islands = []
    invalid = 0
    for island_index, component in enumerate(nx.connected_components(graph)):
        buses = sorted(int(bus) for bus in component)
        slack_buses = [
            bus for bus in buses if psys.buses[bus].type == Bus.SLACK
        ]
        if len(slack_buses) != 1:
            invalid += 1
        islands.append({
            "island_index": int(island_index),
            "bus_count": len(buses),
            "bus_index_sample": buses[:10],
            "slack_bus_count": len(slack_buses),
            "slack_bus_indices": slack_buses,
            "slack_bus_ids": [
                _json_value(psys.buses[bus].id) for bus in slack_buses
            ],
        })

    return {
        "island_count": len(islands),
        "invalid_island_count": invalid,
        "islands": islands,
    }


def _active_set_consistency_diagnostics(psys, pf_solution, tolerance):
    latest_events = {
        int(event["bus_index"]): event for event in pf_solution.q_limit_events
    }
    violations = []
    for bus_index, event in latest_events.items():
        vm = float(pf_solution.v_magnitudes[bus_index])
        vset = float(psys.buses[bus_index].v0m)
        side = event["side"]
        if side == "upper":
            violation = max(vm - vset, 0.0)
        else:
            violation = max(vset - vm, 0.0)
        if violation > tolerance:
            violations.append({
                "bus_index": bus_index,
                "bus_id": _json_value(psys.buses[bus_index].id),
                "side": side,
                "voltage": vm,
                "voltage_setpoint": vset,
                "violation": float(violation),
            })

    violations.sort(key=lambda record: record["violation"], reverse=True)
    return {
        "checked_bus_count": len(latest_events),
        "violation_count": len(violations),
        "violation_max": (
            violations[0]["violation"] if violations else 0.0
        ),
        "violation_top": violations[:10],
    }


def _slack_margin_diagnostics(psys, pf_solution):
    records = []
    for gen_index, gen in enumerate(psys.gens):
        if psys.buses[gen.bus].type != Bus.SLACK:
            continue
        p = float(pf_solution.gen_psch[gen_index])
        q = float(pf_solution.gen_qsch[gen_index])
        records.append({
            "gen_index": int(gen_index),
            "gen_id": str(gen.idx),
            "bus_index": int(gen.bus),
            "bus_id": _json_value(psys.buses[gen.bus].id),
            "p": p,
            "pmin": float(gen.pglb),
            "pmax": float(gen.pgub),
            "p_lower_margin": p - float(gen.pglb),
            "p_upper_margin": float(gen.pgub) - p,
            "q": q,
            "qmin": float(gen.qglb),
            "qmax": float(gen.qgub),
            "q_lower_margin": q - float(gen.qglb),
            "q_upper_margin": float(gen.qgub) - q,
        })
    return records


def validate_power_flow_solution(
        psys,
        pf_solution,
        *,
        residual_tolerance=1e-8,
        generator_limit_tolerance=1e-6,
        voltage_min=None,
        voltage_max=None,
        branch_loading_max=None,
        branch_limit_tolerance=1e-5,
        active_set_voltage_tolerance=1e-6,
):
    """Validate a solved operating point without modifying it."""
    tolerances = {
        "residual_tolerance": residual_tolerance,
        "generator_limit_tolerance": generator_limit_tolerance,
        "branch_limit_tolerance": branch_limit_tolerance,
        "active_set_voltage_tolerance": active_set_voltage_tolerance,
    }
    for name, value in tolerances.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if (
        voltage_min is not None
        and voltage_max is not None
        and voltage_min > voltage_max
    ):
        raise ValueError("voltage_min must not exceed voltage_max")
    if branch_loading_max is not None and branch_loading_max <= 0.0:
        raise ValueError("branch_loading_max must be positive")

    failure_reasons = []

    def fail(reason):
        if reason not in failure_reasons:
            failure_reasons.append(reason)

    residual_norm = pf_solution.residual_norm
    residual_valid = (
        residual_norm is not None
        and np.isfinite(residual_norm)
        and residual_norm <= residual_tolerance
    )
    if not residual_valid:
        fail("pf_residual")

    finite_voltage = bool(
        np.all(np.isfinite(pf_solution.v_magnitudes))
        and np.all(np.isfinite(pf_solution.v_angles))
    )
    if not finite_voltage:
        fail("nonfinite_voltage")

    p_lower = np.array([gen.pglb for gen in psys.gens], dtype=float)
    p_upper = np.array([gen.pgub for gen in psys.gens], dtype=float)
    q_lower = np.array([gen.qglb for gen in psys.gens], dtype=float)
    q_upper = np.array([gen.qgub for gen in psys.gens], dtype=float)
    p_limits = _generator_limit_diagnostics(
        psys,
        pf_solution.gen_psch,
        p_lower,
        p_upper,
        generator_limit_tolerance,
    )
    q_limits = _generator_limit_diagnostics(
        psys,
        pf_solution.gen_qsch,
        q_lower,
        q_upper,
        generator_limit_tolerance,
    )
    if p_limits["violation_count"]:
        fail("gen_p_limit")
    if q_limits["violation_count"]:
        fail("gen_q_limit")

    finite_magnitudes = pf_solution.v_magnitudes[
        np.isfinite(pf_solution.v_magnitudes)
    ]
    solved_voltage_min = (
        float(np.min(finite_magnitudes)) if finite_magnitudes.size else None
    )
    solved_voltage_max = (
        float(np.max(finite_magnitudes)) if finite_magnitudes.size else None
    )
    if (
        voltage_min is not None
        and solved_voltage_min is not None
        and solved_voltage_min < voltage_min
    ):
        fail("voltage_low")
    if (
        voltage_max is not None
        and solved_voltage_max is not None
        and solved_voltage_max > voltage_max
    ):
        fail("voltage_high")

    branch = compute_branch_loading_diagnostics(psys, pf_solution)
    if (
        branch_loading_max is not None
        and branch["loading_max"] is not None
        and branch["loading_max"] > branch_loading_max + branch_limit_tolerance
    ):
        fail("branch_overload")

    active_set = _active_set_consistency_diagnostics(
        psys, pf_solution, active_set_voltage_tolerance,
    )
    if active_set["violation_count"]:
        fail("active_set_inconsistent")

    island_slack = _island_slack_diagnostics(psys)
    if island_slack["invalid_island_count"]:
        fail("invalid_slack_topology")

    diagnostics = {
        "valid": not failure_reasons,
        "failure_reasons": failure_reasons,
        "residual_norm": (
            float(residual_norm)
            if residual_norm is not None and np.isfinite(residual_norm)
            else None
        ),
        "residual_tolerance": float(residual_tolerance),
        "finite_voltage": finite_voltage,
        "voltage_min": solved_voltage_min,
        "voltage_max": solved_voltage_max,
        "voltage_lower_bound": (
            float(voltage_min) if voltage_min is not None else None
        ),
        "voltage_upper_bound": (
            float(voltage_max) if voltage_max is not None else None
        ),
        "generator_limit_tolerance": float(generator_limit_tolerance),
        "gen_p": p_limits,
        "gen_q": q_limits,
        "slack_generators": _slack_margin_diagnostics(psys, pf_solution),
        "branch": {
            **branch,
            "loading_limit": (
                float(branch_loading_max)
                if branch_loading_max is not None
                else None
            ),
            "limit_tolerance": float(branch_limit_tolerance),
        },
        "active_set": {
            **active_set,
            "voltage_tolerance": float(active_set_voltage_tolerance),
        },
        "island_slack": island_slack,
    }
    return diagnostics


def runpf(
        psys,
        verbose=False,
        enforce_q_limits=True,
        q_limit_tolerance=1e-8,
        max_q_limit_iterations=None,
):
    """Solve AC power flow, enforcing non-slack PV generator Q limits by default.

    Q-limit enforcement uses an outer active set. After each Newton solve, the
    required aggregate generator Q at every PV bus is compared with the summed
    generator limits. A violated bus is projected to the corresponding limits,
    switched to PQ, and solved again. Feasible multi-generator buses retain PV
    control and use bounded equal sharing of their aggregate reactive output.

    The input ``psys`` bus types and generator schedules are not modified.
    """
    if q_limit_tolerance < 0.0:
        raise ValueError("q_limit_tolerance must be non-negative")

    # Slack  (1) variables: p, q. parameters: vmag, vang.
    # PV gen (2) variables: q, vang. parameters: P, vmag.
    # PQ load (3) variables: vmag, vang. parameters: P, Q.

    # We create vectors
    # vmag: voltage magnitude vector (buses 1 to n)
    # vang: voltage angle vector (buses 1 to n)
    # x0: vector of unknowns

    original_bus_type = np.zeros(psys.nbuses, dtype=np.int64)
    vmag = np.zeros(psys.nbuses, dtype=float)
    vang = np.zeros(psys.nbuses, dtype=float)
    Pinj = np.zeros(psys.nbuses, dtype=float)

    for i in range(psys.nbuses):
        vmag[i] = psys.buses[i].v0m
        vang[i] = psys.buses[i].v0a
        original_bus_type[i] = psys.buses[i].type

    for gen in psys.gens:
        Pinj[gen.bus] += gen.psch

    for load in psys.loads:
        Pinj[load.bus] -= load.pload
    bus_to_gen = psys.create_bus_to_gen_map()
    working_bus_type = np.copy(original_bus_type)
    q_dispatch = np.array([gen.qsch for gen in psys.gens], dtype=float)
    qinj = _reactive_injections(psys, q_dispatch)
    q_limit_events = []

    if max_q_limit_iterations is None:
        max_q_limit_iterations = int(np.sum(original_bus_type == Bus.PV)) + 1
    if max_q_limit_iterations < 1:
        raise ValueError("max_q_limit_iterations must be at least 1")

    pass_idx = 0
    residual_norm = None
    while True:
        pass_idx += 1
        vmag, vang, residual_norm = _solve_power_flow_once(
            psys,
            working_bus_type,
            vmag,
            vang,
            Pinj,
            qinj,
            verbose,
            pass_idx,
        )
        voltage, sinj_solved, sgen_dispatch = _solved_power_injections(
            psys, vmag, vang,
        )

        if not enforce_q_limits:
            break

        switched = False
        for bus_index in np.where(working_bus_type == Bus.PV)[0]:
            gen_indices = bus_to_gen[bus_index]
            if not gen_indices:
                raise ValueError(
                    f"PV bus {psys.buses[bus_index].id} with no generator"
                )

            lower = np.array(
                [psys.gens[index].qglb for index in gen_indices],
                dtype=float,
            )
            upper = np.array(
                [psys.gens[index].qgub for index in gen_indices],
                dtype=float,
            )
            if np.any(lower > upper):
                raise ValueError(
                    f"Reactive-power bounds are invalid at bus "
                    f"{psys.buses[bus_index].id}"
                )

            q_required = float(sgen_dispatch[2*bus_index + 1])
            lower_sum = float(np.sum(lower))
            upper_sum = float(np.sum(upper))
            side = None
            if q_required > upper_sum + q_limit_tolerance:
                side = "upper"
                fixed_q = upper
            elif q_required < lower_sum - q_limit_tolerance:
                side = "lower"
                fixed_q = lower
            else:
                fixed_q = _project_reactive_dispatch(
                    q_required,
                    lower,
                    upper,
                    tolerance=q_limit_tolerance,
                )

            q_dispatch[gen_indices] = fixed_q
            if side is None:
                continue

            working_bus_type[bus_index] = Bus.PQ
            switched = True
            q_limit_events.append({
                "pass": pass_idx,
                "bus_index": int(bus_index),
                "bus_id": psys.buses[bus_index].id,
                "side": side,
                "q_required": q_required,
                "q_limit": upper_sum if side == "upper" else lower_sum,
                "generator_indices": [int(index) for index in gen_indices],
                "generator_q": [float(value) for value in fixed_q],
            })

        if not switched:
            break
        if pass_idx >= max_q_limit_iterations:
            raise RuntimeError(
                "Reactive-power active set did not stabilize within "
                f"{max_q_limit_iterations} Newton solves"
            )
        qinj = _reactive_injections(psys, q_dispatch)

    pf_solution = PowerFlowSolution(psys.nbuses, psys.ngens)
    pf_solution.v_magnitudes = np.copy(vmag)
    pf_solution.v_angles = np.copy(vang)
    pf_solution.v_vector = voltage
    pf_solution.s_inj_vector = sinj_solved
    pf_solution.bus_types = np.copy(working_bus_type)
    pf_solution.q_limit_enforced = bool(enforce_q_limits)
    pf_solution.q_limit_iterations = pass_idx
    pf_solution.q_limit_events = q_limit_events
    pf_solution.residual_norm = residual_norm

    for gen_index, gen in enumerate(psys.gens):
        pf_solution.gen_psch[gen_index] = gen.psch
        pf_solution.gen_qsch[gen_index] = gen.qsch

    switched_pv_buses = {
        event["bus_index"] for event in q_limit_events
    }
    for bus_index, bus_obj in enumerate(psys.buses):
        gen_indices = bus_to_gen[bus_index]
        if bus_obj.type in (Bus.PV, Bus.SLACK) and not gen_indices:
            raise ValueError(f"Generator bus {bus_obj.id} with no generator")

        if bus_obj.type == Bus.SLACK:
            p_share = sgen_dispatch[2*bus_index] / len(gen_indices)
            q_share = sgen_dispatch[2*bus_index + 1] / len(gen_indices)
            for gen_index in gen_indices:
                pf_solution.gen_psch[gen_index] = p_share
                pf_solution.gen_qsch[gen_index] = q_share
        elif bus_index in switched_pv_buses:
            pf_solution.gen_qsch[gen_indices] = q_dispatch[gen_indices]
        elif bus_obj.type == Bus.PV:
            q_required = float(sgen_dispatch[2*bus_index + 1])
            if enforce_q_limits:
                lower = np.array(
                    [psys.gens[index].qglb for index in gen_indices],
                    dtype=float,
                )
                upper = np.array(
                    [psys.gens[index].qgub for index in gen_indices],
                    dtype=float,
                )
                q_values = _project_reactive_dispatch(
                    q_required,
                    lower,
                    upper,
                    tolerance=q_limit_tolerance,
                )
            else:
                q_values = np.full(
                    len(gen_indices), q_required / len(gen_indices),
                )
            pf_solution.gen_qsch[gen_indices] = q_values

    return pf_solution
