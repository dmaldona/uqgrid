from __future__ import annotations

import numpy as np

from uqgrid.core.psydef import Psystem
from uqgrid.snb import build_index_cache, build_param_selector, closest_snb_fsolve
from uqgrid.snb.params import scatter_lambda
from uqgrid.snb.solver import _prepare_context
from uqgrid.simulation.pflow import jac_wrapper


def build_two_bus_system() -> Psystem:
    psys = Psystem(basemva=1.0)
    psys.add_bus(1, 3)
    psys.add_bus(2, 1)

    for bus in psys.buses:
        bus.set_vinit(1.0, 0.0)

    psys.add_branch(0, 1, r=0.0, x=0.25)
    psys.add_gen(bus=0, idx_name="G1", psch=0.0, qsch=0.0)
    psys.add_load(bus=1, tag="LD1", pload=0.5, qload=0.3)

    psys.assemble()
    psys.createYbusComplex()
    return psys


def build_two_bus_context():
    psys = build_two_bus_system()
    result = closest_snb_fsolve(psys)

    (
        psys_prepped,
        cache,
        selector,
        _x0,
        _lambda0,
        vmag_base,
        vang_base,
        p_fixed,
        q_fixed,
        ybus,
        graph,
    ) = _prepare_context(psys)

    p_load_star, q_load_star = scatter_lambda(result.lambda_star, psys_prepped, cache)
    pinj_star = p_fixed + p_load_star
    qinj_star = q_fixed + q_load_star

    jac_star = jac_wrapper(
        result.x_star,
        vmag_base.copy(),
        vang_base.copy(),
        pinj_star.copy(),
        qinj_star.copy(),
        ybus,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        graph,
    )

    selector_cols = [selector.getcol(j).toarray().ravel() for j in range(selector.shape[1])]

    def fx_apply_x(x_vec: np.ndarray, direction: np.ndarray) -> np.ndarray:
        jac = jac_wrapper(
            np.asarray(x_vec, dtype=float),
            vmag_base.copy(),
            vang_base.copy(),
            pinj_star.copy(),
            qinj_star.copy(),
            ybus,
            cache.bus_type,
            cache.pq_indices,
            cache.pqv_indices,
            graph,
        )
        return np.asarray(jac.dot(np.asarray(direction, dtype=float))).ravel()

    return {
        "psys": psys_prepped,
        "result": result,
        "cache": cache,
        "selector": selector,
        "selector_cols": selector_cols,
        "jac": jac_star,
        "vmag_base": vmag_base,
        "vang_base": vang_base,
        "p_fixed": p_fixed,
        "q_fixed": q_fixed,
        "ybus": ybus,
        "graph": graph,
        "pinj_star": pinj_star,
        "qinj_star": qinj_star,
        "fx_apply_x": fx_apply_x,
        "c_vec": np.ones(cache.n_unknowns, dtype=float),
    }
