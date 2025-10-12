import numpy as np
import pytest

from uqgrid.core.psydef import Psystem
from uqgrid.simulation.pflow import resfun_wrapper
from uqgrid.snb import (
    build_index_cache,
    build_param_selector,
    build_pf_operators,
    extract_lambda,
    pack_params,
    scatter_lambda,
    unpack_params,
)


def _build_two_bus_system():
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


@pytest.fixture
def two_bus_psys():
    return _build_two_bus_system()


def _initial_state_vector(psys, cache):
    x = np.zeros(cache.n_unknowns)
    for bus_idx in cache.pq_buses:
        slot = cache.pq_indices[bus_idx]
        x[slot] = psys.buses[bus_idx].v0m
    for bus_idx in range(psys.nbuses):
        idx = cache.pqv_indices[bus_idx]
        if idx >= 0:
            x[cache.n_pq + idx] = psys.buses[bus_idx].v0a
    return x


def test_pack_unpack_lambda(two_bus_psys):
    cache = build_index_cache(two_bus_psys)
    p = np.array([0.5])
    q = np.array([0.3])

    lam = pack_params(p, q, cache)
    assert lam.shape == (2 * cache.n_pq,)

    p_out, q_out = unpack_params(lam, cache)
    assert np.allclose(p, p_out)
    assert np.allclose(q, q_out)

    lam_current = extract_lambda(two_bus_psys, cache)
    assert np.allclose(lam_current, lam)


def test_f_lambda_finite_difference(two_bus_psys):
    pf_ops = build_pf_operators(two_bus_psys)
    cache = pf_ops.index_cache
    selector = build_param_selector(cache)

    x0 = _initial_state_vector(two_bus_psys, cache)
    base_F = resfun_wrapper(
        x0,
        pf_ops.vmag.copy(),
        pf_ops.vang.copy(),
        pf_ops.pinj.copy(),
        pf_ops.qinj.copy(),
        two_bus_psys.ybus_mat,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        two_bus_psys.graph_mat,
    )

    h = 1e-5
    # Perturb lambda directly (active load component)
    lambda_base = extract_lambda(two_bus_psys, cache)
    p_load_base, q_load_base = scatter_lambda(lambda_base, two_bus_psys, cache)

    lambda_p = lambda_base.copy()
    lambda_p[0] += h
    p_load_p, q_load_p = scatter_lambda(lambda_p, two_bus_psys, cache)
    pinj_p = pf_ops.pinj.copy()
    qinj_p = pf_ops.qinj.copy()
    pinj_p += p_load_p - p_load_base
    qinj_p += q_load_p - q_load_base

    F_p = resfun_wrapper(
        x0,
        pf_ops.vmag.copy(),
        pf_ops.vang.copy(),
        pinj_p,
        qinj_p,
        two_bus_psys.ybus_mat,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        two_bus_psys.graph_mat,
    )

    fd_p = (F_p - base_F) / h
    selector_p = selector[:, 0].toarray().ravel()
    assert np.allclose(fd_p, selector_p, atol=1e-6)

    # Perturb lambda reactive component
    lambda_q = lambda_base.copy()
    lambda_q[cache.n_pq] += h
    p_load_q, q_load_q = scatter_lambda(lambda_q, two_bus_psys, cache)
    pinj_q = pf_ops.pinj.copy()
    qinj_q = pf_ops.qinj.copy()
    pinj_q += p_load_q - p_load_base
    qinj_q += q_load_q - q_load_base

    F_q = resfun_wrapper(
        x0,
        pf_ops.vmag.copy(),
        pf_ops.vang.copy(),
        pinj_q,
        qinj_q,
        two_bus_psys.ybus_mat,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        two_bus_psys.graph_mat,
    )

    fd_q = (F_q - base_F) / h
    selector_q = selector[:, cache.n_pq].toarray().ravel()
    assert np.allclose(fd_q, selector_q, atol=1e-6)

    # Check structure
    assert np.isclose(selector_q[:cache.n_pq], 1.0).all()
    assert np.isclose(selector_p[:cache.n_pq], 0.0).all()
