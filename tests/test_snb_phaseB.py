import numpy as np
import pytest

from uqgrid.core.psydef import Psystem
from uqgrid.simulation.pflow import jac_wrapper
from uqgrid.snb import (
    ClosestSNBResult,
    build_fixed_injections,
    build_index_cache,
    closest_snb_fsolve,
    build_param_selector,
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


def _jacobian_at(psys, cache, x, lambda_vec):
    p_fixed, q_fixed = build_fixed_injections(psys, cache)
    p_load, q_load = scatter_lambda(lambda_vec, psys, cache)
    pinj = p_fixed + p_load
    qinj = q_fixed + q_load

    vmag = np.array([bus.v0m for bus in psys.buses])
    vang = np.array([bus.v0a for bus in psys.buses])

    return jac_wrapper(
        x,
        vmag,
        vang,
        pinj,
        qinj,
        psys.ybus_mat,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        psys.graph_mat,
    )


def test_kkt_properties(two_bus_psys):
    cache = build_index_cache(two_bus_psys)
    selector = build_param_selector(cache)
    result = closest_snb_fsolve(two_bus_psys)

    assert isinstance(result, ClosestSNBResult)
    assert result.diagnostics.ier == 1

    jac = _jacobian_at(two_bus_psys, cache, result.x_star, result.lambda_star)
    jac_dense = jac.toarray()
    assert np.max(np.abs(result.w_star @ jac_dense)) < 1e-7

    normal_direct = np.asarray(selector.transpose().dot(result.w_star)).ravel()
    assert np.allclose(result.normal, normal_direct, atol=1e-8)

    delta_lambda = result.lambda_star - result.lambda0
    assert np.allclose(delta_lambda, result.k_star * result.normal, atol=1e-6)
    assert delta_lambda @ result.normal >= -1e-9
    assert np.max(np.abs(delta_lambda - result.k_star * normal_direct)) < 1e-6
    assert np.isclose(result.w_star.sum(), 1.0, atol=1e-6)

    assert result.angle < 1e-6


def test_two_bus_regression(two_bus_psys):
    cache = build_index_cache(two_bus_psys)
    result = closest_snb_fsolve(two_bus_psys)

    p_load, q_load = unpack_params(result.lambda_star, cache)
    load_vec = np.array([p_load[0], q_load[0]])

    expected = np.array([0.703, 0.877])
    assert np.linalg.norm(load_vec - expected) < 5e-3
    assert result.angle < 1e-3

    constraint = load_vec[0] ** 2 + 4 * load_vec[1] - 4
    assert abs(constraint) < 5e-3

    lambda_reflect = 2 * result.lambda0 - result.lambda_star
    alt_load = np.array([lambda_reflect[0], lambda_reflect[1]])
    base_load = np.array([result.lambda0[0], result.lambda0[1]])
    assert np.linalg.norm(load_vec - base_load) <= np.linalg.norm(alt_load - base_load)

    distance = np.linalg.norm(result.lambda_star - result.lambda0)
    assert np.isclose(result.distance, distance)
