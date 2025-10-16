import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.core.psydef import Psystem
from uqgrid.snb import build_index_cache, build_pf_operators


def _build_two_bus_system():
    psys = Psystem(basemva=1.0)

    # Slack bus (type 3) and PQ bus (type 1)
    psys.add_bus(1, bus_type=3)
    psys.add_bus(2, bus_type=1)

    for bus in psys.buses:
        bus.set_vinit(1.0, 0.0)

    # Simple line between buses with reactance 0.25 pu
    psys.add_branch(0, 1, r=0.0, x=0.25)

    # Slack generator supplies load
    psys.add_gen(bus=0, idx_name="G1", psch=0.0, qsch=0.0)

    # PQ load 0.5 + j0.3 pu
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


def test_t0_mask_counts(two_bus_psys):
    cache = build_index_cache(two_bus_psys)

    assert cache.n_pq == 1
    assert cache.n_pv == 0
    assert cache.n_slack == 1

    # Slack bus should have index -1 in PQ and PQV arrays
    slack_idx = cache.slack_buses[0]
    assert cache.pq_indices[slack_idx] == -1
    assert cache.pqv_indices[slack_idx] == -1

    pq_idx = cache.pq_buses[0]
    assert cache.pq_indices[pq_idx] == 0
    assert cache.pqv_indices[pq_idx] == 0


def test_t0_jacobian_shape_and_sparsity(two_bus_psys):
    pf_ops = build_pf_operators(two_bus_psys)
    cache = pf_ops.index_cache

    x0 = _initial_state_vector(two_bus_psys, cache)

    residual = pf_ops.residual(x0)
    jac = pf_ops.jacobian(x0)

    assert residual.shape == (cache.n_unknowns,)
    assert isinstance(jac, csr_matrix)
    assert jac.shape == (cache.n_unknowns, cache.n_unknowns)

    # Ensure matrix is sparse: number of stored elements << n^2
    n = cache.n_unknowns
    if n > 3:
        assert jac.nnz <= (n * n) // 2
    else:
        assert jac.nnz <= n * n
