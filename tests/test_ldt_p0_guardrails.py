import numpy as np

from uqgrid.snb import (
    build_index_cache,
    build_param_selector,
    closest_snb_fsolve,
    extract_lambda,
)

from tests.ldt_test_utils import build_two_bus_system


def _assert_guardrails(result, selector):
    max_kkt = max(result.kkt_residuals.values())
    assert max_kkt <= 1e-6
    assert result.sigma_min <= 1e-8

    normal = selector.transpose().dot(result.w_star)
    normal = np.asarray(normal).ravel()
    assert np.isfinite(normal).all()
    assert np.linalg.norm(normal) > 1e-12


def test_guardrails_two_bus_euclidean():
    psys = build_two_bus_system()
    cache = build_index_cache(psys)
    selector = build_param_selector(cache)

    result = closest_snb_fsolve(psys)
    _assert_guardrails(result, selector)


def test_guardrails_two_bus_sigma_identity():
    psys = build_two_bus_system()
    cache = build_index_cache(psys)
    selector = build_param_selector(cache)

    lambda0 = extract_lambda(psys, cache)
    sigma_inv = np.ones_like(lambda0)

    result = closest_snb_fsolve(psys, mu=lambda0, Sigma_inv=sigma_inv)
    _assert_guardrails(result, selector)
