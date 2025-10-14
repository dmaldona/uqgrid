import numpy as np
import pytest

from uqgrid.snb import (
    build_index_cache,
    build_param_selector,
    closest_snb_fsolve,
    extract_lambda,
)
from uqgrid.snb.ldt import as_linear_op_Sigma_inv, grad_I, oriented_unit_normal

from tests.ldt_test_utils import build_two_bus_system


def _compute_normal(result, selector):
    normal = selector.transpose().dot(result.w_star)
    return np.asarray(normal).ravel()


def test_oriented_unit_normal_aligns_with_delta_lambda():
    psys = build_two_bus_system()
    cache = build_index_cache(psys)
    selector = build_param_selector(cache)

    result = closest_snb_fsolve(psys)
    delta_lambda = result.lambda_star - result.lambda0
    normal = -_compute_normal(result, selector)  # Flip sign to exercise orientation logic.

    unit, flipped = oriented_unit_normal(delta_lambda, normal)

    assert unit is not None
    assert np.isfinite(unit).all()
    assert np.isclose(np.linalg.norm(unit), 1.0, atol=1e-12)
    assert unit @ delta_lambda >= -1e-14
    assert flipped


def test_oriented_unit_normal_aligns_with_gaussian_gradient():
    psys = build_two_bus_system()
    cache = build_index_cache(psys)
    selector = build_param_selector(cache)

    lambda0 = extract_lambda(psys, cache)
    mu = lambda0 + np.array([-0.1, 0.05])
    sigma_inv = np.array([1.0, 4.0])

    result = closest_snb_fsolve(psys, mu=mu, Sigma_inv=sigma_inv)
    normal = -_compute_normal(result, selector)

    op = as_linear_op_Sigma_inv(sigma_inv, lambda0.shape[0])
    grad = grad_I(result.lambda_star, mu, op)

    unit, flipped = oriented_unit_normal(grad, normal)

    assert unit is not None
    assert np.isfinite(unit).all()
    assert np.isclose(np.linalg.norm(unit), 1.0, atol=1e-12)
    assert unit @ grad >= -1e-14
    assert flipped


def test_oriented_unit_normal_rejects_degenerate_inputs():
    unit, flipped = oriented_unit_normal(np.zeros(2), np.zeros(2))
    assert unit is None
    assert flipped is False

    unit, flipped = oriented_unit_normal(np.array([1.0, np.nan]), np.ones(2))
    assert unit is None
    assert flipped is False

    with pytest.raises(ValueError):
        oriented_unit_normal(np.ones(3), np.ones(2))
