import numpy as np
import pytest

from tests.ldt_test_utils import build_two_bus_context

from uqgrid.snb.ldt import (
    build_alignment,
    build_second_form,
    compute_x_lambda,
    householder_to_e1,
    ldt_first_order,
    ldt_second_order,
    oriented_unit_normal,
    plane_prob,
    second_order_prefactor,
)


@pytest.fixture(scope="module")
def probability_context():
    ctx = build_two_bus_context()
    jac_star = ctx["jac"]

    def fx_solve(rhs):
        from scipy.sparse.linalg import lsmr

        sol, *_ = lsmr(jac_star, rhs, atol=1e-12, btol=1e-12, conlim=1e12)
        return sol

    x_lambda_cols = compute_x_lambda(
        ctx["jac"],
        fx_solve,
        ctx["selector_cols"],
        ctx["result"].w_star,
    )

    II = build_second_form(
        ctx["result"].w_star,
        x_lambda_cols,
        ctx["fx_apply_x"],
        ctx["result"].x_star,
    )

    selector = ctx["selector"]
    N = np.asarray(selector.transpose().dot(ctx["result"].w_star)).ravel()

    return {
        "result": ctx["result"],
        "II": II,
        "N": N,
        "lambda0": ctx["result"].lambda0,
        "lambda_star": ctx["result"].lambda_star,
    }


def _beta_from(lambda_star: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> float:
    diff = np.asarray(lambda_star - mu, dtype=float)
    if Sigma_inv.ndim == 1:
        return float(np.sqrt(np.sum(Sigma_inv * diff * diff)))
    return float(np.sqrt(diff @ (np.asarray(Sigma_inv, dtype=float) @ diff)))


def test_p6_probability_mc_ordering(probability_context):
    data = probability_context
    lambda_star = data["lambda_star"]
    lambda0 = data["lambda0"]
    N = data["N"]

    Sigma = np.array([1.0, 0.4], dtype=float)
    Sigma_inv = 1.0 / Sigma

    n_unit, _ = oriented_unit_normal(lambda_star - lambda0, N)
    assert n_unit is not None

    target_beta = 2.33
    q = float(np.sqrt(np.sum(Sigma_inv * (n_unit ** 2))))
    mu = lambda_star - (target_beta / q) * n_unit
    beta = _beta_from(lambda_star, mu, Sigma_inv)
    assert np.isclose(beta, target_beta, atol=1e-8)

    S, S_perp, _ = build_alignment(Sigma_inv, data["II"], N)
    C, _ = second_order_prefactor(S_perp, beta)
    p_plane = plane_prob(beta)
    p_first = ldt_first_order(beta)
    assert not np.isnan(p_first)
    p_second = ldt_second_order(beta, C)

    rng = np.random.default_rng(2024)
    std = np.sqrt(Sigma)
    eps = 0.20
    z = 2.0
    denom = max(p_plane, 1e-12)
    n_req = int(np.ceil((z * np.sqrt(denom * (1 - denom)) / (eps * denom)) ** 2))
    n_samples = max(60000, n_req)
    samples = rng.standard_normal((n_samples, 2)) * std + mu

    delta_star = (lambda_star - mu).astype(float)
    if Sigma_inv.ndim == 1:
        n_mc = Sigma_inv * delta_star
    else:
        n_mc = np.asarray(Sigma_inv, dtype=float) @ delta_star
    norm_mc = float(np.linalg.norm(n_mc))
    assert norm_mc > 0.0
    n_mc = n_mc / norm_mc

    delta_samples = samples - lambda_star
    signed = np.dot(delta_samples, n_mc)
    p_plane_mc = float(np.mean(signed >= 0.0))
    assert p_plane_mc > 0.0

    rel_plane = abs(p_plane_mc - p_plane) / max(p_plane, 1e-15)
    assert rel_plane <= 0.25

    if Sigma_inv.ndim == 1:
        Sigma_inv_sqrt_mat = np.diag(np.sqrt(Sigma_inv))
    else:
        Sigma_inv_sqrt_mat = np.linalg.cholesky(np.asarray(Sigma_inv, dtype=float))
    g_vec = Sigma_inv_sqrt_mat @ N
    R = householder_to_e1(g_vec)
    A_inv = Sigma_inv_sqrt_mat @ R.T
    norm_atn = np.linalg.norm(A_inv.T @ N)
    y = np.einsum("nj,jk->nk", delta_samples, A_inv) / norm_atn
    y0 = y[:, 0]
    if y.shape[1] > 1:
        y_perp = y[:, 1:]
        quad_term = y0 + 0.5 * np.einsum("ni,ij,nj->n", y_perp, S_perp, y_perp)
    else:
        quad_term = y0
    p_quad_mc = float(np.mean(quad_term >= 0.0))
    assert p_quad_mc > 0.0

    assert abs(p_quad_mc - p_second) < abs(p_quad_mc - p_first)


def test_p6_first_order_gating(probability_context):
    data = probability_context
    lambda_star = data["lambda_star"]
    lambda0 = data["lambda0"]
    N = data["N"]

    Sigma_inv = np.ones_like(N, dtype=float)
    Sigma = np.ones_like(N, dtype=float)

    n_unit, _ = oriented_unit_normal(lambda_star - lambda0, N)
    assert n_unit is not None

    target_beta = 0.58
    q = float(np.sqrt(np.sum(Sigma_inv * (n_unit ** 2))))
    mu = lambda_star - (target_beta / q) * n_unit
    beta = _beta_from(lambda_star, mu, Sigma_inv)
    assert beta < 1.0

    p_plane = plane_prob(beta)
    assert 0.0 < p_plane < 0.4

    p_first = ldt_first_order(beta)
    assert np.isnan(p_first)

    S, S_perp, _ = build_alignment(Sigma_inv, data["II"], N)
    C, _ = second_order_prefactor(S_perp, beta)
    p_second = ldt_second_order(beta, C)

    assert np.isfinite(p_second)
    assert p_second > 0.0
