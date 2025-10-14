import numpy as np
import pytest
from scipy.sparse.linalg import lsmr

from tests.ldt_test_utils import build_two_bus_context

from uqgrid.snb.ldt import build_second_form, compute_x_lambda
from uqgrid.snb.nullspace import normalize_left_vector, smallest_left_singular_vector
from uqgrid.snb.params import scatter_lambda
from uqgrid.simulation.pflow import jac_wrapper, resfun_wrapper


@pytest.fixture(scope="module")
def p3_context():
    ctx = build_two_bus_context()
    jac_star = ctx["jac"]

    def fx_solve(rhs):
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

    context = dict(ctx)
    context.update({
        "fx_solve": fx_solve,
        "x_lambda_cols": x_lambda_cols,
        "II": II,
    })
    return context


def _newton_refine_state(x_guess, lambda_vec, ctx, tol=1e-12, max_iter=4):
    cache = ctx["cache"]
    lambda_arr = np.asarray(lambda_vec, dtype=float).ravel()
    p_load, q_load = scatter_lambda(lambda_arr, ctx["psys"], cache)
    pinj = ctx["p_fixed"] + p_load
    qinj = ctx["q_fixed"] + q_load

    x_curr = np.asarray(x_guess, dtype=float).ravel()

    for _ in range(max_iter):
        F_curr = resfun_wrapper(
            x_curr,
            ctx["vmag_base"].copy(),
            ctx["vang_base"].copy(),
            pinj.copy(),
            qinj.copy(),
            ctx["ybus"],
            cache.bus_type,
            cache.pq_indices,
            cache.pqv_indices,
            ctx["graph"],
        )

        if np.linalg.norm(F_curr) <= tol:
            break

        jac_curr = jac_wrapper(
            x_curr,
            ctx["vmag_base"].copy(),
            ctx["vang_base"].copy(),
            pinj.copy(),
            qinj.copy(),
            ctx["ybus"],
            cache.bus_type,
            cache.pq_indices,
            cache.pqv_indices,
            ctx["graph"],
        )

        dx, *_ = lsmr(jac_curr, -F_curr, atol=1e-12, btol=1e-12, conlim=1e12)
        x_curr = x_curr + np.asarray(dx, dtype=float)

    jac_final = jac_wrapper(
        x_curr,
        ctx["vmag_base"].copy(),
        ctx["vang_base"].copy(),
        pinj.copy(),
        qinj.copy(),
        ctx["ybus"],
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        ctx["graph"],
    )

    return x_curr, jac_final


def test_p3_symmetry(p3_context):
    II = p3_context["II"]
    asym = np.max(np.abs(II - II.T))
    assert asym <= 1e-8


def test_p3_normal_second_difference(p3_context):
    ctx = p3_context
    selector = ctx["selector"]
    cache = ctx["cache"]
    c_vec = ctx["c_vec"]
    result = ctx["result"]
    x_lambda_cols = ctx["x_lambda_cols"]
    II = ctx["II"]

    selector_t = selector.transpose()
    selector_dense = selector_t.toarray() if hasattr(selector_t, "toarray") else np.asarray(selector_t)

    N_star = np.asarray(selector_dense.dot(result.w_star)).ravel()
    n_norm = np.linalg.norm(N_star)
    assert n_norm > 0.0
    n_hat = N_star / n_norm

    P_tan = np.eye(N_star.size) - np.outer(n_hat, n_hat)
    eps = 5e-6

    x_lambda_mat = np.column_stack(x_lambda_cols)
    rng = np.random.default_rng(42)
    checks_run = 0

    for _ in range(max(3, N_star.size)):
        d_raw = rng.standard_normal(N_star.size)
        d = P_tan @ d_raw
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            continue
        d /= d_norm

        lambda_plus = result.lambda_star + eps * d
        lambda_minus = result.lambda_star - eps * d

        x_dir = x_lambda_mat @ d
        x_plus_guess = result.x_star + eps * x_dir
        x_minus_guess = result.x_star - eps * x_dir

        x_plus, jac_plus = _newton_refine_state(x_plus_guess, lambda_plus, ctx)
        x_minus, jac_minus = _newton_refine_state(x_minus_guess, lambda_minus, ctx)

        _, w_plus = smallest_left_singular_vector(jac_plus)
        _, w_minus = smallest_left_singular_vector(jac_minus)

        w_plus = normalize_left_vector(w_plus, c_vec)
        w_minus = normalize_left_vector(w_minus, c_vec)

        if (w_plus @ result.w_star) < 0:
            w_plus = -w_plus
        if (w_minus @ result.w_star) < 0:
            w_minus = -w_minus

        N_plus = np.asarray(selector_dense.dot(w_plus)).ravel()
        N_minus = np.asarray(selector_dense.dot(w_minus)).ravel()

        r_fd = (N_plus - N_minus) / (2.0 * eps)
        II_d = II @ d

        r_fd = P_tan @ r_fd
        II_d = P_tan @ II_d

        r_norm = float(np.linalg.norm(r_fd))
        ii_norm = float(np.linalg.norm(II_d))
        assert r_norm > 0.0
        assert ii_norm > 0.0

        if (r_fd @ II_d) < 0.0:
            r_fd = -r_fd

        cosine = float((r_fd @ II_d) / (r_norm * ii_norm))
        rel_err = float(np.linalg.norm(r_fd - II_d) / max(1.0, ii_norm))

        assert cosine >= 0.95
        assert rel_err <= 0.25

        checks_run += 1

    assert checks_run > 0
