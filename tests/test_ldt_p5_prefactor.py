import numpy as np
import pytest

from tests.ldt_test_utils import build_two_bus_context

from uqgrid.snb.ldt import build_alignment, build_second_form, compute_x_lambda, second_order_prefactor


@pytest.fixture(scope="module")
def prefactor_context():
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
        "II": II,
        "N": N,
    }


def test_p5_prefactor_flat_boundary():
    S_perp = np.zeros((2, 2), dtype=float)
    C, evals = second_order_prefactor(S_perp, beta=2.0)
    assert np.isclose(C, 1.0, atol=1e-12)
    assert np.allclose(evals, 0.0)


def test_p5_prefactor_stability(prefactor_context):
    data = prefactor_context
    Sigma_inv = np.array([3.0, 0.5], dtype=float)
    S, S_perp, _ = build_alignment(Sigma_inv, data["II"], data["N"])

    evals = np.linalg.eigvalsh(S_perp)
    beta = min(0.5, 0.4 / (np.max(np.abs(evals)) + 1e-12))

    C, evals_returned = second_order_prefactor(S_perp, beta)
    assert np.allclose(evals_returned, evals)
    margin = 1.0 - beta * evals_returned
    assert np.min(margin) > 1e-3
    assert C > 0.0
    assert np.isfinite(C)