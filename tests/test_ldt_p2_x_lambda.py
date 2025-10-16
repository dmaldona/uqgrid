import numpy as np
import pytest
from scipy.sparse.linalg import lsmr, aslinearoperator

from tests.ldt_test_utils import build_two_bus_context

from uqgrid.snb.ldt import compute_x_lambda
from uqgrid.snb.params import scatter_lambda
from uqgrid.simulation.pflow import resfun_wrapper


@pytest.fixture(scope="module")
def two_bus_context():
    ctx = build_two_bus_context()
    jac_star = ctx["jac"]

    def fx_solve(rhs):
        sol, *_ = lsmr(jac_star, rhs, atol=1e-12, btol=1e-12, conlim=1e12)
        return sol

    ctx = dict(ctx)
    ctx["fx_solve"] = fx_solve
    return ctx


def _project_residual(residual, w):
    w_norm_sq = float(w @ w)
    if w_norm_sq == 0.0:
        raise ValueError("Left-null vector has zero norm.")
    return residual - ((w @ residual) / w_norm_sq) * w


def test_p2_residual(two_bus_context):
    ctx = two_bus_context
    x_lambda_cols = compute_x_lambda(
        ctx["jac"],
        ctx["fx_solve"],
        ctx["selector_cols"],
        ctx["result"].w_star,
    )

    jac_op = aslinearoperator(ctx["jac"])
    w = ctx["result"].w_star
    for x_lam, col in zip(x_lambda_cols, ctx["selector_cols"]):
        residual = jac_op.matvec(x_lam) + col
        residual_proj = _project_residual(residual, w)
        assert np.linalg.norm(residual_proj) <= 1e-8


def test_p2_gauge(two_bus_context):
    ctx = two_bus_context
    x_lambda_cols = compute_x_lambda(
        ctx["jac"],
        ctx["fx_solve"],
        ctx["selector_cols"],
        ctx["result"].w_star,
    )

    w = ctx["result"].w_star
    for x_lam in x_lambda_cols:
        assert abs(float(w @ x_lam)) <= 1e-10


def test_p2_finite_difference(two_bus_context):
    ctx = two_bus_context
    result = ctx["result"]
    x_lambda_cols = compute_x_lambda(
        ctx["jac"],
        ctx["fx_solve"],
        ctx["selector_cols"],
        result.w_star,
    )

    jac_op = aslinearoperator(ctx["jac"])
    lambda_star = result.lambda_star
    eps = 1e-6

    for j, (x_lam, col) in enumerate(zip(x_lambda_cols, ctx["selector_cols"])):
        pert = np.zeros_like(lambda_star)
        pert[j] = eps
        lambda_plus = lambda_star + pert

        p_load_plus, q_load_plus = scatter_lambda(lambda_plus, ctx["psys"], ctx["cache"])
        pinj_plus = ctx["p_fixed"] + p_load_plus
        qinj_plus = ctx["q_fixed"] + q_load_plus

        x_trial = result.x_star + eps * x_lam

        F_trial = resfun_wrapper(
            x_trial,
            ctx["vmag_base"].copy(),
            ctx["vang_base"].copy(),
            pinj_plus.copy(),
            qinj_plus.copy(),
            ctx["ybus"],
            ctx["cache"].bus_type,
            ctx["cache"].pq_indices,
            ctx["cache"].pqv_indices,
            ctx["graph"],
        )

        residual = jac_op.matvec(x_lam) + col
        approx_error = F_trial / eps - residual
        assert np.linalg.norm(approx_error) <= 1e-4
