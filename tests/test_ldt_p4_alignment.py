import numpy as np
import pytest

from tests.ldt_test_utils import build_two_bus_context

from uqgrid.snb.ldt import build_alignment, build_second_form, compute_x_lambda, householder_to_e1


@pytest.fixture(scope="module")
def alignment_context():
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

    Sigma_inv = np.ones_like(N)

    return {
        "ctx": ctx,
        "x_lambda_cols": x_lambda_cols,
        "II": II,
        "N": N,
        "Sigma_inv": Sigma_inv,
    }


def test_p4_householder_orthogonality(alignment_context):
    data = alignment_context
    N = data["N"]
    Sigma_inv = data["Sigma_inv"]

    g = np.sqrt(Sigma_inv) * N
    R = householder_to_e1(g)

    eye = np.eye(R.shape[0])
    ortho = R.T @ R
    assert np.linalg.norm(ortho - eye, ord=2) <= 1e-12

    aligned = R.T @ g
    assert np.allclose(aligned[1:], 0.0, atol=1e-12)
    assert aligned[0] > 0


def test_p4_alignment_outputs(alignment_context):
    data = alignment_context
    S, S_perp, norm_atn = build_alignment(
        data["Sigma_inv"],
        data["II"],
        data["N"],
    )

    assert S.shape == data["II"].shape
    assert S_perp.shape == (S.shape[0] - 1, S.shape[1] - 1)
    assert norm_atn > 0.0

    Sigma_inv = data["Sigma_inv"]
    N = data["N"]
    g = np.sqrt(Sigma_inv) * N
    R = householder_to_e1(g)
    A = (1.0 / np.sqrt(Sigma_inv))[:, None] * R

    ATN = A.T @ N
    assert np.linalg.norm(ATN[1:]) <= 1e-12
    assert ATN[0] > 0

    S_expected = (A.T @ data["II"] @ A) / np.linalg.norm(ATN)
    S_expected = 0.5 * (S_expected + S_expected.T)

    assert np.allclose(S, S_expected, atol=1e-12)
    assert np.allclose(S_perp, S_expected[1:, 1:], atol=1e-12)
