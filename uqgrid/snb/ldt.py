from __future__ import annotations

from typing import Callable, Iterable, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator, svds, lsmr

Array = np.ndarray
Mat = Union[np.ndarray, sparse.spmatrix, LinearOperator, list, tuple]


def as_linear_op_Sigma_inv(Sigma_inv: Mat, m: int) -> LinearOperator:
    """Wrap Sigma^{-1} as a scipy.sparse.linalg.LinearOperator of shape (m, m).

    Accepts:
      - 1D array of length m (interpreted as diagonal),
      - dense (m x m) ndarray,
      - sparse CSR/CSC matrix,
      - LinearOperator.
    """

    if isinstance(Sigma_inv, LinearOperator):
        return Sigma_inv

    if isinstance(Sigma_inv, (np.ndarray, list, tuple)) and np.ndim(Sigma_inv) == 1:
        d = np.asarray(Sigma_inv, dtype=float).ravel()

        def mv(v: Array) -> Array:
            return d * v

        def rmv(v: Array) -> Array:
            return d * v

        return LinearOperator((m, m), matvec=mv, rmatvec=rmv)

    if sparse.issparse(Sigma_inv):
        Sigma_inv_csr = Sigma_inv.tocsr()
        return LinearOperator(
            (m, m),
            matvec=lambda v: Sigma_inv_csr @ v,
            rmatvec=lambda v: Sigma_inv_csr.transpose() @ v,
        )

    Sigma_inv_arr = np.asarray(Sigma_inv, dtype=float)
    return LinearOperator(
        (m, m),
        matvec=lambda v: Sigma_inv_arr @ v,
        rmatvec=lambda v: Sigma_inv_arr.T @ v,
    )


def grad_I(lambda_vec: Array, mu: Array, Sigma_inv_op: LinearOperator) -> Array:
    """Return ∇_λ I = Σ^{-1} (λ - μ)."""

    return Sigma_inv_op @ (lambda_vec - mu)


def rate_I(lambda_vec: Array, mu: Array, Sigma_inv_op: LinearOperator) -> float:
    """Compute I(λ) = 1/2 (λ - μ)^T Σ^{-1} (λ - μ)."""

    delta = lambda_vec - mu
    return 0.5 * float(delta @ (Sigma_inv_op @ delta))


def first_order_ldt_prob(beta: float) -> float:
    """Evaluate first-order Gaussian LDT probability approximation.

    P_1st ≈ (1 / (sqrt(2π) * β)) * exp(-β^2 / 2). Guard β <= 0.
    """

    if beta <= 0:
        return np.inf
    return (1.0 / (np.sqrt(2.0 * np.pi) * beta)) * np.exp(-0.5 * beta * beta)


def oriented_unit_normal(delta: Array, n: Array, eps: float = 1e-15) -> tuple[Array | None, bool]:
    """Normalize and orient a normal vector relative to a displacement.

    Returns a unit-length vector (or ``None`` when ``n`` is invalid) together with
    a boolean flag indicating whether the vector was flipped to align with
    ``delta``. The returned vector satisfies ``delta · n_unit >= -eps``.
    """

    n_arr = np.asarray(n, dtype=float).ravel()
    if n_arr.size == 0 or not np.isfinite(n_arr).all():
        return None, False

    n_norm = np.linalg.norm(n_arr)
    if n_norm <= eps:
        return None, False

    delta_arr = np.asarray(delta, dtype=float).ravel()
    if delta_arr.shape != n_arr.shape:
        raise ValueError("delta and n must have the same dimension.")
    if not np.isfinite(delta_arr).all():
        return None, False

    unit = n_arr / n_norm
    dot = float(unit @ delta_arr)
    if np.isnan(dot):
        return None, False

    flipped = False
    if dot < -eps:
        unit = -unit
        dot = -dot
        flipped = True

    return unit, flipped


def _ensure_linear_operator(fx_apply: Mat | LinearOperator, n: int) -> LinearOperator:
    if isinstance(fx_apply, LinearOperator):
        if fx_apply.shape != (n, n):
            raise ValueError("LinearOperator shape does not match expected dimensions.")
        return fx_apply

    if sparse.issparse(fx_apply) or isinstance(fx_apply, (np.ndarray, list, tuple)):
        op = aslinearoperator(fx_apply)
        if op.shape != (n, n):
            raise ValueError("Operator shape does not match expected dimensions.")
        return op

    raise TypeError("fx_apply must be a LinearOperator or array-like object.")


def _compute_right_null_vector(fx_apply: Mat | LinearOperator, fx_op: LinearOperator) -> Array | None:
    n = fx_op.shape[1]
    try:
        _, _, vt = svds(fx_op, k=1, which="SM")
        vec = np.asarray(vt[0, :], dtype=float)
    except Exception:
        if sparse.issparse(fx_apply):
            mat = fx_apply.toarray()
        elif isinstance(fx_apply, np.ndarray):
            mat = fx_apply
        else:
            return None
        _, _, vt_dense = np.linalg.svd(mat, full_matrices=False)
        vec = np.asarray(vt_dense[-1, :], dtype=float)

    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return None
    return vec / norm


def _solve_with_alpha(
    fx_op: LinearOperator,
    w: Array,
    col_vec: Array,
    tol: float,
) -> tuple[Array, float]:
    n = w.size

    def matvec(z: Array) -> Array:
        x = z[:n]
        alpha = float(z[-1])
        primary = fx_op.matvec(x) - alpha * w
        gauge = float(w @ x)
        return np.concatenate([np.asarray(primary), np.array([gauge])])

    def rmatvec(y: Array) -> Array:
        y_main = np.asarray(y[:-1])
        y_gauge = float(y[-1])
        x_part = np.asarray(fx_op.rmatvec(y_main)) + y_gauge * w
        alpha_part = -float(w @ y_main)
        return np.concatenate([x_part, np.array([alpha_part])])

    aug_op = LinearOperator((n + 1, n + 1), matvec=matvec, rmatvec=rmatvec)
    b = np.concatenate([-col_vec, np.zeros(1)])
    sol, *_ = lsmr(aug_op, b, atol=tol, btol=tol, conlim=1e12)
    x = np.asarray(sol[:n])
    alpha = float(sol[-1])
    return x, alpha


def compute_x_lambda(
    fx_apply: Mat | LinearOperator,
    fx_solve: Callable[[Array], Array],
    f_lambda_cols: Iterable[Array],
    w_star: Array,
    atol: float = 1e-10,
) -> list[Array]:
    """Solve for directional sensitivities X_λ satisfying gauge and residual constraints."""

    w = np.asarray(w_star, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("w_star must have positive dimension.")
    if not np.isfinite(w).all():
        raise ValueError("w_star contains non-finite values.")

    n = int(w.size)
    fx_op = _ensure_linear_operator(fx_apply, n)

    null_vec = _compute_right_null_vector(fx_apply, fx_op)
    if null_vec is not None and np.linalg.norm(null_vec) > 0.0:
        null_vec = null_vec / np.linalg.norm(null_vec)

    results: list[Array] = []
    tolerance = max(atol, 1e-12)

    for col in f_lambda_cols:
        col_vec = np.asarray(col, dtype=float).ravel()
        if col_vec.shape != (n,):
            raise ValueError("Each f_lambda column must have the same dimension as w_star.")
        if not np.isfinite(col_vec).all():
            raise ValueError("f_lambda columns must contain finite values.")

        rhs = -col_vec

        trial = np.asarray(fx_solve(rhs), dtype=float).ravel()
        if trial.shape != (n,):
            raise ValueError("fx_solve returned an array with incorrect dimension.")

        try:
            x_candidate, alpha = _solve_with_alpha(fx_op, w, col_vec, tolerance)
        except Exception as exc:
            if null_vec is None:
                raise ValueError("Unable to solve directional system with gauge enforcement.") from exc
            w_dot_null = float(w @ null_vec)
            if abs(w_dot_null) <= 1e-14:
                raise ValueError("Right null vector nearly orthogonal to w_star; gauge fails.") from exc
            gauge_correction = -float(w @ trial) / w_dot_null
            x_candidate = trial + gauge_correction * null_vec
            alpha = float((w @ (fx_op.matvec(x_candidate) + col_vec)) / (w @ w))

        residual = fx_op.matvec(x_candidate) + col_vec
        residual_proj = residual - ((w @ residual) / (w @ w)) * w
        res_norm = float(np.linalg.norm(residual_proj))
        if res_norm > tolerance:
            raise ValueError(
                f"Directional solve residual {res_norm:.3e} exceeds tolerance {tolerance:.3e}."
            )

        gauge = float(w @ x_candidate)
        if abs(gauge) > atol:
            if null_vec is None:
                raise ValueError("Gauge condition could not be enforced within tolerance.")
            w_dot_null = float(w @ null_vec)
            if abs(w_dot_null) <= 1e-14:
                raise ValueError("Right null vector nearly orthogonal to w_star; gauge fails.")
            correction = -gauge / w_dot_null
            x_candidate = x_candidate + correction * null_vec
            gauge = float(w @ x_candidate)
            if abs(gauge) > atol:
                raise ValueError("Gauge condition could not be enforced within tolerance.")

        results.append(np.asarray(x_candidate))

    return results


def hvp_fx(
    fx_apply_x: Callable[[Array, Array], Array],
    x_star: Array,
    dir_u: Array,
    dir_v: Array,
    eps: float | None = None,
) -> Array:
    """Approximate f_{xx}(u, v) at ``x_star`` using Jacobian-vector products."""

    x0 = np.asarray(x_star, dtype=float).ravel()
    u = np.asarray(dir_u, dtype=float).ravel()
    v = np.asarray(dir_v, dtype=float).ravel()

    if x0.size == 0 or u.size != x0.size or v.size != x0.size:
        raise ValueError("Directions must match the dimension of x_star.")

    if not (np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(x0).all()):
        raise ValueError("Inputs contain non-finite values.")

    if eps is None:
        scale = max(1.0, np.linalg.norm(u))
        eps = 1e-5 / scale

    fx_forward_v = np.asarray(fx_apply_x(x0 + eps * u, v), dtype=float).ravel()
    fx_backward_v = np.asarray(fx_apply_x(x0 - eps * u, v), dtype=float).ravel()

    return (fx_forward_v - fx_backward_v) / (2.0 * eps)


def build_second_form(
    w_star: Array,
    x_lam_cols: Iterable[Array],
    fx_apply_x: Callable[[Array, Array], Array],
    x_star: Array,
    eps: float | None = None,
) -> Array:
    """Assemble the second fundamental form matrix using Hessian-vector products."""

    w = np.asarray(w_star, dtype=float).ravel()
    if w.size == 0 or not np.isfinite(w).all():
        raise ValueError("w_star must be a finite, non-empty vector.")

    cols = [np.asarray(col, dtype=float).ravel() for col in x_lam_cols]
    if not cols:
        raise ValueError("x_lam_cols must contain at least one direction.")

    x0 = np.asarray(x_star, dtype=float).ravel()

    for col in cols:
        if col.shape != x0.shape:
            raise ValueError("Each x_lambda column must match dimension of x_star.")
        if not np.isfinite(col).all():
            raise ValueError("x_lambda columns must contain only finite values.")

    m = len(cols)
    II = np.zeros((m, m), dtype=float)

    for i in range(m):
        for j in range(i, m):
            hvp = hvp_fx(fx_apply_x, x_star, cols[i], cols[j], eps=eps)
            II[i, j] = float(w @ hvp)
            if j != i:
                II[j, i] = II[i, j]

    return 0.5 * (II + II.T)


def householder_to_e1(g: Array) -> Array:
    """Return an orthogonal matrix ``R`` such that ``R.T @ g = ||g|| e1``.

    Uses a Householder reflection. Raises ``ValueError`` when ``g`` is the
    zero vector or contains non-finite entries.
    """

    g_vec = np.asarray(g, dtype=float).ravel()
    if g_vec.size == 0:
        raise ValueError("Input vector g must have positive dimension.")
    if not np.isfinite(g_vec).all():
        raise ValueError("Input vector g must contain only finite values.")

    norm_g = float(np.linalg.norm(g_vec))
    if norm_g == 0.0:
        raise ValueError("Householder reflection undefined for zero vector.")

    m = g_vec.size
    e1 = np.zeros(m, dtype=float)
    e1[0] = 1.0

    u = g_vec - norm_g * e1
    norm_u = float(np.linalg.norm(u))
    if norm_u <= 1e-15:
        return np.eye(m, dtype=float)

    v = u / norm_u
    R = np.eye(m, dtype=float) - 2.0 * np.outer(v, v)
    return R


def build_alignment(
    Sigma_inv: Mat,
    II: Array,
    N: Array,
) -> tuple[Array, Array, float]:
    """Construct alignment matrices ``S`` and ``S_perp`` for second-order LDT.

    Parameters
    ----------
    Sigma_inv
        Representation of :math:`\\Sigma^{-1}`. Accepts a 1-D diagonal array or
        a full SPD ``(m, m)`` ndarray. LinearOperator inputs currently require a
        dense fallback and are therefore rejected.
    II
        Second fundamental form matrix with shape ``(m, m)``.
    N
        Unscaled normal vector with shape ``(m,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(S, S_perp, norm_ATN)`` where ``S`` is the aligned curvature matrix,
        ``S_perp`` its sub-block excluding the aligned normal direction, and
        ``norm_ATN = ||A^T N||`` for ``A = Σ^{1/2} R``.
    """

    II_arr = np.asarray(II, dtype=float)
    if II_arr.ndim != 2 or II_arr.shape[0] != II_arr.shape[1]:
        raise ValueError("II must be a square matrix.")

    N_vec = np.asarray(N, dtype=float).ravel()
    if N_vec.size != II_arr.shape[0]:
        raise ValueError("Dimension mismatch between II and N.")
    if not np.isfinite(N_vec).all():
        raise ValueError("Normal vector N must contain only finite values.")

    m = N_vec.size

    # Resolve Sigma^{-1} structure.
    if isinstance(Sigma_inv, (list, tuple, np.ndarray, sparse.spmatrix)):
        Sigma_inv_arr = np.asarray(Sigma_inv, dtype=float)
    else:
        Sigma_inv_arr = None

    if Sigma_inv_arr is not None and Sigma_inv_arr.ndim == 1:
        if Sigma_inv_arr.size != m:
            raise ValueError("Diagonal Sigma^{-1} must have length equal to dim of N.")
        if np.any(Sigma_inv_arr <= 0.0) or not np.isfinite(Sigma_inv_arr).all():
            raise ValueError("Diagonal Sigma^{-1} entries must be positive and finite.")

        sigma_inv_sqrt = np.sqrt(Sigma_inv_arr)
        sigma_sqrt = 1.0 / sigma_inv_sqrt

        g_vec = sigma_inv_sqrt * N_vec
        R = householder_to_e1(g_vec)
        A = sigma_sqrt[:, None] * R

    elif Sigma_inv_arr is not None and Sigma_inv_arr.ndim == 2:
        if Sigma_inv_arr.shape != (m, m):
            raise ValueError("Sigma^{-1} matrix must match dimension of N.")
        if not np.allclose(Sigma_inv_arr, Sigma_inv_arr.T, atol=1e-12):
            raise ValueError("Sigma^{-1} must be symmetric positive definite.")

        try:
            chol = np.linalg.cholesky(Sigma_inv_arr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Sigma^{-1} must be positive definite.") from exc

        # chol is lower-triangular such that Sigma_inv = chol @ chol.T.
        Sigma_inv_sqrt = chol
        Sigma_sqrt = np.linalg.solve(Sigma_inv_sqrt, np.eye(m, dtype=float))

        g_vec = Sigma_inv_sqrt @ N_vec
        R = householder_to_e1(g_vec)
        A = Sigma_sqrt @ R

    else:
        raise TypeError(
            "Sigma_inv must be provided as a diagonal array or dense SPD matrix for now."
        )

    g_norm = float(np.linalg.norm(g_vec))
    if g_norm == 0.0:
        raise ValueError("Transformed normal is zero; alignment undefined.")

    AT = A.T
    ATN = AT @ N_vec
    norm_ATN = float(np.linalg.norm(ATN))
    if not np.isfinite(norm_ATN) or norm_ATN <= 0.0:
        raise ValueError("||A^T N|| must be positive and finite.")

    S_mat = (AT @ II_arr @ A) / norm_ATN
    S_mat = 0.5 * (S_mat + S_mat.T)

    S_perp = S_mat[1:, 1:].copy()
    return S_mat, S_perp, norm_ATN
