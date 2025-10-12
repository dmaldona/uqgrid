from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix

from uqgrid.core.psydef import Psystem
from uqgrid.simulation.pflow import jac_wrapper, resfun_wrapper, runpf

from .indexing import PFIndexCache, build_index_cache
from .nullspace import normalize_left_vector, smallest_left_singular_vector
from .params import build_fixed_injections, extract_lambda, scatter_lambda
from .pf import solution_to_state_vector
from .selectors import build_param_selector


@dataclass
class SolverDiagnostics:
    nfev: int
    ier: int
    message: str
    sigma: float


@dataclass
class ClosestSNBResult:
    x_star: np.ndarray
    lambda_star: np.ndarray
    w_star: np.ndarray
    k_star: float
    normal: np.ndarray
    distance: float
    angle: float
    sigma_min: float
    diagnostics: SolverDiagnostics
    lambda0: np.ndarray
    kkt_residuals: Dict[str, float]
    metadata: Dict[str, Any]


def _prepare_context(psys: Psystem) -> tuple[Psystem, PFIndexCache, csr_matrix, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = build_index_cache(psys)
    selector = build_param_selector(cache)

    pf_solution = runpf(psys, verbose=False)
    x0 = solution_to_state_vector(psys, pf_solution, cache)

    lambda0 = extract_lambda(psys, cache)

    vmag_base = pf_solution.v_magnitudes.copy()
    vang_base = pf_solution.v_angles.copy()

    p_fixed, q_fixed = build_fixed_injections(psys, cache)

    ybus = psys.ybus_mat
    graph = psys.graph_mat

    return (
        psys,
        cache,
        selector,
        x0,
        lambda0,
        vmag_base,
        vang_base,
        p_fixed,
        q_fixed,
        ybus,
        graph,
    )


def closest_snb_fsolve(
    psys: Psystem,
    *,
    alpha: float = 1e-3,
    c_vector: Optional[np.ndarray] = None,
    fsolve_kwargs: Optional[Dict[str, float]] = None,
    x_init: Optional[np.ndarray] = None,
    w_init: Optional[np.ndarray] = None,
    lambda_init: Optional[np.ndarray] = None,
    k_init: Optional[float] = None,
) -> ClosestSNBResult:
    (
        psys,
        cache,
        selector,
        x0,
        lambda0,
        vmag_base,
        vang_base,
        p_fixed,
        q_fixed,
        ybus,
        graph,
    ) = _prepare_context(psys)

    n_x = cache.n_unknowns
    n_lambda = 2 * cache.n_pq

    if n_lambda == 0:
        raise ValueError("System has no PQ loads; lambda space is empty.")

    # Jacobian at base point
    p_load0, q_load0 = scatter_lambda(lambda0, psys, cache)
    pinj0 = p_fixed + p_load0
    qinj0 = q_fixed + q_load0

    jac0 = jac_wrapper(
        x0,
        vmag_base.copy(),
        vang_base.copy(),
        pinj0.copy(),
        qinj0.copy(),
        ybus,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        graph,
    )

    c_vec = np.ones(n_x, dtype=float) if c_vector is None else np.asarray(c_vector, dtype=float)
    if c_vec.shape[0] != n_x:
        raise ValueError("Normalization vector length must match number of state variables.")

    if w_init is not None:
        w0 = np.asarray(w_init, dtype=float).copy()
        if w0.shape[0] != n_x:
            raise ValueError("Initial left-null guess has incorrect dimension.")
        w0 = normalize_left_vector(w0, c_vec)
    else:
        _, w0 = smallest_left_singular_vector(jac0)
        w0 = normalize_left_vector(w0, c_vec)

    normal0 = np.asarray(selector.transpose().dot(w0)).ravel()

    if lambda_init is not None:
        lambda_start = np.asarray(lambda_init, dtype=float)
        if lambda_start.shape[0] != n_lambda:
            raise ValueError("Initial lambda guess has incorrect dimension.")
    else:
        lambda_start = lambda0 + alpha * normal0

    if k_init is not None:
        k_start = float(k_init)
    else:
        k_start = max(alpha, 1.0)

    if x_init is not None:
        x_start = np.asarray(x_init, dtype=float)
        if x_start.shape[0] != n_x:
            raise ValueError("Initial state guess has incorrect dimension.")
    else:
        x_start = x0

    z0 = np.concatenate([x_start, lambda_start, w0, np.array([k_start])])

    def residual(z: np.ndarray) -> np.ndarray:
        x = z[:n_x]
        lam = z[n_x:n_x + n_lambda]
        w = z[n_x + n_lambda:n_x + n_lambda + n_x]
        k = float(z[-1])

        p_load, q_load = scatter_lambda(lam, psys, cache)
        pinj = p_fixed + p_load
        qinj = q_fixed + q_load

        F = resfun_wrapper(
            x,
            vmag_base.copy(),
            vang_base.copy(),
            pinj.copy(),
            qinj.copy(),
            ybus,
            cache.bus_type,
            cache.pq_indices,
            cache.pqv_indices,
            graph,
        )

        J = jac_wrapper(
            x,
            vmag_base.copy(),
            vang_base.copy(),
            pinj.copy(),
            qinj.copy(),
            ybus,
            cache.bus_type,
            cache.pq_indices,
            cache.pqv_indices,
            graph,
        )

        delta_lambda = lam - lambda0
        normal = np.asarray(selector.transpose().dot(w)).ravel()
        eq_pf = F
        eq_left = J.transpose().dot(w)
        eq_stationarity = delta_lambda - k * normal
        eq_normalization = np.array([w @ c_vec - 1.0])

        return np.concatenate([eq_pf, np.asarray(eq_left).ravel(), eq_stationarity, eq_normalization])

    fs_kwargs = dict(maxfev=500, xtol=1e-9)
    if fsolve_kwargs:
        fs_kwargs.update(fsolve_kwargs)

    z_star, info, ier, mesg = fsolve(residual, z0, full_output=True, **fs_kwargs)

    x_star = z_star[:n_x]
    lambda_star = z_star[n_x:n_x + n_lambda]
    w_star = z_star[n_x + n_lambda:n_x + n_lambda + n_x]
    k_star = float(z_star[-1])

    p_load_star, q_load_star = scatter_lambda(lambda_star, psys, cache)
    pinj_star = p_fixed + p_load_star
    qinj_star = q_fixed + q_load_star

    jac_star = jac_wrapper(
        x_star,
        vmag_base.copy(),
        vang_base.copy(),
        pinj_star.copy(),
        qinj_star.copy(),
        ybus,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        graph,
    )
    sigma_star, _ = smallest_left_singular_vector(jac_star)

    delta_lambda = lambda_star - lambda0
    normal = np.asarray(selector.transpose().dot(w_star)).ravel()

    distance = float(np.linalg.norm(delta_lambda))

    norm_normal = np.linalg.norm(normal)
    norm_delta = np.linalg.norm(delta_lambda)
    if norm_normal < 1e-14 or norm_delta < 1e-14:
        angle = 0.0
    else:
        cos_theta = float(np.clip((delta_lambda @ normal) / (norm_normal * norm_delta), -1.0, 1.0))
        angle = float(np.arccos(cos_theta))

    eq_pf_star = resfun_wrapper(
        x_star,
        vmag_base.copy(),
        vang_base.copy(),
        pinj_star.copy(),
        qinj_star.copy(),
        ybus,
        cache.bus_type,
        cache.pq_indices,
        cache.pqv_indices,
        graph,
    )
    left_null_star = jac_star.transpose().dot(w_star)
    stationarity_star = delta_lambda - k_star * normal
    normalization_star = float(w_star @ c_vec - 1.0)

    diagnostics = SolverDiagnostics(
        nfev=info.get("nfev", 0),
        ier=ier,
        message=mesg,
        sigma=sigma_star,
    )

    kkt_residuals = {
        "pf": float(np.linalg.norm(eq_pf_star, ord=np.inf)),
        "left_null": float(np.linalg.norm(left_null_star, ord=np.inf)),
        "stationarity": float(np.linalg.norm(stationarity_star, ord=np.inf)),
        "normalization": abs(normalization_star),
    }

    metadata: Dict[str, Any] = {
        "nfev": diagnostics.nfev,
        "ier": diagnostics.ier,
        "message": diagnostics.message,
    }

    return ClosestSNBResult(
        x_star=x_star,
        lambda_star=lambda_star,
        w_star=w_star,
        k_star=k_star,
        normal=normal,
        distance=distance,
        angle=angle,
        sigma_min=sigma_star,
        diagnostics=diagnostics,
        lambda0=lambda0,
        kkt_residuals=kkt_residuals,
        metadata=metadata,
    )
