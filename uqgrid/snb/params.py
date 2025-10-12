from __future__ import annotations

import numpy as np

from uqgrid.core.psydef import Psystem

from .indexing import PFIndexCache


def pack_params(p_load: np.ndarray, q_load: np.ndarray, cache: PFIndexCache) -> np.ndarray:
    """Stack PQ bus load injections into the lambda vector order."""

    if p_load.shape[0] != cache.n_pq or q_load.shape[0] != cache.n_pq:
        raise ValueError("Load vectors must match number of PQ buses.")
    return np.concatenate([p_load, q_load])


def unpack_params(lambda_vec: np.ndarray, cache: PFIndexCache) -> tuple[np.ndarray, np.ndarray]:
    """Split the lambda vector back into (P, Q) load arrays."""

    expected = 2 * cache.n_pq
    if lambda_vec.shape[0] != expected:
        raise ValueError(f"lambda vector length {lambda_vec.shape[0]} does not match expected {expected}.")

    midpoint = cache.n_pq
    p = lambda_vec[:midpoint].copy()
    q = lambda_vec[midpoint:].copy()
    return p, q


def extract_lambda(psys: Psystem, cache: PFIndexCache) -> np.ndarray:
    """Aggregate current PQ-bus injections into the lambda ordering."""

    p = np.zeros(cache.n_pq, dtype=float)
    q = np.zeros(cache.n_pq, dtype=float)

    for local_idx, bus_idx in enumerate(cache.pq_buses):
        p_total = 0.0
        q_total = 0.0
        for load in psys.loads:
            if load.bus == bus_idx:
                p_total += load.pload
                q_total += load.qload
        p[local_idx] = p_total
        q[local_idx] = q_total

    return pack_params(p, q, cache)


def build_fixed_injections(psys: Psystem, cache: PFIndexCache) -> tuple[np.ndarray, np.ndarray]:
    """Return net injections excluding PQ-load parameters."""

    p_fixed = np.zeros(psys.nbuses, dtype=float)
    q_fixed = np.zeros(psys.nbuses, dtype=float)

    pq_set = set(cache.pq_buses.tolist())

    for gen in psys.gens:
        p_fixed[gen.bus] += gen.psch
        q_fixed[gen.bus] += gen.qsch

    for load in psys.loads:
        if load.bus in pq_set:
            continue
        p_fixed[load.bus] -= load.pload
        q_fixed[load.bus] += load.qload

    return p_fixed, q_fixed


def scatter_lambda(lambda_vec: np.ndarray, psys: Psystem, cache: PFIndexCache) -> tuple[np.ndarray, np.ndarray]:
    """Expand a lambda vector into per-bus injection arrays."""

    p_vec, q_vec = unpack_params(lambda_vec, cache)
    p_load = np.zeros(psys.nbuses, dtype=float)
    q_load = np.zeros(psys.nbuses, dtype=float)

    for local_idx, bus_idx in enumerate(cache.pq_buses):
        p_load[bus_idx] = -p_vec[local_idx]
        q_load[bus_idx] = -q_vec[local_idx]

    return p_load, q_load
