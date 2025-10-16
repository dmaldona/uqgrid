from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from uqgrid.core.psydef import Psystem
from uqgrid.simulation.pflow import (PowerFlowSolution, jac_wrapper,
                                     resfun_wrapper, runpf)

from .indexing import PFIndexCache, build_index_cache


@dataclass
class PFFunctions:
    """Callable residual and Jacobian builders for a fixed power system."""

    residual: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], csr_matrix]
    index_cache: PFIndexCache
    pinj: np.ndarray
    qinj: np.ndarray
    vmag: np.ndarray
    vang: np.ndarray


def _ensure_structures(psys: Psystem) -> None:
    if psys.assembled != 1:
        psys.assemble()
    if not hasattr(psys, "ybus_mat"):
        psys.createYbusComplex()


def _build_injections(psys: Psystem) -> tuple[np.ndarray, np.ndarray]:
    pinj = np.zeros(psys.nbuses, dtype=float)
    qinj = np.zeros(psys.nbuses, dtype=float)

    for gen in psys.gens:
        pinj[gen.bus] += gen.psch
        qinj[gen.bus] += gen.qsch

    for load in psys.loads:
        pinj[load.bus] -= load.pload
        qinj[load.bus] += load.qload

    return pinj, qinj


def build_pf_operators(psys: Psystem) -> PFFunctions:
    """Return residual/Jacobian callable pair aligned with the PF unknown ordering."""

    _ensure_structures(psys)

    index_cache = build_index_cache(psys)
    pinj, qinj = _build_injections(psys)

    vmag = np.array([bus.v0m for bus in psys.buses], dtype=float)
    vang = np.array([bus.v0a for bus in psys.buses], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        if x.shape[0] != index_cache.n_unknowns:
            raise ValueError("Unexpected state vector length for residual evaluation.")
        return resfun_wrapper(
            x,
            vmag.copy(),
            vang.copy(),
            pinj.copy(),
            qinj.copy(),
            psys.ybus_mat,
            index_cache.bus_type,
            index_cache.pq_indices,
            index_cache.pqv_indices,
            psys.graph_mat,
        )

    def jacobian(x: np.ndarray) -> csr_matrix:
        return jac_wrapper(
            x,
            vmag.copy(),
            vang.copy(),
            pinj.copy(),
            qinj.copy(),
            psys.ybus_mat,
            index_cache.bus_type,
            index_cache.pq_indices,
            index_cache.pqv_indices,
            psys.graph_mat,
        )

    return PFFunctions(
        residual=residual,
        jacobian=jacobian,
        index_cache=index_cache,
        pinj=pinj,
        qinj=qinj,
        vmag=vmag,
        vang=vang,
    )


def solution_to_state_vector(psys: Psystem, solution: PowerFlowSolution, cache: PFIndexCache) -> np.ndarray:
    """Map a PowerFlowSolution to the reduced state vector ordering."""

    x = np.zeros(cache.n_unknowns, dtype=float)

    for bus_idx in cache.pq_buses:
        slot = cache.pq_indices[bus_idx]
        if slot >= 0:
            x[slot] = solution.v_magnitudes[bus_idx]

    for bus_idx in range(psys.nbuses):
        angle_slot = cache.pqv_indices[bus_idx]
        if angle_slot >= 0:
            x[cache.n_pq + angle_slot] = solution.v_angles[bus_idx]

    return x
